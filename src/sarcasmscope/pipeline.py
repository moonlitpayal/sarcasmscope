"""End-to-end experiment pipeline: load -> audit -> leakage-safe splits -> train -> evaluate -> report.

Protocol guard: by default everything is evaluated on the VALIDATION split. Passing final=True evaluates on
the TEST split and writes FINAL_LOCK.json; a second final run is refused unless force=True. This enforces
"touch the test set once".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .bundle import ModelBundle
from .data.audit import audit, audit_markdown
from .data.loaders import load_files
from .data.splits import make_split
from .eval.metrics import bootstrap_ci, binary_metrics, per_group
from .eval.robustness import robustness_table
from .eval.shortcuts import shortcut_report
from .eval.stats import mcnemar, paired_bootstrap
from .eval.transfer import family_summary, transfer_matrix
from .models.calibration import (TemperatureScaler, aurc, crossfit_temperature, expected_calibration_error,
                                 reliability_table, threshold_for_selective_accuracy, abstain_mask)
from .models.registry import make_model, safe_name
from .models.tfidf import TfidfLR
from .text.normalize import normalize

DEFAULT_CONFIG = {
    "run_name": "my_run",
    "output_dir": "runs",
    "seeds": [42, 43, 44],
    "data": {"files": [{
        "path": "data/raw/my_dataset.csv", "source": "D1",
        "mapping": {"text": "text", "label": "label", "lang": "lang"},
        "lang": None, "label_map": None}]},
    "preprocess": {"lowercase": True, "keep_emoji": True, "hashtags": "strip_symbol"},
    "split": {"ratios": [0.7, 0.15, 0.15], "near_dup_threshold": 0.7, "group_col": None, "dedup": "drop"},
    "models": ["tfidf"],
    "transformer": {"epochs": 4, "lr": 3.0e-5, "batch_size": 16, "max_len": 96},
    "reference_model": "tfidf",
    "abstain": {"target_selective_accuracy": 0.9},
    "shortcut_columns": [],
    "use_context": False,
    "robustness": True,
    "transfer": True,
}


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write(df: pd.DataFrame, base: Path, name: str):
    base.mkdir(parents=True, exist_ok=True)
    df.to_csv(base / f"{name}.csv", index=False)
    (base / f"{name}.md").write_text(df_to_md(df), encoding="utf-8")


def _inputs(frame: pd.DataFrame, spec: str, use_ctx: bool):
    """(texts, contexts) for a model. TF-IDF gets 'context || text'; transformers get sentence pairs."""
    texts = list(frame["text_norm"])
    if not use_ctx:
        return texts, None
    ctx = list(frame["context_norm"])
    if spec == "tfidf":
        return [f"{c} || {t}" if c else t for c, t in zip(ctx, texts)], None
    return texts, ctx


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in user.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) and k != "data" else v
    return cfg


def run(cfg: dict, final: bool = False, force: bool = False, base_dir: str | Path = ".", log=print) -> dict:
    run_dir = Path(cfg.get("output_dir", "runs")) / cfg.get("run_name", "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "FINAL_LOCK.json"
    if final and lock.exists() and not force:
        raise RuntimeError(f"The test set was already used for this run ({lock}). Evaluating it again is test-set "
                           f"peeking. Use a new run_name, or pass force=True only if you accept that.")
    eval_split = "test" if final else "val"
    t0 = time.time()

    # 1. data -------------------------------------------------------------------------------------
    df, load_reports = load_files(cfg["data"]["files"], base_dir)
    pre = cfg.get("preprocess", {})
    df["text_norm"] = [normalize(t, **pre) for t in df["text"]]
    use_ctx = bool(cfg.get("use_context", False))
    if use_ctx:
        df["context_norm"] = [normalize(c, **pre) if c else "" for c in df["context"]]
        if not (df["context_norm"] != "").any():
            raise ValueError("use_context is true but no row has a context. Map a context column in data.files[].mapping.")
    log(f"[1/7] loaded {len(df)} rows from {len(load_reports)} file(s)")

    # 2. audit ------------------------------------------------------------------------------------
    sp = cfg.get("split", {})
    aud = audit(df, near_dup_threshold=sp.get("near_dup_threshold", 0.7))
    (run_dir / "audit.json").write_text(json.dumps(aud, indent=2, default=str), encoding="utf-8")
    (run_dir / "audit.md").write_text(audit_markdown(aud), encoding="utf-8")
    log(f"[2/7] audit done: {len(aud['flags'])} flag(s)  -> {run_dir / 'audit.md'}")

    # 3-5. splits, training, evaluation ---------------------------------------------------------
    seeds, specs = cfg.get("seeds", [42]), cfg.get("models", ["tfidf"])
    reference = cfg.get("reference_model", specs[0])
    target_acc = cfg.get("abstain", {}).get("target_selective_accuracy", 0.9)
    rows, preds_store, first = [], {}, {}
    per_lang_rows, cal_rows, split_reports = [], [], []
    for seed in seeds:
        sdf, rep = make_split(df, seed=seed, ratios=tuple(sp.get("ratios", (0.7, 0.15, 0.15))),
                              group_col=sp.get("group_col"), near_dup_threshold=sp.get("near_dup_threshold", 0.7),
                              dedup=sp.get("dedup", "drop"))
        split_reports.append(rep)
        (run_dir / "splits").mkdir(exist_ok=True)
        sdf[["id", "split", "cluster"]].to_csv(run_dir / "splits" / f"seed_{seed}.csv", index=False)
        tr, va = sdf[sdf.split == "train"], sdf[sdf.split == "val"]
        ev = sdf[sdf.split == eval_split]
        log(f"[3/7] seed {seed}: train {len(tr)} | val {len(va)} | {eval_split} {len(ev)} (evaluating on {eval_split.upper()})")
        if rep["warning"]:
            log("      WARNING: " + rep["warning"])
        if rep["dedup"] == "keep":
            log(f"      note: dedup=keep, {rep['distinct_texts_in_multiple_splits']} distinct texts appear in more than one split")
        for spec in specs:
            tcfg = dict(cfg.get("transformer") or {})
            if use_ctx:
                tcfg["use_context"] = True
            model = make_model(spec, seed, tcfg)
            X_tr, C_tr = _inputs(tr, spec, use_ctx)
            X_va, C_va = _inputs(va, spec, use_ctx)
            X_ev, C_ev = _inputs(ev, spec, use_ctx)
            if spec == "tfidf":
                model.fit(X_tr, tr["label"].values)
            else:
                model.fit(X_tr, tr["label"].values, val_texts=X_va, val_y=va["label"].values,
                          contexts=C_tr, val_contexts=C_va)
            p_val = model.predict_proba(X_va, C_va)[:, 1]
            p_ev = p_val if not final else model.predict_proba(X_ev, C_ev)[:, 1]
            T = TemperatureScaler().fit(p_val, va["label"].values).T
            p_cal = TemperatureScaler(T).transform(p_ev) if final else crossfit_temperature(p_val, va["label"].values, seed)
            tau, cov_val = threshold_for_selective_accuracy(TemperatureScaler(T).transform(p_val), va["label"].values, target_acc)

            y = ev["label"].values
            pred = (p_ev >= 0.5).astype(int)
            m = binary_metrics(y, pred)
            lo, hi = bootstrap_ci(y, pred, seed=seed)
            row = {"model": spec, "seed": seed, "eval_split": eval_split, "n_eval": len(ev),
                   "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                   "macro_f1": m["macro_f1"], "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
                   "ece_raw": expected_calibration_error(p_ev, y), "ece_calibrated": expected_calibration_error(p_cal, y),
                   "temperature": T, "aurc_calibrated": aurc(p_cal, y)}
            if final and tau is not None:
                keep = ~abstain_mask(p_cal, tau)
                row["abstain_threshold"] = tau
                row["coverage_at_threshold"] = float(keep.mean())
                row["selective_accuracy_at_threshold"] = float((pred[keep] == y[keep]).mean()) if keep.any() else np.nan
            rows.append(row)
            pl = per_group(ev.assign(pred=pred), "pred", "lang")
            pl.insert(0, "seed", seed); pl.insert(0, "model", spec)
            per_lang_rows.append(pl)
            rel = reliability_table(p_cal, y); rel.insert(0, "seed", seed); rel.insert(0, "model", spec)
            cal_rows.append(rel)
            preds_store[(spec, seed)] = pd.DataFrame({"id": ev["id"].values, "label": y, "p_raw": p_ev, "p_cal": p_cal})
            (run_dir / "predictions").mkdir(exist_ok=True)
            preds_store[(spec, seed)].to_csv(run_dir / "predictions" / f"{safe_name(spec)}_seed{seed}_{eval_split}.csv", index=False)
            (run_dir / "bundles").mkdir(exist_ok=True)
            ModelBundle(model, T, tau, {"name": f"{spec} seed {seed}", "trained_on": [r["source"] for r in load_reports],
                                        "run": cfg.get("run_name"), "seed": seed}).save(
                str(run_dir / "bundles" / (f"{safe_name(spec)}_seed{seed}.joblib" if spec == "tfidf" else f"{safe_name(spec)}_seed{seed}")))
            if seed == seeds[0]:
                first[spec] = {"model": model, "tr": tr, "ev": ev, "sdf": sdf}
            log(f"      {spec:>10s} seed {seed}: macro-F1 {m['macro_f1']:.4f} (95% CI {lo:.3f}-{hi:.3f}), "
                f"ECE {row['ece_raw']:.3f} -> {row['ece_calibrated']:.3f}")

    res = pd.DataFrame(rows)
    tables = run_dir / "tables"
    _write(res, tables, "all_runs")
    summ = res.groupby("model").agg(
        runs=("seed", "size"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), f1_sarcastic_mean=("f1", "mean"),
        ece_raw_mean=("ece_raw", "mean"), ece_calibrated_mean=("ece_calibrated", "mean")).reset_index()
    _write(summ, tables, "main_results")
    pl_all = pd.concat(per_lang_rows, ignore_index=True)
    pl_sum = pl_all.groupby(["model", "lang"]).agg(n=("n", "first"), macro_f1_mean=("macro_f1", "mean"),
                                                  macro_f1_std=("macro_f1", "std"), share_sarcastic=("share_sarcastic", "first")).reset_index()
    _write(pl_sum, tables, "per_language")
    _write(pd.concat(cal_rows, ignore_index=True), tables, "reliability")

    # 6. significance, shortcuts, robustness, transfer ---------------------------------------
    s0 = seeds[0]
    sig_rows = []
    if reference in specs:
        ref = preds_store[(reference, s0)]
        for spec in specs:
            if spec == reference:
                continue
            other = preds_store[(spec, s0)]
            a, b = (ref["p_raw"].values >= 0.5).astype(int), (other["p_raw"].values >= 0.5).astype(int)
            mc, pb = mcnemar(ref["label"].values, a, b), paired_bootstrap(ref["label"].values, a, b)
            sig_rows.append({"reference": reference, "model": spec, "seed": s0, **mc, **pb})
    sig = pd.DataFrame(sig_rows)
    _write(sig, tables, "significance")

    f0 = first[specs[0]]
    sh = shortcut_report(f0["tr"], f0["ev"], seed=s0, extra_cols=cfg.get("shortcut_columns", []))
    _write(sh, tables, "shortcuts")
    log(f"[6/7] shortcut baselines: {int(sh['flag'].sum())} flagged")

    rob = pd.DataFrame()
    if cfg.get("robustness", True) and use_ctx:
        log("      robustness suite skipped when use_context is on (perturbations apply to the utterance only)")
    elif cfg.get("robustness", True):
        mdl = first[specs[0]]["model"]
        rob = robustness_table(lambda xs: mdl.predict_proba(list(xs))[:, 1], f0["ev"]["text"], f0["ev"]["label"],
                               seed=s0, prep=lambda s: normalize(s, **pre))
        _write(rob, tables, "robustness")
    tr_long, tr_mat, fam = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if cfg.get("transfer", True) and df["lang"].nunique() > 1:
        tr_long, tr_mat = transfer_matrix(f0["sdf"], lambda: TfidfLR(seed=s0), eval_split=eval_split, text_col="text_norm")
        fam = family_summary(tr_long)
        _write(tr_long, tables, "transfer_long")
        _write(tr_mat.reset_index() if len(tr_mat) else tr_mat, tables, "transfer_matrix")
        _write(fam, tables, "transfer_family")

    # 7. report -----------------------------------------------------------------------------------
    if final:
        lock.write_text(json.dumps({"final_evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": seeds,
                                    "models": specs}, indent=2))
    L = [f"# Run report: {cfg.get('run_name')}", "",
         f"Evaluation split: **{eval_split.upper()}**" + ("  (FINAL: test set used; run is now locked)" if final else
                                                           "  (development: the test set has NOT been touched)"),
         f"Seeds: {seeds} | Models: {specs} | Rows used: {split_reports[0]['rows_out']} of {split_reports[0]['rows_in']}", "",
         "## Audit flags", ""]
    L += [f"- **[{f['severity']}] {f['item']}**: {f['message']}" for f in aud["flags"]] or ["No flags raised."]
    L += ["", "## Main results (mean over seeds)", "", df_to_md(summ), "## Per language", "", df_to_md(pl_sum),
          "## Significance versus reference model (first seed)", "", df_to_md(sig),
          "## Shortcut baselines (flag = clearly above majority class without reading the sentence)", "", df_to_md(sh)]
    if len(rob):
        L += ["## Robustness (first model, first seed)", "", df_to_md(rob)]
    if len(fam):
        L += ["## Cross-language transfer (TF-IDF, first seed)", "", df_to_md(fam), df_to_md(tr_mat.reset_index() if len(tr_mat) else tr_mat),
              "_Note: TF-IDF features are script-specific, so transfer across scripts is expected to be near chance. "
              "Use multilingual encoders for meaningful cross-script transfer._", ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"[7/7] report written: {run_dir / 'report.md'}  ({time.time() - t0:.1f}s)")
    return {"run_dir": str(run_dir), "summary": summ, "per_language": pl_sum, "audit": aud, "robustness": rob,
            "shortcuts": sh, "significance": sig, "transfer_family": fam, "eval_split": eval_split}
