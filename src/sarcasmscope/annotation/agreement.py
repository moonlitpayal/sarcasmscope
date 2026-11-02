"""Inter-annotator agreement (Cohen's and Fleiss' kappa), majority labels and an adjudication list."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .workbook import LABELS, read_annotator_workbook


def fleiss_kappa(counts) -> float:
    """counts: (items x categories) matrix of how many raters chose each category; same number of raters per item."""
    counts = np.asarray(counts, dtype=float)
    n_items = counts.shape[0]
    n = counts[0].sum()
    assert np.allclose(counts.sum(axis=1), n), "every item needs the same number of ratings"
    p_j = counts.sum(axis=0) / (n_items * n)
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def landis_koch(k: float) -> str:
    if k < 0:
        return "poor"
    for hi, name in [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if k <= hi:
            return name
    return "almost perfect"


def collect(files) -> pd.DataFrame:
    """Wide table: one row per item, one column per annotator (named after the file)."""
    wide = None
    for f in files:
        name = Path(f).stem
        for suffix in ("_filled", "_final", "_pilot"):
            name = name.replace(suffix, "")
        d = read_annotator_workbook(f).rename(columns={"label": name})
        wide = d if wide is None else wide.merge(d, on="item_id", how="outer")
    cols = [c for c in wide.columns if c != "item_id"]
    wide[cols] = wide[cols].fillna("")
    return wide


def _counts(frame: pd.DataFrame, annotators: list[str], labels: list[str]) -> np.ndarray:
    return np.stack([(frame[annotators] == lab).sum(axis=1).values for lab in labels], axis=1)


def agreement_report(wide: pd.DataFrame) -> dict:
    annotators = [c for c in wide.columns if c != "item_id"]
    wide = wide.copy()
    wide[annotators] = wide[annotators].fillna("")
    bad = {a: sorted(set(wide[a]) - set(LABELS) - {""}) for a in annotators}
    complete = wide[(wide[annotators] != "").all(axis=1)].copy()
    res = {"annotators": annotators, "items_total": len(wide), "items_fully_labelled": len(complete),
           "unrecognised_labels": {a: v for a, v in bad.items() if v}}
    if len(complete) < 2 or len(annotators) < 2:
        res["error"] = "Not enough fully labelled items from at least two annotators."
        return res
    pair = []
    for a, b in combinations(annotators, 2):
        both = complete[(complete[a] != "cannot tell") & (complete[b] != "cannot tell")]
        pair.append({"pair": f"{a} vs {b}",
                     "observed_agreement_3way": float((complete[a] == complete[b]).mean()),
                     "kappa_3way": float(cohen_kappa_score(complete[a], complete[b])),
                     "kappa_binary_excl_cannot_tell": float(cohen_kappa_score(both[a], both[b])) if len(both) > 1 else float("nan"),
                     "n_binary": len(both)})
    res["pairwise"] = pair
    res["fleiss_kappa_3way"] = fleiss_kappa(_counts(complete, annotators, LABELS))
    res["fleiss_band"] = landis_koch(res["fleiss_kappa_3way"])
    binary = complete[(complete[annotators] != "cannot tell").all(axis=1)]
    if len(binary) > 1 and len(annotators) > 1:
        res["fleiss_kappa_binary"] = fleiss_kappa(_counts(binary, annotators, LABELS[:2]))
        res["items_binary"] = len(binary)
    res["label_distribution"] = {a: complete[a].value_counts().to_dict() for a in annotators}
    res["cannot_tell_rate"] = {a: float((complete[a] == "cannot tell").mean()) for a in annotators}
    return res


def adjudicate(wide: pd.DataFrame, overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """Majority among sarcastic / not sarcastic. Ties or too many 'cannot tell' go to a discussion list."""
    annotators = [c for c in wide.columns if c != "item_id"]
    rows = []
    for _, r in wide.iterrows():
        votes = [r[a] for a in annotators]
        s, n, c = votes.count("sarcastic"), votes.count("not sarcastic"), votes.count("cannot tell")
        if s > n and s > len(votes) / 2:
            final, how = "sarcastic", "majority"
        elif n > s and n > len(votes) / 2:
            final, how = "not sarcastic", "majority"
        else:
            final, how = "", "needs_discussion"
        rows.append({"item_id": r["item_id"], "votes_sarcastic": s, "votes_not_sarcastic": n, "votes_cannot_tell": c,
                     "final_label": final, "how": how, **{a: r[a] for a in annotators}})
    out = pd.DataFrame(rows)
    if overrides is not None and len(overrides):
        ov = overrides.set_index("item_id")["label"].to_dict()
        m = out["item_id"].isin(ov)
        out.loc[m, "final_label"] = out.loc[m, "item_id"].map(ov)
        out.loc[m, "how"] = "discussed"
    return out


def report_markdown(res: dict) -> str:
    if "error" in res:
        return f"# Agreement report\n\n{res['error']}\n"
    L = ["# Inter-annotator agreement", "",
         f"- Annotators: {', '.join(res['annotators'])}",
         f"- Items: {res['items_total']} total, {res['items_fully_labelled']} labelled by everyone",
         f"- **Fleiss kappa (3 categories): {res['fleiss_kappa_3way']:.3f} ({res['fleiss_band']})**"]
    if "fleiss_kappa_binary" in res:
        L.append(f"- Fleiss kappa, sarcastic vs not sarcastic only ({res['items_binary']} items with no 'cannot tell'): "
                 f"{res['fleiss_kappa_binary']:.3f} ({landis_koch(res['fleiss_kappa_binary'])})")
    L += ["", "| pair | observed agreement | kappa (3 categories) | kappa (binary) | n binary |", "|---|---|---|---|---|"]
    for p in res["pairwise"]:
        L.append(f"| {p['pair']} | {p['observed_agreement_3way']:.3f} | {p['kappa_3way']:.3f} | "
                 f"{p['kappa_binary_excl_cannot_tell']:.3f} | {p['n_binary']} |")
    L += ["", "| annotator | sarcastic | not sarcastic | cannot tell | cannot-tell rate |", "|---|---|---|---|---|"]
    for a in res["annotators"]:
        d = res["label_distribution"][a]
        L.append(f"| {a} | {d.get('sarcastic', 0)} | {d.get('not sarcastic', 0)} | {d.get('cannot tell', 0)} | {res['cannot_tell_rate'][a]:.1%} |")
    if res["unrecognised_labels"]:
        L += ["", f"**Unrecognised labels found:** {res['unrecognised_labels']}. Fix them in the workbooks."]
    L += ["", "Bands (Landis and Koch): 0.00-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect."]
    return "\n".join(L) + "\n"
