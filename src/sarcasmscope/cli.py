"""Command line interface:  sarcasmscope <command>

  init-config   write a config template
  demo-data     write TOY data (for testing the software only)
  audit         audit a dataset described by a config
  run           run the full pipeline (add --final ONCE, at the very end, to evaluate the test set)
  serve         start the web app (demo model unless --bundle is given)
  prepare-masac convert the MaSaC repository into one clean CSV
  annotate-sample / annotate-agreement   human labelling kit (sampling, workbooks, kappa)
  demo          toy data -> pipeline -> web app, to check that everything works
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _cmd_init_config(a):
    from .pipeline import DEFAULT_CONFIG
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {a.out}. Edit the data.files section to point at your dataset and map its columns.")


def _cmd_demo_data(a):
    from .data.toy import make_toy
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make_toy(n_per_lang=a.n, suffix=True).to_csv(a.out, index=False, encoding="utf-8")
    print(f"Wrote TOY data to {a.out} (software tests only; not a research dataset).")


def _cmd_audit(a):
    from .data.audit import audit, audit_markdown
    from .data.loaders import load_files
    from .pipeline import load_config
    cfg = load_config(a.config)
    df, _ = load_files(cfg["data"]["files"], Path(a.config).parent if a.relative else ".")
    res = audit(df, near_dup_threshold=cfg["split"].get("near_dup_threshold", 0.7))
    print(audit_markdown(res))


def _cmd_run(a):
    from .pipeline import load_config, run
    cfg = load_config(a.config)
    if a.name:
        cfg["run_name"] = a.name
    out = run(cfg, final=a.final, force=a.force)
    print("\n" + out["summary"].to_string(index=False))
    print(f"\nFull report: {out['run_dir']}/report.md")


def _cmd_prepare_masac(a):
    from .data.prepare import prepare_masac
    info = prepare_masac(a.repo, a.out, a.context_turns)
    print(f"Wrote {info['rows']} utterances from {info['dialogues']} dialogue instances "
          f"({info['sarcastic']} sarcastic) to {info['out']}")
    print(f"Alignment checks: speaker match {info['speaker_match']:.3f}, dialogue match {info['dialogue_match']:.3f}")


def _cmd_annotate_sample(a):
    from .annotation.sample import make_sample, reservoir_sample_lines
    from .annotation.workbook import write_annotator_workbook
    out = Path(a.out)
    print(f"Reading a random sample of up to {a.pool} lines from {a.source} ...")
    lines = reservoir_sample_lines(a.source, k=a.pool, seed=a.seed)
    df, rep = make_sample(lines, a.n_random, a.n_enriched, a.pilot, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "master_key.csv", index=False, encoding="utf-8")
    pilot, final = df[df["subset"] == "pilot"], df[df["subset"] != "pilot"]
    for name in a.annotators:
        write_annotator_workbook(pilot, name, out / "pilot" / f"{name}_pilot.xlsx")
        write_annotator_workbook(final, name, out / "final" / f"{name}_final.xlsx")
    print(json.dumps(rep, indent=2))
    print(f"\nWrote {len(pilot)} pilot items and {len(final)} final items for {len(a.annotators)} annotators to {out}/")
    print("KEEP master_key.csv PRIVATE until labelling is finished: it shows which items are random or enriched.")
    print("Give each person ONLY their own workbook. Do not discuss labels until everyone has finished.")


def _cmd_annotate_agreement(a):
    import pandas as pd
    from .annotation.agreement import adjudicate, agreement_report, collect, report_markdown
    files = sorted(Path(a.dir).glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found in {a.dir}")
    wide = collect(files)
    res = agreement_report(wide)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = report_markdown(res)
    (out / "agreement.md").write_text(md, encoding="utf-8")
    print(md)
    if "pairwise" in res:
        pd.DataFrame(res["pairwise"]).to_csv(out / "pairwise_kappa.csv", index=False)
    overrides = pd.read_csv(a.discussed) if a.discussed else None
    adj = adjudicate(wide, overrides)
    key = pd.read_csv(a.key) if a.key else None
    if key is not None:
        adj = adj.merge(key[["item_id", "text", "subset", "script"]], on="item_id", how="left")
    adj.sort_values(["how", "item_id"]).to_csv(out / "adjudication.csv", index=False, encoding="utf-8")
    final = adj[adj["final_label"] != ""]
    final.to_csv(out / "final_labels.csv", index=False, encoding="utf-8")
    print(f"Items needing discussion: {(adj['how'] == 'needs_discussion').sum()} (see {out / 'adjudication.csv'}). "
          f"Items with a final label: {len(final)}.")
    if a.discussed is None and (adj["how"] == "needs_discussion").any():
        print("Discuss those items together, write their agreed labels in a CSV with columns item_id,label, "
              "then re-run with --discussed that file.")


def _cmd_serve(a):
    import os
    import uvicorn
    if a.bundle:
        os.environ["SARCASMSCOPE_BUNDLE"] = a.bundle
    print(f"Open http://{a.host}:{a.port} in your browser (Ctrl+C to stop)")
    uvicorn.run("sarcasmscope.app.api:get_app", factory=True, host=a.host, port=a.port)


def _cmd_demo(a):
    import tempfile
    from .data.toy import make_toy
    from .pipeline import DEFAULT_CONFIG, run
    tmp = Path(tempfile.mkdtemp())
    make_toy(n_per_lang=150, suffix=True).to_csv(tmp / "toy.csv", index=False)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update({"run_name": "demo", "output_dir": str(tmp / "runs"), "seeds": [1, 2], "models": ["tfidf"]})
    cfg["data"] = {"files": [{"path": str(tmp / "toy.csv"), "source": "TOY",
                              "mapping": {"text": "text", "label": "label", "lang": "lang"}}]}
    cfg["split"]["near_dup_threshold"] = None
    out = run(cfg)
    print("\n" + out["summary"].to_string(index=False))
    print(f"\nReport: {out['run_dir']}/report.md")
    print("Everything works. Now start the app with:  sarcasmscope serve")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="sarcasmscope", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-config"); s.add_argument("--out", default="configs/my_run.yaml"); s.set_defaults(f=_cmd_init_config)
    s = sub.add_parser("demo-data"); s.add_argument("--out", default="data/toy_demo.csv"); s.add_argument("--n", type=int, default=100); s.set_defaults(f=_cmd_demo_data)
    s = sub.add_parser("audit"); s.add_argument("--config", required=True); s.add_argument("--relative", action="store_true"); s.set_defaults(f=_cmd_audit)
    s = sub.add_parser("run"); s.add_argument("--config", required=True); s.add_argument("--name")
    s.add_argument("--final", action="store_true", help="evaluate the TEST split (once, at the very end)")
    s.add_argument("--force", action="store_true", help="allow a second final run (test-set peeking!)"); s.set_defaults(f=_cmd_run)
    s = sub.add_parser("prepare-masac", help="convert the MaSaC/MSH-COMICS repository into one clean CSV")
    s.add_argument("--repo", required=True); s.add_argument("--out", default="data/raw/masac.csv")
    s.add_argument("--context-turns", type=int, default=2); s.set_defaults(f=_cmd_prepare_masac)
    s = sub.add_parser("annotate-sample", help="sample sentences and write one labelling workbook per annotator")
    s.add_argument("--source", required=True, help="large text file (one sentence per line) or CSV with a text column")
    s.add_argument("--out", default="annotation"); s.add_argument("--annotators", nargs="+", required=True)
    s.add_argument("--n-random", dest="n_random", type=int, default=450); s.add_argument("--n-enriched", dest="n_enriched", type=int, default=450)
    s.add_argument("--pilot", type=int, default=100); s.add_argument("--pool", type=int, default=200000)
    s.add_argument("--seed", type=int, default=42); s.set_defaults(f=_cmd_annotate_sample)
    s = sub.add_parser("annotate-agreement", help="agreement statistics and adjudication list from filled workbooks")
    s.add_argument("--dir", required=True); s.add_argument("--out", default="annotation/report")
    s.add_argument("--key", help="master_key.csv, to attach text and subset to the final labels")
    s.add_argument("--discussed", help="CSV (item_id,label) with labels agreed in discussion"); s.set_defaults(f=_cmd_annotate_agreement)
    s = sub.add_parser("serve"); s.add_argument("--bundle"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(f=_cmd_serve)
    s = sub.add_parser("demo"); s.set_defaults(f=_cmd_demo)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
