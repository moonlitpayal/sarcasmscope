"""Read CSV/TSV/JSON/JSONL/XLSX files and combine several datasets into one frame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import standardize


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    if suf in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t", encoding="utf-8-sig")
    if suf == ".jsonl":
        return pd.read_json(p, lines=True)
    if suf == ".json":
        try:
            return pd.read_json(p)
        except ValueError:
            return pd.read_json(p, lines=True)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {p.suffix} ({p})")


def load_files(specs: list[dict], base_dir: str | Path = ".") -> tuple[pd.DataFrame, list[dict]]:
    """specs: [{path, source, mapping:{text:..,label:..,lang:..}, lang: 'hi', label_map:{...}}]"""
    frames, reports = [], []
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = Path(base_dir) / path
        raw = read_table(path)
        df, rep = standardize(
            raw, spec.get("mapping"), source=spec.get("source", path.stem),
            lang=spec.get("lang"), label_map=spec.get("label_map"), keep=spec.get("keep"),
        )
        frames.append(df)
        reports.append(rep)
    if not frames:
        raise ValueError("No data files given.")
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["source"].astype(str) + "::" + out["id"].astype(str)
    if out["id"].duplicated().any():
        out["id"] = [f"{i}::{x}" for i, x in enumerate(out["id"])]
    return out, reports
