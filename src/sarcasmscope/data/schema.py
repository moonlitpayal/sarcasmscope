"""Unified dataset schema.

Every dataset is converted to these columns:
  id        unique row id (string)
  text      the utterance
  label     1 = sarcastic, 0 = not sarcastic
  lang      language code (gold label from the dataset; 'unk' if unknown)
  source    dataset name
  context   optional previous turn(s) (empty string if none)
  group     optional group key (conversation id, user id...) used to keep related rows in one split
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["text", "label"]
OPTIONAL = ["id", "lang", "source", "context", "group"]

POS_LABELS = {"1", "true", "yes", "sarcastic", "sarcasm", "sarc", "irony", "ironic", "1.0"}
NEG_LABELS = {"0", "false", "no", "non-sarcastic", "non sarcastic", "not sarcastic", "not_sarcastic",
              "nonsarcastic", "non-sarc", "normal", "literal", "0.0", "non_sarcastic"}


def map_label(value, label_map: dict | None = None):
    if label_map is not None:
        key = str(value).strip()
        if key in label_map:
            return int(label_map[key])
        if value in label_map:
            return int(label_map[value])
    s = str(value).strip().lower()
    if s in POS_LABELS:
        return 1
    if s in NEG_LABELS:
        return 0
    return None


def standardize(df: pd.DataFrame, mapping: dict | None = None, *, source: str = "dataset",
                lang: str | None = None, label_map: dict | None = None,
                keep: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Rename columns, map labels to 0/1, drop unusable rows. Returns (frame, report)."""
    mapping = mapping or {}
    inv = {v: k for k, v in mapping.items()}
    d = df.rename(columns=inv).copy()
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing}. Columns found: {list(df.columns)}. "
            f"Give a mapping such as {{text: <your text column>, label: <your label column>}}."
        )
    n0 = len(d)
    d["text"] = d["text"].astype("string").fillna("").astype(str).str.strip()
    labels = d["label"].map(lambda v: map_label(v, label_map))
    bad_values = sorted({str(v) for v, m in zip(d["label"], labels) if pd.isna(m)})
    d["label"] = labels
    n_bad_label = int(d["label"].isna().sum())
    d = d[d["label"].notna() & (d["text"] != "")].copy()
    d["label"] = d["label"].astype(int)
    if "lang" not in d.columns:
        d["lang"] = lang if lang else "unk"
    elif lang:
        d["lang"] = lang
    d["lang"] = d["lang"].astype(str).str.strip().str.lower()
    d["source"] = source
    for col in ("context", "group"):
        if col not in d.columns:
            d[col] = ""
        d[col] = d[col].fillna("").astype(str)
    if "id" not in d.columns:
        d["id"] = [f"{source}-{i}" for i in range(len(d))]
    d["id"] = d["id"].astype(str)
    extra = [c for c in (keep or []) if c in d.columns]
    d = d[["id", "text", "label", "lang", "source", "context", "group"] + extra].reset_index(drop=True)
    report = {"source": source, "kept_extra_columns": extra, "rows_in": n0, "rows_kept": len(d),
              "dropped_bad_label_or_empty": n0 - len(d), "unrecognised_label_values": bad_values[:10]}
    return d, report
