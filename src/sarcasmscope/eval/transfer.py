"""Cross-language transfer: train on language A, test on language B."""
from __future__ import annotations

import pandas as pd

from .metrics import macro_f1

LANG_FAMILY = {
    "hi": "Indo-Aryan", "mr": "Indo-Aryan", "bn": "Indo-Aryan", "gu": "Indo-Aryan", "pa": "Indo-Aryan",
    "ur": "Indo-Aryan", "or": "Indo-Aryan", "hi-en": "Indo-Aryan", "mr-en": "Indo-Aryan",
    "te": "Dravidian", "ta": "Dravidian", "kn": "Dravidian", "ml": "Dravidian",
    "ta-en": "Dravidian", "ml-en": "Dravidian", "te-en": "Dravidian", "kn-en": "Dravidian",
    "en": "Germanic",
}


def transfer_matrix(df: pd.DataFrame, model_factory, eval_split: str = "val", min_train: int = 40,
                    min_eval: int = 20, text_col: str = "text") -> tuple[pd.DataFrame, pd.DataFrame]:
    train, ev = df[df["split"] == "train"], df[df["split"] == eval_split]
    langs = sorted(set(train["lang"]) & set(ev["lang"]))
    rows = []
    for a in langs:
        tr = train[train["lang"] == a]
        if len(tr) < min_train or tr["label"].nunique() < 2:
            continue
        model = model_factory().fit(tr[text_col], tr["label"].values)
        for b in langs:
            te = ev[ev["lang"] == b]
            if len(te) < min_eval:
                continue
            pred = (model.predict_proba(te[text_col])[:, 1] >= 0.5).astype(int)
            rows.append({"train_lang": a, "eval_lang": b, "n_train": len(tr), "n_eval": len(te),
                         "macro_f1": macro_f1(te["label"].values, pred)})
    long = pd.DataFrame(rows)
    matrix = long.pivot(index="train_lang", columns="eval_lang", values="macro_f1") if len(long) else pd.DataFrame()
    return long, matrix


def family_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=["relation", "mean_macro_f1", "n_pairs"])
    d = long.copy()
    fam = lambda x: LANG_FAMILY.get(x, "Other")
    d["relation"] = [
        "same language" if a == b else ("same family" if fam(a) == fam(b) else "cross family")
        for a, b in zip(d["train_lang"], d["eval_lang"])
    ]
    return d.groupby("relation")["macro_f1"].agg(mean_macro_f1="mean", n_pairs="size").reset_index()
