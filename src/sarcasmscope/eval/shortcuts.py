"""Shortcut baselines: how well can a model do WITHOUT reading the sentence?

If emoji alone, length alone, punctuation alone, the language tag or the data source predicts the label
clearly better than the majority class, headline scores on that dataset can be inflated by the shortcut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from ..text.emoji import extract_emojis
from .metrics import macro_f1


def _punct_feats(texts) -> np.ndarray:
    rows = []
    for t in texts:
        t = str(t)
        rows.append([t.count("!"), t.count("?"), t.count("..."), t.count('"') + t.count("“"),
                     t.count("#"), t.count("@"), int("http" in t), int("<url>" in t), int("<user>" in t)])
    return np.array(rows, dtype=float)


def _len_feats(texts) -> np.ndarray:
    return np.array([[np.log1p(len(str(t))), np.log1p(len(str(t).split()))] for t in texts])


def shortcut_report(train: pd.DataFrame, test: pd.DataFrame, margin: float = 0.05, seed: int = 0,
                    extra_cols: list[str] | None = None) -> pd.DataFrame:
    ytr, yte = train["label"].values, test["label"].values
    maj = int(np.bincount(ytr).argmax())
    maj_f1 = macro_f1(yte, np.full(len(yte), maj))
    rows = [{"shortcut": "majority_class", "macro_f1": maj_f1, "delta_vs_majority": 0.0, "flag": False, "note": ""}]

    def add(name, f1, note=""):
        rows.append({"shortcut": name, "macro_f1": f1, "delta_vs_majority": f1 - maj_f1,
                     "flag": bool(f1 - maj_f1 >= margin), "note": note})

    # emoji only
    cv = CountVectorizer(tokenizer=extract_emojis, token_pattern=None, lowercase=False)
    try:
        Xtr = cv.fit_transform(train["text"])
        if Xtr.shape[1] == 0:
            raise ValueError
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        add("emoji_only", macro_f1(yte, clf.predict(cv.transform(test["text"]))))
    except ValueError:
        add("emoji_only", maj_f1, "no emoji in training text")

    clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed).fit(_len_feats(train["text"]), ytr)
    add("length_only", macro_f1(yte, clf.predict(_len_feats(test["text"]))))

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(_punct_feats(train["text"]), ytr)
    add("punctuation_and_tags_only", macro_f1(yte, clf.predict(_punct_feats(test["text"]))))

    for col in ("lang", "source", *(extra_cols or [])):
        if col in train and train[col].fillna("").nunique() > 1:
            tr_c, te_c = train[[col]].fillna("").astype(str), test[[col]].fillna("").astype(str)
            enc = OneHotEncoder(handle_unknown="ignore")
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            clf.fit(enc.fit_transform(tr_c), ytr)
            add(f"{col}_tag_only", macro_f1(yte, clf.predict(enc.transform(te_c))))
    return pd.DataFrame(rows)
