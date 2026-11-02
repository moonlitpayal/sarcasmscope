from __future__ import annotations

import numpy as np
import pandas as pd


def _counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def binary_metrics(y, pred) -> dict:
    tp, tn, fp, fn = _counts(y, pred)
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos, f1_neg = _f1(tp, fp, fn), _f1(tn, fn, fp)
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec,
            "f1": f1_pos, "macro_f1": (f1_pos + f1_neg) / 2, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def macro_f1(y, pred) -> float:
    tp, tn, fp, fn = _counts(y, pred)
    return (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2


def bootstrap_ci(y, pred, metric=macro_f1, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        vals[b] = metric(y[idx], pred[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def per_group(df: pd.DataFrame, pred_col: str = "pred", group_col: str = "lang", min_n: int = 1) -> pd.DataFrame:
    rows = []
    for g, part in df.groupby(group_col):
        if len(part) < min_n:
            continue
        m = binary_metrics(part["label"].values, part[pred_col].values)
        rows.append({group_col: g, "n": len(part), "share_sarcastic": part["label"].mean(),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "f1_sarcastic": m["f1"]})
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)
