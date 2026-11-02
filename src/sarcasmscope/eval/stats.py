from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from .metrics import macro_f1


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired predictions. 'b_better' = items only model B got right."""
    y = np.asarray(y)
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p}


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the macro-F1 difference (B minus A) over the same examples."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        diffs[i] = macro_f1(y[idx], b[idx]) - macro_f1(y[idx], a[idx])
    return {"mean_diff": float(diffs.mean()), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "prob_b_better": float((diffs > 0).mean())}
