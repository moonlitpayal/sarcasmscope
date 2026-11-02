"""Calibration, selective prediction (abstention) and reliability metrics for binary sarcasm models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def confidence(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1 - p)


class TemperatureScaler:
    """Single-parameter calibration: p' = sigmoid(logit(p) / T). T > 1 softens over-confident models."""

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, p, y) -> "TemperatureScaler":
        z, y = _logit(p), np.asarray(y)

        def nll(logT):
            q = np.clip(_sigmoid(z / np.exp(logT)), 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def transform(self, p) -> np.ndarray:
        return _sigmoid(_logit(p) / self.T)


def crossfit_temperature(p, y, seed: int = 0) -> np.ndarray:
    """Calibrated probabilities for a set without peeking: fit T on one half, apply to the other."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    idx = np.random.RandomState(seed).permutation(len(p))
    a, b = idx[: len(p) // 2], idx[len(p) // 2:]
    out = np.zeros_like(p)
    out[b] = TemperatureScaler().fit(p[a], y[a]).transform(p[b])
    out[a] = TemperatureScaler().fit(p[b], y[b]).transform(p[a])
    return out


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    conf, pred = confidence(p), (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    conf, acc = confidence(p), ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()), "mean_confidence": conf[m].mean(), "accuracy": acc[m].mean()})
    return pd.DataFrame(rows)


def risk_coverage(p, y) -> pd.DataFrame:
    """Selective accuracy when only the most confident `coverage` fraction is answered."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    conf = confidence(p)
    order = np.argsort(-conf, kind="stable")
    correct = ((p >= 0.5).astype(int) == y)[order].astype(float)
    k = np.arange(1, len(p) + 1)
    return pd.DataFrame({"coverage": k / len(p), "selective_accuracy": np.cumsum(correct) / k,
                         "confidence_cutoff": conf[order]})


def aurc(p, y) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    rc = risk_coverage(p, y)
    return float(np.mean(1.0 - rc["selective_accuracy"].values))


def threshold_for_selective_accuracy(p_val, y_val, target: float = 0.9, min_coverage: float = 0.1):
    """Smallest confidence threshold whose selective accuracy on validation data is >= target.

    Returns (threshold, coverage) or (None, 0.0) if the target is never reached with at least
    `min_coverage` of the examples answered.
    """
    rc = risk_coverage(p_val, y_val)
    ok = rc[(rc["selective_accuracy"] >= target) & (rc["coverage"] >= min_coverage)]
    if ok.empty:
        return None, 0.0
    best = ok.iloc[-1]
    return float(best["confidence_cutoff"]), float(best["coverage"])


def abstain_mask(p, tau: float | None) -> np.ndarray:
    return np.zeros(len(p), dtype=bool) if tau is None else confidence(p) < tau
