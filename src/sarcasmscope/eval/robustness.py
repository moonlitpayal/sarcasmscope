"""Controlled perturbation suite: how much does the score drop when text gets noisier?

Character-level operations work on grapheme-like clusters (a letter plus its combining marks,
and virama-joined conjuncts), so Indic words are not broken into invalid fragments.
"""
from __future__ import annotations

import unicodedata
from typing import Callable

import numpy as np
import pandas as pd

from ..text.emoji import strip_emojis
from ..text.scripts import script_of
from .metrics import binary_metrics


def clusters(word: str) -> list[str]:
    out: list[str] = []
    for ch in word:
        attach = bool(out) and (
            unicodedata.category(ch)[0] == "M" or ch in "\u200d\u200c" or unicodedata.combining(out[-1][-1]) == 9
        )
        if attach:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _has_letters(w: str) -> bool:
    return any(script_of(c) for c in w)


def p_char_noise(text: str, rng, p: float = 0.15) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        cl = clusters(w)
        if len(cl) >= 4 and _has_letters(w) and rng.rand() < p:
            k = int(rng.randint(1, len(cl) - 2))
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap":
                cl[k], cl[k + 1] = cl[k + 1], cl[k]
            elif op == "drop":
                del cl[k]
            else:
                cl.insert(k, cl[k])
            words[i] = "".join(cl)
    return " ".join(words)


def p_repeat_letters(text: str, rng, p: float = 0.3) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        cl = clusters(w)
        if len(cl) >= 3 and _has_letters(w) and rng.rand() < p:
            words[i] = "".join(cl) + cl[-1] * int(rng.randint(2, 4))
    return " ".join(words)


def p_remove_emoji(text: str, rng) -> str:
    return " ".join(strip_emojis(text).split())


def p_strip_punct(text: str, rng) -> str:
    kept = "".join(c for c in text if unicodedata.category(c)[0] != "P")
    return " ".join(kept.split())


def p_drop_tags(text: str, rng) -> str:
    return " ".join(w for w in text.split() if not (w.startswith("#") or w.startswith("@")))


def p_truncate(text: str, rng, k: int = 6) -> str:
    return " ".join(text.split()[:k])


def p_shuffle_words(text: str, rng) -> str:
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def p_uppercase(text: str, rng) -> str:
    return text.upper()


PERTURBATIONS: dict[str, Callable] = {
    "char_noise": p_char_noise,
    "repeat_letters": p_repeat_letters,
    "remove_emoji": p_remove_emoji,
    "strip_punct": p_strip_punct,
    "drop_tags": p_drop_tags,
    "truncate_6_words": p_truncate,
    "shuffle_words": p_shuffle_words,
    "uppercase": p_uppercase,
}


def perturb_all(text: str, seed: int = 0, names: list[str] | None = None) -> dict[str, str]:
    out = {}
    for i, name in enumerate(names or PERTURBATIONS):
        out[name] = PERTURBATIONS[name](text, np.random.RandomState(seed + i))
    return out


def robustness_table(predict_p: Callable[[list[str]], np.ndarray], texts, y, seed: int = 0,
                     names: list[str] | None = None, prep: Callable[[str], str] | None = None) -> pd.DataFrame:
    """predict_p maps a list of strings to P(sarcastic). `prep` (e.g. normalize) is applied AFTER perturbing,
    which measures the whole pipeline; pass prep=None to measure the bare model."""
    texts, y = list(texts), np.asarray(y)
    prep = prep or (lambda s: s)
    base_pred = (np.asarray(predict_p([prep(t) for t in texts])) >= 0.5).astype(int)
    base = binary_metrics(y, base_pred)
    rows = [{"perturbation": "none", "macro_f1": base["macro_f1"], "accuracy": base["accuracy"],
             "abs_drop": 0.0, "rel_drop": 0.0, "prediction_flip_rate": 0.0}]
    for i, name in enumerate(names or PERTURBATIONS):
        rng = np.random.RandomState(seed + i)
        pert = [PERTURBATIONS[name](t, rng) for t in texts]
        pred = (np.asarray(predict_p([prep(t) for t in pert])) >= 0.5).astype(int)
        m = binary_metrics(y, pred)
        rows.append({"perturbation": name, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "abs_drop": base["macro_f1"] - m["macro_f1"],
                     "rel_drop": (base["macro_f1"] - m["macro_f1"]) / base["macro_f1"] if base["macro_f1"] else 0.0,
                     "prediction_flip_rate": float((pred != base_pred).mean())})
    return pd.DataFrame(rows)
