"""Model-agnostic token attributions by occlusion (leave-one-token-out)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def occlusion(predict_p: Callable[[list[str]], np.ndarray], text: str, top_k: int = 5) -> dict:
    """delta > 0: removing the token LOWERS P(sarcastic), i.e. the token pushes towards sarcasm."""
    tokens = text.split()
    base = float(np.asarray(predict_p([text]))[0])
    if not tokens:
        return {"p_sarcastic": base, "attributions": [], "top_sarcastic": [], "top_literal": []}
    variants = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    ps = np.asarray(predict_p(variants), dtype=float)
    attr = [{"token": t, "delta": float(base - p)} for t, p in zip(tokens, ps)]
    ranked = sorted(attr, key=lambda a: a["delta"], reverse=True)
    return {
        "p_sarcastic": base,
        "attributions": attr,
        "top_sarcastic": [a for a in ranked[:top_k] if a["delta"] > 0],
        "top_literal": [a for a in ranked[::-1][:top_k] if a["delta"] < 0],
    }
