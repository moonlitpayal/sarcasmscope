"""Create a model from a short spec string:  'tfidf' | 'muril' | 'xlmr' | 'mbert' | 'hf:<hub-id>'."""
from __future__ import annotations

from .tfidf import TfidfLR


def make_model(spec: str, seed: int = 42, transformer_cfg: dict | None = None):
    cfg = dict(transformer_cfg or {})
    if spec == "tfidf":
        return TfidfLR(seed=seed)
    from .transformer import HF_ALIASES, TransformerClassifier  # lazy: needs torch
    if spec in HF_ALIASES:
        return TransformerClassifier(spec, seed=seed, **cfg)
    if spec.startswith("hf:"):
        return TransformerClassifier(spec[3:], seed=seed, **cfg)
    raise ValueError(f"Unknown model spec '{spec}'. Use tfidf, muril, xlmr, mbert or hf:<hub-id>.")


def safe_name(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "_")
