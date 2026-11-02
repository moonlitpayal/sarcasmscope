"""A trained model plus its calibration temperature and abstention threshold, saved as one unit."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np

from .models.calibration import TemperatureScaler, confidence
from .models.tfidf import TfidfLR


@dataclass
class ModelBundle:
    model: object
    temperature: float = 1.0
    abstain_threshold: float | None = None
    meta: dict = field(default_factory=dict)

    def predict_p(self, texts, contexts=None, calibrated: bool = True) -> np.ndarray:
        p = self.model.predict_proba(list(texts), contexts)[:, 1]
        return TemperatureScaler(self.temperature).transform(p) if calibrated else p

    def decide(self, p: float) -> dict:
        conf = float(confidence(p))
        tau = self.abstain_threshold
        return {"label": "sarcastic" if p >= 0.5 else "not sarcastic", "p_sarcastic": float(p),
                "confidence": conf, "abstain": bool(tau is not None and conf < tau), "threshold": tau}

    def save(self, path: str) -> None:
        info = {"temperature": self.temperature, "abstain_threshold": self.abstain_threshold, "meta": self.meta}
        if getattr(self.model, "kind", "") == "tfidf_lr":
            joblib.dump({"model": self.model, **info}, path)
        else:  # transformer: a directory
            os.makedirs(path, exist_ok=True)
            self.model.save(path)
            with open(os.path.join(path, "bundle.json"), "w") as f:
                json.dump(info, f)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        if os.path.isdir(path):
            from .models.transformer import TransformerClassifier
            with open(os.path.join(path, "bundle.json")) as f:
                info = json.load(f)
            return ModelBundle(TransformerClassifier.load(path), **info)
        d = joblib.load(path)
        return ModelBundle(d["model"], d["temperature"], d["abstain_threshold"], d["meta"])


def demo_bundle(seed: int = 0) -> ModelBundle:
    """A small model trained on TOY data so the app can start without any dataset. Not for research use."""
    from .data.toy import make_toy
    from .models.calibration import threshold_for_selective_accuracy
    from .text.normalize import normalize

    df = make_toy(n_per_lang=200, seed=seed, leak=0.3, n_dups=0, suffix=True)
    df["t"] = [normalize(t) for t in df["text"]]
    cut = int(len(df) * 0.8)
    tr, va = df.iloc[:cut], df.iloc[cut:]
    model = TfidfLR(seed=seed).fit(tr["t"], tr["label"])
    p = model.predict_proba(va["t"])[:, 1]
    T = TemperatureScaler().fit(p, va["label"]).T
    tau, _ = threshold_for_selective_accuracy(TemperatureScaler(T).transform(p), va["label"], target=0.9)
    return ModelBundle(model, T, tau if tau is not None else 0.6,
                       {"name": "DEMO model (toy data)", "demo": True, "trained_on": "machine-written toy sentences",
                        "warning": "Demo only. Train on a real dataset with `sarcasmscope run` and serve that bundle."})
