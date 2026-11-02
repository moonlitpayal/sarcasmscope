"""FastAPI service + single-page web UI.

Endpoints: GET /  GET /health  POST /predict  POST /predict_batch  POST /explain  POST /robustness
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..bundle import ModelBundle, demo_bundle
from ..eval.robustness import perturb_all
from ..explain.occlusion import occlusion
from ..text.emoji import emoji_features, extract_emojis
from ..text.langid import guess_language
from ..text.normalize import normalize

WEB_DIR = Path(__file__).parent / "web"
MAX_TEXT_CHARS = 2000
MAX_BATCH = 200


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    context: str | None = Field(None, max_length=MAX_TEXT_CHARS)


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class ExplainIn(PredictIn):
    top_k: int = Field(5, ge=1, le=20)


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    if bundle is None:
        path = os.environ.get("SARCASMSCOPE_BUNDLE")
        bundle = ModelBundle.load(path) if path else demo_bundle()
    app = FastAPI(title="SarcasmScope", version=__version__,
                  description="Calibrated, explainable sarcasm detection for code-mixed Indian-language text.")
    app.state.bundle = bundle

    def _p(texts):
        return bundle.predict_p([normalize(t) for t in texts])

    def _analyze(text: str, context: str | None = None) -> dict:
        norm = normalize(text)
        if not norm:
            raise HTTPException(422, "Text is empty after cleaning.")
        p = float(bundle.predict_p([norm], [normalize(context)] if context else None)[0])
        dec = bundle.decide(p)
        return {**dec, "normalized_text": norm, "language": guess_language(text),
                "emoji": {"emojis": extract_emojis(text), **emoji_features(text)},
                "model": bundle.meta.get("name", "model"), "demo_model": bool(bundle.meta.get("demo")),
                "calibrated_temperature": bundle.temperature}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "model": bundle.meta.get("name", "model"),
                "demo_model": bool(bundle.meta.get("demo")), "temperature": bundle.temperature,
                "abstain_threshold": bundle.abstain_threshold}

    @app.post("/predict")
    def predict(body: PredictIn):
        return _analyze(body.text, body.context)

    @app.post("/predict_batch")
    def predict_batch(body: BatchIn):
        out = []
        for t in body.texts:
            t = t[:MAX_TEXT_CHARS]
            norm = normalize(t)
            if not norm:
                out.append({"text": t, "error": "empty after cleaning"})
                continue
            p = float(bundle.predict_p([norm])[0])
            out.append({"text": t, **bundle.decide(p)})
        return {"n": len(out), "results": out}

    @app.post("/explain")
    def explain(body: ExplainIn):
        norm = normalize(body.text)
        if not norm:
            raise HTTPException(422, "Text is empty after cleaning.")
        res = occlusion(lambda xs: bundle.predict_p(list(xs)), norm, top_k=body.top_k)
        res["method"] = "leave-one-token-out occlusion (delta > 0 pushes towards sarcasm)"
        res["normalized_text"] = norm
        return res

    @app.post("/robustness")
    def robustness(body: PredictIn):
        base = float(bundle.predict_p([normalize(body.text)])[0])
        rows = []
        for name, pert in perturb_all(body.text).items():
            p = float(bundle.predict_p([normalize(pert)])[0])
            rows.append({"perturbation": name, "perturbed_text": pert, "p_sarcastic": p,
                         "flipped": (p >= 0.5) != (base >= 0.5), "delta": p - base})
        return {"baseline_p_sarcastic": base, "results": rows}

    return app


app = None


def get_app() -> FastAPI:
    """Used by `uvicorn sarcasmscope.app.api:get_app --factory`."""
    return create_app()
