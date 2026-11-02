"""Fine-tune a multilingual encoder (MuRIL, XLM-R, mBERT, ...) for sarcasm detection.

Uses a plain PyTorch loop (not the Trainer API) so it keeps working across `transformers`
versions. Needs:  pip install torch transformers.  Use a GPU (Google Colab) for real runs.

NOTE: this module has been syntax-checked but could not be executed in the environment where
the toolkit was written (no GPU / no model downloads). Run scripts/smoke_transformer.py first.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

HF_ALIASES = {
    "muril": "google/muril-base-cased",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}


class TransformerClassifier:
    kind = "transformer"

    def __init__(self, model_name: str = "muril", max_len: int = 96, epochs: int = 4, lr: float = 3e-5,
                 batch_size: int = 16, seed: int = 42, device: str | None = None, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1, patience: int = 2, use_context: bool = False, grad_clip: float = 1.0,
                 verbose: bool = True):
        self.model_name = HF_ALIASES.get(model_name, model_name)
        self.max_len, self.epochs, self.lr, self.batch_size = max_len, epochs, lr, batch_size
        self.seed, self.device, self.weight_decay = seed, device, weight_decay
        self.warmup_ratio, self.patience, self.use_context = warmup_ratio, patience, use_context
        self.grad_clip, self.verbose = grad_clip, verbose
        self.tok = None
        self.model = None

    # ---- helpers -------------------------------------------------------------------------
    def _torch(self):
        import torch
        return torch

    def _device(self):
        torch = self._torch()
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _encode(self, texts, contexts=None):
        texts = list(texts)
        if self.use_context and contexts is not None:
            return self.tok(list(contexts), texts, truncation=True, max_length=self.max_len,
                            padding=True, return_tensors="pt")
        return self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")

    def _seed_everything(self):
        torch = self._torch()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    # ---- API -----------------------------------------------------------------------------
    def fit(self, texts, y, val_texts=None, val_y=None, contexts=None, val_contexts=None):
        torch = self._torch()
        from sklearn.metrics import f1_score
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        self._seed_everything()
        dev = self._device()
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(dev)

        texts, y = list(texts), np.asarray(y)
        ctx = list(contexts) if contexts is not None else None
        n = len(texts)
        counts = np.bincount(y, minlength=2).astype(float)
        weights = torch.tensor(n / (2.0 * np.maximum(counts, 1)), dtype=torch.float, device=dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(n / self.batch_size))
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_ratio * steps), steps)

        best_f1, best_state, bad = -1.0, None, 0
        rng = np.random.RandomState(self.seed)
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self._encode([texts[i] for i in idx], [ctx[i] for i in idx] if ctx else None)
                enc = {k: v.to(dev) for k, v in enc.items()}
                labels = torch.tensor(y[idx], dtype=torch.long, device=dev)
                logits = self.model(**enc).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += float(loss.item()) * len(idx)
            msg = f"epoch {epoch + 1}/{self.epochs} train_loss={total / n:.4f}"
            if val_texts is not None:
                p = self.predict_proba(val_texts, val_contexts)[:, 1]
                f1 = f1_score(val_y, (p >= 0.5).astype(int), average="macro")
                msg += f" val_macro_f1={f1:.4f}"
                if f1 > best_f1:
                    best_f1, bad = f1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(msg + " (early stop)")
                    break
            if self.verbose:
                print(msg, flush=True)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts, contexts=None, batch_size: int = 64) -> np.ndarray:
        torch = self._torch()
        dev = self._device()
        texts = list(texts)
        ctx = list(contexts) if contexts is not None else None
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                enc = self._encode(texts[s:s + batch_size], ctx[s:s + batch_size] if ctx else None)
                enc = {k: v.to(dev) for k, v in enc.items()}
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                out.append(probs)
        return np.vstack(out) if out else np.zeros((0, 2))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        with open(os.path.join(path, "sarcasmscope_meta.json"), "w") as f:
            json.dump({"max_len": self.max_len, "use_context": self.use_context, "base": self.model_name}, f)

    @staticmethod
    def load(path: str, device: str | None = None) -> "TransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        with open(os.path.join(path, "sarcasmscope_meta.json")) as f:
            meta = json.load(f)
        obj = TransformerClassifier(meta["base"], max_len=meta["max_len"], use_context=meta["use_context"], device=device)
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(obj._device())
        return obj
