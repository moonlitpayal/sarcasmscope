"""Check that transformer fine-tuning works on THIS machine (run it first, on Colab GPU or locally).

    python scripts/smoke_transformer.py            # xlm-roberta-base, 1 epoch, 96 toy sentences
    python scripts/smoke_transformer.py muril
"""
import sys

import numpy as np

from sarcasmscope.data.toy import make_toy
from sarcasmscope.models.transformer import TransformerClassifier
from sarcasmscope.text.normalize import normalize

name = sys.argv[1] if len(sys.argv) > 1 else "xlmr"
df = make_toy(n_per_lang=20, seed=0, suffix=True)
texts = [normalize(t) for t in df["text"]]
y = df["label"].values
cut = int(0.75 * len(df))
clf = TransformerClassifier(name, epochs=1, batch_size=8, max_len=48, verbose=True)
clf.fit(texts[:cut], y[:cut], val_texts=texts[cut:], val_y=y[cut:])
p = clf.predict_proba(texts[cut:])
assert p.shape == (len(texts) - cut, 2) and np.allclose(p.sum(1), 1, atol=1e-4)
clf.save("smoke_model")
back = TransformerClassifier.load("smoke_model")
assert np.allclose(back.predict_proba(texts[cut:]), p, atol=1e-3)
print("SMOKE TEST PASSED: training, prediction, save and load all work for", name)
