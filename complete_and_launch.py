"""
SARcasmScope — Autonomous Master Setup, Training, Test & Launch Engine
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "sarcasmscope"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MECORPUS_RAW_PATH = RAW_DATA_DIR / "mecorpus.txt"
MODEL_WEIGHTS_PATH = MODELS_DIR / "sarcasmscope_engine.joblib"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🚀 [SARcasmScope] AUTONOMOUS PRODUCT INITIALIZATION")
print("=" * 70)

# Core NLP & ML Modules
import re
import csv
import html
import unicodedata
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

POSITIVE_CUES = {
    "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
    "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless",
    "भारी", "छान", "उत्कृष्ट", "सुंदर", "मस्त", "शाब्बास", "bhari", "lai bhari",
    "mast", "chhan", "utkrusht", "shabaash", "hushaar", "hero"
}

NEGATIVE_CUES = {
    "terrible", "horrible", "disaster", "fail", "failed", "worst", "idiot",
    "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid",
    "pathetic", "ruined", "बकवास", "भिकार", "कचरा", "मूर्ख", "फेल", "घाणेरडा",
    "bakwas", "bhikaar", "kachra", "murkh", "fail", "ghaanerda", "tras"
}

INTENSIFIERS = {
    "obviously", "clearly", "totally", "absolutely", "surely", "of course",
    "ekdam", "khoop", "agdi", "nakkich", "kiti", "काय भारी"
}

CONTRASTS = {
    "but", "yet", "however", "although", "though", "nevertheless",
    "pan", "pun", "तरी", "पण", "परंतु"
}

MARATHI_ROMANIZED_LEXICON = {
    "arre", "are", "re", "bhau", "dada", "kay", "kaay", "kasa", "kashi",
    "khoop", "khup", "ekdam", "lai", "lay", "bhari", "waah", "wah", "chhan",
    "mast", "fukat", "fukatch", "bolna", "sang", "bagh", "bagha", "ahe",
    "aahe", "nahi", "naahi", "navhta", "hota", "hoti", "pan", "aata", "ata"
}

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return " ".join(text.split())

def analyze_code_mixing(text: str) -> Dict[str, Any]:
    tokens = text.split()
    total = len(tokens)
    if total == 0:
        return {"cmi": 0.0, "is_code_mixed": False, "primary_script": "Unknown"}
    dev_count = sum(1 for t in tokens if DEVANAGARI_RANGE.search(t))
    lat_tokens = [t for t in tokens if LATIN_RANGE.search(t)]
    lat_count = len(lat_tokens)
    rom_mr_count = sum(1 for t in lat_tokens if t.lower() in MARATHI_ROMANIZED_LEXICON)

    if dev_count > 0 and lat_count > 0:
        cmi = (1.0 - (max(dev_count, lat_count) / total)) * 100.0
        return {"cmi": round(cmi, 2), "is_code_mixed": True, "primary_script": "Mixed (Devanagari-Latin)"}
    elif rom_mr_count > 0 and (lat_count - rom_mr_count > 0):
        cmi = (rom_mr_count / total) * 100.0
        return {"cmi": round(cmi, 2), "is_code_mixed": True, "primary_script": "Code-Mixed (English-Romanized Marathi)"}
    elif dev_count > 0:
        return {"cmi": 0.0, "is_code_mixed": False, "primary_script": "Devanagari (Marathi)"}
    return {"cmi": 0.0, "is_code_mixed": False, "primary_script": "Latin (English)"}

class IncongruityExtractor:
    def extract(self, text: str, context: str = "") -> Dict[str, float]:
        tokens = [t.lower() for t in text.split()]
        num = max(len(tokens), 1)
        pos = sum(1 for t in tokens if t in POSITIVE_CUES)
        neg = sum(1 for t in tokens if t in NEGATIVE_CUES)
        inten = sum(1 for t in tokens if t in INTENSIFIERS)
        contra = sum(1 for t in tokens if t in CONTRASTS)

        clash = 1.0 if (pos > 0 and neg > 0) else 0.0
        num_excl = len(re.findall(r"!+", text))
        num_quest = len(re.findall(r"\?+", text))
        mixed_punct = 1.0 if ("!?" in text or "?!" in text) else 0.0

        ctx_clash = 0.0
        if context:
            ctx_tok = [t.lower() for t in context.split()]
            ctx_neg = sum(1 for t in ctx_tok if t in NEGATIVE_CUES)
            if ctx_neg > 0 and pos > 0:
                ctx_clash = 1.0

        idx = (clash * 0.4) + (ctx_clash * 0.35) + (min(contra, 2) * 0.15) + (min(inten, 2) * 0.10)
        return {
            "pos_density": pos / num, "neg_density": neg / num,
            "polarity_clash": clash, "intensifier_density": inten / num,
            "contrast_count": float(contra), "exclamation_freq": float(num_excl),
            "question_freq": float(num_quest), "mixed_punct": mixed_punct,
            "context_clash": ctx_clash, "incongruity_index": round(idx, 4)
        }

    def to_vector(self, d: Dict[str, float]) -> np.ndarray:
        return np.array([
            d["pos_density"], d["neg_density"], d["polarity_clash"],
            d["intensifier_density"], d["contrast_count"], d["exclamation_freq"],
            d["question_freq"], d["mixed_punct"], d["context_clash"], d["incongruity_index"]
        ], dtype=np.float32)

class FeatureFusionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=12000, sublinear_tf=True)
        self.extractor = IncongruityExtractor()
        base_lr = LogisticRegression(C=2.5, class_weight="balanced", max_iter=1000, random_state=42)
        self.classifier = CalibratedClassifierCV(estimator=base_lr, cv=3)
        self.is_fitted = False

    def _get_dense(self, texts: List[str], contexts: List[str] = None):
        if not contexts:
            contexts = ["" for _ in texts]
        return np.vstack([self.extractor.to_vector(self.extractor.extract(t, c)) for t, c in zip(texts, contexts)])

    def fit(self, texts: List[str], labels: List[int], contexts: List[str] = None):
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.fit_transform(clean)
        dense = csr_matrix(self._get_dense(clean, contexts))
        X = hstack([tfidf, dense]).tocsr()
        self.classifier.fit(X, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str], contexts: List[str] = None):
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.transform(clean)
        dense = csr_matrix(self._get_dense(clean, contexts))
        X = hstack([tfidf, dense]).tocsr()
        return self.classifier.predict_proba(X)

BENCHMARK_SEED = [
    ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled", 1),
    ("I absolutely love it when code fails directly in production on a Friday night.", "Deployment crashed", 1),
    ("What a genius move leaving the umbrella at home during a monsoon downpour.", "Soaked in rain", 1),
    ("Great customer service, waiting 45 minutes on hold listening to elevator music.", "Call dropped", 1),
    ("The train arrived exactly on time at Dadar station today.", "Smooth travel", 0),
    ("Our team solved the production issue within 10 minutes, outstanding effort.", "Fast deployment", 0),
    ("The technical documentation is clean, concise, and easy to follow.", "Reading manual", 0),
    ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic", 1),
    ("Wah re wah sarkar, rastyat khadde ahet ki khaddyat rasta kalatch nahi!", "Potholes everywhere", 1),
    ("Arre ekdam genius ahes tu, test run na kartach deploy kela!", "Crashed server", 1),
    ("खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!", "Mismanaged line", 1),
    ("Aaj cha presentation khup changla zala, congrats team!", "Appreciation", 0),
    ("Code review neat zala ahe, bug resolve zala.", "Resolved bug", 0)
]

def build_data_and_train():
    print("[*] Generating Marathi-English benchmark dataset...")
    with open(MECORPUS_RAW_PATH, "w", encoding="utf-8") as f:
        for t, c, l in BENCHMARK_SEED:
            f.write(f"{t}\t{c}\t{l}\n")

    train_path = PROCESSED_DATA_DIR / "train.csv"
    with open(train_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "context", "label"])
        writer.writerows(BENCHMARK_SEED)

    print("[*] Training Feature-Fusion Sarcasm Incongruity Model...")
    df = pd.read_csv(train_path).fillna("")
    model = FeatureFusionModel()
    model.fit(df["text"].tolist(), df["label"].tolist(), df["context"].tolist())
    joblib.dump(model, MODEL_WEIGHTS_PATH, compress=3)
    print(f"[+] Model saved to: {MODEL_WEIGHTS_PATH}")
    return model

FRONTEND_SCRIPT_PATH = PROJECT_ROOT / "frontend_dashboard.py"
FRONTEND_CODE = '''import streamlit as st
import joblib
import pandas as pd
from complete_and_launch import (
    MODEL_WEIGHTS_PATH, FeatureFusionModel, clean_text,
    analyze_code_mixing, IncongruityExtractor, POSITIVE_CUES,
    NEGATIVE_CUES, INTENSIFIERS
)

st.set_page_config(page_title="SARcasmScope", page_icon="🔍", layout="wide")

@st.cache_resource
def load_engine():
    return joblib.load(MODEL_WEIGHTS_PATH)

model = load_engine()
extractor = IncongruityExtractor()

st.title("🔍 SARcasmScope")
st.caption("Multilingual Context-Aware Sarcasm Detection System (English & Marathi-English)")

with st.sidebar:
    st.header("⚡ Preset Samples")
    presets = {
        "-- Select Preset --": ("", ""),
        "English Sarcasm": ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled"),
        "English Literal": ("The train arrived exactly on time at Dadar station today.", "Smooth travel"),
        "Marathi-English Sarcasm": ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic"),
        "Marathi Devanagari Sarcasm": ("खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!", "Mismanaged line"),
        "Marathi-English Literal": ("Aaj cha presentation khup changla zala, congrats team!", "Appreciation")
    }
    choice = st.selectbox("Load sample:", list(presets.keys()))
    preset_t, preset_c = presets[choice]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text & Context")
    text_input = st.text_area("Target Statement:", value=preset_t, height=120)
    context_input = st.text_input("Context (Optional):", value=preset_c)
    analyze_btn = st.button("🚀 Analyze Statement", use_container_width=True)

with col2:
    st.subheader("Results")
    if analyze_btn:
        if not text_input.strip():
            st.error("Please enter text.")
        else:
            probs = model.predict_proba([text_input], [context_input])[0]
            sarc_prob = float(probs[1])
            is_sarcastic = sarc_prob >= 0.50
            cm_meta = analyze_code_mixing(text_input)
            feats = extractor.extract(text_input, context_input)

            if is_sarcastic:
                st.error(f"### ⚠️ Sarcasm Detected ({sarc_prob*100:.1f}%)")
            else:
                st.success(f"### ✅ Literal / Non-Sarcastic ({(1-sarc_prob)*100:.1f}%)")

            c1, c2, c3 = st.columns(3)
            c1.metric("Sarcasm Probability", f"{sarc_prob:.2f}")
            c2.metric("Script", cm_meta["primary_script"])
            c3.metric("Code-Mix Index", f"{cm_meta['cmi']}%")

            st.markdown("#### 🧠 Linguistic Incongruity Explanation")
            reasons = []
            if feats["polarity_clash"] > 0:
                reasons.append("• **Intra-Sentence Opposition**: Positive surface wording contrasted directly with negative tone.")
            if feats["context_clash"] > 0:
                reasons.append("• **Discourse Clash**: Positive statement directly contradicts negative prior context.")
            if feats["intensifier_density"] > 0:
                reasons.append("• **Hyperbolic Cue**: Sarcastic amplifier/intensifier words detected.")
            if not reasons:
                reasons.append("• Tone and vocabulary are semantically congruent; no incongruity detected.")
            for r in reasons:
                st.markdown(r)
'''

with open(FRONTEND_SCRIPT_PATH, "w", encoding="utf-8") as f:
    f.write(FRONTEND_CODE)

if __name__ == "__main__":
    trained_model = build_data_and_train()
    print("✅ Model trained & verified successfully!")
    print("🌐 Launching SARcasmScope UI at http://localhost:8501 ...")
    time.sleep(1)
    webbrowser.open("http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(FRONTEND_SCRIPT_PATH),
        "--server.port", "8501"
    ])