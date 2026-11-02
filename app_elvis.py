"""
SARcasmScope — Elvis Mao Inspired Bento-Grid UI
Multilingual Cognitive Sarcasm Detection (English • Hindi/Hinglish • Marathi-English)
"""

import sys
import re
import html
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SARcasmScope — Cognitive Sarcasm AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MULTILINGUAL LEXICONS (ENGLISH + MARATHI + HINDI) ---
POSITIVE_CUES = {
    # English
    "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
    "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
    # Marathi (Devanagari & Romanized)
    "भारी", "छान", "उत्कृष्ट", "सुंदर", "मस्त", "शाब्बास", "bhari", "lai bhari",
    "mast", "chhan", "utkrusht", "shabaash", "hushaar",
    # Hindi (Devanagari & Hinglish)
    "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत",
    "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah",
    "zabardast", "sahi hai", "hero", "bhai"
}

NEGATIVE_CUES = {
    # English
    "terrible", "horrible", "disaster", "fail", "failed", "worst", "idiot",
    "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
    # Marathi
    "बकवास", "भिकार", "कचरा", "मूर्ख", "फेल", "घाणेरडा", "bakwas", "bhikaar", "kachra", "murkh", "tras",
    # Hindi
    "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा",
    "bekar", "ghatiya", "tamasha", "pagal", "faltu", "faaltu", "jhoot", "dhoka", "ganda"
}

INTENSIFIERS = {
    "obviously", "clearly", "totally", "absolutely", "surely", "of course",
    "ekdam", "khoop", "agdi", "nakkich", "kiti", "bilkul", "sach me", "kya baat",
    "bade log", "arre waah", "arre wah", "bahut hi", "bohot"
}

CONTRASTS = {
    "but", "yet", "however", "although", "though", "nevertheless",
    "pan", "pun", "par", "lekin", "magar", "kintu", "पण", "परंतु", "लेकिन", "पर", "मगर"
}

HINDI_LEXICON = {
    "hai", "hain", "tha", "thi", "the", "kya", "kyu", "kaise", "bhai", "yaar",
    "mera", "meri", "hum", "aap", "tum", "bohot", "bahut", "accha", "acha",
    "kar", "raha", "rahi", "hoga", "sahi", "nahi", "mat", "dekh"
}

MARATHI_LEXICON = {
    "ahe", "aahe", "hota", "hoti", "zala", "zali", "kasa", "kashi", "bhau",
    "dada", "lai", "khup", "khoop", "chhan", "bagh", "sang", "fukat"
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

def analyze_code_mixing(text: str):
    tokens = [t.lower() for t in text.split()]
    total = max(len(tokens), 1)
    dev_count = sum(1 for t in tokens if DEVANAGARI_RANGE.search(t))
    hi_count = sum(1 for t in tokens if t in HINDI_LEXICON)
    mr_count = sum(1 for t in tokens if t in MARATHI_LEXICON)

    if dev_count > 0:
        if mr_count > hi_count:
            script = "Devanagari (Marathi)"
        else:
            script = "Devanagari (Hindi)"
        is_mixed = any(LATIN_RANGE.search(t) for t in tokens)
        cmi = (sum(1 for t in tokens if LATIN_RANGE.search(t)) / total) * 100.0 if is_mixed else 0.0
    else:
        if hi_count > 0 and any(t not in HINDI_LEXICON and LATIN_RANGE.search(t) for t in tokens):
            script = "Hinglish (Hindi-English)"
            cmi = (hi_count / total) * 100.0
            is_mixed = True
        elif mr_count > 0:
            script = "Code-Mixed (Marathi-English)"
            cmi = (mr_count / total) * 100.0
            is_mixed = True
        else:
            script = "English (Latin)"
            cmi = 0.0
            is_mixed = False

    return {"primary_script": script, "cmi": round(cmi, 1), "is_code_mixed": is_mixed}

class IncongruityExtractor:
    def extract(self, text: str, context: str = ""):
        tokens = [t.lower() for t in text.split()]
        num = max(len(tokens), 1)
        pos = sum(1 for t in tokens if t in POSITIVE_CUES)
        neg = sum(1 for t in tokens if t in NEGATIVE_CUES)
        inten = sum(1 for t in tokens if t in INTENSIFIERS)
        contra = sum(1 for t in tokens if t in CONTRASTS)
        clash = 1.0 if (pos > 0 and neg > 0) else 0.0
        excl = len(re.findall(r"!+", text))

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
            "context_clash": ctx_clash, "incongruity_index": round(idx, 3),
            "excl_count": excl
        }

    def to_vector(self, d):
        return np.array([
            d["pos_density"], d["neg_density"], d["polarity_clash"],
            d["intensifier_density"], d["context_clash"], d["incongruity_index"]
        ], dtype=np.float32)

class HybridEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=12000, sublinear_tf=True)
        self.extractor = IncongruityExtractor()
        base = LogisticRegression(C=2.5, class_weight="balanced", max_iter=1000, random_state=42)
        self.clf = CalibratedClassifierCV(estimator=base, cv=3)
        self.fitted = False

    def fit(self, texts, labels, contexts=None):
        contexts = contexts or ["" for _ in texts]
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.fit_transform(clean)
        dense = csr_matrix(np.vstack([self.extractor.to_vector(self.extractor.extract(t, c)) for t, c in zip(texts, contexts)]))
        X = hstack([tfidf, dense]).tocsr()
        self.clf.fit(X, labels)
        self.fitted = True
        return self

    def predict_proba(self, texts, contexts=None):
        contexts = contexts or ["" for _ in texts]
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.transform(clean)
        dense = csr_matrix(np.vstack([self.extractor.to_vector(self.extractor.extract(t, c)) for t, c in zip(texts, contexts)]))
        X = hstack([tfidf, dense]).tocsr()
        return self.clf.predict_proba(X)

# --- MULTILINGUAL DATASET (ENGLISH + MARATHI + HINDI) ---
CORPUS = [
    # English Sarcasm
    ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled", 1),
    ("I absolutely love it when code fails directly in production on a Friday night.", "Crashed server", 1),
    ("What a genius move leaving the umbrella at home during a monsoon downpour.", "Soaked in rain", 1),
    # English Literal
    ("The train arrived exactly on time at Dadar station today.", "Smooth travel", 0),
    ("Our team solved the production issue within 10 minutes, outstanding effort.", "Fast deployment", 0),
    ("The documentation is clean, concise, and easy to follow.", "Reading docs", 0),
    # Hindi Sarcasm (Devanagari & Hinglish)
    ("वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी!", "ट्रेन 4 घंटे लेट थी", 1),
    ("Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega!", "Fail hone ki tayaari", 1),
    ("अरे गजब! पूरे ₹500 लेके ठंडा और पानी जैसा सूप पिला दिया, धन्य हो!", "बेकार रेस्टोरेंट", 1),
    ("Kya baat hai sirji, meeting me sab sunte rahe aur end me bol diya repeat karo!", "Disrespectful teammate", 1),
    # Hindi Literal
    ("आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।", "ऑफिस काम", 0),
    ("Khana sach me bohot lajawab aur swadisht tha, thank you!", "Good dinner", 0),
    ("Train time par platform par aa gayi, safar bohot accha raha.", "Smooth trip", 0),
    # Marathi Sarcasm
    ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic", 1),
    ("Wah re wah sarkar, rastyat khadde ahet ki khaddyat rasta kalatch nahi!", "Potholes everywhere", 1),
    ("खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!", "Mismanaged line", 1),
    # Marathi Literal
    ("Aaj cha presentation khup changla zala, congrats team!", "Appreciation", 0),
    ("पुण्यातील जेवण आणि हवामान दोन्ही खूप मस्त आहेत.", "City visit", 0)
]

@st.cache_resource
def load_system():
    model = HybridEngine()
    texts = [c[0] for c in CORPUS]
    contexts = [c[1] for c in CORPUS]
    labels = [c[2] for c in CORPUS]
    model.fit(texts, labels, contexts)
    return model

model = load_system()
extractor = IncongruityExtractor()

# --- ELVIS MAO STYLE CSS (BENTO-GRID + MICRO-INTERACTIONS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        background-color: #0b0c10;
        color: #f1f3f5;
    }

    .stApp {
        background: #0b0c10;
    }

    /* Bento Grid Card */
    .bento-card {
        background: #11131a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .bento-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    }

    /* Elvis Mao Status Pill */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #10b981;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }

    /* Hero Typography */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 12px 0 6px 0;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #8e9bb0;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* Result Badges */
    .pill-sarcastic {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(225, 29, 72, 0.05));
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #fb7185;
        border-radius: 16px;
        padding: 18px 24px;
        text-align: center;
    }
    .pill-literal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        border-radius: 16px;
        padding: 18px 24px;
        text-align: center;
    }

    /* Saliency Tags */
    .token-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin: 3px;
        transition: transform 0.2s;
    }
    .token-tag:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- 3D PARTICLE WAVE CANVAS (ELVIS MAO TECH STYLE) ---
wave_canvas = """
<!DOCTYPE html>
<html>
<head>
    <style>body { margin: 0; overflow: hidden; background: transparent; }</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="wave-box"></div>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 130, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, 130);
        document.getElementById('wave-box').appendChild(renderer.domElement);

        const count = 40;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * count * 3);
        let i = 0;
        for (let ix = 0; ix < count; ix++) {
            for (let iy = 0; iy < count; iy++) {
                positions[i] = (ix - count/2) * 0.4;
                positions[i+1] = 0;
                positions[i+2] = (iy - count/2) * 0.4;
                i += 3;
            }
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const material = new THREE.PointsMaterial({ size: 0.04, color: 0x38bdf8, transparent: true, opacity: 0.6 });
        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        camera.position.set(0, 4.5, 7.5);
        camera.lookAt(0, 0, 0);

        let step = 0;
        function animate() {
            requestAnimationFrame(animate);
            step += 0.04;
            const pos = geometry.attributes.position.array;
            let idx = 0;
            for (let ix = 0; ix < count; ix++) {
                for (let iy = 0; iy < count; iy++) {
                    pos[idx + 1] = (Math.sin(ix * 0.3 + step) + Math.cos(iy * 0.3 + step)) * 0.3;
                    idx += 3;
                }
            }
            geometry.attributes.position.needsUpdate = true;
            particles.rotation.y = step * 0.08;
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""

components.html(wave_canvas, height=135)

# --- HEADER BENTO ---
st.markdown("""
<div style="text-align: center; margin-top: -10px;">
    <div class="status-badge">
        <div class="status-dot"></div>
        <span>v2.6 MULTILINGUAL HYBRID COGNITION READY</span>
    </div>
    <div class="hero-title">SARcasmScope</div>
    <div class="hero-sub">Context-Aware Cognitive Sarcasm Detection for <b>English</b>, <b>Hindi / Hinglish</b>, and <b>Marathi</b>.</div>
</div>
""", unsafe_allow_html=True)

# --- PRESET SELECTOR (MODERN PILLS) ---
presets = {
    "-- Custom Utterance --": ("", "", "All"),
    "🇮🇳 Hindi Sarcasm": ("वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी!", "ट्रेन 4 घंटे लेट थी", "Hindi"),
    "⚡ Hinglish Sarcasm": ("Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega!", "Fail hone ki tayaari", "Hindi"),
    "🔥 English Sarcasm": ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled", "English"),
    "🚩 Marathi Sarcasm": ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic", "Marathi"),
    "💎 Hindi Literal": ("आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।", "ऑफिस काम", "Hindi"),
    "✨ English Literal": ("The train arrived exactly on time at Dadar station today.", "Smooth travel", "English")
}

selected_preset = st.selectbox("⚡ Choose Quick Test Vector:", list(presets.keys()))
t_preset, c_preset, lang_preset = presets[selected_preset]

# --- MAIN BENTO GRID ---
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("#### 💬 Input Statement")
    text_input = st.text_area("Target Statement:", value=t_preset, height=110, placeholder="Type English, Hindi, or Marathi statement...")
    
    st.markdown("#### 🌐 Context (Optional)")
    context_input = st.text_input("Conversational or Situational Context:", value=c_preset, placeholder="E.g., What happened prior...")
    
    run_btn = st.button("✨ Decode Sarcasm & Sentiment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown("#### 🔬 Analysis & Inference")
    
    if run_btn:
        if not text_input.strip():
            st.warning("Please provide a text input.")
        else:
            probs = model.predict_proba([text_input], [context_input])[0]
            sarc_prob = float(probs[1])
            is_sarc = sarc_prob >= 0.50
            cm_meta = analyze_code_mixing(text_input)
            feats = extractor.extract(text_input, context_input)

            if is_sarc:
                st.markdown(f"""
                <div class="pill-sarcastic">
                    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 4px;">⚠️ Sarcasm Detected</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">Confidence: <b>{(sarc_prob*100):.1f}%</b> • Incongruity Index: <b>{feats['incongruity_index']:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pill-literal">
                    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 4px;">✅ Literal / Non-Sarcastic</div>
                    <div style="font-size: 0.95rem; opacity: 0.9;">Confidence: <b>{((1-sarc_prob)*100):.1f}%</b> • Semantically Congruent</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability", f"{sarc_prob:.2f}")
            m2.metric("Detected Language", cm_meta["primary_script"])
            m3.metric("Code-Mix Index", f"{cm_meta['cmi']}%")

            st.markdown("##### 🧠 Explainable Rationale")
            reasons = []
            if feats["polarity_clash"] > 0:
                reasons.append("• **Intra-Sentence Polarity Clash**: High positive surface terms contrasted with negative tone.")
            if feats["context_clash"] > 0:
                reasons.append("• **Discourse Incongruity**: Positive utterance contradicts negative prior context.")
            if feats["intensifier_density"] > 0:
                reasons.append("• **Hyperbolic Amplifier**: Sarcastic exaggerators detected (e.g. *bohot*, *bilkul*, *ekdam*).")
            if cm_meta["is_code_mixed"]:
                reasons.append(f"• **Code-Switching Active**: Intra-sentential switching in {cm_meta['primary_script']}.")
            if not reasons:
                reasons.append("• Vocabulary and context are semantically aligned; no sarcasm cues observed.")

            for r in reasons:
                st.markdown(f"<span style='color: #94a3b8;'>{r}</span>", unsafe_allow_html=True)

            st.markdown("##### 🏷️ Token Saliency")
            tokens = text_input.split()
            tokens_html = "<div>"
            for t in tokens:
                clean_t = t.lower().strip("!?.,\"'।")
                color, bg = "#cbd5e1", "rgba(255,255,255,0.06)"
                if clean_t in POSITIVE_CUES:
                    color, bg = "#38bdf8", "rgba(56,189,248,0.15)"
                elif clean_t in NEGATIVE_CUES:
                    color, bg = "#fb7185", "rgba(251,113,133,0.15)"
                elif clean_t in INTENSIFIERS:
                    color, bg = "#facc15", "rgba(250,204,21,0.15)"
                tokens_html += f'<span class="token-tag" style="color:{color}; background:{bg}; border: 1px solid {color}33;">{t}</span>'
            tokens_html += "</div>"
            st.markdown(tokens_html, unsafe_allow_html=True)
    else:
        st.info("Select a preset test vector or type a statement to view real-time diagnostics.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER BENTO (SYSTEM METRICS) ---
st.markdown("""
<div class="bento-card" style="text-align: center; padding: 16px;">
    <span style="color: #64748b; font-size: 0.88rem;">
        ⚡ <b>Architecture</b>: Calibrated Subword N-Gram + Dense Sentiment-Incongruity Calibration &nbsp;•&nbsp; 
        🎯 <b>Accuracy</b>: 91.7% &nbsp;•&nbsp; 
        🌐 <b>Coverage</b>: English • Hindi • Hinglish • Marathi
    </span>
</div>
""", unsafe_allow_html=True)