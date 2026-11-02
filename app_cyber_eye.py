"""
SARcasmScope — Luxury Studio Edition
Minimalist, high-end multilingual sarcasm detection (English • Hindi • Hinglish • Marathi).
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
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="SARcasmScope",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- NLP ENGINE & LEXICONS (HINDI + MARATHI + ENGLISH) ---
POSITIVE_CUES = {
    "brilliant", "genius", "amazing", "wonderful", "outstanding", "perfect",
    "love", "lovely", "blessed", "greatest", "fantastic", "superb", "flawless", "hero",
    "बढ़िया", "शानदार", "कमाल", "लाजवाब", "वाह", "गजब", "महान", "शाबाश", "सुंदर", "अद्भुत",
    "badiya", "bohot badiya", "gazab", "shandar", "kamaal", "lajawab", "waah", "wah", "zabardast", "sahi",
    "भारी", "छान", "उत्कृष्ट", "सुंदर", "मस्त", "शाब्बास", "bhari", "lai bhari", "mast", "chhan"
}

NEGATIVE_CUES = {
    "terrible", "horrible", "disaster", "fail", "failed", "worst", "idiot",
    "useless", "garbage", "trash", "broken", "nightmare", "clown", "stupid", "pathetic", "ruined",
    "बेकार", "घटिया", "तमाशा", "पागल", "फालतू", "झूठ", "धोखा", "गंदा", "कचरा",
    "bekar", "ghatiya", "tamasha", "pagal", "faltu", "jhoot", "dhoka", "ganda",
    "बकवास", "भिकार", "मूर्ख", "घाणेरडा", "bakwas", "bhikaar", "kachra", "murkh", "tras"
}

INTENSIFIERS = {
    "obviously", "clearly", "totally", "absolutely", "surely", "of course",
    "ekdam", "khoop", "agdi", "nakkich", "bilkul", "sach me", "kya baat", "bade log", "arre waah", "bohot"
}

CONTRASTS = {
    "but", "yet", "however", "although", "though", "nevertheless",
    "pan", "pun", "par", "lekin", "magar", "kintu", "पण", "परंतु", "लेकिन", "पर", "मगर"
}

HINDI_WORDS = {
    "hai", "hain", "tha", "thi", "the", "kya", "kyu", "kaise", "bhai", "yaar",
    "mera", "meri", "hum", "aap", "tum", "bohot", "bahut", "accha", "kar", "raha", "sahi", "nahi"
}

MARATHI_WORDS = {
    "ahe", "aahe", "hota", "hoti", "zala", "zali", "kasa", "kashi", "bhau",
    "dada", "lai", "khup", "chhan", "bagh", "sang", "fukat"
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
    hi_count = sum(1 for t in tokens if t in HINDI_WORDS)
    mr_count = sum(1 for t in tokens if t in MARATHI_WORDS)

    if dev_count > 0:
        script = "Devanagari (Marathi)" if mr_count > hi_count else "Devanagari (Hindi)"
        is_mixed = any(LATIN_RANGE.search(t) for t in tokens)
        cmi = (sum(1 for t in tokens if LATIN_RANGE.search(t)) / total) * 100.0 if is_mixed else 0.0
    else:
        if hi_count > 0 and any(t not in HINDI_WORDS and LATIN_RANGE.search(t) for t in tokens):
            script = "Hinglish (Hindi-English)"
            cmi = (hi_count / total) * 100.0
            is_mixed = True
        elif mr_count > 0:
            script = "Marathi-English Code-Mixed"
            cmi = (mr_count / total) * 100.0
            is_mixed = True
        else:
            script = "English"
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

        ctx_clash = 0.0
        if context:
            ctx_tok = [t.lower() for t in context.split()]
            ctx_neg = sum(1 for t in ctx_tok if t in NEGATIVE_CUES)
            if ctx_neg > 0 and pos > 0:
                ctx_clash = 1.0

        idx = (clash * 0.4) + (ctx_clash * 0.35) + (min(contra, 2) * 0.15) + (min(inten, 2) * 0.10)
        return {
            "pos_density": float(pos / num),
            "neg_density": float(neg / num),
            "polarity_clash": float(clash),
            "intensifier_density": float(inten / num),
            "context_clash": float(ctx_clash),
            "incongruity_index": float(round(idx, 3))
        }

    def to_vector(self, d):
        return np.array([
            d["pos_density"], d["neg_density"], d["polarity_clash"],
            d["intensifier_density"], d["context_clash"], d["incongruity_index"]
        ], dtype=np.float32)

class CoreEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
        self.extractor = IncongruityExtractor()
        self.clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)

    def fit(self, texts, labels, contexts=None):
        contexts = contexts or ["" for _ in texts]
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.fit_transform(clean)
        dense = csr_matrix(np.vstack([self.extractor.to_vector(self.extractor.extract(t, c)) for t, c in zip(texts, contexts)]))
        X = hstack([tfidf, dense]).tocsr()
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, texts, contexts=None):
        contexts = contexts or ["" for _ in texts]
        clean = [clean_text(t) for t in texts]
        tfidf = self.vectorizer.transform(clean)
        dense = csr_matrix(np.vstack([self.extractor.to_vector(self.extractor.extract(t, c)) for t, c in zip(texts, contexts)]))
        X = hstack([tfidf, dense]).tocsr()
        return self.clf.predict_proba(X)

CORPUS = [
    # English Sarcasm
    ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled", 1),
    ("I absolutely love it when code fails directly in production on a Friday night.", "Crashed server", 1),
    ("What a genius move leaving the umbrella at home during a monsoon downpour.", "Soaked in rain", 1),
    # English Literal
    ("The train arrived exactly on time at Dadar station today.", "Smooth travel", 0),
    ("Our team solved the production issue within 10 minutes, outstanding effort.", "Fast deployment", 0),
    ("The technical documentation is clean, concise, and easy to follow.", "Reading manual", 0),
    # Hindi Sarcasm
    ("वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी!", "ट्रेन 4 घंटे लेट थी", 1),
    ("Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega!", "Fail hone ki tayaari", 1),
    ("अरे गजब! पूरे ₹500 लेके ठंडा और पानी जैसा सूप पिला दिया, धन्य हो!", "बेकार रेस्टोरेंट", 1),
    # Hindi Literal
    ("आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।", "ऑफिस काम", 0),
    ("Train time par platform par aa gayi, safar bohot accha raha.", "Smooth trip", 0),
    ("Khana sach me bohot lajawab aur swadisht tha, thank you!", "Dinner", 0),
    # Marathi Sarcasm
    ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic", 1),
    ("Wah re wah sarkar, rastyat khadde ahet ki khaddyat rasta kalatch nahi!", "Potholes everywhere", 1),
    # Marathi Literal
    ("Aaj cha presentation khup changla zala, congrats team!", "Appreciation", 0),
    ("पुण्यातील जेवण आणि हवामान दोन्ही खूप मस्त आहेत.", "City visit", 0)
]

@st.cache_resource
def get_model():
    m = CoreEngine()
    m.fit([c[0] for c in CORPUS], [c[2] for c in CORPUS], [c[1] for c in CORPUS])
    return m

model = get_model()
extractor = IncongruityExtractor()

# --- LUXURY MINIMALIST STYLING (LINEAR / RAYCAST VIBE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        background: #07080c !important;
        color: #e2e8f0;
    }
    .stApp {
        background: radial-gradient(ellipse at top, #11131f 0%, #07080c 60%);
    }

    /* Minimalist Cards */
    .lux-card {
        background: rgba(15, 17, 26, 0.65);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        transition: border-color 0.3s ease;
    }
    .lux-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
    }

    /* Clean Studio Pill */
    .studio-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        margin-bottom: 8px;
    }

    /* Modern Title */
    .studio-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .studio-sub {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 24px;
    }

    /* Result Banners */
    .verdict-sarcasm {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(244, 63, 94, 0.02) 100%);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .verdict-literal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0.02) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .token-chip {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3D AMBIENT QUANTUM ORB (SUBTLE & INTERACTIVE) ---
quantum_orb_js = """
<!DOCTYPE html>
<html>
<head>
    <style>body { margin: 0; overflow: hidden; background: transparent; }</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="orb-box"></div>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / 120, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, 120);
        document.getElementById('orb-box').appendChild(renderer.domElement);

        // Core Quantum Rings
        const group = new THREE.Group();
        scene.add(group);

        const ringGeo1 = new THREE.TorusGeometry(1.5, 0.015, 16, 100);
        const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.65 });
        const ring1 = new THREE.Mesh(ringGeo1, ringMat1);
        group.add(ring1);

        const ringGeo2 = new THREE.TorusGeometry(1.2, 0.015, 16, 100);
        const ringMat2 = new THREE.MeshBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0.5 });
        const ring2 = new THREE.Mesh(ringGeo2, ringMat2);
        ring2.rotation.x = Math.PI / 3;
        group.add(ring2);

        const ringGeo3 = new THREE.TorusGeometry(0.9, 0.012, 16, 100);
        const ringMat3 = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.45 });
        const ring3 = new THREE.Mesh(ringGeo3, ringMat3);
        ring3.rotation.y = Math.PI / 4;
        group.add(ring3);

        // Ambient particles
        const particleCount = 120;
        const pGeo = new THREE.BufferGeometry();
        const pPos = new Float32Array(particleCount * 3);
        for(let i=0; i < particleCount * 3; i++) {
            pPos[i] = (Math.random() - 0.5) * 5;
        }
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        const pMat = new THREE.PointsMaterial({ size: 0.03, color: 0x818cf8, transparent: true, opacity: 0.6 });
        const particleMesh = new THREE.Points(pGeo, pMat);
        scene.add(particleMesh);

        camera.position.z = 4.2;

        let mouseX = 0, mouseY = 0;
        window.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth) - 0.5;
            mouseY = (e.clientY / 120) - 0.5;
        });

        function animate() {
            requestAnimationFrame(animate);
            ring1.rotation.x += 0.006;
            ring1.rotation.y += 0.008;
            ring2.rotation.y += 0.007;
            ring3.rotation.z += 0.009;

            group.rotation.x += (mouseY * 0.4 - group.rotation.x) * 0.05;
            group.rotation.y += (mouseX * 0.4 - group.rotation.y) * 0.05;

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            renderer.setSize(window.innerWidth, 120);
            camera.aspect = window.innerWidth / 120;
            camera.updateProjectionMatrix();
        });
    </script>
</body>
</html>
"""

components.html(quantum_orb_js, height=125)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; margin-top: -10px;">
    <div class="studio-pill">✦ SARcasmScope Studio</div>
    <div class="studio-title">Cognitive Sarcasm Intelligence</div>
    <div class="studio-sub">Context-aware sentiment incongruity detection across English, Hindi, Hinglish, and Marathi.</div>
</div>
""", unsafe_allow_html=True)

# --- PRESET SELECTOR ---
presets = {
    "Select a pre-loaded sample...": ("", ""),
    "🇮🇳 Hindi Sarcasm": ("वाह क्या बात है, 4 घंटे ट्रेन लेट करने के लिए रेलवे को 21 तोपों की सलामी!", "ट्रेन 4 घंटे लेट थी"),
    "⚡ Hinglish Sarcasm": ("Bohot badiya bhai, exam se ek raat pehle phone pe reels dekh, pakka top karega!", "Fail hone ki tayaari"),
    "🔥 English Sarcasm": ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled"),
    "🚩 Marathi Sarcasm": ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic"),
    "💎 Hindi Literal": ("आज की बैठक बहुत ही उत्पादक रही और समय पर समाप्त हो गई।", "ऑफिस काम"),
    "✨ English Literal": ("The train arrived exactly on time at Dadar station today.", "Smooth travel")
}

choice = st.selectbox("", list(presets.keys()), label_visibility="collapsed")
t_val, c_val = presets[choice]

# --- MAIN WORKSPACE ---
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-weight: 600; font-size: 0.95rem; margin-bottom: 6px;'>Target Statement</p>", unsafe_allow_html=True)
    in_text = st.text_area("Input statement", value=t_val, height=115, placeholder="Enter statement in English, Hindi, or Marathi...", label_visibility="collapsed")

    st.markdown("<p style='font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; margin-top: 14px;'>Conversational Context <span style='color: #64748b; font-weight: 400;'>(Optional)</span></p>", unsafe_allow_html=True)
    in_context = st.text_input("Context", value=c_val, placeholder="What happened prior (e.g. Flight was delayed)", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✦ Analyze Sarcasm & Sentiment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_out:
    st.markdown('<div class="lux-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-weight: 600; font-size: 0.95rem; margin-bottom: 12px;'>Cognitive Verdict</p>", unsafe_allow_html=True)

    if run_btn:
        if not in_text.strip():
            st.warning("Please enter a statement.")
        else:
            probs = model.predict_proba([in_text], [in_context])[0]
            sarc_prob = float(probs[1])
            is_sarc = sarc_prob >= 0.50
            cm = analyze_code_mixing(in_text)
            feats = extractor.extract(in_text, in_context)

            if is_sarc:
                st.markdown(f"""
                <div class="verdict-sarcasm">
                    <div style="font-size: 1.35rem; font-weight: 700; color: #fb7185; margin-bottom: 4px;">⚠️ Sarcastic Intent Detected</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1;">Confidence: <b>{sarc_prob*100:.1f}%</b> • Incongruity Index: <b>{feats['incongruity_index']:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-literal">
                    <div style="font-size: 1.35rem; font-weight: 700; color: #34d399; margin-bottom: 4px;">✦ Literal Statement</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1;">Confidence: <b>{(1-sarc_prob)*100:.1f}%</b> • Semantically Congruent</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability", f"{sarc_prob:.2f}")
            m2.metric("Language / Script", cm["primary_script"])
            m3.metric("Code-Mix Index", f"{cm['cmi']}%")

            st.markdown("<p style='font-weight: 600; font-size: 0.88rem; color: #94a3b8; margin-top: 14px; margin-bottom: 6px;'>WHY THIS VERDICT</p>", unsafe_allow_html=True)
            reasons = []
            if feats["polarity_clash"] > 0:
                reasons.append("• **Intra-Sentence Opposition**: Positive wording directly contradicted by underlying negative tone.")
            if feats["context_clash"] > 0:
                reasons.append("• **Contextual Clash**: Statement opposes the negative preceding context.")
            if feats["intensifier_density"] > 0:
                reasons.append("• **Hyperbolic Amplifier**: Exaggerators detected typical of sarcastic delivery.")
            if cm["is_code_mixed"]:
                reasons.append(f"• **Code-Switching Active**: Intra-sentential switching detected in {cm['primary_script']}.")
            if not reasons:
                reasons.append("• Tone and vocabulary are semantically congruent with the context.")

            for r in reasons:
                st.markdown(f"<span style='color: #cbd5e1; font-size: 0.88rem;'>{r}</span>", unsafe_allow_html=True)

            st.markdown("<p style='font-weight: 600; font-size: 0.88rem; color: #94a3b8; margin-top: 14px; margin-bottom: 6px;'>TOKEN ATTRIBUTION</p>", unsafe_allow_html=True)
            tokens = in_text.split()
            tokens_html = "<div>"
            for t in tokens:
                clean_t = t.lower().strip("!?.,\"'।")
                color, bg = "#94a3b8", "rgba(255,255,255,0.05)"
                if clean_t in POSITIVE_CUES:
                    color, bg = "#38bdf8", "rgba(56,189,248,0.15)"
                elif clean_t in NEGATIVE_CUES:
                    color, bg = "#fb7185", "rgba(251,113,133,0.15)"
                elif clean_t in INTENSIFIERS:
                    color, bg = "#facc15", "rgba(250,204,21,0.15)"
                tokens_html += f'<span class="token-chip" style="color:{color}; background:{bg}; border: 1px solid {color}33;">{t}</span>'
            tokens_html += "</div>"
            st.markdown(tokens_html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #64748b; font-size: 0.9rem; padding: 20px 0;'>Select a preset sample above or type a custom statement to see real-time analysis.</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)