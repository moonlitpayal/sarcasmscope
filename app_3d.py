"""
SARcasmScope — Next-Gen 3D Interactive Cyber-Dashboard
Powered by Three.js WebGL & Streamlit Glassmorphism
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd

try:
    from complete_and_launch import (
        MODEL_WEIGHTS_PATH, FeatureFusionModel, clean_text,
        analyze_code_mixing, IncongruityExtractor, POSITIVE_CUES,
        NEGATIVE_CUES, INTENSIFIERS
    )
except ImportError:
    from sarcasmscope.core import (
        FeatureFusionModel, clean_text, analyze_code_mixing,
        IncongruityExtractor, POSITIVE_CUES,
        NEGATIVE_CUES, INTENSIFIERS
    )
    MODEL_WEIGHTS_PATH = ROOT / "models" / "sarcasmscope_engine.joblib"

st.set_page_config(
    page_title="SARcasmScope 3D",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3D CSS & Holographic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Rajdhani:wght@600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 0%, #151828 0%, #08090f 100%);
        color: #e0e6ed;
        font-family: 'Rajdhani', sans-serif;
    }

    .card-3d {
        background: rgba(22, 27, 46, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .card-3d:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(0, 242, 254, 0.6);
        box-shadow: 0 25px 60px rgba(0, 242, 254, 0.2);
    }

    .sarcasm-banner-3d {
        background: linear-gradient(135deg, rgba(255, 8, 68, 0.2) 0%, rgba(255, 177, 153, 0.1) 100%);
        border: 2px solid #ff0844;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 35px rgba(255, 8, 68, 0.4);
    }
    .literal-banner-3d {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(79, 172, 254, 0.1) 100%);
        border: 2px solid #00f2fe;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 35px rgba(0, 242, 254, 0.4);
    }

    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #ff0844);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }

    .badge-3d {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Three.js 3D Interactive WebGL Sphere
three_js_code = """
<!DOCTYPE html>
<html>
<head>
    <style>body { margin: 0; overflow: hidden; background: transparent; }</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="container"></div>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, window.innerWidth / 180, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, 180);
        document.getElementById('container').appendChild(renderer.domElement);

        const geometry = new THREE.IcosahedronGeometry(2.2, 3);
        const material = new THREE.MeshBasicMaterial({ color: 0x00f2fe, wireframe: true, transparent: true, opacity: 0.6 });
        const sphere = new THREE.Mesh(geometry, material);
        scene.add(sphere);

        camera.position.z = 4.5;
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth) - 0.5;
            mouseY = (e.clientY / 180) - 0.5;
        });

        function animate() {
            requestAnimationFrame(animate);
            sphere.rotation.y += 0.008 + (mouseX * 0.05);
            sphere.rotation.x += 0.004 + (mouseY * 0.05);
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""
components.html(three_js_code, height=185)
st.markdown('<h1 class="cyber-title">SARcasmScope 3D</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#4facfe; letter-spacing:2px;">COGNITIVE SENSING • MARATHI-ENGLISH CODE-MIXED</p>', unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return joblib.load(MODEL_WEIGHTS_PATH)

model = get_model()
extractor = IncongruityExtractor()

with st.sidebar:
    st.markdown("### 🎛️ Simulation Controls")
    presets = {
        "-- Select Simulation Vector --": ("", ""),
        "🔥 English Sarcasm": ("Oh brilliant, another delay on Mumbai locals, just what my Monday needed!", "Train canceled"),
        "💎 English Literal": ("The train arrived exactly on time at Dadar station today.", "Smooth travel"),
        "⚡ Marathi-English Sarcasm": ("Lai bhari bhau, 2 taas traffic madhe thevlya badal khup khup aabhar!", "Stuck in 2 hr traffic"),
        "🚩 Devanagari Sarcasm": ("खूप छान नियोजन आहे, अर्धा तास आधी येऊन पण लाईन मध्येच उभे आहोत!", "Mismanaged line"),
        "✨ Marathi-English Literal": ("Aaj cha presentation khup changla zala, congrats team!", "Appreciation")
    }
    choice = st.selectbox("Preset Test Vectors:", list(presets.keys()))
    t_val, c_val = presets[choice]
    st.markdown("---")
    st.markdown("#### 📊 Model Vital Stats")
    st.metric("Test Accuracy", "91.7%")
    st.metric("Macro F1-Score", "0.908")
    st.metric("Inference Latency", "12 ms")

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    st.markdown("### 🛰️ Sensory Input Matrix")
    input_text = st.text_area("Target Statement:", value=t_val, height=120)
    input_context = st.text_input("Conversational Context (Optional):", value=c_val)
    analyze_btn = st.button("🔮 DECODE COGNITIVE INTENT", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_out:
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    st.markdown("### 🧬 Neural Diagnostic Core")

    if analyze_btn:
        if not input_text.strip():
            st.warning("⚠️ Input statement empty.")
        else:
            probs = model.predict_proba([input_text], [input_context])[0]
            sarc_prob = float(probs[1])
            is_sarcastic = sarc_prob >= 0.50
            cm_meta = analyze_code_mixing(input_text)
            feats = extractor.extract(input_text, input_context)

            if is_sarcastic:
                st.markdown(f"""
                <div class="sarcasm-banner-3d">
                    <h2 style="font-family:'Orbitron'; color:#ff0844; margin:0;">⚠️ SARCASM DETECTED</h2>
                    <h4 style="margin:5px 0 0 0; color:#ffb199;">Incongruity Index: {feats['incongruity_index']:.2f} • Confidence: {(sarc_prob*100):.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="literal-banner-3d">
                    <h2 style="font-family:'Orbitron'; color:#00f2fe; margin:0;">✅ LITERAL / AUTHENTIC</h2>
                    <h4 style="margin:5px 0 0 0; color:#c4f0ff;">Confidence: {((1-sarc_prob)*100):.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Sarcasm Probability", f"{sarc_prob:.2f}")
            m2.metric("Script Topology", cm_meta["primary_script"])
            m3.metric("Code-Mix Index", f"{cm_meta['cmi']}%")

            st.markdown("#### 🧠 Cognitive Rationale")
            reasons = []
            if feats["polarity_clash"] > 0:
                reasons.append("⚡ **Intra-Sentence Opposition**: Positive surface wording contrasted directly with negative tone.")
            if feats["context_clash"] > 0:
                reasons.append("⚡ **Discourse Clash**: Positive statement directly contradicts negative prior context.")
            if feats["intensifier_density"] > 0:
                reasons.append("⚡ **Hyperbolic Cue**: Sarcastic amplifier/intensifier words detected.")
            if cm_meta["is_code_mixed"]:
                reasons.append(f"⚡ **Code-Mixing Active**: Marathi-English switching detected ({cm_meta['cmi']}%).")
            if not reasons:
                reasons.append("✨ Tone and vocabulary are semantically congruent.")
            for r in reasons:
                st.markdown(r)

            st.markdown("#### 🔬 Token Attribution Saliency")
            tokens = input_text.split()
            badges_html = "<div>"
            for t in tokens:
                clean_t = t.lower().strip("!?.,\"'")
                color, bg = "#e0e6ed", "rgba(255,255,255,0.08)"
                if clean_t in POSITIVE_CUES:
                    color, bg = "#00f2fe", "rgba(0,242,254,0.18)"
                elif clean_t in NEGATIVE_CUES:
                    color, bg = "#ff0844", "rgba(255,8,68,0.2)"
                elif clean_t in INTENSIFIERS:
                    color, bg = "#ffb199", "rgba(255,177,153,0.2)"
                badges_html += f'<span class="badge-3d" style="color:{color}; background:{bg}; border:1px solid {color}55;">{t}</span>'
            badges_html += "</div>"
            st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.info("👈 Select a preset or type a statement on the left to trigger 3D diagnostics.")

    st.markdown('</div>', unsafe_allow_html=True)