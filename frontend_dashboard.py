import streamlit as st
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
