# SARcasmScope Model Card

## Model Details
- **Model Name:** SARcasmScope (Calibrated Feature-Fusion Sarcasm Engine)
- **Model Version:** 4.0.0 (Stage 4 Publication Freeze)
- **Architecture:** Hybrid Feature-Fusion combining:
  1. Client-Side Linguistic Incongruity Engine (Polarity clash, morphology, phonetic folding, rhetorical markers, irony emojis)
  2. Subword & N-Gram Statistical Semantic Representations
  3. Cost-Sensitive Contextual Transformer Backbone (MuRIL / XLM-RoBERTa transfer)
  4. Temperature-Scaled Selective Prediction with Epistemic Uncertainty Gating
- **Authors:** Payal Mehta (`payal.12620246@vit.edu`)
- **License:** Open Academic Research License
- **Framework:** TypeScript 5.7 (Browser/Node Runtime) + PyTorch 2.0 (Transformer GPU Training)

## Intended Use
- **Primary Use Cases:**
  - Detecting sarcasm and verbal irony in multilingual text (English, Hindi, Marathi, Telugu) and code-mixed conversational text (Hinglish, Marathi-Latin).
  - Selective content moderation: identifying high-confidence sarcasm vs escalating ambiguous text to human review.
  - Explainable sentiment analysis: outputting token attributions and linguistic incongruity justifications.
- **Out-of-Scope Use Cases:**
  - Fully autonomous censorship or automated punitive account banning without human verification.
  - Sarcasm detection in unrepresented languages (e.g., non-Indic script families without morphological adaptations).

## Training & Evaluation Data
- **Corpus Provenance:** 5 independently published peer-reviewed datasets:
  1. ACL 2020 Sarcasm Shared Task (Twitter, English, Ghosh et al. 2020)
  2. Misra & Arora News Headlines Dataset (The Onion vs HuffPost, English, 2019)
  3. IIIT Delhi MaSaC Code-Mixed Dialogue Corpus (Sarabhai vs Sarabhai, Bedi et al. 2021)
  4. L3Cube-MahaEmotions (Marathi Twitter, Kulkarni et al. 2023)
  5. IIIT Hyderabad Telugu Sarcasm Dataset (Marreddy et al. 2022)
- **Data Partitioning (Deterministic Seed 42):**
  - **Training:** 3,373 records (60%)
  - **Validation:** 1,104 records (20%)
  - **Held-Out Test:** 1,123 records (20%, unsealed only at Stage 4 publication freeze)
  - Zero cross-split leakage verified via hash audit.

## Quantitative Performance (Held-Out Test Split, N = 1,123)
- **Accuracy:** 56.63% (95% CI: [53.77%, 59.39%])
- **Macro-F1:** 56.57% (95% CI: [53.71%, 59.29%])
- **Sarcasm Recall:** 57.23% (compared to 3.08% for pure linguistic baseline)
- **Specificity:** 56.13%
- **Expected Calibration Error (ECE):** 42.88%
- **Inference Latency (P95):** 2.31 ms (Local Client Runtime)
- **Memory Footprint:** < 2 MB (Browser engine) / 110 MB (INT8 ONNX)

## Ethical Considerations & Bias
- **Class Imbalance in Indic Text:** High imbalance in low-resource sets (e.g., Telugu 1:6 ratio) can lead uncalibrated models to default to majority literal classes.
- **Selective Abstention:** At threshold $\tau = 0.58$, the model abstains on 46.13% of ungrounded text, treating ambiguity transparently rather than forcing arbitrary guesses.
