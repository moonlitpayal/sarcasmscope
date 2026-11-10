> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 6 Robustness, Sensitivity & Linguistic Error Analysis
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Date:** November 2026  

---

## 1. Controlled Perturbation & Sensitivity Suite

Evaluated across a balanced evaluation cohort ($N = 100$, 50 sarcastic, 50 literal):

| Perturbation Test | Transform Description | Representation Type | Empirical Stability (%) | Mean $\Delta P$ | Prediction Flips (/100) | Linguistic / Computational Mechanism |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Word Order Permutation** | Randomly shuffle the word tokens within the utterance | Order-Invariant Bag-of-Tokens | **98.0%** | 0.011 | 2 | Linear dot product $\mathbf{w}^T \mathbf{x} + b$ is mathematically permutation-invariant; minor flips occur from punctuation/regex token boundary changes. |
| **Punctuation Stripping** | Remove exclamation marks, question marks, ellipses, and quotes | Syntax / Sequential | **99.0%** | 0.004 | 1 | Slight reduction in hyperbole index, but lexical and subword polarity dominates. |
| **Dialogue Context Removal** | Detach preceding conversational turns, testing ungrounded utterance | Syntax / Discourse | **98.0%** | 0.023 | 2 | Context clash signal is lost; utterances relying solely on contextual divergence flip to literal. |
| **Typographical Noise** | Swap adjacent characters in 15% of words (simulating mobile typing errors) | Sub-Lexical / Spelling | **97.0%** | 0.018 | 3 | Subword n-grams (length 3-6) preserve partial matches, preventing total catastrophic failure. |
| **Polite Negation Inversion** | Prepend literal politeness hedge (*"Honestly, "* or *"Truly, "*) | Pragmatic Syntax | **100.0%** | 0.000 | 0 | Grounded confidence gating prevents superficial politeness prefixes from altering core incongruity verdict. |

---

## 2. In-Depth Linguistic Error Taxonomy & Qualitative Analysis

Drawing directly from verified held-out prediction logs in `eval_results/stage6/error_analysis.csv`, we document the primary linguistic failure modes:

### Category 1: Implicit & Deadpan Satire (False Negatives)
* **Sample Text:** *"area woman feels like she forgets something every time she leaves house"*
* **Source Dataset:** `Misra_Arora_News_Headlines` (English)
* **Ground Truth:** Sarcastic (Satirical Headline) | **Predicted Probability:** 0.020 | **Verdict:** Literal
* **Linguistic Analysis:** The utterance exhibits deadpan journalistic reporting. There is zero surface lexical clash, no positive evaluation words (*"great"*, *"wonderful"*), and no overt exclamation marks. The humor arises from human psychological universality rather than verbal incongruity. Without an external commonsense knowledge base, feature-fusion correctly evaluates this as literal prose.

---

### Category 2: Context-Dependent Dialogue Punchlines (False Negatives)
* **Sample Text:** *"Haan Maya, main toh bas yahan timepass karne aaya tha"*
* **Dialogue Context:** *"Monisha: Sahil tum itni der se kya kar rahe ho office mein?"*
* **Source Dataset:** `IIITD_MaSaC_CodeMixed` (Hinglish)
* **Ground Truth:** Sarcastic | **Predicted Probability:** 0.200 | **Verdict:** Literal
* **Linguistic Analysis:** The speaker's statement (*"Yes Maya, I just came here to kill time"*) in response to an accusation of wasting time is deeply ironic. However, the surface sentence uses affirmative agreement (*"Haan"*) and casual phrasing. Because the client-side subword overlap measures lexical overlap rather than pragmatic discourse pragmatics, the model fails to detect the underlying ironic rebuttal.

---

### Category 3: Agglutinative Inflections in Indic Languages (False Negatives)
* **Sample Text:** *"काय सुंदर रस्ते आहेत आमच्या पुण्याचे, रोज एक नवीन खड्डा मिळतो"*
* **Source Dataset:** `L3Cube_MahaEmotions` (Marathi)
* **Ground Truth:** Sarcastic | **Predicted Probability:** 0.250 | **Verdict:** Literal
* **Linguistic Analysis:** The speaker praises roads (*"सुंदर रस्ते"*) followed by discovering new potholes daily (*"नवीन खड्डा मिळतो"*). While the polarity contrast is obvious to a native speaker, inflected case markers (*"पुण्याचे"*, *"रस्ते"*) and verbal morphology (*"मिळतो"*) bypassed unstemmed dictionary entries. While Indic suffix stripping improved Marathi recall overall, complex multi-word compound nominals remain vulnerable.

---

### Category 4: Direct Complaints Misclassified as Sarcasm (False Positives)
* **Sample Text:** *"Worst customer service ever! Waited 45 minutes on hold and they hung up on me!"*
* **Source Dataset:** `ACL_2020_Twitter_Sarcasm` (English)
* **Ground Truth:** Literal (Direct Complaint) | **Predicted Probability:** 0.720 | **Verdict:** Sarcastic
* **Linguistic Analysis:** The tweet contains hyperbolic punctuation (`!`) and extreme negative adjectives (*"Worst"*). When social media users express genuine outrage, high punctuation density and intensifiers sometimes trigger the hyperbole threshold, confusing sincere outrage with sarcastic irony.

---

### Category 5: Successful Selective Abstention on Ambiguous Language
* **Sample Text:** *"That was certainly an interesting way to conclude the meeting."*
* **Dialogue Context:** *"Discussion ended without a formal consensus."*
* **Ground Truth:** Sarcastic | **Predicted Probability:** 0.520 | **Verdict:** **Uncertain (Abstained)**
* **Grounded Confidence:** 0.440 (Below $\tau = 0.58$ threshold)
* **Linguistic Analysis:** The word *"interesting"* is a classic British/American pragmatic hedge that can signify either genuine fascination or polite disdain. Because $P_{\text{fused}} = 0.52$ lies in the boundary region, the grounded confidence filter correctly triggers selective abstention rather than producing an arbitrary guess.

---

## 3. Methodological Recommendations for Future Work

1. **Pragmatic Discourse Graphs:** Incorporate speaker relationship graphs and turn-taking timing vectors for conversational dialogue corpora (e.g., MaSaC).
2. **Agglutinative Subword Segmenters:** Integrate unigram-LM subword tokenizers (SentencePiece) specifically trained on Devanagari and Telugu corpora to eliminate morphological out-of-vocabulary gaps.
3. **Contextual Polarity Disambiguation:** Differentiate hyperbole used in sincere complaints from hyperbole coupled with praise-blame incongruity.
