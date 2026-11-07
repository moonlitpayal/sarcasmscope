# SARcasmScope — Stage 7 Multi-Lingual Robustness & Sensitivity Report
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Date:** November 2026  

---

## 1. Controlled Perturbation Benchmarks

Evaluated across a balanced evaluation cohort ($N = 100$, 50 sarcastic, 50 literal):

| Perturbation Test | Target Language Variety | Linguistic Scope | Empirical Stability (%) | Mean $\Delta P$ | Prediction Flips (/100) | Architectural Rationale |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Word Order Permutation** | English & Hinglish | Word Sequence | **98.0%** | 0.017 | 2 | Bag-of-subwords feature vector $\mathbf{w}^T \mathbf{x} + b$ is permutation-invariant; minor flips arise from sentence boundary regex shifts. |
| **Vowel / Grapheme Elongation** | Hinglish (*"waaaah"*, *"soooo"*) | Phonetic Orthography | **98.0%** | 0.022 | 2 | Romanized phonetic normalizer folds repeated characters ($[c]\{3,\} \to c$), neutralizing orthographic noise. |
| **Punctuation Stripping** | All Languages | Hyperbole & Syntax | **100.0%** | 0.005 | 0 | Subword and lexical incongruity features dominate even when exclamation marks (`!`, `?`) are deleted. |
| **Emoji Stripping** | English & Hinglish | Pragmatic Markers | **100.0%** | 0.000 | 0 | Textual polarity and subwords maintain classification verdict. |
| **Dialogue Context Removal** | English & Hinglish | Discourse Grounding | **99.0%** | 0.007 | 1 | Isolated utterances relying purely on contextual contradiction revert to the literal prior. |
| **Typographical Noise (Transposition)** | All Languages | Mobile Typing Noise | **99.0%** | 0.008 | 1 | Subword chunks (length 3 to 6) retain partial n-gram matches, preventing catastrophic token failure. |

---

## 2. Multi-Lingual Qualitative Case Studies

### 1. English: Polarity Clash under Punctuation Stripping
- **Original Utterance:** *"Oh brilliant, another four-hour delay on my connecting flight! Just what I needed!"*
- **Perturbed Utterance:** *"Oh brilliant another four hour delay on my connecting flight Just what I needed"*
- **Original Probability:** 0.880 | **Perturbed Probability:** 0.865 | **Verdict:** Sarcastic $\to$ Sarcastic (**Stable**)
- *Linguistic Analysis:* Although hyperbolic punctuation (`!`) is stripped, the intra-sentence opposition between positive praise (*"brilliant"*, *"needed"*) and situational failure (*"four hour delay"*) is successfully captured.

### 2. Hinglish: Phonetic Elongation & Digraph Normalization
- **Original Utterance:** *"Wah kya baat hai yaar, pura din traffic me nikal gaya"*
- **Perturbed Utterance:** *"Waaaaahhh kyaaa baaaattt hai yaar pura din traffic me nikal gaya"*
- **Original Probability:** 0.760 | **Perturbed Probability:** 0.745 | **Verdict:** Sarcastic $\to$ Sarcastic (**Stable**)
- *Linguistic Analysis:* The Romanized normalizer regex (`/([a-z])\1{2,}/gi`) collapses *"Waaaaahhh"* into *"Wah"* and *"baaaattt"* into *"bat"*, preserving lexicon matching and subword hash alignment.

### 3. Marathi: Agglutinative Inflection Resilience
- **Original Utterance:** *"खूप छान काम केलंत, सगळे पैसे पाण्यात गेले"*
- **Perturbed Utterance:** *"खूप छान काम केलंत सगळे पैसे पाण्यात गेले"* (Punctuation removed)
- **Original Probability:** 0.780 | **Perturbed Probability:** 0.780 | **Verdict:** Sarcastic $\to$ Sarcastic (**Stable**)
- *Linguistic Analysis:* Suffix stripping for Indic verbal endings preserves the root contrast between positive evaluation (*"छान काम"*) and loss (*"पाण्यात गेले"*).

### 4. Telugu: Sarcasm under Dialogue Context Detachment
- **Original Target:** *"చాలా మంచి నిర్ణయం తీసుకున్నారు"* (*"A very good decision was taken"*)
- **Dialogue Context:** *"కంపెనీ భారీ నష్టాల్లో కూరుకుపోయింది"* (*"Company plunged into heavy financial losses"*)
- **With Context:** $P = 0.650$ (**Sarcastic**)
- **Without Context:** $P = 0.220$ (**Literal**) — **Pragmatic Shift**
- *Linguistic Analysis:* In the absence of context, the Telugu sentence is completely literal and positive (*"very good decision"*). Sarcasm is purely situational; removing the dialogue turn appropriately causes the model to classify it as literal.

---

## 3. Order Invariance vs Sequential Syntax Disclosure

In adherence to scientific integrity principles:
1. **Bag-of-Subwords Permutation Invariance:** The primary statistical representation relies on subword frequency vectors where addition is commutative. Permuting word order will not alter term frequency counts.
2. **Contextual Incongruity Sequentiality:** The contextual subword overlap metric ($C_{\text{sim}}$) and punctuation density counters provide partial structural sensitivity, but full dependency syntax requires deep transformer self-attention.
