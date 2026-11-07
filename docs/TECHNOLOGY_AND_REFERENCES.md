# SARcasmScope: Technology Stack, Academic References & Dataset Documentation

**Author / Maintainer:** Payal (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope — Multilingual & Code-Mixed Cognitive Sarcasm Detection System  
**Framework Version:** React 19 + TypeScript + Vite 6 + Tailwind CSS v4  

---

## 1. Overview
SARcasmScope is an explainable, calibrated sarcasm detection system specifically engineered for multilingual and code-mixed Indian-language discourse (English, Hindi, Hinglish, Marathi, and Dravidian languages like Telugu and Tamil). It addresses key limitations of standard sentiment models: label overconfidence, lack of discourse context anchoring, inability to handle intra-sentential script alternation, and vulnerability to social-media noise.

---

## 2. Software Technologies & Frameworks

| Technology | Category | Version / Spec | Role in SARcasmScope |
| :--- | :--- | :--- | :--- |
| **React** | Frontend UI Framework | `^19.0.0` | Powers the reactive single-page interface, real-time input analysis, tabs router, and responsive state synchronization. |
| **TypeScript** | Programming Language | `~5.7.2` | Implements end-to-end type safety across mathematical NLP formulas, token attribution structures, and dataset pipelines. |
| **Vite** | Build Tool & Bundler | `^6.2.0` | Provides ultra-fast development server execution on port 3000, optimized Rollup bundling, and Hot Module Replacement. |
| **Tailwind CSS** | Styling Framework | `^4.0.0` | Powers the luxury dark studio aesthetic (`#07080c`), linear gradients, responsive metric grids, and custom scrollbars. |
| **Three.js** | 3D WebGL Engine | `^0.174.0` | Renders the interactive 3D ambient orbital quantum rings and celestial particle mesh in the header, responding dynamically to mouse coordinates and verdict intensity. |
| **Lucide React** | Iconography | `^1.16.0` | Provides high-clarity status icons, badges, and warning indicators for cognitive verdicts. |

---

## 3. Scientific & Academic Research References

### 3.1 Probability Calibration & Temperature Scaling
* **Primary Citation:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *"On Calibration of Modern Neural Networks."* Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70:1321–1330.
* **Implementation in SarcasmScope:**
  $$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right), \quad z = \frac{\text{logit}(p)}{T}, \quad P_{\text{calibrated}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  where $T > 0$ is the learned or adjusted temperature parameter.
* **Selective Abstention:**
  $$\text{Confidence} = 2 \times |P_{\text{calibrated}} - 0.5|$$
  If $\text{Confidence} < \tau$ (default threshold $\tau = 0.60$), the model abstains from guessing and outputs `"Uncertain"`.

### 3.2 Model-Agnostic Token Influence via Occlusion
* **Primary Citations:**
  * Zeiler, M. D., & Fergus, R. (2014). *"Visualizing and Understanding Convolutional Networks."* ECCV 2014.
  * Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?: Explaining the Predictions of Any Classifier."* ACM SIGKDD 2016.
* **Implementation in SarcasmScope:**
  Evaluates leave-one-token-out perturbation:
  $$\Delta_i = P(\text{sarcastic} \mid S) - P(\text{sarcastic} \mid S_{\setminus i})$$
  * $\Delta_i > 0$: Removing the token lowers the sarcasm score; the token pushes towards **sarcastic intent** (highlighted in rose/red).
  * $\Delta_i < 0$: Removing the token increases the sarcasm score; the token serves as a **literal anchor** (highlighted in sky/blue).

### 3.3 Code-Mixing Index (CMI)
* **Primary Citation:** Gambäck, B., & Das, A. (2014/2016). *"Comparing the Level of Code-Switching in Corpora."* Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC).
* **Implementation in SarcasmScope:**
  Exact script-based alternation index measuring the proportion of tokens belonging to secondary linguistic scripts relative to the dominant matrix language:
  $$\text{CMI} = 100 \times \left(1 - \frac{\max_{s} N_s}{N}\right)$$
  where $N$ is total alphabetic tokens and $N_s$ is token count per Unicode script block.

### 3.4 Inter-Annotator Reliability Metrics
* **Primary Citations:**
  * Cohen, J. (1960). *"A coefficient of agreement for nominal scales."* Educational and Psychological Measurement.
  * Krippendorff, K. (2011). *"Computing Krippendorff's Alpha-Reliability."*
* **Implementation in SarcasmScope:**
  Used in `docs/ANNOTATION_GUIDELINES.md` to compute cross-annotator agreement score ($\kappa$ and $\alpha$) among native Indian speakers evaluating ambiguous sarcasm.

### 3.5 Foundational Transformer Architectures (Evaluated in Baseline Suite)
* **MuRIL:** Khanuja et al. (2021). *"MuRIL: Multilingual Representations for Indian Languages."* Google Research.
* **XLM-RoBERTa:** Conneau et al. (2020). *"Unsupervised Cross-lingual Representation Learning at Scale."* ACL 2020.
* **mBERT:** Devlin et al. (2019). *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* NAACL 2019.

---

## 4. Datasets & Corpora Referenced

### 4.1 MaSaC (Hindi-English Dialogue Sarcasm)
* **Full Title:** Multi-modal Sarcasm and Humor in Hindi-English Conversations.
* **Authors:** Bedi, M., Kumar, S., Akhtar, M. S., & Chakraborty, T. (LCS2 Lab, IIIT Delhi).
* **Key Characteristics:** Real human dialogue dataset containing paired utterances with preceding conversational context turns, critical for evaluating context-clash features.

### 4.2 L3Cube-MeCorpus (Marathi-English Code-Mixed Corpus)
* **Authors:** Joshi et al., L3Cube Pune.
* **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
* **Key Characteristics:** Authentic social media corpus capturing Marathi in both Devanagari and Latin (Romanized) scripts, foundational for Marathi sarcasm lexicon calibration.

---

## 5. Algorithmic Modules Implemented in SarcasmScope

1. **Linguistic Incongruity Extractor:**
   Detects semantic contradictions between surface positive praise cues (e.g., *"brilliant"*, *"kamaal"*, *"shandar"*, *"lai bhari"*) and negative failure terms (e.g., *"delayed"*, *"terrible"*, *"crashed"*, *"ghatiya"*), combined with hyperbolic intensifiers (*"obviously"*, *"bilkul"*, *"ekdam"*).
2. **Context Anchoring Module:**
   Cross-references statement sentiment against preceding conversational discourse (e.g., praising efficiency immediately following a 4-hour delay).
3. **Indic Grapheme Cluster Engine:**
   Prevents Indic words from breaking into invalid orthographic fragments during noise perturbation by clustering base consonants with associated combining *matras*, *viramas*, and ZWJ/ZWNJ characters.
4. **Controlled Robustness Perturbation Suite:**
   Applies controlled character swaps, letter elongations, punctuation stripping, tag dropping, and word order perturbations to calculate empirical classification stability.
5. **Batch Processor & CSV Exporter:**
   In-memory client-side batch processing supporting up to 200 items with exportable RFC 4180 CSV generation.

---
*Document verified and formatted for academic and project submission.*
