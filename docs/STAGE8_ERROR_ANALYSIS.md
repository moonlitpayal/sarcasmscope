> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 8 Qualitative Error Analysis & Robustness Stress Testing

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Date:** November 7, 2026  
**Audit Phase:** 8G — Linguistic Error Taxonomy & Perturbation Robustness  

---

## 1. Executive Summary

Understanding why a model fails is as critical to scientific rigor as reporting its successes. In Stage 8, we conduct an exhaustive qualitative error taxonomy and controlled perturbation stress test on SARcasmScope across its four target language regimes: English, Hindi-English code-mixing (Hinglish), Marathi (Devanagari script), and Telugu (Telugu script).

Our evaluation demonstrates that:
1. **Multi-lingual Robustness:** The subword and phonetic normalization pipelines provide high stability ($98\% - 100\%$) under typological, graphemic, and emoji perturbations.
2. **Failure Modes:** Primary remaining failures stem from deadpan journalistic satire lacking overt sentiment markers, multi-party dialogue pragmatics requiring external world knowledge, and extreme dialectal inflection in low-resource Dravidian text.
3. **Selective Epistemic Abstention:** When an utterance exhibits borderline probability ($p \approx 0.50$), the calibrated uncertainty gate ($\tau = 0.58$) correctly marks the sample as *Uncertain / Abstained*, protecting the user from erratic decisions.

---

## 2. Multi-Lingual Qualitative Error Taxonomy

Representative failure and success cases extracted from `eval_results/stage8/error_analysis.csv` are analyzed below:

### Case 1: Deadpan Satire False Negative (English)
- **Dataset:** Misra-Arora Headlines | **Ground Truth:** Sarcastic (1) | **Predicted:** Literal (0) ($p = 0.02$)
- **Utterance:** *"Area woman feels like she forgets something every time she leaves house"*
- **Dialogue Context:** None
- **Linguistic Mechanism:** The headline employs deadpan, understated satire typical of *The Onion*. It contains no hyperbole, exclamation marks, contrastive conjunctions, or explicit affective polarity clash. The humor relies strictly on pragmatic human world knowledge regarding ubiquitous domestic anxiety. Because purely linguistic and statistical subwords detect neither negative sentiment nor overt irony, the engine outputs a low sarcastic probability ($0.02$).

### Case 2: Multi-Party Conversational False Negative (Hinglish)
- **Dataset:** IIITD MaSaC | **Ground Truth:** Sarcastic (1) | **Predicted:** Literal (0) ($p = 0.20$)
- **Utterance:** *"Haan Maya, main toh bas yahan timepass karne aaya tha"*
- **Dialogue Context:** *Monisha: "Sahil tum itni der se kya kar rahe ho office mein?"*
- **Linguistic Mechanism:** The target utterance begins with an affirmative marker (*"Haan"*, meaning *Yes*). While the phrase *"timepass karne aaya tha"* conveys sarcastic self-deprecation in Indian English, the surface polarity clash with the preceding question is subtle. Without deep relational modeling of the hierarchical interpersonal tension between the characters, the model classifies the utterance as an earnest explanation.

### Case 3: Agglutinative Inflection False Negative (Marathi)
- **Dataset:** L3Cube MahaEmotions | **Ground Truth:** Sarcastic (1) | **Predicted:** Literal (0) ($p = 0.25$)
- **Utterance:** *"काय सुंदर रस्ते आहेत आमच्या पुण्याचे, रोज एक नवीन खड्डा मिळतो"*  
  *(Translation: "How beautiful the roads of Pune are, every day we get a new pothole")*
- **Dialogue Context:** None
- **Linguistic Mechanism:** The utterance expresses classic ironic praise followed by a jarring reality. While *"सुंदर"* (beautiful) has positive polarity and *"खड्डा"* (pothole) has negative polarity, the heavy inflectional genitive ending in *"पुण्याचे"* (Pune's) and verbal agreement in *"मिळतो"* (is received) dilute the exact sentiment token match. Although the Devanagari normalizer strips common case markers, dialect-specific noun inflections require broader contextual representation.

### Case 4: Extreme Class Imbalance False Negative (Telugu)
- **Dataset:** IIIT Hyderabad Telugu Sarcasm | **Ground Truth:** Sarcastic (1) | **Predicted:** Literal (0) ($p = 0.15$)
- **Utterance:** *"చాలా మంచి నిర్ణయం తీసుకున్నారు"*  
  *(Translation: "A very good decision was taken")*
- **Dialogue Context:** *"కంపెనీ భారీ నష్టాల్లో కూరుకుపోయింది"* *(The company sank into massive losses)*
- **Linguistic Mechanism:** The natural class distribution of this corpus is skewed ($84.3\%$ literal, $15.7\%$ sarcastic). While the preceding context contains strong negative economic words (*"భారీ నష్టాల్లో"*), the model's calibrated Bayesian prior heavily penalizes false sarcasm alarms to avoid catastrophic precision loss, resulting in a false negative.

### Case 5: Direct Complaint False Positive (English)
- **Dataset:** ACL 2020 Twitter Sarcasm | **Ground Truth:** Literal (0) | **Predicted:** Sarcastic (1) ($p = 0.72$)
- **Utterance:** *"Worst customer service ever! Waited 45 minutes on hold and they hung up on me!"*
- **Dialogue Context:** None
- **Linguistic Mechanism:** The utterance is an earnest, angry customer complaint. However, the presence of multiple exclamation marks (*"!"*), superlative negative hyperbole (*"Worst customer service ever"*), and strong negative verbs triggered an elevated irony score, misclassifying genuine outrage as hyperbolic sarcasm.

### Case 6: Selective Epistemic Abstention Success
- **Dataset:** ACL 2020 Twitter Sarcasm | **Ground Truth:** Sarcastic (1) | **Predicted:** Uncertain / Abstained ($p = 0.52$)
- **Utterance:** *"That was certainly an interesting way to conclude the meeting."*
- **Dialogue Context:** *"Discussion ended without a formal consensus."*
- **Linguistic Mechanism:** The utterance contains the euphemistic adjective *"interesting"*, which can be either genuinely neutral or politely ironic. The model outputs a borderline calibrated probability of $p = 0.52$. Under standard binary classification, this would be an unconfident flip; however, because the confidence metric ($|0.52 - 0.50| \times 2 = 0.04 < \tau = 0.58$), the system safely abstains, marking the verdict as **Uncertain**.

---

## 3. Controlled Robustness Perturbation Evaluation

To test anti-fragility against adversarial noise and social media irregularities, we subjected 100 randomly sampled test utterances to six controlled perturbations (`eval_results/stage8/robustness_results.csv`):

| Perturbation Test | Linguistic Mechanism Targeted | Tested Count | Stability (%) | Mean $|\Delta p|$ | Prediction Flips | Scientific Explanation |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Typographical Noise** | Random interior character transposition | 100 | **100.0%** | 0.005 | 0 / 100 | Subword character $n$-grams (lengths 3-6) preserve partial matches, preventing total lexical breakdown. |
| **Vowel / Grapheme Elongation** | *sooo / waaaah / kyuuun* | 100 | **98.0%** | 0.022 | 2 / 100 | Romanized phonetic folding (`romanizedNormalizer.ts`) collapses character runs ($[c]{3,} \to c$), neutralizing elongation. |
| **Emoji Stripping** | Removal of all Unicode pictographs | 100 | **100.0%** | 0.000 | 0 / 100 | Core sentiment clash and subword scores maintain prediction even without visual markers. |
| **Punctuation Stripping** | Removal of `!`, `?`, `...`, quotes | 100 | **100.0%** | 0.005 | 0 / 100 | Incongruity polarity clash dominates over exclamation marks. |
| **Context Removal** | Detaching dialogue context | 100 | **99.0%** | 0.007 | 1 / 100 | Utterances relying on external dialogue divergence revert to the literal baseline when context is absent. |
| **Word Order Permutation** | Shuffling token sequence | 100 | **98.0%** | 0.017 | 2 / 100 | Bag-of-subwords representation is permutation-invariant; minor shifts occur only from sentence-boundary markers. |

---

## 4. Conclusion & Implications for Journal Review

The error analysis reveals that SARcasmScope's calibrated hybrid is resilient against real-world social media orthographic noise. Its limitations are typical of computational pragmatics: resolving deadpan satire without world knowledge and processing deep Dravidian inflection. Crucially, the implementation of **selective prediction** prevents the system from making confident blunders when encountering boundary cases.
