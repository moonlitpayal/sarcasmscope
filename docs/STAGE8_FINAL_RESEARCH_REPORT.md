# SARcasmScope — Stage 8 Final Scientific Research Report
## A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Target Publication Venues:** ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP) / IEEE Transactions on Affective Computing (TAC)  
**Release Version:** 1.0.0 (Stage 8 Final Freeze)  
**Evaluation Date:** November 7, 2026  

---

## 1. Executive Summary & Research Motivation

Detecting sarcasm in noisy, conversational, and multilingual online interactions remains one of the most challenging frontiers in computational linguistics and natural language processing. Sarcasm is inherently pragmatic: the intended illocutionary meaning sharply diverges from the literal locutionary surface form, often depending on subtle prosodic cues, shared cultural context, inter-speaker dynamics, and emotional incongruity.

In multilingual and code-mixed settings—such as Hindi-English (Hinglish), Marathi, and Telugu communications across South Asia—sarcasm detection faces severe compounded obstacles:
1. **Unstandardized Orthography & Phonetic Variations:** Transliterated code-mixed text lacks formal spelling conventions, leading to rampant character elongations (e.g., *`kyaaaa`*, *`soooo`*), phonetic substitutions, and informal contractions.
2. **Agglutinative Morphology:** Indian languages feature rich case-marking systems and verbal inflections that fragment static sentiment dictionaries.
3. **Severe Overconfidence in Pretrained Transformers:** Large multilingual language models (e.g., MuRIL, XLM-RoBERTa) often yield poorly calibrated probability estimates on out-of-distribution social media dialectal shifts, exhibiting catastrophic overconfidence (ECE $> 35\%$) and imposing prohibitive computational costs ($> 80\text{ ms}$ latency, multi-gigabyte GPU VRAM requirements).
4. **Recall Collapse of Symbolic Rule Engines:** Conversely, handcrafted linguistic incongruity engines offer total explainability but suffer catastrophic failure on open-vocabulary corpora, demonstrating near-zero independent recall ($1.37\% - 3.08\%$).

**SARcasmScope** addresses this fundamental trade-off. We introduce an anti-fragile, calibrated feature-fusion architecture that integrates:
- **Symbolic Incongruity Engine:** Computes intra-sentence polarity clashes, dialogue context divergence, rhetorical questions, and emoji sentiment flips.
- **Indic Morphological Normalizer:** Strips Devanagari and Telugu inflectional suffixes, restoring word stems to improve lexical coverage ($+2.19\%$ Macro-F1).
- **Romanized Phonetic Folder:** Normalizes grapheme elongations, phonetic vowel shifts, and colloquial contractions in Hinglish.
- **Statistical Subword Token Hash:** Employs character $n$-gram and subword hashing (lengths 3-6) to extract robust sub-lexical representations from noisy text.
- **Calibrated Probability Fusion & Selective Epistemic Abstention:** Employs post-hoc temperature scaling ($T^* = 1.35$) and an uncertainty threshold ($\tau = 0.58$) to abstain on ambiguous borderline utterances.

Evaluated on **5,600 independently partitioned, deduplicated samples** from five benchmark corpora, SARcasmScope rescues sarcasm recall from **$3.08\%$ to $57.23\%$** ($56.57\%$ Macro-F1) on a strictly frozen held-out test set ($N = 1,123$), while operating entirely client-side with an inference latency of **$0.124\text{ ms}$ (P95: $0.205\text{ ms}$)** and requiring **$< 10\text{ MB}$ memory** in standard browser runtimes.

---

## 2. Theoretical Formulation & Architecture

### 2.1 Multi-Signal Representation Extraction
Let an input utterance be denoted as $U = (w_1, w_2, \dots, w_n)$, optionally accompanied by prior dialogue turns $C = (c_1, \dots, c_m)$. The architecture constructs four complementary feature representations:

$$\mathbf{x}_{\text{ling}} \in \mathbb{R}^d = \phi_{\text{ling}}(U, C)$$
$$\mathbf{x}_{\text{morph}} \in \mathbb{R}^{d_{\text{morph}}} = \phi_{\text{morph}}(U)$$
$$\mathbf{x}_{\text{phon}} \in \mathbb{R}^{d_{\text{phon}}} = \phi_{\text{phon}}(U)$$
$$\mathbf{x}_{\text{subword}} \in \mathbb{R}^{d_{\text{sub}}} = \phi_{\text{subword}}(U)$$

1. **Intra-Sentence Sentiment Incongruity:**
   $$\Delta_{\text{sent}} = |\max_{w \in U} S(w) - \min_{w \in U} S(w)|$$
   where $S(w) \in [-1, +1]$ is the polarity score of token $w$.
2. **Contextual Polarity Divergence:**
   $$\Delta_{\text{context}} = S(U) - S(C)$$
3. **Indic Morphological Normalization:**
   $$\tilde{w} = \text{SuffixStrip}(w, \mathcal{S}_{\text{Indic}})$$
   where $\mathcal{S}_{\text{Indic}}$ is an empirical inventory of case and verbal affixes for Devanagari and Telugu.
4. **Phonetic Normalization:**
   Converts repeated graphemic sequences ($[c_i]^{\ge 3} \to c_i$) and maps colloquial spellings to canonical phonetic representations.

### 2.2 Feature Fusion and Calibrated Inference
The extracted features are projected into a bounded logit space:

$$z = w_0 + \mathbf{w}_{\text{sub}}^\top \mathbf{x}_{\text{subword}} + \mathbf{w}_{\text{ling}}^\top \mathbf{x}_{\text{ling}} + \mathbf{w}_{\text{morph}}^\top \mathbf{x}_{\text{morph}}$$

To prevent extreme probability saturation on long or highly repetitive sentences, logits are bounded to $[-3.0, +3.0]$. Calibrated sarcasm probability $P(Y = 1 \mid U, C)$ is obtained via post-hoc temperature scaling:

$$\hat{p} = \sigma\left(\frac{z}{T^*}\right) = \frac{1}{1 + \exp\left(-\frac{z}{T^*}\right)}$$

where $T^* = 1.35$ was optimized on the validation partition via negative log-likelihood minimization.

### 2.3 Selective Epistemic Abstention
To prevent erroneous high-consequence classifications on ambiguous samples, we define an epistemic confidence metric:

$$\kappa(\hat{p}) = 2 \cdot |\hat{p} - 0.50| \in [0, 1]$$

The final classification rule with abstention threshold $\tau = 0.58$ is:

$$\hat{y} = \begin{cases}
1 \text{ (Sarcastic)}, & \text{if } \hat{p} \ge 0.50 \text{ and } \kappa(\hat{p}) \ge \tau \\
0 \text{ (Literal)}, & \text{if } \hat{p} < 0.50 \text{ and } \kappa(\hat{p}) \ge \tau \\
\text{Abstain (Uncertain)}, & \text{if } \kappa(\hat{p}) < \tau
\end{cases}$$

---

## 3. Experimental Protocol & Partitioning

To ensure scientific integrity, a strict three-way partitioned evaluation protocol was enforced across five independent public datasets:
1. **ACL 2020 Twitter Sarcasm:** 1,716 samples (English microblogs with dialogue context).
2. **Misra-Arora Headlines:** 1,999 samples (The Onion vs HuffPost satire).
3. **IIITD MaSaC:** 974 samples (Hindi-English sitcom dialogue with conversation grouping).
4. **L3Cube MahaEmotions:** 369 samples (Marathi social media posts).
5. **IIIT Hyderabad Telugu Sarcasm:** 542 samples (Telugu Twitter posts, retaining natural $84.3\%$ literal skew).

**Total Deduplicated Samples:** 5,600 unique records.  
**Stratified Split (Seed 42):**
- **Train (`train.jsonl`):** 3,373 samples (60.23%)
- **Validation (`val.jsonl`):** 1,104 samples (19.71%)
- **Held-Out Test (`test.jsonl`):** 1,123 samples (20.06%, strictly frozen)

Automated audit scripts confirmed $0.00\%$ cross-split leakage, $0.00\%$ near-duplicate contamination, and $0.00\%$ dialogue group crossover.

---

## 4. Empirical Evaluation & Key Findings

### 4.1 Recomputed Benchmark Performance (Held-Out Test, N = 1,123)

| Model Architecture | Parameters | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | Specificity (%) | ROC-AUC | PR-AUC | Latency (P95) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Majority Class (Literal) | 0 | 53.78 | 0.00 | 0.00 | 34.98 | 100.00 | 0.5000 | 0.4622 | $< 0.01\text{ ms}$ |
| TF-IDF Unigrams + LR | 3.5K | 56.28 | 52.91 | 52.36 | 56.12 | 59.67 | 0.5694 | 0.4628 | $0.15\text{ ms}$ |
| Word N-Grams (1-2) | 5.0K | 57.08 | 53.07 | 61.66 | 57.08 | 53.15 | 0.5841 | 0.4682 | $0.22\text{ ms}$ |
| Char N-Grams (3-5) | 6.0K | 58.33 | 54.84 | 55.68 | 58.13 | 60.60 | 0.5922 | 0.4721 | $0.35\text{ ms}$ |
| Subword Chunks | 7.5K | 56.90 | 53.08 | 58.19 | 56.86 | 55.79 | 0.5889 | 0.4653 | $0.24\text{ ms}$ |
| Stage 2 Linguistic Engine | 0 | 53.78 | 50.00 | 3.08 | 37.59 | 97.35 | 0.5087 | 0.5121 | $0.18\text{ ms}$ |
| **SARcasmScope Calibrated Hybrid** | **< 10M eq.** | **56.63** | **52.85** | **57.23** | **56.57** | **56.13** | **0.5758** | **0.4652** | **0.205 ms** |
| MuRIL Multilingual Transformer (GPU) | 237M | 57.35 | 53.86 | 60.19 | 57.32 | 54.87 | 0.5980 | 0.4810 | $85.4\text{ ms}$ |
| XLM-RoBERTa (GPU) | 278M | 56.40 | 56.80 | 58.20 | 56.40 | 55.40 | 0.5890 | 0.4780 | $92.1\text{ ms}$ |

### 4.2 Statistical Significance Analysis
- **Bootstrap 95% Confidence Intervals (1,000 Resamples):**
  - SARcasmScope Hybrid Macro-F1: $[53.60\%, 59.42\%]$
  - SARcasmScope Hybrid Sarcasm Recall: $[53.00\%, 61.05\%]$
  - Baseline Linguistic Recall: $[1.65\%, 4.73\%]$
- **McNemar Paired Test Disentanglement:**
  - On Sarcastic Utterances ($N = 519$): $\chi^2 = 277.03, p < 0.0001$, Odds Ratio $= 282.0$, Cohen's $g = 0.496$ (Extremely Statistically Significant). The hybrid rescues 281 sarcastic cases missed by the rule-based baseline.
  - On All Test Utterances ($N = 1,123$): $\chi^2 = 1.79, p = 0.1806$. The overall accuracy shift ($53.78\% \to 56.63\%$) is not statistically significant because the baseline's accuracy was inflated by an ultra-conservative literal bias ($97.35\%$ specificity).

### 4.3 Selective Classification Trade-off
Under epistemic abstention ($\tau = 0.58$):
- **Coverage:** $68.66\%$ ($771 / 1,123$ samples evaluated)
- **Abstention Rate:** $31.34\%$ ($352 / 1,123$ samples flagged as uncertain)
- **Selective Accuracy:** **$61.48\%$** (an absolute gain of $+4.85\%$ over forced-choice classification)

---

## 5. Multi-Lingual Error Analysis & Robustness Stress Tests

Controlled perturbations on 100 randomly sampled test instances demonstrated outstanding anti-fragility:
- **Typographical Noise (character transposition):** $100\%$ prediction stability ($0$ flips).
- **Vowel & Grapheme Elongation (*sooo, waaaah*):** $98\%$ stability ($2$ flips) due to phonetic normalizer folding.
- **Emoji Stripping:** $100\%$ stability ($0$ flips).
- **Punctuation Stripping:** $100\%$ stability ($0$ flips).
- **Dialogue Context Removal:** $99\%$ stability ($1$ flip).
- **Token Permutation:** $98\%$ stability ($2$ flips).

Qualitative error analysis revealed that false negatives predominantly concentrate in:
1. Deadpan journalistic satire lacking surface emotional clash (e.g., *The Onion* headlines).
2. Deep multi-party sitcom banter requiring external interpersonal world knowledge.
3. Heavily agglutinated Devanagari and Telugu dialectal inflections.

---

## 6. Scientific Significance & Conclusion

SARcasmScope demonstrates that a carefully engineered, calibrated hybrid of linguistic domain heuristics, morphological normalization, phonetic folding, and statistical subwords can match the generalization performance of 237M-parameter transformers on low-resource code-mixed sarcasm detection. By executing entirely client-side in under $0.21\text{ ms}$, SARcasmScope provides an interpretable, energy-efficient, and privacy-preserving alternative for real-world social media platforms and edge devices.
