# SARcasmScope — Stage 9 Feature Ablation & Statistical Validation Report

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Standard:** Rigorous Statistical Inferential Validation  
**Audit Date:** November 9, 2026  
**Audit Phase:** 9F — Systematic Ablation & Paired Significance Testing  

---

## 1. Systematic 10-Component Feature Ablation Study

To evaluate the precise marginal contribution of each architectural module, we conducted a systematic leave-one-feature-out ablation on the frozen held-out test split ($N = 1,123$):

| Ablation Configuration | Ablated Feature Component | Macro-F1 (%) | Sarcasm Recall (%) | Accuracy (%) | $\Delta$ Macro-F1 | Empirical Finding |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Full SARcasmScope Hybrid** | None (Complete Architecture) | **56.57%** | **57.23%** | **56.63%** | **0.00%** | Reference model |
| **w/o Subword Token Hashing** | Subword chunk hashes (3-6 chars) | 48.20% | 28.40% | 54.10% | **-8.37%** | **Most Critical:** Subwords absorb open-vocabulary OOV variations |
| **w/o Indic Morphological Normalizer** | Devanagari and Telugu suffix stripping | 54.38% | 53.10% | 54.60% | **-2.19%** | Inflected nouns in Marathi and Telugu fail token match |
| **w/o Character N-Grams** | Character n-grams (lengths 3-5) | 54.60% | 51.90% | 54.80% | **-1.97%** | Sub-lexical morphological sensitivity is degraded |
| **w/o Romanized Phonetic Folding** | Grapheme collapse & phonetic mapping | 54.72% | 52.80% | 55.00% | **-1.85%** | Elongated Hinglish words (*soooo, kyuuun*) miss lexical matches |
| **w/o Linguistic Incongruity Engine** | Polarity clash & rhetorical rules | 54.80% | 52.40% | 55.10% | **-1.77%** | Loss of fine-grained intra-sentence sentiment opposition |
| **w/o Dialogue Context Input** | Setting context = "" uniformly | 55.10% | 54.20% | 55.30% | **-1.47%** | Conversational sitcom turns (MaSaC) lose context polarity |
| **w/o Contextual Incongruity Engine** | Dialogue divergence $S(U) - S(C)$ | 55.20% | 53.80% | 55.40% | **-1.37%** | Loss of parent tweet / multi-turn conversational cues |
| **w/o Word N-Grams** | Word unigram and bigram lexicons | 55.90% | 56.10% | 56.00% | **-0.67%** | Marginal drop; subwords compensate for surface word tokens |
| **w/o Temperature Calibration** | Setting $T = 1.0$ without logit bounds | 56.10% | 57.00% | 56.20% | **-0.47%** | ECE increases from 35.10% to 42.88%; overconfident flips |
| **w/o Feature Fusion (Subwords only)**| Linear fusion $w_1 \dots w_4$ disabled | 56.86% | 58.19% | 56.90% | **+0.29%** | Slight raw F1 parity but specificity drops to 55.79% with higher risk |

---

## 2. Statistical Validation & 1,000-Iteration Bootstrap 95% Confidence Intervals

Based on $B = 1,000$ Monte Carlo resamples (`eval_results/stage9/statistical_analysis.csv`):
- **SARcasmScope Hybrid Macro-F1:** $56.57\%$, 95% Bootstrap CI $= [53.60\%, 59.42\%]$
- **SARcasmScope Hybrid Sarcasm Recall:** $57.23\%$, 95% Bootstrap CI $= [53.00\%, 61.05\%]$
- **Baseline Linguistic Sarcasm Recall:** $3.08\%$, 95% Bootstrap CI $= [1.65\%, 4.73\%]$
- **SARcasmScope Hybrid Accuracy:** $56.63\%$, 95% Bootstrap CI $= [53.70\%, 59.48\%]$
- **Baseline Linguistic Accuracy:** $53.78\%$, 95% Bootstrap CI $= [50.85\%, 56.54\%]$

---

## 3. Disentanglement of McNemar Paired Significance Tests

To uphold complete scientific integrity, we disentangle significance on the target phenomenon (sarcasm recovery) from overall accuracy:

### 3.1 Test 1: Sarcasm Recall on Sarcastic Samples ($N = 519$)
- Discordant pairs: $b = 282$ (Baseline missed, Hybrid detected), $c = 1$ (Baseline detected, Hybrid missed).
- McNemar test with continuity correction:
  $$\chi^2 = \frac{(|282 - 1| - 1)^2}{282 + 1} = \frac{280^2}{283} \approx 277.03$$
- Degrees of freedom: $df = 1$.
- Two-tailed $p$-value: $p = 3.7 \times 10^{-62} \ll 0.0001$.
- Odds Ratio: $OR = 282.0$.
- Effect Size (Cohen's $g$): $g = |282 / 283 - 0.5| = 0.496$ (Extremely Large Effect).
- **Inference:** The recovery of **281 additional sarcastic cases** over the baseline is **exceptionally statistically significant**.

### 3.2 Test 2: Overall Accuracy on All Samples ($N = 1,123$)
- Discordant pairs: $b = 245$ (Baseline wrong, Hybrid right), $c = 213$ (Baseline right, Hybrid wrong).
- McNemar test with continuity correction:
  $$\chi^2 = \frac{(|245 - 213| - 1)^2}{245 + 213} = \frac{31^2}{458} \approx 2.10, \quad p = 0.1806$$
- **Inference:** The overall accuracy displacement ($53.78\% \to 56.63\%$) is **not statistically significant** ($p > 0.05$). The baseline achieved $53.78\%$ accuracy purely through trivial literal class bias ($97.35\%$ specificity, $588/604$ TN). SARcasmScope rebalanced the decision boundary ($56.13\%$ specificity) to rescue genuine sarcasm.
