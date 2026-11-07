# SARcasmScope — Stage 7 Final Verified Experimental Results
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Evaluation Standard:** Held-Out Frozen Test Set ($N = 1,123$) Across 5 Corpora  
**Date:** November 2026  

---

## 1. Primary Empirical Evaluation Matrix

Evaluated across 1,123 independent test records from ACL 2020 Twitter, Misra-Arora Headlines, IIITD MaSaC Hinglish, L3Cube MahaEmotions Marathi, and IIIT-H Telugu:

| Model System | Architecture Paradigm | Parameter Size | Accuracy (%) | Precision (%) | Sarcasm Recall (%) | Macro-F1 (%) | Specificity (%) | ROC-AUC | PR-AUC | Conf-ECE (%) | Brier Score | Latency (P95) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Class Baseline** | Always Literal (Class 0) | 0 | 53.78 | 0.00 | 0.00 | 34.98 | 100.00 | 0.5000 | 0.4622 | 46.22 | 0.4622 | < 0.01 ms |
| **TF-IDF Unigram + LR** | Linear Sparse N-Gram | ~3,500 | 56.28 | 52.91 | 52.36 | 56.12 | 59.67 | 0.5780 | 0.4510 | 38.10 | 0.3810 | 0.15 ms |
| **Word N-Grams (1-2)** | Sparse Surface Sequences | ~5,000 | 57.08 | 53.07 | 61.66 | 57.08 | 53.15 | 0.5792 | 0.4551 | 36.40 | 0.4123 | 0.22 ms |
| **Character N-Grams (3-5)** | Character Substrings | ~6,000 | 58.33 | 54.84 | 55.68 | 58.13 | 60.60 | 0.5877 | 0.4652 | 36.90 | 0.4130 | 0.35 ms |
| **Subword Chunks (3-6)** | Subword Token Hash | ~7,500 | 56.90 | 53.08 | 58.19 | 56.86 | 55.79 | 0.5889 | 0.4653 | 37.10 | 0.4251 | 0.24 ms |
| **Stage 2 Linguistic Engine** | Handcrafted Rules + Lexicons | 0 | 53.78 | 50.00 | 3.08 | 37.59 | 97.35 | 0.5087 | 0.5121 | 23.36 | 0.3075 | 0.18 ms |
| **SARcasmScope Hybrid (Ours)** | **Calibrated Feature Fusion** | **< 10M eq.** | **56.63** | **52.85** | **57.23** | **56.57** | **56.13** | **0.5758** | **0.4652** | **35.10** | **0.4275** | **0.124 ms** |
| **MuRIL Transformer (Server)** | Pre-Trained Cross-Attention | 237M | 57.35 | 53.86 | 60.19 | 57.32 | 54.87 | 0.6010 | 0.5490 | 28.40 | 0.3640 | 85.4 ms |
| **XLM-RoBERTa (Server)** | Cross-Lingual Masked LM | 278M | 56.40 | 56.80 | 58.20 | 56.40 | 55.40 | 0.5910 | 0.5340 | 31.20 | 0.3780 | 92.1 ms |

---

## 2. Statistical Reliability & Bootstrap 95% Confidence Intervals

Computed via 1,000-iteration bootstrap resampling on the test partition:

### Global 95% Bootstrap Confidence Intervals:
- **SARcasmScope Hybrid Macro-F1:** $[53.60\%, 59.42\%]$
- **SARcasmScope Hybrid Accuracy:** $[53.70\%, 59.48\%]$
- **SARcasmScope Hybrid Recall:** $[53.00\%, 61.05\%]$
- **SARcasmScope Hybrid Precision:** $[48.75\%, 57.07\%]$

### Language-Specific Bootstrap Breakdown:
| Language / Variety | Sample Count ($N$) | Accuracy (%) [95% CI] | Macro-F1 (%) [95% CI] | Sarcasm Recall (%) [95% CI] | Precision (%) [95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|
| **English** | 745 | 54.23% $[50.34-57.58]$ | 52.55% $[48.71-55.75]$ | 74.52% $[69.76-78.95]$ | 52.31% $[47.98-56.56]$ |
| **Hindi-English (Hinglish)** | 194 | 53.61% $[46.39-60.31]$ | 48.70% $[41.42-55.74]$ | 22.00% $[13.46-30.53]$ | 64.71% $[47.22-80.65]$ |
| **Marathi** | 75 | 46.67% $[36.00-58.67]$ | 33.86% $[27.18-41.26]$ | 2.70% $[0.00-8.89]$ | 20.00% $[0.00-66.67]$ |
| **Telugu** | 109 | 85.32% $[78.90-91.74]$ | 55.96% $[45.23-68.41]$ | 11.76% $[0.00-28.57]$ | 66.67% $[0.00-100.0]$ |

---

## 3. Paired Significance & Effect Size Analysis

Conducted via McNemar's paired chi-square test with Edwards' continuity correction:

1. **Sarcasm Recall Recovery ($N_{\text{sarc}} = 519$):**
   - Contingency Table: $n_{01} = 282$ (Hybrid correct, Linguistic wrong), $n_{10} = 1$ (Linguistic correct, Hybrid wrong)
   - $\chi^2 = 277.03, \quad p < 0.0001$ (**Extremely Significant**, Bonferroni $\alpha = 0.0167$)
   - **Effect Size:** Odds Ratio $\text{OR} = 282.0$, Cohen's $g = 0.496$ (Large Effect Size)
2. **Overall Accuracy ($N = 1,123$):**
   - Contingency Table: $n_{01} = 284$, $n_{10} = 252$
   - $\chi^2 = 1.79, \quad p = 0.1806$ (**Not Statistically Significant** on overall accuracy)
   - **Effect Size:** Odds Ratio $\text{OR} = 1.13$, Cohen's $g = 0.030$ (Negligible raw accuracy shift due to specificity trade-off)
3. **Hybrid vs Subword Model ($N = 1,123$):**
   - Contingency Table: $n_{01} = 2$, $n_{10} = 5$
   - $\chi^2 = 0.57, \quad p = 0.4497$ (**Statistically Comparable**)
   - Demonstrates that hybrid feature-fusion preserves subword statistical recall while introducing interpretable linguistic grounding and calibrated abstention.

---

## 4. Selective Classification & Risk-Coverage Frontier

Applying selective prediction gating ($\tau \in [0.45, 0.80]$):

| Rejection Threshold ($\tau$) | Empirical Coverage (%) | Accepted Count | Abstained Count | Selective Accuracy (%) | Selective Macro-F1 (%) | Selective Risk (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.45 | 96.88 | 1,088 | 35 | 57.08 | 57.01 | 42.92 |
| 0.50 | 89.40 | 1,004 | 119 | 58.17 | 58.09 | 41.83 |
| 0.55 | 77.20 | 867 | 256 | 59.86 | 59.74 | 40.14 |
| **0.58 (Standard)** | **68.66** | **771** | **352** | **61.48** | **61.34** | **38.52** |
| 0.60 | 62.42 | 701 | 422 | 62.62 | 62.49 | 37.38 |
| 0.65 | 45.95 | 516 | 607 | 65.50 | 65.29 | 34.50 |
| 0.70 | 28.76 | 323 | 800 | 70.28 | 70.04 | 29.72 |
| 0.75 | 14.87 | 167 | 956 | 76.65 | 76.22 | 23.35 |
| 0.80 | 5.34 | 60 | 1,063 | 83.33 | 82.91 | 16.67 |

*Takeaway:* Selective abstention systematically trades coverage for precision, reducing classification risk from $43.37\%$ to **$16.67\%$** at high confidence ($\tau = 0.80$).
