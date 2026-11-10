# SARcasmScope — Unified Experimental Results & Empirical Analysis
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Source of Truth:** `eval_results/stage9/master_summary.json` & `eval_results/stage9/*.csv`

---

## 1. Frozen Held-Out Test Evaluation Matrix (`test.jsonl`, $N = 1,123$)

Evaluated in a single pass on frozen `eval_results/independent/test.jsonl` (519 Sarcastic, 604 Literal). Zero test-set hyperparameter tuning or weight updates.

| Model Identifier | Params | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | Specificity (%) | ROC-AUC | PR-AUC | Conf-ECE (%) | Brier | NLL | Latency (P95) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `Majority_Class_Literal` | 0 | 53.78 | 0.00 | 0.00 | 34.97 | 100.00 | 0.5000 | 0.4691 | 46.22 | 0.4622 | 12.7698 | < 0.01 ms | MEASURED |
| `v1_Prev_Symbolic` (Nov 7) | 0 | 53.52 | 28.57 | 0.39 | 35.21 | 99.17 | 0.5101 | 0.5102 | 34.98 | 0.3707 | 1.0957 | 0.112 ms | MEASURED |
| `v2_Symbolic_Linguistic` | 0 | 53.78 | 50.00 | 3.08 | 37.59 | 97.35 | 0.5087 | 0.5121 | 23.36 | 0.3075 | 0.9010 | 1.509 ms | MEASURED |
| `WordLex_1_2` | 8,000 | 68.39 | 67.30 | 61.46 | 67.96 | 74.34 | 0.7324 | 0.6901 | 2.80 | 0.2081 | 0.6021 | 0.200 ms | MEASURED |
| `CharSub_3_5` | 8,000 | 65.36 | 63.21 | 59.92 | 65.01 | 70.03 | 0.7294 | 0.6652 | 2.11 | 0.2090 | 0.6029 | 0.378 ms | MEASURED |
| `Statistical_OOF_Ensemble` | 16,000 | 69.55 | 69.11 | 61.66 | 69.06 | 76.32 | 0.7582 | 0.7024 | 4.99 | 0.2016 | 0.5876 | 0.881 ms | MEASURED |
| `Uncalibrated_Hybrid_T1` | 16,006 | 68.03 | 67.09 | 60.50 | 67.56 | 74.50 | 0.7536 | 0.7015 | 2.28 | 0.2013 | 0.5856 | 2.856 ms | MEASURED |
| **`SARcasmScope_Calibrated_Hybrid`** | **16,006** | **68.03** | **67.09** | **60.50** | **67.56** | **74.50** | **0.7535** | **0.7014** | **1.36** | **0.2005** | **0.5826** | **2.856 ms** | **MEASURED** |
| `MuRIL_FineTuned_GPU` | 237M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` |
| `XLM_RoBERTa_GPU` | 278M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` |

---

## 2. Bootstrap Confidence Intervals & Paired McNemar Significance

### 1,000-Iteration Bootstrap 95% CIs (`Seed = 42`)
- **SARcasmScope Calibrated Hybrid:** Accuracy `[65.36%, 70.79%]`, Macro-F1 `[64.79%, 70.29%]`, Precision `[62.89%, 71.24%]`, Recall `[56.00%, 64.77%]`, Conf-ECE `[1.05%, 4.54%]`.
- **`v2` Symbolic Linguistic:** Accuracy `[50.93%, 56.81%]`, Macro-F1 `[35.73%, 39.66%]`, Recall `[1.69%, 4.68%]`.

### McNemar Tests (`Calibrated Hybrid` vs Baselines)
- **Overall Accuracy vs `v2` ($N=1,123$):** $\chi^2 = 55.20, p < 0.0001, OR = 2.07, g = 0.175$ ($n_{01}=309, n_{10}=149$).
- **Sarcasm Recall vs `v2` ($N=519$):** $\chi^2 = 290.16, p < 0.0001, OR = 100.33, g = 0.490$ ($n_{01}=301, n_{10}=3$).
- **Overall Accuracy vs `v1` ($N=1,123$):** $\chi^2 = 56.44, p < 0.0001, OR = 2.08$ ($n_{01}=314, n_{10}=151$).
- **Overall Accuracy vs `Majority` ($N=1,123$):** $\chi^2 = 54.02, p < 0.0001, OR = 2.04$ ($n_{01}=314, n_{10}=154$).
- **Overall Accuracy vs `Statistical_OOF_Ensemble` ($N=1,123$):** $\chi^2 = 3.94, p = 0.0472, OR = 0.59, g = 0.131$ ($n_{01}=24, n_{10}=41$). Stat OOF leads on unweighted 0.50-threshold accuracy (`+1.52%`), whereas Calibrated Hybrid reduces **Conf-ECE by 3.67×** (`4.99% -> 1.36%`) and improves proper scoring (`NLL 0.5876 -> 0.5826`, `Brier 0.2016 -> 0.2005`).

---

## 3. Systematic 11-Configuration Runtime Ablation Study (`ablation_results.csv`)

| Configuration | Accuracy (%) | Recall (%) | Macro-F1 (%) | Specificity (%) | Conf-ECE (%) | NLL | $\Delta$ MF1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full SARcasmScope Hybrid** | **68.03** | **60.50** | **67.56** | **74.50** | **1.36** | **0.5826** | **0.00** |
| w/o WordLex Expert | 64.02 | 60.12 | 63.77 | 67.38 | 4.49 | 0.6208 | **-3.79** |
| w/o CharSub Expert | 66.96 | 61.46 | 66.62 | 71.69 | 4.73 | 0.6154 | **-0.94** |
| w/o Linguistic Engine Logit | 67.05 | 76.69 | 67.00 | 58.77 | 2.22 | 0.5914 | **-0.56** |
| w/o Incongruity Index Feature | 68.03 | 60.50 | 67.56 | 74.50 | 1.21 | 0.5824 | 0.00 |
| w/o Lexical Coverage Feature | 68.12 | 60.50 | 67.64 | 74.67 | 1.61 | 0.5828 | +0.08 |
| w/o Conversational Context | 64.29 | 52.02 | 63.33 | 74.83 | 4.32 | 0.6136 | **-4.23** |
| w/o Indic Morphological Stemmer | 68.03 | 60.50 | 67.56 | 74.50 | 1.34 | 0.5816 | 0.00 |
| w/o Romanized Phonetic Folding | 68.12 | 61.46 | 67.71 | 73.84 | 0.77 | 0.5823 | +0.15 |
| w/o Word Bigrams | 68.83 | 61.66 | 68.39 | 75.00 | 2.97 | 0.5841 | +0.83 |
| w/o Temperature Scaling ($T=1.0$) | 68.03 | 60.50 | 67.56 | 74.50 | 2.28 | 0.5856 | 0.00 |

---

## 4. Selective Prediction & Risk-Coverage Profile (`risk_coverage.csv`)

| Threshold ($\tau$) | Coverage (%) | Accepted | Abstained | Selective Acc (%) | Selective MF1 (%) | Risk (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 100.00 | 1,123 | 0 | 68.03 | 67.56 | 31.97 |
| 0.10 | 83.53 | 938 | 185 | 71.86 | 71.32 | 28.14 |
| 0.20 | 67.68 | 760 | 363 | 73.95 | 73.21 | 26.05 |
| **0.35 (Default)** | **45.68** | **513** | **610** | **80.31** | **78.92** | **19.69** |
| 0.50 | 28.32 | 318 | 805 | 87.42 | 85.17 | 12.58 |
| 0.60 | 19.32 | 217 | 906 | 88.48 | 84.69 | 11.52 |
