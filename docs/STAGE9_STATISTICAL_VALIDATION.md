# SARcasmScope — Stage 9 Feature Ablation & Statistical Validation Report

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9F — 11-Configuration Runtime Ablations, Bootstrap CIs & McNemar Tests (`ablation_results.csv`, `statistical_analysis.csv`)

---

## 1. Systematic 11-Configuration Runtime Ablation Study (`test.jsonl`, $N = 1,123$)

Executed directly through `predictHybrid` (`src/engine/hybridFusion.ts`) with modular component toggles:

| Ablation Configuration | Toggle / Component | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | Specificity (%) | Conf-ECE (%) | NLL | $\Delta$ MF1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full SARcasmScope Hybrid** | Complete 3-Stage Framework | **68.03** | **67.09** | **60.50** | **67.56** | **74.50** | **1.36** | **0.5826** | **0.00** |
| **w/o WordLex Expert** | `useWord=false` | 64.02 | 61.30 | 60.12 | 63.77 | 67.38 | 4.49 | 0.6208 | **-3.79** |
| **w/o CharSub Expert** | `useChar=false` | 66.96 | 65.10 | 61.46 | 66.62 | 71.69 | 4.73 | 0.6154 | **-0.94** |
| **w/o Linguistic Engine Logit** | `useLinguistic=false` | 67.05 | 61.51 | 76.69 | 67.00 | 58.77 | 2.22 | 0.5914 | **-0.56** |
| **w/o Incongruity Index** | `useIncongruity=false` | 68.03 | 67.09 | 60.50 | 67.56 | 74.50 | 1.21 | 0.5824 | 0.00 |
| **w/o Lexical Coverage** | `useCoverage=false` | 68.12 | 67.24 | 60.50 | 67.64 | 74.67 | 1.61 | 0.5828 | +0.08 |
| **w/o Conversational Context** | `useContext=false` | 64.29 | 63.98 | 52.02 | 63.33 | 74.83 | 4.32 | 0.6136 | **-4.23** |
| **w/o Indic Morph Stemmer** | `useMorph=false` | 68.03 | 67.09 | 60.50 | 67.56 | 74.50 | 1.34 | 0.5816 | 0.00 |
| **w/o Romanized Phonetic Fold** | `usePhonetic=false` | 68.12 | 66.88 | 61.46 | 67.71 | 73.84 | 0.77 | 0.5823 | +0.15 |
| **w/o Word Bigrams** | `useBigrams=false` | 68.83 | 67.94 | 61.66 | 68.39 | 75.00 | 2.97 | 0.5841 | +0.83 |
| **w/o Temperature Scaling** | `useCalibration=false` ($T=1.0$) | 68.03 | 67.09 | 60.50 | 67.56 | 74.50 | 2.28 | 0.5856 | 0.00 |

### Key Ablation Takeaways
1. **Conversational Context (`-4.23%` MF1) and `WordLex_1_2` (`-3.79%` MF1) are the largest empirical drivers** of held-out generalization and log-likelihood (`+0.0310` and `+0.0382` NLL degradation when removed).
2. **`CharSub_3_5` and `Temperature Scaling` sharpen calibration:** Removing `CharSub` spikes Conf-ECE from `1.36%` to `4.73%`; removing $T^*=0.845$ raises Conf-ECE from `1.36%` to `2.28%`.
3. **Symbolic Linguistic Logit acts as a false-positive guard:** Disabling `useLinguistic` drops specificity sharply from **`74.50%` to `58.77%`** (`FP` rises `154 -> 249`).
4. **`useMorph` / `usePhonetic` at inference:** Surface word unigrams and char 3–5 n-grams largely absorb common surface inflections on balanced English-dominated aggregate scores, leaving zero aggregate delta when disabled alone despite aiding symbolic matches and rare OOV stems.

---

## 2. Bootstrap 95% Confidence Intervals ($B = 1,000$, Seed `42`)

- **Calibrated Hybrid Accuracy:** `68.03%`, 95% CI `[65.36%, 70.79%]`
- **Calibrated Hybrid Macro-F1:** `67.56%`, 95% CI `[64.79%, 70.29%]`
- **Calibrated Hybrid Recall:** `60.50%`, 95% CI `[56.00%, 64.77%]`
- **Calibrated Hybrid Precision:** `67.09%`, 95% CI `[62.89%, 71.24%]`
- **Calibrated Hybrid Conf-ECE:** `1.36%`, 95% CI `[1.05%, 4.54%]`
- **`v2` Symbolic Macro-F1:** `37.59%`, 95% CI `[35.73%, 39.66%]` (complete CI separation)

---

## 3. Paired McNemar Tests (`statistical_analysis.csv`)

| Hypothesis Test | Cohorts ($N$) | Discordant ($n_{01} / n_{10}$) | $\chi^2$ | $p$-Value | Odds Ratio | Cohen's $g$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **H1: Overall Acc vs `v2` Symbolic** | `1,123` | `309 / 149` | **55.20** | **< 0.0001** | 2.07 | 0.175 |
| **H2: Sarcasm Recall vs `v2` (Positives)** | `519` | `301 / 3` | **290.16** | **< 0.0001** | 100.33 | 0.490 |
| **H3: Overall Acc vs `v1` Prev** | `1,123` | `314 / 151` | **56.44** | **< 0.0001** | 2.08 | 0.175 |
| **H4: Overall Acc vs `Majority` Class** | `1,123` | `314 / 154` | **54.02** | **< 0.0001** | 2.04 | 0.171 |
| **H5: Overall Acc vs `Statistical_OOF`** | `1,123` | `24 / 41` | 3.94 | **0.0472** | 0.59 | 0.131 |
