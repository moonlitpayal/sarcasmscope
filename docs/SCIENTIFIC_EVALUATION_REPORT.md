# SARcasmScope: Comprehensive Scientific Evaluation & Experimental Analysis
**Paper Title:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Codebase Maintainer / Audit Lead:** Payal Mehta (`payal.12620246@vit.edu`)  
**Status:** Reproducible Empirical Benchmark Report (**Submission Blocked Until 5-Author Order, Corresponding Authorship & Dataset/GPU Scope Are Approved by All Co-Authors**)

---

## Executive Summary

This report documents two distinct evaluation tiers side-by-side so closed-loop development diagnostics are never conflated with independent held-out generalization:
1. **Closed-Loop Diagnostic Suite (`BENCHMARK_CORPUS`, $N = 30$, 5 languages):**
   - **Nov 7 `v1` Prev Symbolic (`prevPredict`):** **`66.67%` Accuracy, `100.00%` Precision, `33.33%` Recall, `62.50%` Macro-F1** (`TP=5, FP=0, TN=15, FN=10`, Conf-ECE `25.55%`, NLL `0.9770`). Missed 10/15 sarcastic examples due to strict surface rules.
   - **Stage 2 `v2` Symbolic (`predict`):** `96.67%` Accuracy, `93.75%` Precision, `100.00%` Recall, `96.66%` Macro-F1 (`15/1/14/0`).
   - **Stage 9 `Calibrated Hybrid` (`predictHybrid`):** `86.67%` Accuracy, `78.95%` Precision, `100.00%` Recall, `86.43%` Macro-F1 (`15/4/11/0`).
2. **Independent Frozen Held-Out Test Split (`eval_results/independent/test.jsonl`, $N = 1,123$, 5 public corpora):**
   - **`v1` Prev Symbolic:** `53.52%` Acc, `28.57%` Prec, `0.39%` Rec, `35.21%` Macro-F1 (`2/5/599/517`, `34.98%` Conf-ECE).
   - **`v2` Symbolic Linguistic:** `53.78%` Acc, `50.00%` Prec, `3.08%` Rec, `37.59%` Macro-F1 (`16/16/588/503`, `23.36%` Conf-ECE).
   - **`Statistical_OOF_Ensemble` (Level-1 WordLex+CharSub):** `69.55%` Acc, `69.11%` Prec, `61.66%` Rec, `69.06%` Macro-F1 (`320/143/461/199`, `4.99%` Conf-ECE, `0.5876` NLL).
   - **`SARcasmScope_Calibrated_Hybrid` (3-Stage OOF + $T^*=0.845, \tau^*=0.35$):** **`68.03%` Accuracy** (`[65.36%, 70.79%]`), **`67.09%` Precision**, **`60.50%` Recall** (`[56.00%, 64.77%]`), **`67.56%` Macro-F1** (`[64.79%, 70.29%]`), **`74.50%` Specificity**, **`0.7535` ROC-AUC**, **`0.7014` PR-AUC**, **`1.36%` Conf-ECE** (`0.5826` NLL, `0.2005` Brier; `TP=314, FP=154, TN=450, FN=205`), and **`80.31%` selective accuracy** (`45.68%` coverage at $\tau^*=0.35$).

---

## 1. Feature-Wise Runtime Capabilities (`src/engine/`)

| Module / Feature | Source File | Verified Behavior & Measurement |
|:---|:---|:---|
| **NFKC & Punctuation Normalizer** | `src/engine/sarcasmEngine.ts` | Empty strings safely yield prior output without throwing exceptions; preserves exclamation/question emphasis. |
| **Indic Morphological Stemmer** | `src/engine/sarcasmEngine.ts` | Strips conservative Devanagari/Telugu suffixes (`stemIndicToken`), generating `st:` keys in `WordLex_1_2` + symbolic rules. |
| **Romanized Phonetic Folder** | `src/engine/sarcasmEngine.ts` | Collapses elongation & transliteration variants (`normalizeRomanized`), emitting `ph:` keys (`100%` spelling invariance test stability). |
| **Unicode Script & CMI Analyzer** | `src/engine/sarcasmEngine.ts` | Identifies Devanagari, Telugu, Latin, Tamil, Bengali, Gujarati, etc. and computes Code-Mixing Index (`cmi`). |
| **Level-1 Base Experts (`WordLex`, `CharSub`)** | `src/engine/statisticalModel.ts` | Sparse $V=8,000$ TF-IDF LR experts fitted exclusively on `train.jsonl` ($N=3,373$) via bit-identical `src/engine/textFeatures.ts`. |
| **3-Stage Calibrated Stacking** | `src/engine/hybridFusion.ts` | 5-fold group OOF MAP Ridge meta-learner + validation temperature $T^*=0.845$ lowering Test Conf-ECE from `4.99%` to **`1.36%`**. |
| **Selective Epistemic Gate** | `src/engine/hybridFusion.ts` | Confidence $= 2|P_{\text{fused}}-0.5| < \tau^* (0.35)$ yields `"Uncertain"`, boosting Test accuracy `68.03% -> 80.31%` at `45.68%` coverage. |
| **LOTO Occlusion Attribution** | `src/engine/hybridFusion.ts` | Index-safe leave-one-token-out probability deltas ($\Delta_i = P - P_{\setminus i}$) exported in `explainability_results.csv`. |
| **9-Perturbation Stress Suite** | `eval/run_stage9_master_execution.ts` | Deterministic seed-42 stress test on `testRecords` ($N=100$): 95–100% on 7 transforms, 75% on triple vowel elongation, 70% on context removal. |
| **Latency & Memory Profile** | `eval_results/stage9/software_audit.csv` | Mean **`1.6319 ms`**, Median **`1.4230 ms`**, P95 **`2.8557 ms`**, P99 **`4.7743 ms`**; bundle **`405.36 KB`**, heap `70.53 MB`. |

---

## 2. Reconciled Closed-Loop ($N=30$) vs Frozen Independent Test ($N=1,123$)

### 2.1 Historical Closed-Loop Confusion Matrix (`v1` Nov 7 on `BENCHMARK_CORPUS`, $N=30$)
```
                    Predicted Literal (0)   Predicted Sarcastic (1)
Actual Literal (0)           15 (TN)                  0 (FP)
Actual Sarcastic (1)         10 (FN)                  5 (TP)
```
Accuracy $= 20/30 = 66.67\%$, Precision $= 5/5 = 100.00\%$, Recall $= 5/15 = 33.33\%$, Macro-F1 $= 62.50\%$.

### 2.2 Canonical Held-Out Confusion Matrix (`SARcasmScope_Calibrated_Hybrid` on `test.jsonl`, $N=1,123$)
```
                    Predicted Literal (0)   Predicted Sarcastic (1)
Actual Literal (0)          450 (TN)                154 (FP)
Actual Sarcastic (1)        205 (FN)                314 (TP)
```
Accuracy $= 764/1,123 = 68.03\%$, Precision $= 314/468 = 67.09\%$, Recall $= 314/519 = 60.50\%$, Specificity $= 450/604 = 74.50\%$, Macro-F1 $= 67.56\%$, Conf-ECE $= 1.36\%$, ROC-AUC $= 0.7535$.

### 2.3 Validation Calibration & Risk-Coverage Curve (`test.jsonl`)
- **Validation Calibration (`val.jsonl`, $N=1,104$):** $T^* = 0.845$ minimizes validation NLL ($0.5798 \to 0.5777$); grid selects $\tau^* = 0.35$ (`81.02%`–`81.06%` selective accuracy across `~49.7%` coverage).
- **Held-Out Test Risk-Coverage:**
  - $\tau = 0.00$: `100.00%` cov, `68.03%` acc, `31.97%` risk
  - $\tau = 0.20$: `67.68%` cov, `73.95%` acc, `26.05%` risk
  - **$\tau^* = 0.35$ (Default):** **`45.68%` cov** ($513/1,123$), **`80.31%` acc** (`78.92%` MF1), **`19.69%` risk**
  - $\tau = 0.50$: `28.32%` cov, `87.42%` acc, `12.58%` risk
  - $\tau = 0.60$: `19.32%` cov, `88.48%` acc, `11.52%` risk

---

## 3. Uncertainty Separation, Dataset Caveats, GPU Scope & Authorship Disclosures

- **Four Uncertainty Constructs:** (1) Calibrated class probability $P_{\text{fused}} = \sigma(z_{\text{raw}}/T^*) \in (0,1)$ with $T^*=0.845$ fitted by validation NLL (aggregate test `Conf-ECE = 1.36%`, Bin 5 `[0.9-1.0]` now `68` samples with `0.026` error vs legacy `1,106`-sample `0.428` defect; sub-corpus Hinglish remains `15.57%` Conf-ECE); (2) Margin confidence $\text{Confidence} = 2|P_{\text{fused}}-0.5| \in [0,1]$; (3) Abstention threshold $\tau^* = 0.35$ (`val.jsonl`); (4) Selective utility (`80.31%` selective acc @ `45.68%` coverage on `test.jsonl`).
- **Dataset Construct Boundaries:** `L3Cube_MahaEmotions` (`Sarcasm` $n=185$ vs `Neutral` $n=184$, excluding anger/sadness) is an exploratory emotion-proxy subset largely from Kowtal & Joshi's (2025) CoTR-annotated training pool; `Misra_Arora` (`context=""`) tests editorial news satire; `IIITD_MaSaC` evaluates isolated utterances (`context=""`, no audio/video prosody). Both `eval_results/stage9/` and `eval_results/independent/` pass independent row-level recomputation (`eval/verify_independent_recomputation.ts`: `1,123/1,123`, `max|Δp|=0`).
- **GPU & Legacy Files:** Deep neural baselines (`google/muril-base-cased`, `xlm-roberta-base`) pass `py_compile` + `--dry_run` and are marked **`NOT_EXECUTED_LOCALLY`** (no MuRIL parity claimed). `RESULTS_STATUS_AND_AUDIT.md` does not exist in the workspace. Final 5-author list & corresponding authorship await co-author approval before submission.
