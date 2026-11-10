# SARcasmScope — Investigation of Accuracy Discrepancies & Calibrated Feature-Fusion Formalization
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Status:** Canonical Audit Reconciling Nov 7 Evaluation, Pre-Refactor Registers, and Frozen Stage 9 Results

---

## 1. Scope of Investigation

This document directly resolves the two methodological questions raised for peer-review readiness:
1. **Version & Benchmark Reconciliation:** Identifying the exact code version behind the Nov 7 evaluation (`66.7%` Acc, `100%` Prec, `33.3%` Rec, `62.5%` Macro-F1, `TP=5, FP=0, TN=15, FN=10` on $N=30$), investigating historical `56.63%` / `56.57%` registers (noting explicitly that `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace), and re-executing **all versions head-to-head on identical test items** (`BENCHMARK_CORPUS` $N=30$, `UNSEEN_TEST_CASES` $N=15$, and frozen `eval_results/independent/test.jsonl` $N=1,123$).
2. **Establishing Genuine 3-Stage Calibrated Feature Fusion:** Replacing ad-hoc heuristic score summation with a mathematically defined stacked OOF meta-learner fitted strictly on `train.jsonl` and calibrated strictly on `val.jsonl`.

---

## 2. Reconciling Code Versions on Identical Examples (`eval/verify_comparison.ts`)

| Evaluation Suite ($N$) | `v1` Prev Symbolic (Nov 7 `prevPredict`) | `v2` Symbolic Linguistic (`predict`) | Level-1 `Statistical OOF` (`WordLex+CharSub`) | **3-Stage `Calibrated Hybrid` (`predictHybrid`)** |
|:---|:---|:---|:---|:---|
| **`BENCHMARK_CORPUS` ($N=30$, closed-loop)** | **66.67% Acc**, **62.50% MF1**, 100.00% Prec, **33.33% Rec** (`TP=5,FP=0,TN=15,FN=10`) | **96.67% Acc**, 96.66% MF1, 93.75% Prec, 100.00% Rec (`15/1/14/0`) | 56.67% Acc, 56.23% MF1, 58.33% Prec, 46.67% Rec (`7/5/10/8`) | **86.67% Acc**, 86.43% MF1, 78.95% Prec, 100.00% Rec (`15/4/11/0`) |
| **`UNSEEN_TEST_CASES` ($N=15$, regression)** | 73.33% Acc, 58.33% MF1, 100% Prec, 20.00% Rec (`1/0/10/4`) | 100.00% Acc, 100% MF1 (`5/0/10/0`) | 66.67% Acc, 60.32% MF1, 50.00% Prec, 40.00% Rec (`2/2/8/3`) | **93.33% Acc**, 92.82% MF1, 83.33% Prec, 100.00% Rec (`5/1/9/0`) |
| **Frozen Held-Out `test.jsonl` ($N=1,123$)** | **53.52% Acc**, 35.21% MF1, 28.57% Prec, **0.39% Rec** (`2/5/599/517`) | **53.78% Acc**, 37.59% MF1, 50.00% Prec, **3.08% Rec** (`16/16/588/503`) | **69.55% Acc**, **69.06% MF1**, 69.11% Prec, 61.66% Rec, `4.99%` ECE (`320/143/461/199`) | **68.03% Acc**, **67.56% MF1**, 67.09% Prec, **60.50% Rec**, **`1.36%` Conf-ECE**, `0.5826` NLL (`314/154/450/205`) |

### Key Conclusions from Head-to-Head Verification
1. **Why Nov 7 (`v1`) showed 66.7% and `v2` showed 96.7% on $N=30$:** Those 30 items were internal closed-loop examples (`eval/benchmarkCorpus.ts`), where `v2` added explicit lexical rules tailored to cover those patterns (`33.3% -> 100%` recall). On real independent text (`test.jsonl`, $N=1,123$), both symbolic versions collapse (`v1 = 0.39%` recall, `v2 = 3.08%` recall), proving rule tuning on small handcrafted suites does NOT generalize to natural text.
2. **Legacy `56.63%` / `56.57%` Register & Absence of `RESULTS_STATUS_AND_AUDIT.md`:** `RESULTS_STATUS_AND_AUDIT.md` is not present in the repository. Re-evaluating all models reproducibly on frozen `test.jsonl` replaces that unverified historical register with audited measurements: **`68.03%` Acc / `67.56%` MF1 / `1.36%` Conf-ECE** for the 3-Stage Calibrated Hybrid (`+29.97%` Macro-F1 and `+57.42%` Recall over `v2`, McNemar $\chi^2 = 55.20, p < 0.0001$ overall Acc; $\chi^2 = 290.16, p < 0.0001$ on sarcastic samples).

---

## 3. Verifying Genuine Feature Fusion & Non-Circular Calibration (`eval/train_and_calibrate.ts`)

1. **Non-Circular 3-Stage Hierarchy:**
   - **Stage 1 (`train.jsonl`, $N=3,373$ only):** Fits `WordLex_1_2` ($V=8,000$) and `CharSub_3_5` ($V=8,000$) via full-batch Adam + $L_2$ regularization alongside deterministic `LinguisticEngine`.
   - **Stage 2 (`train.jsonl` 5-fold group-stratified OOF only):** Groups folds by `dataset__conversation_id` to prevent within-conversation overfitting and fits a MAP Ridge logistic meta-learner (`w0=0.3483, w1_word=0.54817, w2_char=0.54648, w3_linguistic=0.26486, w4_incongruity=0.19418, w5_coverage=0.20441`).
   - **Stage 3 (`val.jsonl`, $N=1,104$ only):** Fits scalar temperature $T^* = 0.845$ (NLL $0.5798 \to 0.5777$) and chooses selective abstention threshold $\tau^* = 0.35$. **`test.jsonl` is never accessed during training or calibration.**
2. **Empirical Benefit Over Single Experts & Uncalibrated Stacking:**
   - Although the unweighted statistical logit mean reaches slightly higher raw 0.50-threshold test accuracy (`69.55%` vs `68.03%`, McNemar $\chi^2 = 3.94, p = 0.0472$), it suffers **`4.99%` Conf-ECE** and **`0.5876` NLL**.
   - Adding Stage-2 OOF fusion + Stage-3 temperature scaling reduces **Conf-ECE by 3.67×** (`4.99% -> 1.36%`, and `2.28% -> 1.36%` vs uncalibrated $T=1.0$), lowers proper scoring (`NLL 0.5876 -> 0.5826`, `Brier 0.2016 -> 0.2005`), and boosts specificity over removing the linguistic logit (`58.77% -> 74.50%`), enabling reliable selective gating (`80.31%` Acc @ `45.68%` coverage for $\tau^*=0.35$).
