# SARcasmScope: Scientific Evaluation & Experimental Report

**Lead Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Repository:** `https://github.com/moonlitpayal/sarcasmscope`

---

## 1. Executive Empirical Summary

### Closed-Loop Nov 7 Reproduction (`BENCHMARK_CORPUS`, $N = 30$)
- **`v1` Prev Symbolic (`prevPredict`):** `66.67%` Acc, `100.00%` Prec, `33.33%` Rec, `62.50%` Macro-F1 (`TP=5, FP=0, TN=15, FN=10`).
- **`v2` Symbolic (`predict`):** `96.67%` Acc, `100.00%` Rec, `96.66%` Macro-F1 (`15/1/14/0`).
- **`Calibrated Hybrid` (`predictHybrid`):** `86.67%` Acc, `100.00%` Rec, `86.43%` Macro-F1 (`15/4/11/0`).

### Frozen Held-Out Test Split (`test.jsonl`, $N = 1,123$, Single Pass)
- **`v1` Prev Symbolic:** `53.52%` Acc, `0.39%` Rec, `35.21%` MF1 (`2/5/599/517`, `34.98%` Conf-ECE).
- **`v2` Symbolic Linguistic:** `53.78%` Acc, `3.08%` Rec, `37.59%` MF1 (`16/16/588/503`, `23.36%` Conf-ECE).
- **`WordLex_1_2` / `CharSub_3_5`:** `68.39%` Acc (`67.96%` MF1) / `65.36%` Acc (`65.01%` MF1).
- **`Statistical_OOF_Ensemble`:** `69.55%` Acc, `69.06%` MF1, `61.66%` Rec, `4.99%` Conf-ECE, `0.5876` NLL.
- **`SARcasmScope_Calibrated_Hybrid` ($T^* = 0.845, \tau^* = 0.35$):** **`68.03%` Acc** (`[65.36%, 70.79%]`), **`67.09%` Prec**, **`60.50%` Rec** (`[56.00%, 64.77%]`), **`67.56%` MF1** (`[64.79%, 70.29%]`), **`74.50%` Spec**, **`0.7535` ROC-AUC**, **`0.7014` PR-AUC**, **`1.36%` Conf-ECE** (`[1.05%, 4.54%]`), **`0.5826` NLL**, **`0.2005` Brier** (`TP=314, FP=154, TN=450, FN=205`).
- **Statistical Significance (McNemar vs `v2`):** Overall Acc $\chi^2 = 55.20, p < 0.0001$; Sarcasm Recall $\chi^2 = 290.16, p < 0.0001$.
- **Calibration & Selective Utility:** 5 monotonic reliability bins (`|diff| <= 0.026`). At $\tau^* = 0.35$, selective accuracy reaches **`80.31%`** (`78.92%` MF1) across **`45.68%` coverage** ($513/1,123$ accepted; risk `31.97% -> 19.69%`).
- **Key Ablations:** `w/o Context` (`-4.23%` MF1), `w/o WordLex` (`-3.79%` MF1), `w/o CharSub` (`ECE 1.36% -> 4.73%`), `w/o Linguistic Logit` (`Specificity 74.50% -> 58.77%`), `w/o Temp Scaling` (`ECE 1.36% -> 2.28%`).
- **CPU Runtime Profiling (`software_audit.csv`):** `1.632 ms` mean, `1.423 ms` median, `2.856 ms` P95, `405.36 KB` bundle, `70.53 MB` heap.
- **Note on Missing Legacy File & GPU Scripts:** `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace; GPU scripts are syntax- & `--dry_run`-verified locally and marked `NOT_EXECUTED_LOCALLY`.
