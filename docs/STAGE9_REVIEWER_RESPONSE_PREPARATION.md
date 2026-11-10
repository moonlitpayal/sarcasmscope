# SARcasmScope — Stage 9 Peer-Reviewer Response & Rebuttal Guide

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9J — Methodological Defense Grounded in Verified Code & CSV Evidence

---

## Preemptive Author Responses to Anticipated Reviewer Critiques

### Q1. Accuracy Comparison with Nov 7 ($N=30$) & Missing Historical Audit File
**Concern:** *How do the Nov 7 `66.7%` ($N=30$, 5/0/15/10) figures relate to `RESULTS_STATUS_AND_AUDIT.md` (`56.63%`) and your new results?*  
**Response:** `RESULTS_STATUS_AND_AUDIT.md` does not exist in the repository; all results were independently recomputed cleanly via `eval/verify_comparison.ts` and `eval/run_stage9_master_execution.ts`. On the exact same 30 internal diagnostic examples, `v1` (`prevPredict`) reproduces `66.67%` Acc / `33.33%` Rec (`TP=5, FP=0, TN=15, FN=10`), `v2` achieves `96.67%`, and `Calibrated Hybrid` achieves `86.67%`. However, on the independent frozen `test.jsonl` ($N=1,123$), `v1` and `v2` collapse (`0.39%` and `3.08%` recall), whereas our 3-stage `Calibrated Hybrid` rises to **`68.03%` Accuracy, `67.56%` Macro-F1, `60.50%` Recall** (McNemar $\chi^2 = 55.20$ Acc, $\chi^2 = 290.16$ Rec, $p < 0.0001$).

### Q2. Is "Calibrated Feature Fusion" Genuine or Just Heuristic Tuning?
**Response:** It is a mathematically standard **3-stage stacked generalization + validation calibration pipeline** (`eval/train_and_calibrate.ts`):
- **Stage 1 (`train.jsonl` only):** Trains sparse `WordLex_1_2` ($V=8,000$) and `CharSub_3_5` ($V=8,000$) TF-IDF LR experts alongside deterministic `LinguisticEngine`.
- **Stage 2 (5-fold group OOF on `train.jsonl`):** Fits MAP Ridge logistic weights (`w0=0.3483, w1_word=0.54817, w2_char=0.54648, w3_linguistic=0.26486, w4_incongruity=0.19418, w5_coverage=0.20441`).
- **Stage 3 (`val.jsonl` only):** Fits $T^* = 0.845$ and selects $\tau^* = 0.35$. While unweighted `Statistical_OOF` has slightly higher 0.50-threshold test accuracy (`69.55%` vs `68.03%`, McNemar $p=0.0472$), 3-stage calibrated fusion **reduces Conf-ECE by 3.67× (`4.99% -> 1.36%`)**, lowers NLL (`0.5876 -> 0.5826`), and keeps specificity high (`74.50%` vs `58.77%` without symbolic signal).

### Q3. Near-Duplicates & Dialogue Grouping
**Response:** `eval/split_independent_datasets.ts` enforces `0` exact normalized text overlaps and `0` shared `conversation_id`s across splits. An exhaustive $\binom{5600}{2}$ word 3-shingle Jaccard $\ge 0.80$ scan reveals **`13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, `8` label-conflicting)**—all short recurring sitcom lines inside `IIITD_MaSaC_CodeMixed` (`974` records). Rather than silently modifying frozen `test.jsonl`, we report the exact count transparently.

### Q4. Telugu Recall (`0.00%`) vs Ranking (`0.7455` ROC-AUC)
**Response:** Telugu retains its real-world `15.60%` sarcastic prevalence (`17/109`). Under a conservative minority prior, calibrated probabilities stay below the global `0.50` cutoff (`0.00%` Hybrid recall), while ranking discrimination remains strong (**`0.7455` ROC-AUC**, `0.1148` Brier, `0.3797` NLL; `WordLex_1_2` reaches `55.21%` MF1). We document threshold tuning per low-resource prior as an immediate clinical/deployment requirement.

### Q5. Why Are GPU Transformer Metrics Listed as `NOT_EXECUTED_LOCALLY`?
**Response:** Because local workspace sandboxes lack CUDA GPU and `torch`, we refuse to report unexecuted cloud benchmarks as local measurements. Instead, we syntax-check and `--dry_run` verify `eval/train_transformer.py`, `eval/evaluate_transformers.py`, and `eval/export_onnx.py`, remove any unverified MuRIL parity claim, and restrict numerical tables strictly to audited local CPU runs (`2.856 ms` P95, `405.36 KB` artifact).

### Q6. Was Calibration Actually Fixed Given Earlier `0.428` Bin-5 Overconfidence (`1,106` Samples in `[0.9-1.0]`)?
**Response:** Yes—and verified on both splits separately via `eval/verify_independent_recomputation.ts`. Fitting $T^*=0.845$ on `val.jsonl` (`N=1,104`, NLL `0.5798 -> 0.5777`; `Val SelAcc=81.06% @ 49.73%` coverage at $\tau^*=0.35$) transfers cleanly to untouched `test.jsonl` (`N=1,123`), reducing NLL (`0.5856 -> 0.5826`), Brier (`0.2013 -> 0.2005`), Class-ECE (`4.40% -> 2.80%`), and Conf-ECE (`2.28% -> 1.36%`). Bin 5 `[0.9-1.0]` now contains `68/1,123` samples (`0.930` mean confidence vs `0.956` empirical accuracy, `0.026` error), while `80.31%` selective accuracy is achieved across `45.68%` coverage. We explicitly separate $P_{\text{fused}}$, Margin Confidence, $\tau^*$, and Selective Utility, and disclose Hinglish (`15.57%` Conf-ECE) as uncalibrated out-of-distribution dialogue without A/V cues.

### Q7. How Do We Justify Dataset Labels (`MahaEmotions`, `Misra`, `MaSaC`) & Fix the `5,712` CSV Mismatch?
**Response:** `eval/run_independent_benchmarks.ts` was refactored to read frozen `test.jsonl` (`N=1,123`: `344+401+194+75+109`), eliminating the stale `5,712` CSV. Across all 5 corpora (`eval/verify_independent_recomputation.ts`), only `ACL_2020_Twitter_Sarcasm` includes non-empty `context` (`1,716/1,716`); `IIITD_MaSaC` uses `conversation_id` solely for split isolation (`context=""` on isolated text turns, no A/V prosody); `Misra` tests editorial satire (`context=""`). For `L3Cube_MahaEmotions`, we explicitly retained native `Sarcasm` ($n=185$) vs `Neutral` ($n=184$) rather than mapping `Anger`/`Sadness` to literal, while disclosing that Kowtal & Joshi (*PACLIC 39*, 2025) annotated `train.csv` with CoTR LLM prompting vs human-gold `test.csv`—explaining why Marathi is framed strictly as an exploratory subset.

### Q8. How Is the Five-Author vs Sole-Author Discrepancy Handled?
**Response:** All Stage 9 codebase audit logs identify Payal Mehta strictly as repository maintainer, while `STAGE9_FINAL_MANUSCRIPT.md`, `STAGE9_PUBLICATION_READINESS.md`, `CITATION.cff`, and `README.md` carry an explicit **Pre-Submission Blocker Notice**: final 5-author ordering, CRediT contributions, corresponding-author designation (originally the supervisor/collaborator), and all co-authors' sign-off must be completed before submission.
