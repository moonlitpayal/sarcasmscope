# SARcasmScope: Independent Generalization Audit & Version Reconciliation

**Research Paper:** "SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text"  
**Lead Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Source of Truth:** `eval_results/stage9/master_summary.json` & `eval_results/independent/version_comparison_report.json`

---

## 1. Executive Reconciliation Across Code Versions

Re-executing every model version identically via `eval/verify_comparison.ts` and `eval/run_stage9_master_execution.ts` (note: `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace):

| Model Version | Closed-Loop `BENCHMARK_CORPUS` ($N=30$) | Regression `UNSEEN_TEST_CASES` ($N=15$) | Frozen Held-Out `test.jsonl` ($N=1,123$) | Conf-ECE (`test`) |
|:---|:---|:---|:---|:---:|
| **`v1` Prev Symbolic (Nov 7 `prevPredict`)** | `66.67%` Acc / `62.50%` MF1 / `33.33%` Rec (`TP=5,FP=0,TN=15,FN=10`) | `73.33%` Acc / `58.33%` MF1 / `20.00%` Rec (`1/0/10/4`) | `53.52%` Acc / `35.21%` MF1 / **`0.39%` Rec** (`2/5/599/517`) | `34.98%` |
| **`v2` Symbolic Linguistic (`predict`)** | `96.67%` Acc / `96.66%` MF1 / `100.00%` Rec (`15/1/14/0`) | `100.00%` Acc / `100.00%` MF1 (`5/0/10/0`) | `53.78%` Acc / `37.59%` MF1 / **`3.08%` Rec** (`16/16/588/503`) | `23.36%` |
| **`Statistical_OOF_Ensemble` (`WordLex+CharSub`)** | `56.67%` Acc / `56.23%` MF1 / `46.67%` Rec (`7/5/10/8`) | `66.67%` Acc / `60.32%` MF1 / `40.00%` Rec (`2/2/8/3`) | **`69.55%` Acc / `69.06%` MF1 / `61.66%` Rec** (`320/143/461/199`) | `4.99%` |
| **`SARcasmScope_Calibrated_Hybrid` (3-Stage OOF)** | `86.67%` Acc / `86.43%` MF1 / `100.00%` Rec (`15/4/11/0`) | `93.33%` Acc / `92.82%` MF1 / `100.00%` Rec (`5/1/9/0`) | **`68.03%` Acc / `67.56%` MF1 / `60.50%` Rec** (`314/154/450/205`) | **`1.36%`** |

### Key Conclusions
1. **Why Nov 7 (`v1`) showed `66.7%` ($N=30$):** Small closed-loop suites measure synthetic pattern matches; both `v1` (`0.39%`) and `v2` (`3.08%`) collapse on natural independent corpora (`test.jsonl`).
2. **True Generalization Improvement:** The 3-Stage Calibrated Hybrid rescues independent sarcasm recall from `3.08% -> 60.50%` (`+57.42%`, McNemar $\chi^2 = 290.16, p < 0.0001$) and lifts Macro-F1 from `37.59% -> 67.56%` (`+29.97%`, $\chi^2 = 55.20, p < 0.0001$), while cutting Conf-ECE `3.67x` vs the uncalibrated `Statistical_OOF_Ensemble` (`4.99% -> 1.36%`).

---

## 2. Corpus Provenance, Splits & Leakage Audit (`5,600` Records)

| Dataset Identifier | Language | Train / Val / Test ($N$) | Citation & License | Near-Dups ($\ge 0.80$) |
|:---|:---|:---:|:---|:---:|
| `ACL_2020_Twitter_Sarcasm` | English | `1,029 / 343 / 344` (`1,716`) | Ghosh et al. (2020, FigLang/ACL); Research-Only (Twitter Terms) | `0` total / `0` Tr-Te |
| `Misra_Arora_News_Headlines` | English | `1,199 / 399 / 401` (`1,999`) | Misra & Arora (2019/2023, *AI Open*); Public Domain / CC0 | `0` total / `0` Tr-Te |
| `IIITD_MaSaC_CodeMixed` | Hinglish + Devanagari Hindi | `599 / 181 / 194` (`974`) | Bedi et al. (2021, *IEEE TAFFC*); Research Purpose Only | **`13` total / `5` Tr-Te** |
| `L3Cube_MahaEmotions` | Marathi | `221 / 73 / 75` (`369`) | Kulkarni et al. (2023); Kowtal & Joshi (2025); CC-BY-4.0 | `0` total / `0` Tr-Te |
| `IIIT_Hyderabad_Telugu_Sarcasm` | Telugu (`15.7%` pos) | `325 / 108 / 109` (`542`) | Marreddy et al. (2022, LREC/TALLIP); Research Only | `0` total / `0` Tr-Te |

Exact cross-split overlap $= 0$; `conversation_id` overlap $= 0$; exhaustive 3-shingle Jaccard $\ge 0.80$ detects `13` near-duplicate pairs (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

---

## 3. Per-Language Held-Out Test Performance (`test.jsonl`, $N = 1,123$)

- **English ($745$):** `71.54%` Acc, `71.48%` MF1, `68.22%` Rec, `0.7826` ROC-AUC (`+34.15%` MF1 vs `v2`).
- **Marathi ($75$):** `62.67%` Acc, `62.66%` MF1, `62.16%` Rec, `1.29%` Conf-ECE (`+29.03%` MF1 vs `v2`).
- **Hinglish ($194$):** `47.42%` Acc, `47.33%` MF1, `42.00%` Rec (`+14.69%` MF1 vs `v2`).
- **Telugu ($109$):** `84.40%` Acc, `45.77%` MF1, `0.00%` Rec at `0.50` cutoff (`0.7455` ROC-AUC, `0.1148` Brier under `15.60%` positive prior).
- **GPU Transformers:** `eval/train_transformer.py`, `eval/evaluate_transformers.py`, `eval/export_onnx.py` verified via syntax + `--dry_run`; GPU metrics marked `NOT_EXECUTED_LOCALLY`.
