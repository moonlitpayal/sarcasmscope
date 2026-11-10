# Independent Generalization Audit & Version Reconciliation Report
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Lead Investigator:** Payal Mehta (`payal.12620246@vit.edu`)

---

## 1. Purpose & Historical Context

This audit reconciles every closed-loop diagnostic check against the independent 5-corpus benchmark (`5,600` records total: `3,373` train, `1,104` val, `1,123` held-out test) and explicitly notes that `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, so every metric below is freshly recomputed by executable scripts (`eval/verify_comparison.ts`, `eval/run_stage9_master_execution.ts`).

| System Version | Closed-Loop `BENCHMARK_CORPUS` ($N=30$) | Regression `UNSEEN_TEST_CASES` ($N=15$) | Frozen Held-Out `test.jsonl` ($N=1,123$) | Conf-ECE (`test`) |
|:---|:---|:---|:---|:---:|
| **`v1` Prev Symbolic (Nov 7)** | `66.67%` Acc / `62.50%` MF1 / `33.33%` Rec (`5/0/15/10`) | `73.33%` Acc / `58.33%` MF1 / `20.00%` Rec (`1/0/10/4`) | `53.52%` Acc / `35.21%` MF1 / **`0.39%` Rec** (`2/5/599/517`) | `34.98%` |
| **`v2` Symbolic Linguistic** | `96.67%` Acc / `96.66%` MF1 / `100.00%` Rec (`15/1/14/0`) | `100.00%` Acc / `100.00%` MF1 (`5/0/10/0`) | `53.78%` Acc / `37.59%` MF1 / **`3.08%` Rec** (`16/16/588/503`) | `23.36%` |
| **`Statistical_OOF_Ensemble`** | `56.67%` Acc / `56.23%` MF1 / `46.67%` Rec (`7/5/10/8`) | `66.67%` Acc / `60.32%` MF1 / `40.00%` Rec (`2/2/8/3`) | **`69.55%` Acc / `69.06%` MF1 / `61.66%` Rec** (`320/143/461/199`) | `4.99%` |
| **`SARcasmScope_Calibrated_Hybrid`** | `86.67%` Acc / `86.43%` MF1 / `100.00%` Rec (`15/4/11/0`) | `93.33%` Acc / `92.82%` MF1 / `100.00%` Rec (`5/1/9/0`) | **`68.03%` Acc / `67.56%` MF1 / `60.50%` Rec** (`314/154/450/205`) | **`1.36%`** |

---

## 2. Corpus Provenance & Per-Dataset Generalization (`test.jsonl`, $N = 1,123$)

| Dataset (`N` test) | `v2` Symbolic (Acc / MF1 / Rec) | `WordLex_1_2` (MF1) | `Statistical_OOF` (MF1) | `Calibrated Hybrid` (Acc / MF1 / Rec / ROC-AUC / ECE) |
|:---|:---|:---:|:---:|:---|
| `ACL_2020_Twitter_Sarcasm` ($344$) | `52.62% / 41.31% / 9.15%` | `64.92%` | `70.30%` | **`68.60% / 68.17% / 59.76% / 0.7393 / 3.62%`** |
| `Misra_Arora_News_Headlines` ($401$) | `49.63% / 33.60% / 0.50%` | `74.31%` | `75.56%` | **`74.06% / 74.06% / 75.12% / 0.8162 / 5.43%`** |
| `IIITD_MaSaC_CodeMixed` ($194$) | `48.45% / 32.64% / 0.00%` | `54.52%` | `48.18%` | `47.42% / 47.33% / 42.00% / 0.5048 / 15.57%` |
| `L3Cube_MahaEmotions` ($75$) | `50.67% / 33.63% / 0.00%` | `63.90%` | `65.33%` | **`62.67% / 62.66% / 62.16% / 0.6789 / 1.29%`** |
| `IIIT_Hyderabad_Telugu_Sarcasm` ($109$) | `84.40% / 45.77% / 0.00%` | `55.21%` | `45.77%` | `84.40% / 45.77% / 0.00% / 0.7455 / 5.77%` |
| **Total Held-Out (`test.jsonl`)** ($1,123$) | `53.78% / 37.59% / 3.08%` | `67.96%` | `69.06%` | **`68.03% / 67.56% / 60.50% / 0.7535 / 1.36%`** |

---

## 3. Leakage & Near-Duplicate Audit Summary

- **Exact Text & Conversation ID Leakage:** `0` exact overlaps across splits (`train-val`, `train-test`, `val-test`) and `0` shared `conversation_id` entries across splits.
- **Exhaustive Word 3-Shingle Near-Duplicates ($\text{Jaccard} \ge 0.80$):** **13 pairs corpus-wide** (`8` cross-split, `5` train-test, `8` label-conflicting), all confined to `IIITD_MaSaC_CodeMixed`.

---

## 4. Scientific Takeaways
1. Small internal sets (`N=30` and `N=15`) verify feature wiring and Nov-7 reproduction (`66.67% / 33.33%`), never real-world accuracy.
2. On frozen `test.jsonl`, 3-stage calibrated feature fusion lifts Macro-F1 by **`+29.97%`** (`37.59% -> 67.56%`, $\chi^2 = 55.20, p < 0.0001$) and sarcasm recall by **`+57.42%`** (`3.08% -> 60.50%`, $\chi^2 = 290.16, p < 0.0001$) over `v2` symbolic rules, while cutting Conf-ECE to **`1.36%`** and elevating accepted-set selective accuracy to **`80.31%`** ($\tau^*=0.35$).
