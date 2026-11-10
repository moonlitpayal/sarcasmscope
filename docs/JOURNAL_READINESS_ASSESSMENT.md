# SARcasmScope — Journal Readiness & Peer-Review Dossier
**Repository Maintainer / Audit Lead:** Payal Mehta (`payal.12620246@vit.edu`)  
**Article Title:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Status:** Reproducible Client-Side Verification Complete; **Not Ready for Journal Upload Until 5-Author Reconciliation / Corresponding Author Agreement & Dataset/GPU Scope Are Approved by All Co-Authors**

---

## 1. Candidate Publication Venues (Scopus Q1 / Q2)

1. **ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)** — *Primary Target* (Indic languages, code-mixing, low-resource morphology, calibration).
2. **IEEE Transactions on Affective Computing (TAC)** — *High-Reach Alternative* (irony/sarcasm incongruity, selective risk-coverage calibration).
3. **Language Resources and Evaluation (LREV)** — Cross-corpus provenance, leakage auditing, and reproducibility.

---

## 2. Technical Readiness Verification Matrix

| Dimension | Audit Evidence in Repository | Status |
|:---|:---|:---|
| **Version & Nov 7 Reconciliation** | 4-way reproduction (`v1`, `v2`, `StatOOF`, `Hybrid`) across `N=30`, `N=15`, `test N=1,123` (`eval/verify_comparison.ts`); explicitly discloses `RESULTS_STATUS_AND_AUDIT.md` absent. | **COMPLETE** |
| **Non-Circular 3-Stage Stacking** | Level-1 `WordLex`+`CharSub` fitted on `train` ($3,373$); Level-2 MAP Ridge meta-learner on 5-fold group OOF; Level-3 $T^*=0.845, \tau^*=0.35$ on `val` ($1,104$). | **COMPLETE** |
| **Single-Pass Frozen Test Evaluation & Verification** | `68.03%` Acc, `67.56%` MF1, `60.50%` Rec, `1.36%` Conf-ECE, `0.5826` NLL, `0.7535` ROC-AUC on immutable `test.jsonl` ($N=1,123$); `eval/verify_independent_recomputation.ts` proves `1123/1123` CSV parity (`max|Δp|=0`). | **COMPLETE** |
| **Calibration & Selective Utility** | Val-fitted $T^*=0.845, \tau^*=0.35$ (`81.06%` val acc @ `49.73%` cov) transferred to sealed test: `80.31%` acc @ `45.68%` cov, Conf-ECE cut `4.99% -> 1.36%` (`validation_vs_test_calibration_and_selective.csv`). | **COMPLETE** |
| **Statistical Significance & CIs** | $B=1,000$ bootstrap CIs + 6 paired McNemar tests ($\chi^2 = 55.20$ Acc / $\chi^2 = 290.16$ Rec vs `v2`, $p < 0.0001$). | **COMPLETE** |
| **11-Config Ablation & Stress Tests** | Runtime ablations (`w/o Context -4.23% MF1`, `w/o WordLex -3.79%`) + 9 perturbations ($N=100$, seed 42). | **COMPLETE** |
| **Corpus Provenance, Labels & Bibliography** | `0` exact / `conversation_id` overlaps; `13` near-dups (`8` cross-split, `5` train-test in `MaSaC`); verifies `context=""` on `MaSaC`, `MahaEmotions` (`Sarcasm` vs `Neutral` + CoTR train provenance), and full DOIs/pages (`TECHNOLOGY_AND_REFERENCES.md`). | **COMPLETE** |
| **GPU Transformer Baselines** | `eval/train_transformer.py`, `eval/evaluate_transformers.py`, `eval/export_onnx.py` pass syntax + `--dry_run`; actual T4 GPU execution pending in Colab (`NOT_EXECUTED_LOCALLY`). | **DOCUMENTED** |
| **Build, Lint & Profiling** | Single-digit ms CPU latency (`1.632 ms` mean, `2.856 ms` P95), `405.36 KB` TS artifact, `70.53 MB` heap. | **COMPLETE** |

---

## 3. Recommended Pre-Submission Action (Colab GPU Only)
Execute the Colab commands in `docs/COLAB.md` once on a T4 GPU to replace `NOT_EXECUTED_LOCALLY` transformer rows with empirical MuRIL (`google/muril-base-cased`) and XLM-RoBERTa (`xlm-roberta-base`) test metrics before journal upload.
