# SARcasmScope — Journal Submission Checklist & Reviewer Dossier
**Repository Maintainer / Audit Lead:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** Vellore Institute of Technology, Vellore, India  
**Article Title:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Submission Gate:** **NOT READY FOR SUBMISSION UNTIL CHECKBOXES BELOW (INCLUDING 5-AUTHOR SIGN-OFF) ARE COMPLETED**

---

## 1. Pre-Submission Checklist

- [ ] **Five-Author Reconciliation & Approval (CRITICAL BLOCKER):** Original manuscript lists **5 co-authors** with supervisor/collaborator as corresponding author; Stage 9 technical docs list Payal Mehta as codebase lead. Confirm final author order, CRediT roles, corresponding author, and co-author sign-off across manuscript, `CITATION.cff`, and `README.md`.
- [x] **Nov 7 vs Stage 9 Reconciliation:** All 4 code versions (`v1`, `v2`, `Statistical OOF`, `Calibrated Hybrid`) compared on `N=30`, `N=15`, and `test.jsonl` ($N=1,123$); `RESULTS_STATUS_AND_AUDIT.md` noted absent in workspace.
- [x] **Deployed Engine & Trained Model Integration:** `src/engine/hybridFusion.ts` directly consumes `trainedModelData.ts` (`v2.0.0-oof-calibrated`, `405.36 KB`) + `statisticalModel.ts`; obsolete `"subword_hash"` fallback removed.
- [x] **Non-Circular 3-Stage Calibration & 4-Metric Distinction:** Level-1 experts fitted on `train.jsonl`, Level-2 MAP Ridge on 5-fold group OOF, Level-3 $T^*=0.845$ and $\tau^*=0.35$ fitted strictly on `val.jsonl`. Explicitly separates $P_{\text{fused}}$, Margin Confidence, $\tau^*$, and Selective Acc/Coverage.
- [x] **Held-Out Test Metrics & Independent Recomputation:** `eval/verify_independent_recomputation.ts` confirms `1123/1123` CSV parity (`max|Δp|=0`) and independently recomputes `68.03%` Acc, `67.56%` MF1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE, `0.7535` ROC-AUC (`80.31%` selective acc @ `45.68%` coverage; val-to-test reliability curve saved in `validation_vs_test_calibration_and_selective.csv`).
- [x] **Honest Dataset Validity, Leakage & Verified Bibliography Audit:** Corrected 5 dataset licenses & complete DOIs/pages (`docs/TECHNOLOGY_AND_REFERENCES.md`); verifies `HasContext=true` only on `ACL 2020`, `MaSaC` isolated utterances (`context=""`, no A/V prosody), `13` 3-shingle near-dups (`8` cross-split, `5` train-test), and `MahaEmotions` (`Sarcasm` vs `Neutral` proxy + CoTR train provenance).
- [x] **Honest Hardware & Baseline Qualification:** Client CPU latency measured (`1.632 ms` mean, `2.856 ms` P95); GPU PyTorch baselines marked `NOT_EXECUTED_LOCALLY`; MuRIL parity claim removed.
- [ ] **Optional Colab GPU Run:** Execute `eval/train_transformer.py --eval_test` on Google Colab T4 GPU (`docs/COLAB.md`) if empirical neural baselines are required by the target venue.

---

## 2. Canonical Dossier Manifest (`eval_results/stage9/` & `docs/`)

- `docs/STAGE9_FINAL_MANUSCRIPT.md` — Complete journal-formatted manuscript
- `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md` — Version reconciliation & 3-stage stacking proof
- `docs/DATASET_PROVENANCE_AND_LICENSES.md` — Corpus citations, licenses & near-duplicate report
- `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/COLAB.md`
- `eval_results/stage9/master_summary.json` + all 15 Stage 9 CSVs
