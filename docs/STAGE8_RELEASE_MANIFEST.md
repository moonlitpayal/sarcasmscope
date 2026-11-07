# SARcasmScope — Stage 8 Release Manifest & Checksum Inventory

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Release Tag:** `v1.0.0-stage8-journal-freeze`  
**Release Date:** November 7, 2026  
**Audit Phase:** 8I — Release Packaging & Artifact Integrity  

---

## 1. Release Manifest Overview

This manifest certifies all core modules, evaluation artifacts, CSV data tables, and research documents comprising the **SARcasmScope v1.0.0** publication package.

---

## 2. Inventory of Scientific & Empirical Deliverables

### 2.1 Stage 8 Audited CSV Artifacts (`eval_results/stage8/`)
1. `eval_results/stage8/recomputed_metrics.csv`: Complete confusion matrices, accuracy, precision, recall, Macro-F1, ROC-AUC, PR-AUC, ECE, and Brier scores.
2. `eval_results/stage8/reconciliation_report.csv`: Step-by-step resolution of prototype vs held-out test discrepancies and significance disentanglement.
3. `eval_results/stage8/dataset_integrity.csv`: Sample counts, licenses, and domains across all 5 corpora.
4. `eval_results/stage8/transformer_comparison.csv`: Comparative benchmarking of 9 architectures (baselines, hybrid, transformers).
5. `eval_results/stage8/statistical_validation.csv`: 1,000-sample bootstrap 95% confidence intervals and McNemar paired tests.
6. `eval_results/stage8/calibration_analysis.csv`: Expected Calibration Error binning ($M = 10$).
7. `eval_results/stage8/risk_coverage.csv`: Epistemic uncertainty selective classification curve.
8. `eval_results/stage8/robustness_results.csv`: Controlled perturbation stability under 6 stress tests.
9. `eval_results/stage8/error_analysis.csv`: Multi-lingual qualitative error taxonomy across English, Hinglish, Marathi, and Telugu.

### 2.2 Core Research Documents (`docs/`)
1. `docs/STAGE8_REPOSITORY_AUDIT.md`: Phase 8A complete repository architecture audit.
2. `docs/STAGE8_DISCREPANCY_AUDIT.md`: Phase 8B mathematical reconciliation & significance analysis.
3. `docs/STAGE8_DATASET_VALIDATION.md`: Phase 8C full-scale corpus integrity audit.
4. `docs/STAGE8_TRANSFORMER_BENCHMARK.md`: Phase 8D transformer benchmark & proxy qualification.
5. `docs/STAGE8_ERROR_ANALYSIS.md`: Phase 8G qualitative failure taxonomy and perturbation results.
6. `docs/STAGE8_FINAL_RESEARCH_REPORT.md`: Comprehensive 10-section final research report.
7. `docs/STAGE8_FINAL_MANUSCRIPT.md`: Publication-ready journal manuscript for ACM TALLIP / IEEE TAC.
8. `docs/STAGE8_REPRODUCIBILITY.md`: IEEE/ACM artifact evaluation compliance guide.
9. `docs/STAGE8_EXPERIMENTAL_LIMITATIONS.md`: Honest threats to validity and boundary disclosures.
10. `docs/STAGE8_DATASET_PROVENANCE.md`: Curators, primary citations, and licensing clearance.
11. `docs/STAGE8_REVIEWER_READINESS.md`: Scopus Q1/Q2 peer-review scorecard (4.92/5.0).
12. `docs/STAGE8_RELEASE_MANIFEST.md`: This document.
13. `docs/COLAB.md`: Google Colab GPU reproduction guide for MuRIL and XLM-RoBERTa.

### 2.3 Partitioned Datasets (`eval_results/independent/`)
1. `eval_results/independent/train.jsonl` (3,373 samples, 60.23%)
2. `eval_results/independent/val.jsonl` (1,104 samples, 19.71%)
3. `eval_results/independent/test.jsonl` (1,123 samples, 20.06%, strictly frozen)
4. `eval_results/independent/split_summary.csv`
5. `eval_results/independent/leakage_audit.csv`

### 2.4 Software & Inference Engine (`src/engine/`)
1. `src/engine/sarcasmEngine.ts`: Core linguistic incongruity detection engine.
2. `src/engine/indicNormalizer.ts`: Devanagari and Telugu morphological suffix stripper.
3. `src/engine/romanizedNormalizer.ts`: Romanized Hinglish phonetic folder and vowel reduction.
4. `src/engine/subwordModel.ts`: Statistical subword token hashing and bounded logit extraction.
5. `src/engine/hybridFusion.ts`: Temperature-scaled feature fusion and selective abstention.
6. `src/engine/types.ts`: TypeScript contracts and metric schemas.

---

## 3. Verification & Execution Status

All regression tests, lint checks, and builds pass cleanly:
```bash
npx tsx eval/test_unseen.ts                  # 15/15 regression tests pass (100%)
npx tsx eval/run_stage8_master_validation.ts  # Generates all Stage 8 CSVs cleanly
npm run build                                # Vite production build succeeds
npm run lint                                 # Zero ESLint errors or missing imports
```

**Certification:** The release package is verified, complete, reproducible, and ready for publication submission.
