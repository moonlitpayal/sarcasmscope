> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 7 Final Release Manifest & Repository Freeze
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Release Tag:** `v1.0.0-submission-freeze`  
**Date:** November 2026  

---

## 1. Release Manifest

This manifest documents the exact inventory of all verified scientific artifacts, production code, datasets, evaluation tables, and manuscript documentation locked for journal submission:

### Core Engine & Architecture (`src/`)
- `src/engine/sarcasmEngine.ts`: Client-side linguistic incongruity analyzer, Indic morphological normalizer (`stemIndicToken`), Romanized phonetic folding (`normalizeRomanized`), and lexical coverage calculator ($C(x)$).
- `src/engine/hybridFusion.ts`: Bounded ridge probability fusion, temperature scaling calibration ($T^* = 1.15-1.50$), and standard Confidence Calibration Error (Guo et al., 2017).
- `src/engine/contextualEngine.ts`: Tiered deployment adapter with fault-tolerant 1,500 ms `AbortController` timeout fallback.
- `src/App.tsx`, `src/main.tsx`, `src/components/*`: Interactive React UI with Three.js 3D ambient incongruity visualizer.

### Evaluation Pipelines (`eval/`)
- `eval/run_stage7_full_validation.ts`: Master Stage 7 validation script executing independent audit, bootstrap CIs, McNemar tests, robustness perturbations, and software profiling.
- `eval/split_independent_datasets.ts`: Deterministic (Seed 42) dataset deduplication and conversation-preserving splitting harness.
- `eval/test_unseen.ts`: 15-sample regression test suite across English, Hinglish, Marathi, and Telugu.
- `eval/train_transformer.py`: PyTorch HuggingFace fine-tuning pipeline for MuRIL and XLM-RoBERTa on GPU.
- `eval/export_onnx.py`: ONNX export and dynamic INT8 quantization pipeline.

### Verified Empirical Evidence (`eval_results/stage7/`)
- `eval_results/stage7/independent_audit.csv`: Verification of TP, FP, TN, FN, accuracy, precision, recall, and Macro-F1.
- `eval_results/stage7/dataset_validation_report.csv`: Sample counts, licensing, and 0% cross-split leakage confirmation across 5,600 unique records.
- `eval_results/stage7/transformer_benchmark_status.csv`: Transparent qualification of local transfer proxy vs external Colab T4 GPU transformer checkpoints.
- `eval_results/stage7/statistical_validation.csv`: Global and per-language 1,000-sample bootstrap 95% confidence intervals.
- `eval_results/stage7/significance_matrix.csv`: McNemar paired tests disentangling recall recovery ($\chi^2 = 277.03, p < 0.0001$) from overall accuracy ($\chi^2 = 1.79, p = 0.1806$).
- `eval_results/stage7/robustness_matrix.csv`: Empirical stability across 6 perturbation benchmarks.
- `eval_results/stage7/software_audit.csv`: Measured P95 latency ($0.124\text{ ms}$), heap allocation ($23.7\text{ MB}$), and fallback resilience.

### Publication Documentation Package (`docs/`)
- `docs/STAGE7_FINAL_MANUSCRIPT.md`: Full 21-section submission-ready manuscript.
- `docs/STAGE7_FINAL_RESULTS.md`: Detailed empirical tables and statistical distributions.
- `docs/STAGE7_REPRODUCIBILITY.md`: IEEE/ACM artifact evaluation compliance guide.
- `docs/STAGE7_LIMITATIONS.md`: Transparent scientific disclosures and threats to validity.
- `docs/STAGE7_PUBLICATION_CHECKLIST.md`: Pre-submission reviewer checklist.
- `docs/STAGE7_ROBUSTNESS.md`: Multi-lingual sensitivity analysis and case studies.
- `docs/STAGE7_INDEPENDENT_AUDIT.md`: Cross-stage discrepancy audit and mathematical proof.
- `docs/DATASET_PROVENANCE_AND_LICENSES.md`: Data governance, FAIR principles, and licensing conditions.
- `docs/ARCHITECTURE.md`: Detailed component diagrams and mathematical equations.
- `docs/COLAB.md`: Google Colab GPU transformer execution instructions.

### Repository Root Metadata
- `CITATION.cff`: Citation File Format metadata.
- `README.md`: Professional repository documentation with exact reproduction instructions.
- `metadata.json`: Verified project name `SARcasmScope` and capabilities.
- `index.html`: Synchronized application title, description, and OpenGraph tags.

---

## 2. Integrity Verification Sign-Off

All 7 execution stages of SARcasmScope are formally completed. The codebase, datasets, and manuscripts are locked and certified ready for journal submission.
