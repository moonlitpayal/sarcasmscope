# SARcasmScope — Paper-to-Artifact Traceability Map

All empirical claims, equations, and dataset disclosures map directly to executable source files and reproducible outputs:

| Paper Section | Canonical Workspace Source | Notes |
|:---|:---|:---|
| Abstract & Headline Tables | `eval_results/stage9/master_summary.json`, `eval_results/stage9/model_comparison.csv` | Frozen `test.jsonl` ($N=1,123$) + diagnostic suites ($N=30, 15$). |
| Nov 7 Reconciliation | `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, `eval_results/independent/version_comparison_report.json` | 4-way reproduction via `eval/verify_comparison.ts`; notes `RESULTS_STATUS_AND_AUDIT.md` absent. |
| 3-Stage Architecture & Equations | `docs/ARCHITECTURE.md`, `src/engine/hybridFusion.ts`, `src/engine/sarcasmEngine.ts`, `src/engine/statisticalModel.ts`, `src/engine/textFeatures.ts` | $16,006$ params, $T^*=0.845, \tau^*=0.35$, OOF MAP Ridge stacking. |
| Datasets, Licenses & Leakage | `docs/DATASET_PROVENANCE_AND_LICENSES.md`, `eval_results/stage9/dataset_integrity.csv` | 5,600 records; `0` exact / conv overlaps; `13` near-dup pairs (`8` cross-split, `5` train-test). |
| Per-Language & Error Analysis | `docs/STAGE9_MULTILINGUAL_ANALYSIS.md`, `eval_results/stage9/multilingual_improvements.csv` | Includes Telugu `0.00%` recall @ 0.50 cutoff vs `0.7455` ROC-AUC analysis. |
| Ablations, McNemar & CIs | `docs/STAGE9_STATISTICAL_VALIDATION.md`, `eval_results/stage9/{ablation_results,statistical_analysis}.csv` | 11-configuration ablations, $B=1,000$ bootstrap CIs, 6 McNemar comparisons. |
| Calibration & Risk-Coverage | `eval_results/stage9/{calibration_results,risk_coverage}.csv`, `eval_results/independent/training_calibration_summary.json` | 5 monotonic bins (`1.36%` Conf-ECE), `80.31%` selective acc @ $\tau^*=0.35$. |
| Robustness & Explainability | `docs/STAGE9_ROBUSTNESS_REPORT.md`, `eval_results/stage9/{robustness_results,explainability_results}.csv` | 9 perturbations ($N=100$, seed 42) + 5 LOTO attribution case studies. |
| GPU Transformer Protocol | `docs/COLAB.md`, `eval/train_transformer.py`, `eval/evaluate_transformers.py`, `eval/export_onnx.py` | Syntax & `--dry_run` verified locally; GPU benchmarks marked `NOT_EXECUTED_LOCALLY`. |
| Deployment & UI | `docs/STAGE9_DEPLOYMENT_REPORT.md`, `eval_results/stage9/software_audit.csv`, `src/components/*` | `StudioView`, `BatchView`, `RobustnessView`, `ProtocolView`, `QuantumOrb`. |
