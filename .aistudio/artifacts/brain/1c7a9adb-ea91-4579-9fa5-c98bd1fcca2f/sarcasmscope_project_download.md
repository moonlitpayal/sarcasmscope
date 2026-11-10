# SARcasmScope — Verified Project Structure & Artifacts

All source code, trained artifacts, evaluation scripts, documentation, and CSV outputs are synchronized across the repository:

### Core Architecture & Engine (`src/`)
- `src/App.tsx`: Main React 19 shell, navigation tabs, and live $T^*=0.845 / \tau^*=0.35$ calibration controls.
- `src/components/StudioView.tsx`: Interactive utterance + dialogue context analyzer, calibrated probability meter, verdict card, and leave-one-token-out (LOTO) occlusion breakdown.
- `src/components/RobustnessView.tsx`: Controlled 8-perturbation stress suite (`perturbHybridAll`).
- `src/components/BatchView.tsx`: Multi-line batch evaluation and CSV export.
- `src/components/ProtocolView.tsx`: Live calibration temperature ($T$) and selective abstention ($\tau$) sliders + lexicon reference.
- `src/components/QuantumOrb.tsx`: Interactive Three.js 3D ambient orbital canvas.
- `src/engine/sarcasmEngine.ts`: Stage 2 symbolic linguistic analyzer (`stemIndicToken`, `normalizeRomanized`, `analyzeCodeMixing`, `extractIncongruity`, `predict`, `prevPredict`).
- `src/engine/textFeatures.ts`: Bit-identical feature extraction shared between training and runtime (`extractWordFeatures`, `extractCharFeatures`, `boundedLogit`, `stableSigmoid`, Mulberry32 PRNG).
- `src/engine/statisticalModel.ts`: Runtime Level-1 sparse experts (`WordLex_1_2` & `CharSub_3_5`).
- `src/engine/trainedModelData.ts`: Serialized `v2.0.0-oof-calibrated` weights (`405.36 KB`, `16,006` effective params, $T^*=0.845, \tau^*=0.35$).
- `src/engine/hybridFusion.ts`: 3-Stage OOF MAP Ridge calibrated feature fusion engine, LOTO attribution, and ECE calculator.
- `src/engine/contextualEngine.ts`: Mode A offline client inference + Mode B `/api/predict-contextual` fallback adapter.

### Reproducible Evaluation & Training (`eval/` & `eval_results/`)
- `eval/train_and_calibrate.ts`: Non-circular 3-stage training & validation calibration pipeline (`train.jsonl` OOF + `val.jsonl`).
- `eval/verify_comparison.ts`: 4-way comparison across `N=30`, `N=15`, and frozen `test.jsonl` (`N=1,123`).
- `eval/run_stage9_master_execution.ts`: Master suite emitting `eval_results/stage9/master_summary.json` and all 15 Stage 9 CSVs.
- `eval/{train_transformer,evaluate_transformers,export_onnx}.py`: Google Colab T4 GPU scripts (syntax-checked & `--dry_run` verified locally; GPU metrics `NOT_EXECUTED_LOCALLY`).
- `docs/`: Stage 9 manuscript, architecture, model card, dataset licenses/provenance, and investigation reports.
