# SARcasmScope: Reproducibility Protocol & Verification Commands
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)

---

## 1. Deterministic Split & Training Specifications
- **PRNG & Seed:** Seed `42` across data partitioning, OOF fold generation, bootstrap resampling ($B=1,000$), and perturbation stress testing (`createSeededPRNG` Mulberry32 in `src/engine/textFeatures.ts`).
- **Frozen Partitions (`eval_results/independent/`):**
  - `train.jsonl`: $3,373$ records (`60.23%`) — Stage 1 base expert training & 5-fold group-stratified OOF stacking.
  - `val.jsonl`: $1,104$ records (`19.71%`) — Stage 3 temperature optimization ($T^* = 0.845$) and abstention selection ($\tau^* = 0.35$).
  - `test.jsonl`: $1,123$ records (`20.06%`) — Frozen held-out evaluation split (`0` exact overlaps, `0` conversation overlaps; `13` near-duplicate 3-shingle pairs $\ge 0.80$ across the 5,600 corpus / `5` train-test inside `IIITD_MaSaC_CodeMixed`).

---

## 2. Step-by-Step Verification Commands

```bash
# 1. Regenerate dataset provenance CSV
npx tsx eval/generate_provenance_csvs.ts

# 2. Verify 4-way code comparison across N=30, N=15, and test.jsonl (N=1,123)
npx tsx eval/verify_comparison.ts

# 3. Verify closed-loop regression suite (v2 15/15; Hybrid 14/15 + 6/15 abstentions)
npx tsx eval/test_unseen.ts

# 4. Run full Stage 9 deterministic verification suite (all 15 CSVs + master_summary.json)
npx tsx eval/run_stage9_master_execution.ts

# 5. Synchronize Stage 6 / Stage 7 / Stage 8 CSV deliverables from Stage 9
npx tsx eval/run_stage6_integrity_and_benchmarking.ts
npx tsx eval/run_stage7_full_validation.ts
npx tsx eval/run_stage8_master_validation.ts

# 6. Syntax & dry-run verify GPU Colab transformer pipeline
python3 -m py_compile eval/train_transformer.py eval/evaluate_transformers.py eval/export_onnx.py
python3 eval/train_transformer.py --dry_run
python3 eval/evaluate_transformers.py --dry_run
python3 eval/export_onnx.py --dry_run

# 7. TypeScript typecheck & production Vite bundle
npm run lint && npm run build
```

*Note on Model Training (`eval/train_and_calibrate.ts`):* Fits all Level-1 (`WordLex`, `CharSub`), Level-2 (MAP Ridge OOF), and Level-3 ($T^*, \tau^*$) artifacts using only `train.jsonl` and `val.jsonl` without accessing `test.jsonl`.
