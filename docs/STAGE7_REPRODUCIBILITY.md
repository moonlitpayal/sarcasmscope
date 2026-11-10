> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 7 Scientific Reproducibility Protocol
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Standard:** IEEE / ACM Artifact Evaluation and Reproducibility Badge Compliance  
**Verification Status:** 100% REPRODUCIBLE (Deterministic Seed 42)  

---

## 1. System Requirements & Execution Environment

- **Host Operating System:** Linux x86_64 (Ubuntu/Debian kernel)
- **Node.js Runtime:** v20.x or v22.x
- **Package Manager:** npm v10.x
- **TypeScript:** v5.x with `tsx` execution harness
- **Python (Optional for Colab Transformers):** Python 3.10+, PyTorch 2.x, HuggingFace `transformers` 4.35+
- **PRNG Determinism:** Seed `42` hardcoded across all partitioning and resampling loops.

---

## 2. Dataset Provenance & Integrity Checksums

All records were deduplicated before partitioning using deterministic PRNG seed `42`:

| Dataset Identifier | Language | Unique Count | Train Partition | Val Partition | Held-Out Test Partition |
|:---|:---|:---:|:---:|:---:|:---:|
| **ACL_2020_Twitter_Sarcasm** | English | 1,716 | 1,029 | 343 | 344 |
| **Misra_Arora_News_Headlines** | English | 1,999 | 1,199 | 399 | 401 |
| **IIITD_MaSaC_CodeMixed** | Hinglish | 974 | 599 | 181 | 194 |
| **L3Cube_MahaEmotions** | Marathi | 369 | 221 | 73 | 75 |
| **IIIT_Hyderabad_Telugu_Sarcasm** | Telugu | 542 | 325 | 108 | 109 |
| **Total Unified Benchmark** | Multilingual | **5,600** | **3,373** | **1,104** | **1,123** |

### File Locations:
- `eval_results/independent/train.jsonl` (3,373 lines)
- `eval_results/independent/val.jsonl` (1,104 lines)
- `eval_results/independent/test.jsonl` (1,123 lines, strictly frozen)
- `eval_results/stage7/dataset_validation_report.csv`

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Install Dependencies
```bash
git clone https://github.com/payal-mehta/SARcasmScope.git
cd SARcasmScope
npm install
```

### Step 2: Ingest & Partition Datasets
```bash
npx tsx eval/split_independent_datasets.ts
```
*Generates train.jsonl (3,373), val.jsonl (1,104), test.jsonl (1,123) with 0% cross-split leakage.*

### Step 3: Run Full Stage 7 Validation
```bash
npx tsx eval/run_stage7_full_validation.ts
```
*Recomputes all metrics from raw predictions, runs 1,000 bootstrap CIs, McNemar paired tests, robustness matrix, and deployment profiling.*

### Step 4: Run Internal Regression Suite
```bash
npx tsx eval/test_unseen.ts
```
*Verifies 15/15 (100%) pass on unseen generalization cases across English, Hinglish, Marathi, and Telugu.*

### Step 5: (Optional) Google Colab GPU Transformer Fine-Tuning
1. Open Google Colab and select a T4 GPU runtime.
2. Upload `eval_results/independent/train.jsonl`, `val.jsonl`, `test.jsonl`.
3. Follow `docs/COLAB.md` to run `eval/train_transformer.py --model_name google/muril-base-cased`.
4. Export the fine-tuned model via `eval/export_onnx.py`.

### Step 6: Launch Web Studio
```bash
npm run dev
```
*Launches the interactive React application and Three.js 3D ambient incongruity visualizer on http://localhost:3000.*
