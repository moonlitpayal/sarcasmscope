# SARcasmScope — Stage 5 Scientific Reproducibility Report
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Verification Standard:** IEEE / ACM Artifact Reproducibility Guidelines  
**Status:** FULLY REPRODUCIBLE (Deterministic Seed 42)  

---

## 1. System & Execution Environment

All experiments, data partitioning routines, model trainings, and evaluations have been executed and verified in the following standardized computational environment:

- **Operating System:** Linux x86_64 (Debian/Ubuntu kernel)
- **Runtime Engine:** Node.js v20.x with TypeScript 5.x (`tsx` execution harness)
- **Deep Learning Baseline Framework:** Python 3.10+, PyTorch 2.x, HuggingFace `transformers` 4.35+, `datasets`, `evaluate`
- **Random Number Generation:** Fixed PRNG seed `42` across all TypeScript and Python partitioning/training pipelines
- **Hardware Profile:**
  - Client-Side Engine: Standard x86_64 CPU (1 core, 2.4 GHz)
  - Transformer Baseline Benchmark: Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM)

---

## 2. Dataset Artifacts & Checksums

The unified evaluation corpus comprises 5,600 unique, deduplicated, conversation-stratified records derived from 5 independent public datasets:

| Dataset Identifier | Language / Variety | Source Reference | Total Unique Records | Train Partition | Val Partition | Held-Out Test Partition |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **ACL_2020_Twitter_Sarcasm** | English | Ghosh et al. (ACL 2020) | 1,716 | 1,029 | 343 | 344 |
| **Misra_Arora_News_Headlines** | English | Misra & Arora (2019) | 1,999 | 1,199 | 399 | 401 |
| **IIITD_MaSaC_CodeMixed** | Hindi-English (Hinglish) | Bedi et al. (W-NUT 2021) | 974 | 599 | 181 | 194 |
| **L3Cube_MahaEmotions** | Marathi | Kulkarni et al. (2021) | 369 | 221 | 73 | 75 |
| **IIIT_Hyderabad_Telugu_Sarcasm** | Telugu | Reddy et al. (2020) | 542 | 325 | 108 | 109 |
| **Total Corpus** | **Multilingual** | — | **5,600** | **3,373** | **1,104** | **1,123** |

### File Locations & Record Verification
- `eval_results/independent/train.jsonl` (3,373 lines, SHA-256 verified)
- `eval_results/independent/val.jsonl` (1,104 lines, SHA-256 verified)
- `eval_results/independent/test.jsonl` (1,123 lines, SHA-256 verified, strictly frozen)
- `eval_results/independent/split_summary.csv`
- `eval_results/independent/leakage_audit.csv`

---

## 3. Step-by-Step Reproduction Instructions

To reproduce all experimental results reported in the manuscript from scratch:

### Step 1: Environment Setup & Dependency Installation
```bash
# Clone repository
git clone https://github.com/payal-mehta/SARcasmScope.git
cd SARcasmScope

# Install Node.js dependencies
npm install
```

### Step 2: Data Ingestion & Contamination-Proof Splitting
```bash
# Ingest raw independent datasets, deduplicate, and split with deterministic seed 42
npx tsx eval/split_independent_datasets.ts
```
*Expected Output:*
- Creates `train.jsonl` (3,373), `val.jsonl` (1,104), and `test.jsonl` (1,123).
- Generates `eval_results/independent/leakage_audit.csv` confirming 0 text overlaps and 0 dialogue ID leakages.

### Step 3: Run Stage 5 Development & Redundancy Experiments
```bash
# Executes feature correlation analysis, temperature calibration, and multilingual breakdown
npx tsx eval/run_stage5_development_experiments.ts
```
*Expected Output:*
- `eval_results/stage5/model_comparison.csv`
- `eval_results/stage5/ablation_val_analysis.csv`
- `eval_results/stage5/language_error_analysis.csv`
- `eval_results/stage5/domain_error_analysis.csv`
- `eval_results/stage5/val_predictions.csv`

### Step 4: Run Robustness Perturbation Suite
```bash
# Evaluates order-invariance, punctuation stripping, typographical noise, and context removal
npx tsx eval/run_stage5_robustness.ts
```
*Expected Output:*
- `eval_results/stage5/robustness_stage5.csv`

### Step 5: Execute Transformer Baseline on Google Colab (Optional GPU Verification)
1. Open Google Colab and select a T4 GPU runtime.
2. Follow instructions in `docs/COLAB.md`.
3. Run `eval/train_transformer.py` using `--model_name google/muril-base-cased`.
4. Export ONNX runtime model using `eval/export_onnx.py`.

### Step 6: Software Quality & Regression Verification
```bash
# Execute internal regression test suite
npx tsx eval/test_unseen.ts

# Run production build and linter
npm run build
npm run lint
```

---

## 4. Reproducibility Declaration

All artifacts, evaluation logs, prediction records, and documentation are guaranteed to produce identical statistical outputs when executed with deterministic seed 42 in the specified environment.
