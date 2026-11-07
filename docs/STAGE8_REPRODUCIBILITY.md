# SARcasmScope — Stage 8 Scientific Reproducibility & Replication Protocol

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Standard:** IEEE / ACM Artifact Review and Badging (Artifacts Available & Reusable)  
**Evaluation Date:** November 7, 2026  

---

## 1. Reproducibility Overview

To support complete independent scientific verification, every dataset partition, evaluation script, and metric reported in the SARcasmScope manuscript can be reproduced using deterministic commands with zero manual intervention.

### Core Guarantees:
1. **Deterministic Random Seed:** All data partitioning, Monte Carlo bootstrapping ($B = 1,000$), and synthetic perturbation suites use deterministic seed `42`.
2. **Frozen Held-Out Evaluation:** The test split (`eval_results/independent/test.jsonl`, $N = 1,123$) is immutable and sealed with verified SHA-256 signatures.
3. **Hardware Independence:** Client-side inference and statistical tests run on standard x86_64 / ARM CPU hardware with zero external cloud dependencies.
4. **Reproducible GPU Scripts:** Complete Google Colab pipelines are provided for reproducing external 237M-parameter MuRIL transformer fine-tuning.

---

## 2. Environment Specifications

### 2.1 Software Requirements
- **Node.js:** v18.0.0 or higher (v20.x recommended)
- **Package Manager:** npm v9.0.0+
- **TypeScript:** v5.0+
- **Operating System:** Linux (Ubuntu 20.04/22.04), macOS (12+), or Windows (WSL2)

### 2.2 Hardware Requirements
- **Client Inference / Evaluation:**
  - CPU: 2 cores or higher
  - RAM: 2 GB minimum (Node.js heap footprint $< 50\text{ MB}$)
  - Disk: $< 500\text{ MB}$ for repository and dependencies
- **External GPU Transformer Fine-Tuning (Optional):**
  - GPU: NVIDIA T4 / V100 / A100 (Google Colab free tier is sufficient)
  - Python: 3.10+
  - PyTorch: 2.1+

---

## 3. Step-by-Step Replication Protocol

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/moonlitpayal/sarcasmscope.git
cd sarcasmscope
npm install
```

### Step 2: Verify Partition Integrity & Leakage Check
```bash
# Audits all 5 datasets and verifies 0% cross-split leakage
npx tsx eval/partition_independent_datasets.ts
```
Expected output:
```text
Deduplicated samples: 5,600
Train: 3,373 | Val: 1,104 | Test: 1,123
Exact text overlap (Train vs Test): 0 (0.00%)
Dialogue ID overlap (MaSaC): 0 (0.00%)
Near-duplicate pairs (Jaccard > 0.85): 0 (0.00%)
```

### Step 3: Run Stage 8 Master Validation & Statistical Verification
```bash
# Executes all Stage 8 validations, bootstrap CIs, McNemar tests, and generates all 9 CSV deliverables
npx tsx eval/run_stage8_master_validation.ts
```
This produces all deliverables directly in `eval_results/stage8/`:
- `recomputed_metrics.csv`
- `reconciliation_report.csv`
- `dataset_integrity.csv`
- `transformer_comparison.csv`
- `statistical_validation.csv`
- `calibration_analysis.csv`
- `risk_coverage.csv`
- `robustness_results.csv`
- `error_analysis.csv`

### Step 4: Run Robustness Perturbation Suite
```bash
# Evaluates 6 linguistic perturbation stress tests across languages
npx tsx eval/run_stage5_robustness.ts
```

### Step 5: Run Architectural Regression Suite
```bash
# Runs 15 unseen regression unit tests verifying core algorithms
npx tsx eval/test_unseen.ts
```

### Step 6: Launch Interactive Web Application
```bash
# Starts local Vite development server
npm run dev
# Open http://localhost:3000 to interact with the Three.js 3D latent space visualizer
```

---

## 4. Colab Transformer GPU Replication

To independently reproduce the fine-tuned MuRIL transformer results on Google Colab:
1. Open a new Google Colab notebook and select **Runtime > Change runtime type > T4 GPU**.
2. Run the following cells:
```python
!git clone https://github.com/moonlitpayal/sarcasmscope.git
%cd sarcasmscope
!pip install -q torch transformers datasets accelerate scikit-learn

!python eval/train_transformer.py \
    --model_name google/muril-base-cased \
    --train_file eval_results/independent/train.jsonl \
    --val_file eval_results/independent/val.jsonl \
    --test_file eval_results/independent/test.jsonl \
    --batch_size 16 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --output_dir results/muril_gpu
```
3. Evaluate the output summary:
```python
import json
with open("results/muril_gpu/eval_summary.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

---

## 5. Artifact Verification Checksum Signatures

The integrity of critical evaluation artifacts can be verified using standard SHA-256 checksums:

```bash
sha256sum eval_results/independent/test.jsonl
sha256sum eval_results/stage8/recomputed_metrics.csv
sha256sum eval_results/stage8/statistical_validation.csv
```

All data partitions, evaluation scripts, and source modules are permanently archived under MIT Open Source licensing.
