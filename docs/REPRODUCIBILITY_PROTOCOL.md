# SARcasmScope: Reproducibility Protocol & Experimental Blueprint
**Project:** SARcasmScope — A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Standard Compliance:** ACM / IEEE Reproducibility Guidelines, NeurIPS / ACL Experiment Standards

---

### 1. Deterministic Experimental Foundations

All experimental partitions, seeds, and algorithms are deterministic:
- **Global Pseudo-Random Seed:** `42` across all data loaders, splits, and training scripts.
- **Data Splitting PRNG:** Mulberry32 32-bit deterministic state generator:
  $$t = a + \text{0x6d2b79f5}; \quad a \leftarrow t$$
- **Split Ratios:** 60% Training ($N = 3,373$), 20% Validation ($N = 1,104$), 20% Held-Out Test ($N = 1,123$).
- **Stratification:** Stratified by class label and dataset domain. Dialogue turns from the same TV episode in MaSaC are strictly kept in the same partition (`conversation_id` preservation).
- **Exact Deduplication:** 5,600 unique samples derived from 5,712 raw downloaded records with zero cross-split leakage (`eval_results/independent/leakage_audit.csv`).

---

### 2. Software & Hardware Environment

#### Local Runtime (Client / Edge Execution)
- **Node.js:** v20+ with TypeScript 5.7+ (`tsx` execution engine)
- **Framework:** React 19 SPA, Tailwind CSS v4, Three.js (r174)
- **Host OS:** Linux x86_64
- **Client Latency:** Sub-millisecond ($< 0.5\text{ ms}$) per prediction, heap footprint $< 2\text{ MB}$.

#### Cloud Runtime (Deep Contextual Transformer Training)
- **Platform:** Google Colab / Linux Compute Instance
- **Accelerator:** NVIDIA Tesla T4 GPU (16 GB VRAM)
- **PyTorch:** 2.0+ with CUDA 11.8
- **Transformers:** HuggingFace Transformers 4.35+
- **Precision:** FP16 mixed precision training, INT8 ONNX dynamic quantization

---

### 3. Step-by-Step Reproduction Commands

#### A. Data Partitioning & Leakage Audit
```bash
# Ingest independent datasets, deduplicate, dialogue-group, and partition with Seed 42:
npx tsx eval/split_independent_datasets.ts
```
*Outputs generated in `eval_results/independent/`: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_summary.csv`, `leakage_audit.csv`.*

#### B. Training-Only Lexicon Induction
```bash
# Mine high-precision cues exclusively from train.jsonl (support >= 3, precision >= 70%):
npx tsx eval/mine_training_lexicons.ts
```
*Outputs generated: `eval_results/independent/mined_training_lexicons.json`.*

#### C. Stage 3B Representation Investigation
```bash
# Compare Word N-grams, Char N-grams, Subwords, and Linguistic Engine:
npx tsx eval/investigate_embeddings.ts
```
*Outputs generated: `eval_results/independent/stage3b_representations_comparison.csv`.*

#### D. Stage 3C & 3D Hybrid Feature Fusion Training
```bash
# Fit regularized fusion weights and calibrate on validation split:
npx tsx eval/train_fusion.ts
```
*Outputs generated: `eval_results/independent/stage3_system_comparison.csv`, `eval_results/independent/stage3_evaluation_report.json`.*

#### E. Stage 4 Final Held-Out Test Evaluation & Ablation Suite
```bash
# Unseal held-out test split (test.jsonl, N=1,123) for one-time final evaluation:
npx tsx eval/run_stage4_full_evaluation.ts
```
*Outputs generated in `eval_results/stage4/`: all 12 CSV validation matrices.*

#### F. Contextual Transformer Baseline (Google Colab GPU)
Follow the step-by-step instructions in `docs/COLAB.md`:
```bash
# Train MuRIL
python3 eval/train_transformer.py --model_name "google/muril-base-cased" --seed 42

# Train XLM-RoBERTa
python3 eval/train_transformer.py --model_name "xlm-roberta-base" --seed 42

# Export & Quantize to INT8 ONNX
python3 eval/export_onnx.py --checkpoint_dir "runs/transformer_muril/best_checkpoint"
```

---

### 4. Verification Checkpoints

1. **Zero Data Leakage:** Ensure `eval_results/independent/leakage_audit.csv` shows 0 overlapping samples between `train.jsonl`, `val.jsonl`, and `test.jsonl`.
2. **Deterministic Outputs:** All predictions match exact floating-point values down to 4 decimal places given seed 42.
3. **Pristine Test Split:** The file size and checksum of `test.jsonl` must match the original Stage 1 output ($906,151$ bytes, $1,123$ lines).
