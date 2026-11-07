# SARcasmScope: Google Colab GPU Training Guide
## Fine-Tuning Contextual Transformers (MuRIL & XLM-RoBERTa) on Free T4 GPU

This document provides the complete, self-contained protocol to train and evaluate the deep contextual transformer baselines for **SARcasmScope**:
- **Model A: MuRIL** (`google/muril-base-cased`) — Optimized for Indic scripts, Hinglish, code-mixing, and transliterated morphology.
- **Model B: XLM-RoBERTa** (`xlm-roberta-base`) — Cross-lingual representation backbone across English, Hindi, Marathi, and Telugu.
- **Model C: ONNX Dynamic INT8 Quantization** — Compress models from ~440 MB to ~110 MB for fast CPU and web inference.

---

### 1. Colab Runtime Setup
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** -> **Change runtime type** -> **Hardware accelerator: T4 GPU** (or A100 if Colab Pro).
3. Verify GPU availability in Cell 1:
```python
!nvidia-smi
```

---

### 2. Environment & Dependency Installation
Run the following cell to install PyTorch, HuggingFace Transformers, Accelerate, and ONNX Runtime:
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers datasets accelerate scikit-learn onnx onnxruntime
```

---

### 3. Repository & Data Ingestion
Clone or upload the repository to `/content/sarcasmscope`:
```python
# Option A: If cloning from your Git repository:
# !git clone <REPO_URL> /content/sarcasmscope
# %cd /content/sarcasmscope

# Option B: If uploading ZIP directly:
!unzip -q /content/sarcasmscope.zip -d /content/sarcasmscope
%cd /content/sarcasmscope

# Verify that the partitioned independent splits are present:
!ls -lh eval_results/independent/
```
Expected partition files:
- `train.jsonl` (3,373 samples — used for training)
- `val.jsonl` (1,104 samples — used for validation & early stopping)
- `test.jsonl` (1,123 samples — **HELD-OUT, DO NOT TOUCH**)

---

### 4. Train Model A: MuRIL (`google/muril-base-cased`)
Execute the reproducible training pipeline:
```python
!python3 eval/train_transformer.py \
  --model_name "google/muril-base-cased" \
  --train_file "eval_results/independent/train.jsonl" \
  --val_file "eval_results/independent/val.jsonl" \
  --output_dir "runs/transformer_muril" \
  --max_length 128 \
  --batch_size 16 \
  --lr 2e-5 \
  --epochs 4 \
  --seed 42 \
  --early_stopping_patience 2
```
*Expected training duration: ~4–6 minutes on T4 GPU.*  
Checkpoints and metrics are automatically saved to `runs/transformer_muril/`.

---

### 5. Train Model B: XLM-RoBERTa (`xlm-roberta-base`)
```python
!python3 eval/train_transformer.py \
  --model_name "xlm-roberta-base" \
  --train_file "eval_results/independent/train.jsonl" \
  --val_file "eval_results/independent/val.jsonl" \
  --output_dir "runs/transformer_xlmr" \
  --max_length 128 \
  --batch_size 16 \
  --lr 1.5e-5 \
  --epochs 4 \
  --seed 42 \
  --early_stopping_patience 2
```

---

### 6. Evaluate Checkpoints & Latency Profiling
Evaluate the best checkpoints on the validation partition, computing Accuracy, Precision, Recall, Macro-F1, Specificity, Brier Score, ECE, and P95 latency:
```python
# Evaluate MuRIL
!python3 eval/evaluate_transformers.py \
  --checkpoint_dir "runs/transformer_muril/best_checkpoint" \
  --eval_file "eval_results/independent/val.jsonl" \
  --output_file "eval_results/independent/val_evaluation_muril.json"

# Evaluate XLM-RoBERTa
!python3 eval/evaluate_transformers.py \
  --checkpoint_dir "runs/transformer_xlmr/best_checkpoint" \
  --eval_file "eval_results/independent/val.jsonl" \
  --output_file "eval_results/independent/val_evaluation_xlmr.json"
```

---

### 7. Export to ONNX & INT8 Dynamic Quantization
Export the best model to dynamic-axis ONNX format and quantize to INT8 for deployment:
```python
!python3 eval/export_onnx.py \
  --checkpoint_dir "runs/transformer_muril/best_checkpoint" \
  --output_onnx "runs/transformer_muril/model_fp32.onnx" \
  --output_quantized "runs/transformer_muril/model_int8.onnx"
```

---

### 8. Download Evaluation Artifacts
Zip and download the resulting run logs, evaluation metrics, and quantized model weights:
```python
!zip -r sarcasmscope_stage3_results.zip runs/ eval_results/independent/
from google.colab import files
files.download('sarcasmscope_stage3_results.zip')
```

---

### 9. Protocol Checklist & Integrity Rules
- [x] **Seed 42:** All model initializations and data loaders use deterministic seed 42.
- [x] **Context-Utterance Pair:** Inputs formatted as `[CONTEXT] {c} [SEP] [UTTERANCE] {u}`.
- [x] **Class Weighting:** Automatically adjusts Cross-Entropy loss for imbalanced sets (e.g. Telugu).
- [x] **Held-Out Test Set Untouched:** Only `val.jsonl` is evaluated during model selection. `test.jsonl` remains completely unobserved until Stage 4 publication freeze.
