# SARcasmScope: Google Colab GPU Training Guide
## Fine-Tuning Contextual Transformers (MuRIL & XLM-RoBERTa) on T4 GPU

> **Execution Status Disclosure:** Local workspace sandboxes lack NVIDIA CUDA and `torch`. All Python scripts (`eval/train_transformer.py`, `eval/evaluate_transformers.py`, `eval/export_onnx.py`) have been **syntax-compiled and verified via `--dry_run`**, while actual GPU fine-tuning metrics are marked **`NOT_EXECUTED_LOCALLY`** until run using the commands below on Google Colab T4 GPU or equivalent CUDA hardware.

---

### 1. Colab Runtime Setup
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** -> **Change runtime type** -> **Hardware accelerator: T4 GPU**.
3. Verify GPU availability:
```python
!nvidia-smi
```

---

### 2. Environment & Dependency Installation
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers datasets accelerate scikit-learn onnx onnxruntime
```

---

### 3. Repository & Data Ingestion
```python
!git clone https://github.com/moonlitpayal/sarcasmscope.git /content/sarcasmscope
%cd /content/sarcasmscope
!ls -lh eval_results/independent/
```
Required frozen split files:
- `train.jsonl` (3,373 samples — training)
- `val.jsonl` (1,104 samples — validation & early stopping)
- `test.jsonl` (1,123 samples — **strictly held out; evaluate once only via `--eval_test` or `evaluate_transformers.py`**)

---

### 4. Train Model A: MuRIL (`google/muril-base-cased`)
```python
!python3 eval/train_transformer.py \
  --model_name "google/muril-base-cased" \
  --train_file "eval_results/independent/train.jsonl" \
  --val_file "eval_results/independent/val.jsonl" \
  --test_file "eval_results/independent/test.jsonl" \
  --output_dir "runs/transformer_muril" \
  --max_length 128 \
  --batch_size 16 \
  --lr 2e-5 \
  --epochs 4 \
  --seed 42 \
  --early_stopping_patience 2 \
  --eval_test
```

---

### 5. Train Model B: XLM-RoBERTa (`xlm-roberta-base`)
```python
!python3 eval/train_transformer.py \
  --model_name "xlm-roberta-base" \
  --train_file "eval_results/independent/train.jsonl" \
  --val_file "eval_results/independent/val.jsonl" \
  --test_file "eval_results/independent/test.jsonl" \
  --output_dir "runs/transformer_xlmr" \
  --max_length 128 \
  --batch_size 16 \
  --lr 1.5e-5 \
  --epochs 4 \
  --seed 42 \
  --early_stopping_patience 2 \
  --eval_test
```

---

### 6. Standalone Evaluation & Latency Profiling
```python
!python3 eval/evaluate_transformers.py \
  --checkpoint_dir "runs/transformer_muril/best_checkpoint" \
  --eval_file "eval_results/independent/test.jsonl" \
  --output_file "eval_results/independent/test_evaluation_muril.json"

!python3 eval/evaluate_transformers.py \
  --checkpoint_dir "runs/transformer_xlmr/best_checkpoint" \
  --eval_file "eval_results/independent/test.jsonl" \
  --output_file "eval_results/independent/test_evaluation_xlmr.json"
```

---

### 7. Export to ONNX & INT8 Dynamic Quantization
```python
!python3 eval/export_onnx.py \
  --checkpoint_dir "runs/transformer_muril/best_checkpoint" \
  --output_onnx "runs/transformer_muril/model_fp32.onnx" \
  --output_quantized "runs/transformer_muril/model_int8.onnx"
```

---

### 8. Local Verification Command (CPU Sandbox)
```bash
python3 -m py_compile eval/train_transformer.py eval/evaluate_transformers.py eval/export_onnx.py
python3 eval/train_transformer.py --dry_run
```
