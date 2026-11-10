# SARcasmScope — Stage 9 Transformer Benchmarking & Execution Protocol

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9D — Client-Side Baselines vs External Transformer Execution Status (`eval_results/stage9/transformer_results.csv`)

---

## 1. Transparent Hardware & Execution Disclosure

- **Local Workspace Environment:** CPU-only Node.js/Python container without CUDA GPU or PyTorch (`import torch` raises `ModuleNotFoundError: No module named 'torch'`).
- **Verified Local Status:** All three Python scripts (`eval/train_transformer.py`, `eval/evaluate_transformers.py`, `eval/export_onnx.py`) pass `python3 -m py_compile` and `--dry_run` checks locally, including `--test_file ... --eval_test` support.
- **Zero Fabricated Transformer Numbers:** Because GPU training has not been executed inside this container, **`MuRIL_FineTuned_GPU` (`google/muril-base-cased`, 237M) and `XLM_RoBERTa_GPU` (`xlm-roberta-base`, 278M) are strictly reported as `NOT_EXECUTED_LOCALLY` (`N/A`)** across all benchmark tables. Only genuinely executed client-side models are numerically reported.

---

## 2. Measured Client-Side Training & Frozen Test Results (`test.jsonl`, $N = 1,123$)

| Model Identifier | Representation / Architecture | Params | Execution Hardware | Latency (P95) | Accuracy | Precision | Recall | Macro-F1 | Conf-ECE | NLL | Audit Qualification |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`WordLex_1_2`** | Word 1-2 + Morph + Phonetic + Ctx LR | 8,000 | Local CPU | 0.200 ms | 68.39% | 67.30% | 61.46% | 67.96% | 2.80% | 0.6021 | **MEASURED** (Level-1 Base Expert) |
| **`CharSub_3_5`** | Character 3-5 TF-IDF LR | 8,000 | Local CPU | 0.378 ms | 65.36% | 63.21% | 59.92% | 65.01% | 2.11% | 0.6029 | **MEASURED** (Level-1 Base Expert) |
| **`Statistical_OOF_Ensemble`** | Logit-Space WordLex + CharSub Mean | 16,000 | Local CPU | 0.881 ms | 69.55% | 69.11% | 61.66% | 69.06% | 4.99% | 0.5876 | **MEASURED** (Statistical Baseline) |
| **`SARcasmScope_Calibrated_Hybrid`** | **3-Stage OOF MAP Ridge Fusion ($T^*=0.845$)** | **16,006** | **Local CPU** | **2.856 ms** | **68.03%** | **67.09%** | **60.50%** | **67.56%** | **1.36%** | **0.5826** | **MEASURED** (Full Hybrid Engine) |
| **`MuRIL_FineTuned_GPU`** | `google/muril-base-cased` | 237M | Colab T4 GPU | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | **`NOT_EXECUTED_LOCALLY`** |
| **`XLM_RoBERTa_GPU`** | `xlm-roberta-base` | 278M | Colab T4 GPU | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | **`NOT_EXECUTED_LOCALLY`** |

---

## 3. Standalone Google Colab Replication Commands

Run on a Google Colab T4 GPU to generate actual fine-tuned MuRIL and XLM-RoBERTa metrics (input format `[CONTEXT] {c} [SEP] [UTTERANCE] {u}`, seed `42`):
```bash
!pip install -q torch transformers datasets accelerate scikit-learn onnx onnxruntime

!python3 eval/train_transformer.py \
  --model_name "google/muril-base-cased" \
  --train_file "eval_results/independent/train.jsonl" \
  --val_file "eval_results/independent/val.jsonl" \
  --test_file "eval_results/independent/test.jsonl" \
  --output_dir "runs/transformer_muril" \
  --batch_size 16 --epochs 4 --lr 2e-5 --seed 42 --eval_test
```
See `docs/COLAB.md` for full XLM-RoBERTa and INT8 ONNX export commands.
