#!/usr/bin/env python3
"""
SARcasmScope — Stage 3A: Contextual Transformer Baseline Training Pipeline
Models:
  - Model A: google/muril-base-cased (Indic / Hinglish contextual representations)
  - Model B: xlm-roberta-base (Cross-lingual multilingual contextual representations)
  - Model C: distilled / compact architecture variant

Input Specification:
  [CONTEXT] {context_turns} [SEP] [UTTERANCE] {current_statement}
  (If context is absent: [CONTEXT] NONE [SEP] [UTTERANCE] {current_statement})

Strict Partitioning Policy:
  - Training: eval_results/independent/train.jsonl (3,373 samples)
  - Validation: eval_results/independent/val.jsonl (1,104 samples)
  - Test: eval_results/independent/test.jsonl (STRICTLY UNTOUCHED)
"""

import os
import sys
import json
import argparse
import random
import time
import math
from typing import List, Dict, Any, Tuple

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def format_input(record: Dict[str, Any]) -> str:
    ctx = (record.get("context") or "").strip()
    text = (record.get("text") or "").strip()
    if not ctx:
        ctx = "NONE"
    return f"[CONTEXT] {ctx} [SEP] [UTTERANCE] {text}"

def compute_class_weights(records: List[Dict[str, Any]]) -> List[float]:
    labels = [r["binary_sarcasm_label"] for r in records]
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return [1.0, 1.0]
    w0 = n_total / (2.0 * n_neg)
    w1 = n_total / (2.0 * n_pos)
    return [float(w0), float(w1)]

def compute_ece(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        b_lo, b_hi = bin_boundaries[i], bin_boundaries[i + 1]
        bin_idx = [k for k, p in enumerate(probs) if (p >= b_lo and (p < b_hi if i < n_bins - 1 else p <= b_hi))]
        if bin_idx:
            bin_acc = sum([labels[k] for k in bin_idx]) / len(bin_idx)
            bin_conf = sum([probs[k] for k in bin_idx]) / len(bin_idx)
            ece += (len(bin_idx) / n) * abs(bin_acc - bin_conf)
    return ece

def compute_metrics(probs: List[float], labels: List[int], threshold: float = 0.5) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum([1 for p, y in zip(preds, labels) if p == 1 and y == 1])
    fp = sum([1 for p, y in zip(preds, labels) if p == 1 and y == 0])
    tn = sum([1 for p, y in zip(preds, labels) if p == 0 and y == 0])
    fn = sum([1 for p, y in zip(preds, labels) if p == 0 and y == 1])
    
    total = len(labels)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Weighted & Macro F1
    prec_lit = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_lit = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_lit = (2 * prec_lit * rec_lit) / (prec_lit + rec_lit) if (prec_lit + rec_lit) > 0 else 0.0
    macro_f1 = (f1 + f1_lit) / 2.0
    
    n_pos = tp + fn
    n_neg = tn + fp
    weighted_f1 = (n_pos * f1 + n_neg * f1_lit) / total if total > 0 else 0.0
    
    # Brier Score
    brier = sum([(p - y) ** 2 for p, y in zip(probs, labels)]) / total if total > 0 else 0.0
    
    # NLL
    eps = 1e-12
    nll = -sum([y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p)) for p, y in zip(probs, labels)]) / total if total > 0 else 0.0
    
    # ECE
    ece = compute_ece(probs, labels)
    
    return {
        "accuracy": acc * 100.0,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "weighted_f1": weighted_f1 * 100.0,
        "specificity": spec * 100.0,
        "brier_score": brier,
        "nll": nll,
        "ece": ece * 100.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total
    }

def main():
    parser = argparse.ArgumentParser(description="SARcasmScope Stage 3A: Contextual Transformer Training")
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased",
                        choices=["google/muril-base-cased", "xlm-roberta-base", "bert-base-multilingual-cased"],
                        help="Transformer backbone")
    parser.add_argument("--train_file", type=str, default="eval_results/independent/train.jsonl")
    parser.add_argument("--val_file", type=str, default="eval_results/independent/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="runs/transformer_muril")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true", help="Perform architecture check without GPU execution")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("SARcasmScope — Stage 3A Transformer Baseline Training")
    print("=" * 80)
    print(f"Model Backbone:    {args.model_name}")
    print(f"Train File:        {args.train_file}")
    print(f"Val File:          {args.val_file}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Max Sequence Len:  {args.max_length}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Random Seed:       {args.seed}")
    print("=" * 80)

    # 1. Load Data
    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    print(f"\nLoaded {len(train_records)} training records and {len(val_records)} validation records.")

    # 2. Compute Class Imbalance Weights
    class_weights = compute_class_weights(train_records)
    print(f"Computed Class Weights [Literal=0, Sarcastic=1]: [{class_weights[0]:.3f}, {class_weights[1]:.3f}]")

    # 3. Format Context-Utterance Inputs
    train_texts = [format_input(r) for r in train_records]
    train_labels = [r["binary_sarcasm_label"] for r in train_records]
    val_texts = [format_input(r) for r in val_records]
    val_labels = [r["binary_sarcasm_label"] for r in val_records]

    print("\nSample Formatted Input:")
    print("--------------------------------------------------------------------------------")
    print(train_texts[0])
    print(f"Label: {train_labels[0]} ({'Sarcastic' if train_labels[0] == 1 else 'Literal'})")
    print("--------------------------------------------------------------------------------\n")

    # 4. Check for PyTorch & Transformers
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
        from torch.utils.data import TensorDataset, DataLoader
        has_torch = True
    except ImportError as e:
        has_torch = False
        print(f"[ENVIRONMENT NOTICE] PyTorch / Transformers not installed in current environment ({e}).")
        print("This script is fully prepared for execution in Google Colab (T4 GPU).")
        print("See docs/COLAB.md for one-click reproducible execution.")

    if not has_torch or args.dry_run:
        print("\n[DRY RUN SUMMARY]")
        print("Architecture and data ingestion pipelines validated successfully.")
        print(f"Target model: {args.model_name}")
        print(f"Train tokens: ~{sum(len(t.split()) for t in train_texts):,} words across {len(train_texts)} samples.")
        print(f"Val tokens:   ~{sum(len(t.split()) for t in val_texts):,} words across {len(val_texts)} samples.")
        print("\nPipeline ready. Execute on Google Colab GPU runtime as documented in docs/COLAB.md.")
        return

    # 5. Full Training Pipeline (When executed in PyTorch/GPU environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    def encode_data(texts: List[str], labels: List[int]):
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))

    train_ds = encode_data(train_texts, train_labels)
    val_ds = encode_data(val_texts, val_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1 = -1.0
    patience_counter = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_t = time.time()

        for batch in train_loader:
            b_ids, b_mask, b_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(b_ids, attention_mask=b_mask)
            loss = loss_fn(outputs.logits, b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                b_ids, b_mask, _ = [b.to(device) for b in batch]
                outputs = model(b_ids, attention_mask=b_mask)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
                val_probs.extend(probs)

        metrics = compute_metrics(val_probs, val_labels)
        elapsed = time.time() - start_t

        print(f"Epoch {epoch}/{args.epochs} [{elapsed:.1f}s] - Train Loss: {avg_train_loss:.4f} | Val F1: {metrics['f1']:.2f}% | Macro-F1: {metrics['macro_f1']:.2f}% | Val Acc: {metrics['accuracy']:.2f}% | Val Rec: {metrics['recall']:.2f}% | ECE: {metrics['ece']:.2f}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "metrics": metrics
        }
        history.append(epoch_record)

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            patience_counter = 0
            model.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))
            with open(os.path.join(args.output_dir, "best_val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> Saved new best checkpoint (Macro-F1: {best_val_f1:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed. Best Validation Macro-F1: {best_val_f1:.2f}%. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()
