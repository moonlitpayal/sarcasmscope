# SARcasmScope: Comprehensive Scientific Evaluation & Experimental Analysis

**Paper Title:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Lead Investigator & Author:** Payal Mehta (`payal.12620246@vit.edu`)  
**Evaluation Date:** November 7, 2026  
**Status:** Reproducible Empirical Benchmark Report  

---

## Executive Summary
This document provides an independent, end-to-end experimental evaluation of the SARcasmScope system across 15 rigorous testing dimensions. All reported numbers are derived from executing the actual source code (`src/engine/sarcasmEngine.ts`) against a multi-script, multi-language ground-truth benchmark corpus ($N=30$ items across English, Hindi, Hinglish, Marathi, and Telugu).

### Core Findings:
1. **Classification Performance:** On the multi-language benchmark, SARcasmScope achieves **66.7% overall accuracy**, **100.0% precision** (zero false positives), **33.3% recall**, and **62.5% Macro-F1** (95% bootstrap confidence interval: $[44.4\%, 79.9\%]$).
2. **Context Clashing Significance:** Ablation experiments demonstrate that discourse context clashing contributes a **+4.9% gain in Macro-F1** (Macro-F1 drops from $62.5\%$ to $57.6\%$ when context is disabled).
3. **Calibration Validation:** Temperature scaling ($T=1.5$) successfully reduces Expected Calibration Error (ECE) from $0.2853$ ($T=0.5$) down to $0.2228$, with Negative Log-Likelihood (NLL) dropping from $1.6991$ to $0.7858$.
4. **Inference Latency:** Median inference latency is **0.058 ms (58 microseconds)** per sentence. A full batch of 200 items executes in **12.43 ms**, proving the viability of client-side browser deployment.
5. **Major Scientific Weakness:** The system displays high precision but low recall in regional languages where figurative sarcasm lacks explicit dictionary cues (Hindi Devanagari recall: $0\%$, Telugu recall: $0\%$).

---

## 1. Feature-Wise Evaluation Table

| Module / Feature | Functionality Tested | Execution Status | Empirical Result | Weakness / Limitation Identified |
| :--- | :--- | :--- | :--- | :--- |
| **Input Validation** | Empty, whitespace, 5000+ char strings | Executed | Handled safely | Empty string returns fallback probability (0.10) without throwing errors. |
| **Incongruity Extractor** | Polarity clash between positive & negative cues | Executed | Clash detected ($1.0$) for opposing cues | Fails on sarcasm using pure figurative metaphor without surface positive words. |
| **Context Clashing** | Target statement opposition to preceding context | Executed | $\Delta P = +0.28$ shift when context clashes | Context requires negative cue words to trigger discourse opposition. |
| **Script Identifier** | Devanagari, Latin, Telugu Unicode blocks | Executed | 100% correct script mapping | Romanized Indic text mapped via token heuristic, not subword phonetic parser. |
| **Code-Mix Index (CMI)** | Script and language alternation ratio | Executed | Correctly scales from 0.0% to 50.0% | Script-based CMI is 0% for Romanized Hindi written entirely in Latin characters. |
| **Emoji Classifier** | Irony vs. positive/negative sentiment | Executed | 100% categorization of irony emojis | Mixed emojis (`😂🙄`) produce competing signals. |
| **Temperature Scaler** | Post-hoc logit scaling ($T \in [0.2, 2.5]$) | Executed | ECE drops by $0.0625$ at $T=1.5$ | Temperature is static rather than class-conditionally learned via vector scaling. |
| **Selective Abstention** | Rejection threshold $\tau$ triggering `"Uncertain"` | Executed | Accuracy increases from $66.7\%$ to $70.4\%$ | Coverage drops to $70\%$ when $\tau=0.60$. |
| **Token Occlusion** | Leave-one-out sensitivity ($\Delta = P - P_{\setminus i}$) | Executed | Correctly ranks trigger tokens | Occlusion reflects model sensitivity, not necessarily causal discourse rationale. |
| **Perturbation Suite** | 8 controlled social-media noise transforms | Executed | 100% classification stability on sample | Truncation and emoji removal cause probability drops of $\Delta = -0.24$. |
| **Batch Processor** | 200 item multi-line processing + CSV export | Executed | 12.43 ms total execution time | Input limited to 200 items in browser memory. |

---

## 2. Benchmark Classification Results & Confusion Matrix

### 2.1 Multilingual Benchmark Corpus ($N=30$)
* **Ground-Truth Composition:** 15 Sarcastic ($y=1$), 15 Literal ($y=0$).
* **Languages Covered:** English ($n=8$), Hindi ($n=6$), Hinglish ($n=6$), Marathi ($n=6$), Telugu ($n=4$).

### 2.2 Confusion Matrix
```
                    Predicted Literal (0)   Predicted Sarcastic (1)
Actual Literal (0)           15 (TN)                  0 (FP)
Actual Sarcastic (1)         10 (FN)                  5 (TP)
```
* **True Positives (TP):** 5
* **False Positives (FP):** 0
* **True Negatives (TN):** 15
* **False Negatives (FN):** 10

### 2.3 Aggregate Metrics
* **Accuracy:** $66.7\%$
* **Precision:** $100.0\%$
* **Recall:** $33.3\%$
* **F1 Score:** $50.0\%$
* **Macro-F1:** $62.5\%$
* **95% Bootstrap Confidence Interval:** $[44.4\%, 79.9\%]$

---

## 3. Language-Wise Disaggregated Performance

| Language | Sample Size ($N$) | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Macro-F1 (%) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Marathi** | 6 | **83.3%** | 100.0% | **66.7%** | **80.0%** | **82.9%** | **0.1561** |
| **English** | 8 | **75.0%** | 100.0% | **50.0%** | **66.7%** | **73.3%** | 0.2559 |
| **Hinglish** | 6 | **66.7%** | 100.0% | 33.3% | 50.0% | 62.5% | 0.2975 |
| **Hindi (Devanagari)** | 6 | 50.0% | 0.0% | 0.0% | 0.0% | 33.3% | 0.4020 |
| **Telugu** | 4 | 50.0% | 0.0% | 0.0% | 0.0% | 33.3% | 0.2751 |

**Analysis:**
* Marathi and English achieve high Macro-F1 ($82.9\%$ and $73.3\%$) because their sarcasm lexicons contain dense coverage of idiomatic failure expressions (*"lai bhari"*, *"khup changla"*, *"brilliant"*, *"genius"*).
* Hindi Devanagari and Telugu suffer from zero recall when figurative sarcasm does not match predefined root dictionary keys.

---

## 4. Ablation Study Results

To evaluate the contribution of each algorithmic module, we systematically masked individual feature extractors across the 30-item benchmark:

| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Macro-F1 (%) | $\Delta$ Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full SARcasmScope (All Features)** | **66.7%** | **100.0%** | **33.3%** | **50.0%** | **62.5%** | **Baseline** |
| **Ablation 1: No Context Clashing** | 63.3% | 100.0% | 26.7% | 42.1% | 57.6% | **-4.9%** |
| **Ablation 2: No Polarity Clash** | 63.3% | 100.0% | 26.7% | 42.1% | 57.6% | **-4.9%** |
| **Ablation 3: No Intensifiers** | 66.7% | 100.0% | 33.3% | 50.0% | 62.5% | 0.0% |
| **Ablation 4: No Irony Emoji Features**| 66.7% | 100.0% | 33.3% | 50.0% | 62.5% | 0.0% |

**Scientific Interpretation:**
Discourse context clashing and intra-sentential polarity clashing are the two most critical drivers of performance, each contributing a **4.9% improvement in Macro-F1**.

---

## 5. Baseline Comparison

| Baseline Architecture | Accuracy (%) | Macro-F1 (%) | Scientific Note |
| :--- | :---: | :---: | :--- |
| **Majority Class (Always Sarcastic)** | 50.0% | 33.3% | Predicts $y=1$ unconditionally. |
| **Keyword Negativity Lookup Only** | 60.0% | 55.0% | Predicts $y=1$ if any negative word is present. High false alarm rate. |
| **Irony Emoji Shortcut Only** | 60.0% | 52.4% | Predicts $y=1$ if irony emoji is present. Fails on text-only sarcasm. |
| **SARcasmScope (Calibrated Feature-Fusion)** | **66.7%** | **62.5%** | **Outperforms heuristic baselines by +7.5% to +29.2% Macro-F1.** |

---

## 6. Calibration & Selective Prediction

### 6.1 Temperature Scaling Optimization
* At $T=0.5$ (overconfident): $\text{ECE} = 0.2853$, $\text{Brier} = 0.2978$, $\text{NLL} = 1.6991$.
* At $T=1.0$ (standard): $\text{ECE} = 0.2346$, $\text{Brier} = 0.2760$, $\text{NLL} = 0.9770$.
* At $T=1.5$ (calibrated): $\text{ECE} = 0.2228$, $\text{Brier} = 0.2581$, $\text{NLL} = 0.7858$.

### 6.2 Selective Risk-Coverage Trade-Off
When the model is permitted to abstain ($\tau > 0$), selective accuracy improves:
* **$\tau = 0.0$ (Full Coverage):** Coverage = $100\%$, Accuracy = $66.7\%$, Risk = $33.3\%$.
* **$\tau = 0.2$:** Coverage = $93.3\%$, Accuracy = $67.9\%$, Risk = $32.1\%$.
* **$\tau = 0.4$:** Coverage = $90.0\%$, Accuracy = **$70.4\%$**, Risk = **$29.6\%$**.
* **$\tau = 0.6$:** Coverage = $70.0\%$, Accuracy = $66.7\%$, Risk = $33.3\%$.

---

## 7. Latency and Computational Efficiency
Benchmarked across 1,000 independent client-side executions:
* **Minimum Latency:** 0.037 ms
* **Median Latency:** 0.058 ms
* **Mean Latency:** 0.069 ms
* **95th Percentile ($P_{95}$):** 0.110 ms
* **Maximum Latency:** 1.244 ms
* **Batch Throughput:** 200 sentences in 12.43 ms (approx. 16,000 sentences/second).

---

## 8. Academic Assessment for Scopus-Indexed NLP Journals

### Candid Evaluation:
* **Is the system functional?** Yes. It executes flawlessly with zero crashes, microsecond latency, zero false positives, and robust client-side explainability.
* **Is the current experimental evidence ready for submission to a Q1/Q2 Scopus NLP journal?**
  **Not yet.** While the software architecture and evaluation protocol are sound, a reputable journal (such as *IEEE Access*, *ACM TALIP*, or *Language Resources and Evaluation*) will reject the paper if it only evaluates a 30-sentence benchmark. 
* **What is required before final submission:**
  1. Fine-tune the provided MuRIL and XLM-RoBERTa transformer models on Google Colab using the full MaSaC dataset (approx. 1,500+ dialogues).
  2. Compute McNemar's statistical significance test between SARcasmScope's calibrated predictions and standard BERT baselines.
  3. Expand the regional lexicon (Hindi and Telugu) to resolve the recall bottleneck.
