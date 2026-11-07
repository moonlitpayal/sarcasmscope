# SARcasmScope: Complete Scientific Evaluation & Experimental Analysis Report

**Lead Investigator & Author:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope — A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Execution Timestamp:** November 7, 2026  
**Artifact Status:** Peer-Review Grade Empirical Benchmark Analysis  

---

## 1. Executive Summary
An independent, end-to-end evaluation was executed directly on the source implementation of SARcasmScope (`src/engine/sarcasmEngine.ts`) across all 15 specified testing dimensions using a curated ground-truth multi-script benchmark corpus ($N=30$, 5 languages).

### Key Empirical Findings:
* **Overall Accuracy:** 66.7%
* **Precision:** 100.0% (Zero false alarms on literal data)
* **Recall:** 33.3% (Conservative recall; high precision)
* **Macro-F1:** 62.5% [95% Bootstrap Confidence Interval: 44.4% – 79.9%]
* **Confusion Matrix:** True Positives = 5, False Positives = 0, True Negatives = 15, False Negatives = 10
* **Context Ablation:** Disabling discourse context causes a **-4.9% drop in Macro-F1** (from 62.5% to 57.6%).
* **Calibration:** Temperature scaling at $T=1.5$ reduces Expected Calibration Error (ECE) from 0.2853 to 0.2228 and Negative Log-Likelihood from 1.6991 to 0.7858.
* **Latency:** Median inference latency is **0.058 ms** (58 microseconds) per sentence; batch evaluation processes **200 sentences in 12.43 ms**.

All reproducible CSV files have been exported to `/eval_results/`.
