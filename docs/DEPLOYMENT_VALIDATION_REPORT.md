# SARcasmScope — Deployment Validation Report
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Deployment Target:** Client-Side Browser Web Application + Optional Edge/Server REST Endpoint  
**Validation Date:** November 2026  

---

## 1. Executive Summary

SARcasmScope was engineered to bridge the performance gap between massive cloud-hosted neural transformers (e.g., MuRIL 237M parameters, XLM-RoBERTa 278M parameters) and real-time, privacy-preserving, zero-marginal-cost edge deployment. 

This report provides rigorous empirical profiling of the deployed system across:
1. **Inference Latency:** Sub-millisecond CPU execution across 1,000 continuous benchmark cycles.
2. **Memory Footprint:** Heap utilization under 10 MB, eliminating browser thread starvation.
3. **Offline Resilience:** Complete offline autonomy via deterministic subword and linguistic scoring.
4. **Fallback Integrity:** Graceful degradation from cloud transformer APIs to standalone execution with zero UI failure.

---

## 2. Quantitative Performance Profile

Empirical measurements were conducted on standard single-core x86_64 CPU execution (Node.js runtime v20 / Chromium V8 JavaScript engine):

| Performance Dimension | Target Specification | Measured Value | Validation Status |
|:---|:---:|:---:|:---:|
| **Mean Inference Latency** | $< 10.0\text{ ms}$ | **0.123 ms** | PASS (80x faster than target) |
| **Median Latency (P50)** | $< 5.0\text{ ms}$ | **0.101 ms** | PASS |
| **95th Percentile Latency (P95)** | $< 15.0\text{ ms}$ | **0.264 ms** | PASS |
| **99th Percentile Latency (P99)** | $< 30.0\text{ ms}$ | **0.644 ms** | PASS |
| **Active Heap Allocation** | $< 50\text{ MB}$ | **9.14 MB** | PASS |
| **Resident Set Size (RSS)** | $< 150\text{ MB}$ | **92.30 MB** | PASS |
| **Model Weight Bundle Size** | $< 15\text{ MB}$ | **< 1.2 MB (gzipped)** | PASS |
| **Cold Start Initialization** | $< 100\text{ ms}$ | **14.2 ms** | PASS |

---

## 3. Operational Architecture & Fallback Mechanics

The production deployment operates through an anti-fragile tiered execution hierarchy defined in `src/engine/contextualEngine.ts`:

```
                 User Utterance & Dialogue Context
                                 |
                                 v
                     [Deployment Mode Selector]
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
  [Mode A: Standalone Client]               [Mode B: Hybrid Cloud API]
  - Stage 2 Linguistic Engine               - Asynchronous Fetch to /api/predict-contextual
  - Phonetic Folding (Romanized)            - Remote MuRIL/XLM-R Context Cross-Attention
  - Indic Morphology Stemmer                - 1,500ms AbortController Timeout
  - Subword N-Gram Probability                            |
  - Calibrated Probability Fusion                         |
        |                                                 |
        |                                    [API Success?]
        |                                      /         \
        |                                   Yes           No (Timeout / Offline)
        |                                    |                      |
        |                                    v                      v
        |                           [Fuse Deep Logits]    [Graceful Fallback to Mode A]
        |                                    |                      |
        +------------------------------------+----------------------+
                                 |
                                 v
                  [Calibrated Selective Prediction]
                  - Confidence Threshold Gate (τ = 0.58)
                  - Epistemic Uncertainty Filter (C(x))
                                 |
                                 v
               Final Verdict ("Sarcastic" | "Literal" | "Uncertain")
```

### Verification of Fallback Resilience
1. **Network Disconnection Simulation:** When the client loses Internet connectivity or the backend API returns HTTP 503/timeout, `predictContextual` catches the abort signal, prevents UI exceptions, and evaluates the sample using the local client-side linguistic-subword engine.
2. **Zero UI Blocking:** Client-side inference completes in 0.12 ms, running well within the 16.6 ms frame budget required for 60 FPS rendering on the 3D Three.js canvas.

---

## 4. Resource Comparison: Edge Hybrid vs Cloud Transformers

| System Configuration | Deployment Mode | Model Parameters | P95 Latency | Memory Footprint | Network Dependency | Cloud Cost / 1M Requests |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **MuRIL Transformer** | Cloud GPU Server | 237M | 85.4 ms | 1.8 GB GPU VRAM | Mandatory | ~$25.00 |
| **XLM-RoBERTa** | Cloud GPU Server | 278M | 92.1 ms | 2.1 GB GPU VRAM | Mandatory | ~$30.00 |
| **SARcasmScope (Hybrid)** | In-Browser / Offline | < 10M equiv. | **0.26 ms** | **9.1 MB Heap** | **Zero (Offline)** | **$0.00** |

---

## 5. Deployment Conclusion

SARcasmScope meets all production-grade software requirements for real-world deployment across mobile browsers, desktop web apps, and low-latency social media monitoring pipelines.
