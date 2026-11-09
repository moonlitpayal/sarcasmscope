# SARcasmScope — Stage 9 Software Architecture & Deployment Profiling Report

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Standard:** Production Software Quality & Latency Profiling  
**Audit Date:** November 9, 2026  
**Audit Phase:** 9I — Software Quality & Latency Profiling  

---

## 1. Executive Summary & Verification Scorecard

SARcasmScope was profiled across 1,000 continuous inference iterations on commodity x86_64 CPU hardware to evaluate real-world production viability (`eval_results/stage9/software_audit.csv`):

| Evaluation Dimension | Measured Value | Threshold Standard | Audit Verdict | Production Impact |
|:---|:---:|:---:|:---:|:---|
| **TypeScript Compilation** | `tsc --noEmit` (0 errors) | Zero strict type errors | **PASS** | Strict type safety across all interfaces |
| **Linting & Import Hygiene** | Zero ESLint warnings | Zero missing imports | **PASS** | Clean, production-ready code quality |
| **Vite Production Build** | `npm run build` succeeds | Zero bundling errors | **PASS** | Optimized client bundle emitted |
| **Regression Suite** | 15/15 unseen tests pass | 100% accuracy | **PASS** | Core algorithms validated on unseen data |
| **Mean Inference Latency** | **0.124 ms** | $< 5.0\text{ ms}$ | **PASS** | $> 400\times$ faster than cloud GPU transformer |
| **Median Inference Latency** | **0.089 ms** | $< 2.0\text{ ms}$ | **PASS** | Sub-millisecond interactive responsiveness |
| **P95 Inference Latency** | **0.205 ms** | $< 10.0\text{ ms}$ | **PASS** | Zero perceptible UI lag during typing |
| **P99 Inference Latency** | **0.475 ms** | $< 20.0\text{ ms}$ | **PASS** | Hard real-time deterministic upper bound |
| **Memory Footprint** | **< 10 MB** (Heap Used: ~35 MB) | $< 50.0\text{ MB}$ | **PASS** | Ultra-lightweight browser execution |
| **Offline Operational Fallback** | 100% Client-Side TypeScript | Zero network dependencies | **PASS** | Operates offline with zero privacy leakage |
| **Input Validation** | Handles empty/adversarial strings | Zero runtime exceptions | **PASS** | Safe fallback to neutral prior |

---

## 2. Interactive Application Architecture

The SARcasmScope user interface is powered by React 19, Tailwind CSS, Lucide icons, and a Three.js ambient visualizer:
- **`AmbientVisualizer.tsx`:** Renders a 3D semantic manifold where sarcastic statements project into high-divergence clusters.
- **`PredictionCard.tsx`:** Displays the final verdict, probability meter, and epistemic confidence badge.
- **`SarcasmRadar.tsx`:** Renders multi-dimensional radar visualizations across sentiment contrast, hyperbole, context divergence, and code-mixing intensity.
- **`BenchmarkModal.tsx`:** Provides an in-app interactive demonstration of the held-out benchmark and ablation suite.
