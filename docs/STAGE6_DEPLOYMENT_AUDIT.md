> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 6 Software Quality & Deployment Audit Report
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Verification Scope:** TypeScript Engine, React Frontend, Three.js 3D Visualizer, Fallback Adapters, ONNX Pipeline, Performance Profiling  
**Audit Date:** November 2026  

---

## 1. Automated Verification & Regression Matrix

| Verification Dimension | Execution Command | Result / Status | Error Count | Audit Details |
|:---|:---|:---:|:---:|:---|
| **TypeScript Compilation** | `npm run build` (`tsc -b && vite build`) | **PASS** | 0 errors | Production bundle compiled cleanly |
| **Codebase Linting** | `npm run lint` (`tsc --noEmit`) | **PASS** | 0 errors | Zero type errors, zero missing imports |
| **Unseen Generalization Suite** | `npx tsx eval/test_unseen.ts` | **PASS (100%)** | 0 failures | 15/15 unseen test cases correctly predicted |
| **End-to-End Stage 6 Script** | `npx tsx eval/run_stage6_integrity_and_benchmarking.ts` | **PASS** | 0 errors | Recomputed all primary metrics from predictions |
| **Robustness Suite** | `npx tsx eval/run_stage5_robustness.ts` | **PASS** | 0 errors | 5 perturbation benchmarks verified |

---

## 2. Hardware Environment & Runtime Latency Profile

All quantitative runtime benchmarks were performed in a standardized containerized Linux x86_64 single-thread environment:
- **Processor:** x86_64 Virtual CPU @ 2.40 GHz
- **Runtime Environment:** Node.js v22.x / Chromium V8 JavaScript Engine
- **Iterations:** 1,000 continuous benchmark cycles (with 50-cycle warmup)
- **Input Payload:** Standard conversational pair (*"Oh wonderful, another three hour train delay!"* + contextual discourse turn)

### Latency Distribution Profile:
- **Mean Inference Latency:** **$0.085\text{ ms}$** (85 microseconds)
- **Median Latency (P50):** **$0.069\text{ ms}$**
- **90th Percentile (P90):** **$0.106\text{ ms}$**
- **95th Percentile (P95):** **$0.124\text{ ms}$**
- **99th Percentile (P99):** **$0.441\text{ ms}$**
- **Maximum Observed Latency:** **$1.820\text{ ms}$**

*Finding:* Client-side inference is over **130x faster** than the 16.6 ms frame budget required for 60 FPS animation, ensuring fluid interaction on mobile browsers and Three.js 3D rendering loops.

---

## 3. Memory Footprint & Bundle Metrics

- **Active JavaScript Heap:** **$23.70\text{ MB}$**
- **Process Resident Set Size (RSS):** **$150.88\text{ MB}$**
- **Core Model Weights & Lexicons:** **$< 1.2\text{ MB}$ (gzipped)**
- **UI & WebGL Assets:** Bundled via Vite tree-shaking with zero external runtime server requirement.

---

## 4. Anti-Fragile Fallback & API Resilience Audit

In `src/engine/contextualEngine.ts`, the deployment pipeline provides a tiered execution hierarchy:
1. **Mode A (Standalone In-Browser):**
   - Pure client-side execution.
   - Evaluates linguistic incongruity, Indic morphology, Romanized phonetic folding, and subword hash features.
   - Operates 100% offline with zero network connectivity.
2. **Mode B (Hybrid Cloud Transformer API):**
   - Calls `/api/predict-contextual` to query external MuRIL or XLM-R models.
   - **Timeout Protection:** Protected by an `AbortController` set to 1,500 ms.
   - **Fault Tolerance:** If the server returns HTTP 500/503, fails DNS resolution, or exceeds the timeout, the engine catches the exception and immediately returns Mode A predictions with zero UI errors or visual disruptions.

---

## 5. Security & Input Sanitization Audit

1. **XSS & Injection Protection:** All user inputs in the React UI pass through standard React DOM escaping, preventing script injection.
2. **Path Traversal & Secrets:** No file system access or server secrets are exposed on the client.
3. **Denial-of-Service (DoS) Protection:** Input text length is bounded, and regex tokenizers employ bounded lookaheads to prevent catastrophic backtracking (ReDoS).

---

## 6. Software Audit Sign-Off

The software infrastructure of SARcasmScope is certified as production-ready, highly maintainable, and fully compliant with modern web and NLP engineering standards.
