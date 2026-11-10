# SARcasmScope — Stage 9 Software Architecture & Deployment Profiling Report

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9I — Software Quality & CPU Latency Profiling (`eval_results/stage9/software_audit.csv`)

---

## 1. Verification Scorecard (1,000 Warmed Iterations on x86_64 Node CPU)

| Evaluation Dimension | Measured Value | Threshold Standard | Status | Production Impact |
|:---|:---:|:---:|:---:|:---|
| **TypeScript Check & Build** | `npm run lint && npm run build` | Zero TS/Vite errors | **PASS** | Strict type safety across engine and UI |
| **Regression Suite (`eval/test_unseen.ts`)** | `v2` 15/15; `Hybrid` 14/15 full-cov | Core regression validated | **PASS** | Verifies symbolic & fusion pipelines |
| **Mean Inference Latency** | **`1.6319 ms`** | $< 5.0\text{ ms}$ | **PASS** | Single-digit ms CPU inference |
| **Median Latency (P50)** | **`1.4230 ms`** | $< 2.0\text{ ms}$ | **PASS** | Responsive within browser frame budget |
| **P90 / P95 Latency** | **`2.2050 ms` / `2.8557 ms`** | $< 5.0 / 10.0\text{ ms}$ | **PASS** | Predictable tail latency for batching |
| **P99 Latency** | **`4.7743 ms`** | $< 20.0\text{ ms}$ | **PASS** | Bounded worst-case execution |
| **TS Model Bundle Size** | **`405.36 KB`** | $< 1024\text{ KB}$ | **PASS** | Ships directly inside Vite bundle (`trainedModelData.ts`) |
| **Memory Profile** | **`70.53 MB` Heap / `215.24 MB` RSS** | $< 100 / 250\text{ MB}$ | **PASS** | Lightweight runtime memory |
| **Offline Autonomy** | **100% Client-Side (Mode A)** | Zero API dependency | **PASS** | Privacy-preserving browser inference |

Component P95 latencies (`model_comparison.csv`): `v1` `0.112 ms`, `WordLex` `0.200 ms`, `CharSub` `0.378 ms`, `Statistical_OOF` `0.881 ms`, `v2` `1.509 ms`, `Calibrated Hybrid` `2.856 ms`.

---

## 2. Actual UI Application Structure (`src/App.tsx` & `src/components/`)

- `src/components/QuantumOrb.tsx`: Three.js ambient WebGL particle orbital canvas in the header.
- `src/components/StudioView.tsx`: Interactive statement & dialogue context workbench showing Verdict badge, Calibrated probability bar, Level-1 vs Fused signal breakdown, linguistic triggers, and LOTO token occlusion inspector.
- `src/components/RobustnessView.tsx`: Controlled 8-perturbation stress suite (`perturbHybridAll`).
- `src/components/BatchView.tsx`: Multi-line batch evaluation and CSV downloader.
- `src/components/ProtocolView.tsx`: Interactive temperature ($T$) and selective abstention ($\tau$) sliders alongside lexicon and protocol specifications.
