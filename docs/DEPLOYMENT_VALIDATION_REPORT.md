# SARcasmScope — Deployment Validation Report
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)

---

## 1. Runtime Performance Profile (`eval_results/stage9/software_audit.csv`)

Warmed 1,000-iteration CPU benchmark on Node.js x86_64 running `predictHybrid` (`src/engine/hybridFusion.ts`) with the `405.36 KB` `trainedModelData.ts` bundle:

| Performance Dimension | Target Threshold | Measured Value | Status |
|:---|:---:|:---:|:---:|
| **Mean Inference Latency** | $< 5.0\text{ ms}$ | **`1.6319 ms`** | **PASS** |
| **Median Latency (P50)** | $< 2.0\text{ ms}$ | **`1.4230 ms`** | **PASS** |
| **P90 Latency** | $< 5.0\text{ ms}$ | **`2.2050 ms`** | **PASS** |
| **P95 Latency** | $< 10.0\text{ ms}$ | **`2.8557 ms`** | **PASS** |
| **P99 Latency** | $< 20.0\text{ ms}$ | **`4.7743 ms`** | **PASS** |
| **Serialized TS Model Artifact** | $< 1024\text{ KB}$ | **`405.36 KB`** | **PASS** |
| **Node Heap Used** | $< 100\text{ MB}$ | **`70.53 MB`** | **PASS** |
| **Process RSS** | $< 250\text{ MB}$ | **`215.24 MB`** | **PASS** |
| **Offline Privacy Autonomy** | Zero External Calls Required | **100% Client-Side (Mode A)** | **PASS** |

Component P95 latencies (`model_comparison.csv`): `v1_Prev_Symbolic` `0.112 ms`, `WordLex_1_2` `0.200 ms`, `CharSub_3_5` `0.378 ms`, `Statistical_OOF_Ensemble` `0.881 ms`, `v2_Symbolic_Linguistic` `1.509 ms`, `Calibrated Hybrid` `2.856 ms`.

---

## 2. Mode A / Mode B Execution Flow (`src/engine/contextualEngine.ts`)

- **Mode A (Default — Standalone Client-Side):** Runs Stage 2 symbolic incongruity + `WordLex_1_2` + `CharSub_3_5` + MAP Ridge meta-learner + validation temperature scaling ($T^*=0.845$) + abstention ($\tau^*=0.35$) completely offline.
- **Mode B (Optional Cloud Transformer Extension):** Queries `/api/predict-contextual` with a `1,500 ms` `AbortController` timeout and falls back seamlessly to Mode A upon network failure or timeout.

---

## 3. UI Module Inventory (`src/components/` & `src/App.tsx`)
- `src/components/StudioView.tsx`: Interactive statement/context studio, calibrated probability meter, verdict badge, and leave-one-token-out (LOTO) attribution inspector.
- `src/components/BatchView.tsx`: Multi-line batch evaluation and CSV export.
- `src/components/RobustnessView.tsx`: Controlled perturbation stress tester (`perturbHybridAll`).
- `src/components/ProtocolView.tsx`: Live calibration temperature ($T$) and abstention threshold ($\tau$) controls + lexicon/protocol reference.
- `src/components/QuantumOrb.tsx`: Three.js ambient 3D orbital canvas.
