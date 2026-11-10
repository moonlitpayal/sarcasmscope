# SARcasmScope — Stage 9 Complete Repository & Discrepancy Audit

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Audit Phase:** 9A — Code Version Reconciliation & Scientific Verification (`eval_results/stage9/initial_audit.csv`)

---

## 1. Executive Summary

This Stage 9 audit reconciles every code version, historical claim, and file in SARcasmScope against live deterministic execution over `BENCHMARK_CORPUS` ($N=30$), `UNSEEN_TEST_CASES` ($N=15$), and frozen `eval_results/independent/test.jsonl` ($N=1,123$). Notably, `RESULTS_STATUS_AND_AUDIT.md` is not present in this workspace; all previously unverified registers (`56.63%`, `56.57%`) are replaced by reproducible computations from immutable splits.

---

## 2. Historical Discrepancy Reconciliation

### 2.1 Nov 7 `prevPredict` vs `v2` vs `Statistical_OOF` vs `Calibrated Hybrid` (`eval/verify_comparison.ts`)
- **`BENCHMARK_CORPUS` ($N=30$ closed-loop diagnostic):**
  - `v1` (`prevPredict`, Nov 7): **`66.67%` Acc, `100.00%` Prec, `33.33%` Rec, `62.50%` MF1** (`TP=5, FP=0, TN=15, FN=10`). Exact reproduction of the Nov 7 headline numbers.
  - `v2` (`predict`): **`96.67%` Acc, `93.75%` Prec, `100.00%` Rec, `96.66%` MF1** (`15/1/14/0`). Rule additions resolved all 10 missed synthetic examples.
  - `Statistical_OOF_Ensemble`: `56.67%` Acc, `56.23%` MF1 (`7/5/10/8`).
  - `Calibrated Hybrid` (`predictHybrid`): `86.67%` Acc, `86.43%` MF1, `100.00%` Rec (`15/4/11/0`).
- **Frozen `test.jsonl` ($N=1,123$ independent held-out):**
  - `v1`: `53.52%` Acc, **`0.39%` Rec**, `35.21%` MF1 (`2/5/599/517`).
  - `v2`: `53.78%` Acc, **`3.08%` Rec**, `37.59%` MF1 (`16/16/588/503`). Confirms $N=30$ gains reflected closed-loop pattern coverage, not open-world generalization.
  - `Statistical_OOF_Ensemble`: **`69.55%` Acc**, **`69.06%` MF1**, `61.66%` Rec, `4.99%` Conf-ECE (`320/143/461/199`).
  - **`SARcasmScope_Calibrated_Hybrid`**: **`68.03%` Acc**, **`67.56%` MF1**, `67.09%` Prec, **`60.50%` Rec**, **`1.36%` Conf-ECE**, `0.5826` NLL, `0.7535` ROC-AUC (`314/154/450/205`).

### 2.2 Non-Circular 3-Stage Stacking & Validation Calibration
- **Level-1 Base Experts (`train.jsonl`, $N=3,373$ only):** `WordLex_1_2` ($V=8,000$) + `CharSub_3_5` ($V=8,000$) + deterministic `LinguisticEngine`.
- **Level-2 Meta-Learner (`train.jsonl` 5-fold group-stratified OOF):** MAP Ridge logistic weights (`w0=0.3483, w1_word=0.54817, w2_char=0.54648, w3_linguistic=0.26486, w4_incongruity=0.19418, w5_coverage=0.20441`).
- **Level-3 Post-Hoc Calibration (`val.jsonl`, $N=1,104$ only):** $T^* = 0.845$, $\tau^* = 0.35$. Reduces test Conf-ECE $3.67\times$ (`4.99% -> 1.36%`) and boosts selective accuracy to **`80.31%`** (`45.68%` coverage).

### 2.3 GPU Transformers & Near-Duplicate Disclosures
- **Transformers (`google/muril-base-cased`, `xlm-roberta-base`):** Workspace has no PyTorch/CUDA GPU (`ModuleNotFoundError: No module named 'torch'`). All historical `57.32%` / `pytorch_model.bin` references are removed; Python scripts pass `py_compile` + `--dry_run` and GPU baselines are marked **`NOT_EXECUTED_LOCALLY`**.
- **Exhaustive Near-Duplicate Audit:** Exact text overlap = `0`, `conversation_id` overlap = `0`. Word 3-shingle Jaccard $\ge 0.80$ scan finds **13 near-duplicate pairs corpus-wide** (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

---

## 3. Verified Repository Source File Inventory

- `src/engine/sarcasmEngine.ts`: Stage 2 linguistic rules, `stemIndicToken`, `normalizeRomanized`, `analyzeCodeMixing`, `prevPredict`, `predict`.
- `src/engine/textFeatures.ts`: Bit-identical feature extractors (`extractWordFeatures`, `extractCharFeatures`), `boundedLogit`, `stableSigmoid`.
- `src/engine/trainedModelData.ts`: Serialized `v2.0.0-oof-calibrated` weights (`405.36 KB`, $16,006$ params).
- `src/engine/statisticalModel.ts`: Runtime `WordLex_1_2` and `CharSub_3_5` scorer.
- `src/engine/hybridFusion.ts`: 3-stage calibrated fusion engine, LOTO occlusion, perturbation suite, `calculateECE`.
- `src/engine/contextualEngine.ts`: Mode A offline + Mode B `/api/predict-contextual` fallback adapter.
- `src/components/{StudioView,BatchView,RobustnessView,ProtocolView,QuantumOrb}.tsx`: Production React 19 UI.
