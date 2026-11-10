# SARcasmScope Model Card

## Model Details
- **Model Name:** SARcasmScope (`v2.0.0-oof-calibrated`)
- **Architecture:** 3-Stage Calibrated Stacked Feature-Fusion Engine (`16,006` effective parameters; `405.36 KB` bundled TypeScript artifact `src/engine/trainedModelData.ts`):
  1. **Level-1 Base Experts (Train-only):** `WordLex_1_2` ($V=8,000$ TF-IDF LR w/ Indic stems `st:`, phonetic keys `ph:`, emojis `em:`, context `cw:`), `CharSub_3_5` ($V=8,000$ char 3–5 TF-IDF LR), and deterministic `LinguisticEngine` (`src/engine/sarcasmEngine.ts`).
  2. **Level-2 Meta-Learner (5-Fold Group OOF):** MAP Ridge logistic stacking over bounded expert logits + `incongruity_index` + `lexical_coverage`.
  3. **Level-3 Post-Hoc Calibration (Val-only):** Scalar temperature scaling ($T^* = 0.845$) and margin confidence gate ($\tau^* = 0.35$).
- **Author:** Payal Mehta (`payal.12620246@vit.edu`)
- **License:** MIT (code); source datasets governed by individual corpus licenses (see `docs/DATASET_PROVENANCE_AND_LICENSES.md`).
- **Runtime:** Pure client-side TypeScript 5.7 (React 19 + Vite 6). External GPU scripts provided in `eval/*.py` (`NOT_EXECUTED_LOCALLY`).

## Intended Use
- **Primary Use Cases:**
  - Real-time client-side sarcasm triage across English, Hinglish, Marathi, and Telugu with leave-one-token-out (LOTO) occlusion attributions.
  - High-precision selective escalation (`80.31%` selective accuracy at `45.68%` coverage using $\tau^* = 0.35$) where uncertain boundary examples are deferred to human reviewers.
- **Out-of-Scope Use Cases:**
  - Autonomous punitive moderation without human oversight.
  - Zero-shot deployment on Telugu under a `0.50` probability cutoff without threshold recalibration (`0.7455` ROC-AUC but `0.00%` recall at `0.50` due to `15.6%` positive class prevalence).

## Training & Evaluation Data
- **Corpora:** `ACL_2020_Twitter_Sarcasm` ($1,716$), `Misra_Arora_News_Headlines` ($1,999$), `IIITD_MaSaC_CodeMixed` ($974$), `L3Cube_MahaEmotions` ($369$), `IIIT_Hyderabad_Telugu_Sarcasm` ($542$).
- **Splits (`5,600` records, Seed `42`):** `train.jsonl` ($3,373$), `val.jsonl` ($1,104$), `test.jsonl` ($1,123$).
- **Integrity:** `0` exact cross-split overlaps, `0` `conversation_id` overlaps; exhaustive word 3-shingle Jaccard $\ge 0.80$ scan reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all within `IIITD_MaSaC_CodeMixed`).

## Quantitative Performance (Frozen Held-Out Test Split, $N = 1,123$)
- **Accuracy:** `68.03%` (95% CI `[65.36%, 70.79%]`)
- **Macro-F1:** `67.56%` (95% CI `[64.79%, 70.29%]`)
- **Sarcasm Precision / Recall / Specificity:** `67.09%` / `60.50%` (95% CI `[56.00%, 64.77%]`) / `74.50%`
- **ROC-AUC / PR-AUC:** `0.7535` / `0.7014`
- **Calibration (Guo et al., 2017):** **Conf-ECE `1.36%`** (95% CI `[1.05%, 4.54%]`; Class-ECE `2.80%`; Brier `0.2005`; NLL `0.5826`) — a **3.67× reduction in Conf-ECE** vs the uncalibrated statistical OOF ensemble (`4.99%`).
- **Selective Prediction ($\tau^* = 0.35$):** `80.31%` Selective Accuracy (`78.92%` Macro-F1) across `45.68%` Coverage (`513/1,123` accepted; `19.69%` risk vs `31.97%` full-coverage risk).
- **Latency & Memory (Node.js x86_64 CPU):** `1.632 ms` mean, `1.423 ms` median, `2.856 ms` P95, `4.774 ms` P99; `405.36 KB` model bundle, `70.53 MB` Node heap.
- **Confusion Matrix:** `TP = 314, FP = 154, TN = 450, FN = 205`.
