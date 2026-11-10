# SARcasmScope
### A Calibrated 3-Stage Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text

**Repository Maintainer / Audit Lead:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** Vellore Institute of Technology, Vellore, India  
**Authorship Governance Status:** *Pending Co-Author Sign-Off* — Original manuscript draft lists **five authors** (with project advisor/collaborator as corresponding author); Stage 9 audit documents record the codebase maintainer alone. Final author order, CRediT roles, corresponding author, and co-author approval are **required prior to journal submission**.  
**Research Classification:** Computational Linguistics, Affective Computing & Calibrated Machine Learning

---

## 1. Abstract & Research Motivation

Detecting sarcasm in noisy, conversational, and multilingual online interactions requires reconciling symbolic pragmatic explainability with statistical open-vocabulary coverage. Purely rule-based systems achieve high precision on canonical patterns (`97.35%–99.17%` specificity) yet collapse catastrophically on open-vocabulary sarcasm (`0.39%` recall on `v1`, `3.08%` recall on `v2` over frozen `test.jsonl`). Conversely, massive multilingual neural transformers (e.g., MuRIL, XLM-RoBERTa) demand GPU servers not present in privacy-first edge browser environments.

**SARcasmScope** implements a genuine, non-circular **3-Stage Calibrated Stacked Feature-Fusion Framework** executed 100% client-side in TypeScript:
1. **Level-1 Base Experts (fitted strictly on `train.jsonl`, $N = 3,373$):**
   - `WordLex_1_2` ($V = 8,000$): L2-normalized sublinear TF-IDF logistic regression over word 1–2 n-grams enriched with Devanagari/Telugu morphological stems (`st:`), Romanized phonetic keys (`ph:`), emoji cues (`em:`), and conversational context tokens (`cw:`).
   - `CharSub_3_5` ($V = 8,000$): L2-normalized sublinear TF-IDF logistic regression over boundary-marked character 3–5 grams.
   - `LinguisticEngine` (`src/engine/sarcasmEngine.ts`): Deterministic Stage-2 symbolic incongruity engine extracting intra-sentence polarity clash, contextual divergence, rhetorical patterns, irony emojis, Unicode script identification, and Code-Mixing Index (CMI).
2. **Level-2 MAP Ridge Meta-Learner (fitted strictly on 5-fold group-stratified `train.jsonl` OOF predictions):** Combines bounded log-odds of the three experts with `incongruity_index` and `lexical_coverage` ($16,006$ total effective parameters; `405.36 KB` artifact).
3. **Level-3 Post-Hoc Calibration & Selective Prediction (fitted strictly on `val.jsonl`, $N = 1,104$):** Optimizes scalar temperature $T^* = 0.845$ and selective confidence gate $\tau^* = 0.35$ ($\text{Confidence}(x) = 2|P_{\text{fused}} - 0.5|$).

Evaluated once on the frozen held-out test split (`eval_results/independent/test.jsonl`, $N = 1,123$), SARcasmScope lifts test performance from `53.78%` Accuracy / `37.59%` Macro-F1 (`v2` symbolic) to **`68.03%` Accuracy**, **`67.56%` Macro-F1**, and **`60.50%` Sarcasm Recall** (`0.7535` ROC-AUC), while achieving **`1.36%` Conf-ECE** (3.67× lower calibration error than the uncalibrated statistical OOF ensemble `4.99%`), **`80.31%` selective accuracy** across `45.68%` accepted coverage at $\tau^* = 0.35$, and **`2.856 ms` P95** (`1.632 ms` mean) CPU latency.

---

## 2. Benchmark Results on Frozen Held-Out Test Split (`test.jsonl`, $N = 1,123$)

| Model Identifier | Architecture | Parameters | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC | Conf-ECE | Latency (P95) | Status Qualification |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Majority_Class_Literal | Trivial Literal Prior (`y=0`) | 0 | 53.78% | 0.00% | 0.00% | 34.97% | 0.5000 | 46.22% | < 0.01 ms | Measured Baseline |
| v1_Prev_Symbolic | Nov 7 Pre-Fix Heuristic (`prevPredict`) | 0 | 53.52% | 28.57% | 0.39% | 35.21% | 0.5101 | 34.98% | 0.112 ms | Measured Historical Baseline |
| v2_Symbolic_Linguistic | Stage 2 Symbolic Engine (`predict`) | 0 | 53.78% | 50.00% | 3.08% | 37.59% | 0.5087 | 23.36% | 1.509 ms | Measured Symbolic Baseline |
| WordLex_1_2 | Word 1-2 + Morph + Phonetic + Ctx LR | 8,000 | 68.39% | 67.30% | 61.46% | 67.96% | 0.7324 | 2.80% | 0.200 ms | Measured Level-1 Expert |
| CharSub_3_5 | Character 3-5 TF-IDF LR | 8,000 | 65.36% | 63.21% | 59.92% | 65.01% | 0.7294 | 2.11% | 0.378 ms | Measured Level-1 Expert |
| Statistical_OOF_Ensemble | Unweighted Logit Mean (WordLex+CharSub) | 16,000 | 69.55% | 69.11% | 61.66% | 69.06% | 0.7582 | 4.99% | 0.881 ms | Measured Statistical Baseline |
| Uncalibrated_Hybrid_T1 | 5-Fold OOF MAP Ridge Fusion ($T=1.0$) | 16,006 | 68.03% | 67.09% | 60.50% | 67.56% | 0.7536 | 2.28% | 2.856 ms | Measured Ablation |
| **SARcasmScope_Calibrated_Hybrid** | **3-Stage OOF Fusion ($T^*=0.845, \tau^*=0.35$)** | **16,006** | **68.03%** | **67.09%** | **60.50%** | **67.56%** | **0.7535** | **1.36%** | **2.856 ms** | **Measured Client Framework** |
| MuRIL_FineTuned_GPU | `google/muril-base-cased` | 237M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` (Colab GPU script only) |
| XLM_RoBERTa_GPU | `xlm-roberta-base` | 278M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` (Colab GPU script only) |

### Key Statistical & Calibration Validation
- **Bootstrap 95% CIs ($B=1,000$, Seed `42`):** Accuracy `[65.36%, 70.79%]`, Macro-F1 `[64.79%, 70.29%]`, Recall `[56.00%, 64.77%]`, Conf-ECE `[1.05%, 4.54%]`.
- **McNemar Paired Tests (Calibrated Hybrid vs `v2` Symbolic):** Overall Accuracy ($N=1,123$): $\chi^2 = 55.20, p < 0.0001, OR = 2.07$; Sarcasm Recall ($N=519$): $\chi^2 = 290.16, p < 0.0001, OR = 100.33$. Vs `Statistical_OOF_Ensemble`: $\chi^2 = 3.94, p = 0.0472$ (Stat OOF higher raw threshold accuracy by `+1.52%`, while Calibrated Hybrid lowers Conf-ECE from `4.99%` to `1.36%` and NLL from `0.5876` to `0.5826`).
- **Selective Prediction ($\tau^* = 0.35$):** **`80.31%` Selective Accuracy** (`78.92%` Macro-F1) across **`45.68%` Coverage** ($513 / 1,123$ accepted; selective risk drops from `31.97%` to `19.69%`).
- **Closed-Loop Diagnostic Comparison ($N = 30$ `BENCHMARK_CORPUS`):** `v1` (Nov 7): `66.67%` Acc / `33.33%` Rec / `62.50%` MF1 (`TP=5, FP=0, TN=15, FN=10`); `v2`: `96.67%` Acc / `100.00%` Rec / `96.66%` MF1; `Calibrated Hybrid`: `86.67%` Acc / `100.00%` Rec / `86.43%` MF1. Note: `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace; all historical numbers were independently recomputed cleanly.

---

## 3. Dataset Inventory, Partitions & Near-Duplicate Audit (5,600 Records)

Conversation-grouped stratified split (Deterministic Seed `42`, `0` exact text overlaps, `0` `conversation_id` overlaps across splits; exhaustive word 3-shingle Jaccard $\ge 0.80$ scan detects **13 near-duplicate pairs corpus-wide**, all in `IIITD_MaSaC_CodeMixed`: **8 cross-split**, **5 train-test**, **8 label-conflicting**):
1. **ACL 2020 Twitter Sarcasm** (Ghosh et al., 2020, FigLang/ACL; Research-Only Twitter Terms): 1,716 (1,029 / 343 / 344) — English microblogs + conversational context.
2. **Misra-Arora News Headlines** (Misra & Arora, 2019/2023; Public Domain / CC0): 1,999 (1,199 / 399 / 401) — *Editorial headline satire* (*The Onion* vs *HuffPost*, no dialogue context; distinct from conversational social-media sarcasm).
3. **IIITD MaSaC Code-Mixed** (Bedi et al., 2021, IEEE TAFFC; Research Purpose Only): 974 (599 / 181 / 194) — *Text-only sitcom dialogue transcripts* (`text` + preceding `context` inside atomic episodes; audio prosody & visual cues from the original multimodal release are omitted).
4. **L3Cube MahaEmotions** (Kulkarni et al., 2023; Kowtal & Joshi, 2025, PACLIC 39; CC-BY-4.0): 369 (221 / 73 / 75) — *Restricted emotion-taxonomy proxy* (only `Sarcasm` vs `Neutral` labels retained; non-neutral emotions excluded, requiring future native binary sarcasm validation).
5. **IIIT Hyderabad Telugu Sarcasm** (Marreddy et al., 2022, LREC/TALLIP; Research Only): 542 (325 / 108 / 109) — Telugu posts retaining `15.7%` natural positive skew.

Partitions: `train.jsonl` ($3,373$), `val.jsonl` ($1,104$), `test.jsonl` ($1,123$). Note that on Telugu (`test` $N=109$, 15.6% sarcastic), Hybrid achieves `0.7455` ROC-AUC and `0.1148` Brier score but `0.00%` recall at the global `0.50` probability threshold due to minority prior suppression; likewise Hinglish retains `15.57%` Conf-ECE, so global calibration does not guarantee uniform calibration across all sub-corpora.

---

## 4. Key Documentation & Code Layout

- **Source Engine (`src/engine/`):** `sarcasmEngine.ts`, `textFeatures.ts`, `trainedModelData.ts`, `statisticalModel.ts`, `hybridFusion.ts`, `contextualEngine.ts`
- **Web Application (`src/components/`):** `StudioView.tsx`, `BatchView.tsx`, `RobustnessView.tsx`, `ProtocolView.tsx`, `QuantumOrb.tsx`
- **Canonical Stage 9 Reports (`docs/`):** `STAGE9_FINAL_MANUSCRIPT.md`, `STAGE9_INITIAL_AUDIT.md`, `STAGE9_DATASET_AUDIT.md`, `STAGE9_TRANSFORMER_BENCHMARK.md`, `STAGE9_MULTILINGUAL_ANALYSIS.md`, `STAGE9_STATISTICAL_VALIDATION.md`, `STAGE9_ROBUSTNESS_REPORT.md`, `STAGE9_DEPLOYMENT_REPORT.md`, `STAGE9_PUBLICATION_READINESS.md`, `STAGE9_REVIEWER_RESPONSE_PREPARATION.md`, `ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, `ARCHITECTURE.md`, `DATASET_PROVENANCE_AND_LICENSES.md`, `COLAB.md`.

---

## 5. Quickstart & Verification Commands

```bash
# 1. Install dependencies
npm install

# 2. Reproduce full Stage 9 master suite (writes 15 CSVs + master_summary.json)
npx tsx eval/run_stage9_master_execution.ts

# 3. Reproduce 4-way historical version comparison (N=30, N=15, test N=1,123)
npx tsx eval/verify_comparison.ts

# 4. Syntax-check & dry-run GPU transformer Colab pipeline (torch optional locally)
python3 -m py_compile eval/train_transformer.py eval/evaluate_transformers.py eval/export_onnx.py
python3 eval/train_transformer.py --dry_run

# 5. Build, lint, and launch React studio on port 3000
npm run lint && npm run build && npm run dev
```

---

## 6. Citation (Draft Pending Co-Author Finalization)

```bibtex
@misc{mehta2026sarcasmscope,
  title={SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text},
  author={Mehta, Payal and Co-Authors},
  note={Pre-submission software & benchmark repository; confirm final 5-author order and corresponding author prior to journal submission},
  year={2026},
  url={https://github.com/moonlitpayal/sarcasmscope}
}
```
