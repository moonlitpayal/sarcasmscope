# SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text

**Manuscript Status:** *Technical Verification Draft — Pending Five-Author Reconciliation & Co-Author Approval Before Submission*  
**Codebase & Stage 9 Audit Maintainer:** Payal Mehta (`payal.12620246@vit.edu`), School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India  
**Authorship Governance Notice (Pre-Submission Blocker):** The original working manuscript lists **five co-authors with the project advisor/collaborator designated as corresponding author**. Final author list, author order, CRediT statements, corresponding-author designation, and approval by all five co-authors **must be finalized** before journal submission.  
Repository: `https://github.com/moonlitpayal/sarcasmscope`

---

## Abstract

Detecting sarcasm in noisy, multilingual, and code-mixed social communication poses a severe trade-off between symbolic interpretability and open-vocabulary generalization. Handcrafted rule engines preserve high specificity (`97.35%–99.17%`) on literal inputs but suffer catastrophic recall collapse on independent benchmarks (`0.39%` recall in `v1`, `3.08%` in `v2` on $N=1,123$). Meanwhile, massive transformer models depend on GPU servers unavailable in privacy-preserving offline client browsers. In this work, we formulate and validate **SARcasmScope**, a non-circular **3-Stage Calibrated Stacked Feature-Fusion Framework** running 100% client-side in TypeScript (`16,006` effective parameters; `405.36 KB` artifact). Stage 1 fits regularized sublinear TF-IDF sparse logistic experts (`WordLex_1_2` with Indic morphological stems, Romanized phonetic keys, emoji markers, and dialogue context tokens; and `CharSub_3_5` char n-grams) exclusively on `train.jsonl` ($N=3,373$) alongside a deterministic linguistic incongruity engine. Stage 2 learns a MAP Ridge logistic meta-learner on 5-fold group-stratified train out-of-fold predictions (`dataset__conversation_id`). Stage 3 fits scalar temperature scaling ($T^*=0.845$) and a selective confidence gate ($\tau^*=0.35$) strictly on held-out `val.jsonl` ($N=1,104$). Evaluated once on frozen `test.jsonl` ($N=1,123$ spanning five public corpora in English, Hinglish, Marathi, and Telugu), SARcasmScope lifts performance from `53.78%` Accuracy / `37.59%` Macro-F1 / `3.08%` Recall (`v2`) to **`68.03%` Accuracy**, **`67.56%` Macro-F1**, **`60.50%` Recall**, and **`0.7535` ROC-AUC** (McNemar $\chi^2 = 55.20$ Acc, $\chi^2 = 290.16$ Rec, $p < 0.0001$), reduces **Conf-ECE by 3.67× (`4.99% -> 1.36%`)** versus uncalibrated statistical stacking, elevates accepted-subset accuracy to **`80.31%`** (`45.68%` coverage at $\tau^*=0.35$), and executes on commodity CPUs at **`2.856 ms` P95** (`1.632 ms` mean) latency.

---

## 1. Introduction & Reconciliation Motivation

Sarcasm conveys an intended meaning opposite to literal valence using intra-utterance contradiction, dialogue turn opposition, rhetorical exaggeration, and cultural world knowledge (Joshi et al., 2015; Ghosh et al., 2020). In South Asian online spaces (Hinglish, Marathi, Telugu), surface lexical rules struggle against non-standardized transliteration, agglutinative inflections, and deadpan banter.

Equally critical for scientific rigor is separating closed-loop diagnostic checks from true out-of-sample evaluation:
1. On our initial 30-item internal benchmark (`BENCHMARK_CORPUS`), the November 7 symbolic predictor (`v1` `prevPredict`) scored `66.67%` Acc / `100.00%` Prec / `33.33%` Rec / `62.50%` Macro-F1 (`TP=5, FP=0, TN=15, FN=10`), and rule expansions (`v2`) reached `96.67%` Acc / `100.00%` Rec.
2. When evaluated on independent public test data (`test.jsonl`, $N=1,123$), `v1` collapsed to `0.39%` recall (`35.21%` MF1) and `v2` to `3.08%` recall (`37.59%` MF1). Moreover, because an earlier legacy file (`RESULTS_STATUS_AND_AUDIT.md`) is absent from the workspace, all historical numbers were independently recomputed end-to-end.

---

## 2. Corpus Provenance, Splits & Leakage Audit (`5,600` Records)

All records derive from five published corpora (`eval/split_independent_datasets.ts`, seed `42`; `112` exact normalized duplicates removed pre-split from `5,712` raw candidates):
1. **`ACL_2020_Twitter_Sarcasm`** (Ghosh et al., 2020, *FigLang/ACL*; Research-Only Twitter Terms): $1,716$ (`1,029 / 343 / 344`) — English conversational replies with preceding parent tweet `context` (`100%` populated).
2. **`Misra_Arora_News_Headlines`** (Misra & Arora, 2023, *AI Open*; Public Domain / CC0): $1,999$ (`1,199 / 399 / 401`) — *Editorial news satire* (*The Onion* vs *HuffPost*, `context=""`). Framed explicitly as journalistic satire rather than conversational social-media sarcasm.
3. **`IIITD_MaSaC_CodeMixed`** (Bedi et al., 2023/2021, *IEEE TAFFC*, $\kappa = 0.81$; Research Purpose Only): $974$ (`599 / 181 / 194`) — *Isolated text sitcom utterances* (`text` with `context=""`; episode `conversation_id` used strictly for atomic train/val/test split grouping, with surrounding turn text and audio/video prosody omitted).
4. **`L3Cube_MahaEmotions`** (Kowtal & Joshi, 2025, *PACLIC 39*; built on MahaSent, Kulkarni et al., 2021; CC-BY-4.0): $369$ (`221 / 73 / 75`) — *Restricted emotion-taxonomy subset*: only explicit `Sarcasm` ($n=185$) vs `Neutral` ($n=184$) labels retained (excluding `Anger/Disgust`/`Sadness`). Because L3Cube's `train.csv` used CoTR LLM synthetic labeling while `test.csv` is manual gold, all Neutral controls and ~93% of `Sarcasm` rows come from CoTR-annotated splits; treated strictly as an exploratory proxy.
5. **`IIIT_Hyderabad_Telugu_Sarcasm`** (Marreddy et al., 2023, *ACM TALLIP*; Research Only): $542$ (`325 / 108 / 109`) — Telugu posts retaining `15.7%` natural positive skew (`15.60%` in `test.jsonl`).

**Leakage Audit (`dataset_integrity.csv`):** Cross-split exact normalized text overlap $= 0$; cross-split `conversation_id` overlap $= 0$. Exhaustive pairwise word 3-shingle Jaccard $\ge 0.80$ detects **`13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, `8` label-conflicting)** exclusively inside `IIITD_MaSaC_CodeMixed` short dialogue turns. Both `eval_results/stage9/` and synchronized `eval_results/independent/` CSVs report exact held-out $N=1,123$ per-dataset counts (`344 + 401 + 194 + 75 + 109`).

---

## 3. 3-Stage Calibrated Feature-Fusion Architecture (`src/engine/`)

### 3.1 Stage 1: Base Experts (Fitted Strictly on `train.jsonl`, $N = 3,373$)
- **`WordLex_1_2` ($V=8,000$):** Sublinear L2-normalized TF-IDF logistic regression ($\text{epochs}=70, \text{lr}=0.05, L_2=10^{-4}$) over unigrams, conservative Indic suffixes (`st:` via `stemIndicToken`), Romanized phonetic keys (`ph:` via `normalizeRomanized`), emojis (`em:`), bigrams (`bg:`), and dialogue context tokens (`cw:`, `cw_st:`).
- **`CharSub_3_5` ($V=8,000$):** Sublinear L2-normalized TF-IDF logistic regression ($\text{epochs}=70, \text{lr}=0.05, L_2=2\times 10^{-4}$) over boundary-marked character 3–5 grams.
- **`LinguisticEngine` (`sarcasmEngine.ts`):** Deterministic score (`idx = 0.40*clash + 0.35*ctxClash + 0.15*min(contrastCount,2) + 0.10*min(intensifierCount,2)`), lexical coverage $C(x)$, script detection, and Code-Mixing Index (CMI).

### 3.2 Stage 2: 5-Fold Group-Stratified OOF MAP Ridge Meta-Learner
Using folds grouped by `dataset__conversation_id` inside `train.jsonl` and Gaussian prior $\boldsymbol{\mu}_{\text{prior}}=[0.5,0.5,0.4,0.2,0.2], \lambda=0.25$:
$$z_{\text{raw}}(x, c) = w_0 + w_1 \cdot \text{logit}(P_{\text{word}}, 4) + w_2 \cdot \text{logit}(P_{\text{char}}, 4) + w_3 \cdot \text{logit}(P_{\text{ling}}, 3) + w_4 \cdot \text{Idx}_{\text{incongruity}} + w_5 \cdot C(x)$$
with fitted coefficients ($16,006$ total params; `405.36 KB` `trainedModelData.ts`): `w0=0.3483, w1=0.54817, w2=0.54648, w3=0.26486, w4=0.19418, w5=0.20441`.

### 3.3 Stage 3: Validation Temperature Scaling & Four Uncertainty Quantities (`val.jsonl`, $N = 1,104$)
Minimizing validation NLL yields $T^* = 0.845$. Crucially, four non-interchangeable uncertainty constructs are separated in both implementation and reporting:
1. **Calibrated Class Probability:** $P_{\text{fused}} = \sigma(z_{\text{raw}}/T^*) \in (0, 1)$, evaluated via Guo et al. (2017) 5-bin `Conf-ECE` over $\max(P, 1-P)$ (`1.36%`), `Class-ECE` (`2.80%`), `Brier` (`0.2005`), and `NLL` (`0.5826`). (Legacy heuristic overconfidence—where 1,106/1,123 items clustered in `[0.9-1.0]` with `0.428` error—is resolved in Stage 9: Bin 5 `[0.9-1.0]` now contains `68` samples with `0.930` mean confidence vs `0.956` empirical accuracy, error `0.026`.)
2. **Decision Margin Confidence:** $\text{Confidence} = 2|P_{\text{fused}} - 0.5| \in [0, 1]$ (used exclusively for abstention gating, distinct from Stage-2 symbolic heuristic score).
3. **Abstention Threshold ($\tau^*$):** $\tau^* = 0.35$ fitted on `val.jsonl` (`Uncertain` iff $\text{Confidence} < 0.35$, i.e., $P_{\text{fused}} \in (0.325, 0.675)$).
4. **Selective Utility (`test.jsonl`):** `80.31%` Selective Accuracy (`78.92%` Selective Macro-F1; risk `31.97% -> 19.69%`) over **`45.68%` coverage** (`513/1,123` accepted predictions). (Geifman & El-Yaniv, 2017.)

---

## 4. Frozen Held-Out Test Results (`test.jsonl`, $N = 1,123$)

| Model Identifier | Params | P95 Latency | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC | Conf-ECE | NLL | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Majority Class (Literal) | 0 | < 0.01 ms | 53.78% | 0.00% | 0.00% | 34.97% | 0.5000 | 46.22% | 12.7698 | Measured |
| `v1_Prev_Symbolic` (Nov 7) | 0 | 0.112 ms | 53.52% | 28.57% | 0.39% | 35.21% | 0.5101 | 34.98% | 1.0957 | Measured |
| `v2_Symbolic_Linguistic` | 0 | 1.509 ms | 53.78% | 50.00% | 3.08% | 37.59% | 0.5087 | 23.36% | 0.9010 | Measured |
| `WordLex_1_2` | 8,000 | 0.200 ms | 68.39% | 67.30% | 61.46% | 67.96% | 0.7324 | 2.80% | 0.6021 | Measured |
| `CharSub_3_5` | 8,000 | 0.378 ms | 65.36% | 63.21% | 59.92% | 65.01% | 0.7294 | 2.11% | 0.6029 | Measured |
| `Statistical_OOF_Ensemble` | 16,000 | 0.881 ms | 69.55% | 69.11% | 61.66% | 69.06% | 0.7582 | 4.99% | 0.5876 | Measured |
| `Uncalibrated_Hybrid_T1` | 16,006 | 2.856 ms | 68.03% | 67.09% | 60.50% | 67.56% | 0.7536 | 2.28% | 0.5856 | Measured |
| **`SARcasmScope_Calibrated_Hybrid`** | **16,006** | **2.856 ms** | **68.03%** | **67.09%** | **60.50%** | **67.56%** | **0.7535** | **1.36%** | **0.5826** | **Measured** |
| `MuRIL_FineTuned_GPU` | 237M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` |
| `XLM_RoBERTa_GPU` | 278M | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `NOT_EXECUTED_LOCALLY` |

---

## 5. Statistical Validation, Per-Language Breakdown, Ablations & Stress Tests

- **Bootstrap CIs & McNemar:** 95% CIs (`[65.36%, 70.79%]` Acc; `[64.79%, 70.29%]` MF1; `[1.05%, 4.54%]` Conf-ECE). McNemar vs `v2`: $\chi^2 = 55.20, p < 0.0001$ Acc; $\chi^2 = 290.16, p < 0.0001$ Rec. Vs `Statistical_OOF`: $\chi^2 = 3.94, p = 0.0472$ (where `Calibrated Hybrid` cuts Conf-ECE `4.99% -> 1.36%` and lowers NLL `0.5876 -> 0.5826`).
- **Selective Abstention ($\tau^* = 0.35$):** **`80.31%` Selective Accuracy** (`78.92%` Macro-F1; risk `31.97% -> 19.69%`) over `45.68%` accepted coverage (`513/1,123`), rising to `87.42%` at $\tau=0.50$ (`28.32%` coverage).
- **Per-Language & Calibration Scope Boundaries:** Aggregate `1.36%` Conf-ECE does **not** guarantee uniform sub-corpus calibration: English (`71.48%` MF1, `4.03%` Conf-ECE) and Marathi (`62.66%` MF1, `1.29%` Conf-ECE, on restricted `Sarcasm`-vs-`Neutral` subset) calibrate well, whereas text-only Hinglish (`47.33%` MF1, `15.57%` Conf-ECE) exhibits substantial miscalibration without audio/visual cues, and Telugu (`45.77%` MF1, `5.77%` Conf-ECE) suffers `0.00%` recall at the global `0.50` threshold under `15.60%` prevalence despite `0.7455` ROC-AUC and `0.1148` Brier.
- **Ablations & Stress Suite:** Removing conversational context (`-4.23%` MF1) and `WordLex` (`-3.79%` MF1) cause the largest drops; removing `Linguistic` logit collapses specificity `74.50% -> 58.77%`. On $N=100$ perturbations (`seed=42`), spelling variation (`100%`), dialectal shift (`100%`), emoji removal (`99%`), punctuation removal (`98%`), code-mixing (`98%`), OOV (`97%`), and character transposition (`95%`) hold steady while vowel elongation (`75%`) and context removal (`70%`) expose genuine sensitivity.
- **CPU Runtime & GPU Scope:** `1.632 ms` mean, `1.423 ms` median, `2.856 ms` P95, `70.53 MB` heap (`software_audit.csv`). Transformer scripts (`MuRIL`, `XLM-R`) remain `NOT_EXECUTED_LOCALLY` pending Colab GPU execution (`docs/COLAB.md`), and no parity claim with neural transformers is asserted here.

---

## References

- Bedi, M., Kumar, S., Akhtar, M. S., & Chakraborty, T. (2023). Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations. *IEEE Transactions on Affective Computing*, 14(2), 1363–1375. DOI: `10.1109/TAFFC.2021.3083522`.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems 30 (NIPS / NeurIPS 2017)*, pp. 4878–4887.
- Ghosh, D., Vajpayee, A., & Muresan, S. (2020). A Report on the 2020 Sarcasm Detection Shared Task. *Proceedings of the Second Workshop on Figurative Language Processing (FigLang 2020)*, pp. 1–11. DOI: `10.18653/v1/2020.figlang-1.1`.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML 2017*, PMLR Vol. 70, pp. 1321–1330.
- Joshi, A., Sharma, V., & Bhattacharyya, P. (2015). Harnessing Context Incongruity for Sarcasm Detection. *Proceedings of ACL 2015 (Short Papers)*, pp. 757–762 (`P15-2124`).
- Kowtal, N., & Joshi, R. (2025). L3Cube-MahaEmotions: A Marathi Emotion Recognition Dataset with Synthetic Annotations using CoTR prompting and Large Language Models. *Proceedings of PACLIC 39*, pp. 419–427 (`2025.paclic-1.38`; DOI: `10.48550/arXiv.2506.00863`). *(MahaSent base: Kulkarni, A., et al., WASSA 2021, pp. 213–220.)*
- Marreddy, M., Oota, S. R., Vakada, L. S., Chinni, V. C., & Mamidi, R. (2023). Am I a Resource-Poor Language? Data Sets, Embeddings, Models and Analysis for four different NLP Tasks in Telugu Language. *ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)*, 22(1), Article 18, 1–34. DOI: `10.1145/3531535`.
- Misra, R., & Arora, P. (2023). Sarcasm detection using news headlines dataset. *AI Open*, 4, 13–18. DOI: `10.1016/j.aiopen.2023.01.001`.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of ACM SIGKDD 2016*, pp. 1135–1144. DOI: `10.1145/2939672.2939778`.
- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259. DOI: `10.1016/S0893-6080(05)80023-1`.
