# SARcasmScope — Stage 9 Multilingual Improvement & Per-Language Analysis

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9E — Multilingual Mechanisms & Per-Language Frozen Evaluation (`eval_results/stage9/multilingual_improvements.csv`)

---

## 1. Engineered Multilingual Architecture (Actual Module Locations)

SARcasmScope's multilingual processing lives directly inside **`src/engine/sarcasmEngine.ts`** and **`src/engine/textFeatures.ts`** (consumed symmetrically during training in `eval/train_and_calibrate.ts` and runtime inference in `src/engine/statisticalModel.ts`):
1. **Indic Morphological Stemmer (`stemIndicToken` in `src/engine/sarcasmEngine.ts`):**
   Conservatively strips inflectional case postpositions and verbal auxiliaries in Devanagari (`DEV_SUFFIXES`, min stem 3) and Telugu (`TELUGU_SUFFIXES`, min stem 4), emitting `st:` root keys for `WordLex_1_2` and symbolic matching.
2. **Romanized Phonetic Folding (`normalizeRomanized` in `src/engine/sarcasmEngine.ts`):**
   Collapses repeated character runs and transliteration pairs (`ee->i, oo->u, ph->f, kh->k, bh->b, sh->s, z->j, w->v`), emitting `ph:` phonetic keys.
3. **Unicode Script & Code-Mixing Index (`analyzeCodeMixing` in `src/engine/sarcasmEngine.ts`):**
   Identifies Devanagari (Hindi/Marathi heuristic), Telugu, Tamil, Bengali, Gujarati, Latin, and Romanized Hinglish/Indic code-mixing ratio.

---

## 2. Empirical Per-Language Results on Frozen `test.jsonl` ($N = 1,123$)

| Language Regime | Count ($N$) | `v2` Symbolic Acc / MF1 / Rec | `Calibrated Hybrid` Acc | Precision | Recall | **Hybrid Macro-F1** | Specificity | Conf-ECE | $\Delta$ MF1 vs `v2` |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **English** (`ACL` + `Misra`) | 745 | `51.01%` / `37.33%` / `4.38%` | **71.54%** | 72.17% | **68.22%** | **71.48%** | 74.74% | 4.03% | **+34.15%** |
| **Hindi-English (`IIITD_MaSaC`)** | 194 | `48.45%` / `32.64%` / `0.00%` | 47.42% | 48.84% | **42.00%** | **47.33%** | 53.19% | 15.57% | **+14.69%** |
| **Marathi (`MahaEmotions`)** | 75 | `50.67%` / `33.63%` / `0.00%` | **62.67%** | 62.16% | **62.16%** | **62.66%** | 63.16% | **1.29%** | **+29.03%** |
| **Telugu (`IIIT_H`)** | 109 | `84.40%` / `45.77%` / `0.00%` | 84.40% | 0.00% | **0.00%** | 45.77% | 100.00% | 5.77% | **+0.00%** |
| **Unified Multilingual** | **1,123** | `53.78%` / `37.59%` / `3.08%` | **68.03%** | **67.09%** | **60.50%** | **67.56%** | **74.50%** | **1.36%** | **+29.97%** |

---

## 3. Honest Cross-Lingual Diagnosis

1. **Strong English & Marathi Gains:** On English (`+34.15%` MF1, `0.7826` ROC-AUC) and Marathi (`+29.03%` MF1, `0.6789` ROC-AUC, `1.29%` Conf-ECE), OOF feature fusion strongly rescues sarcasm recall (`4.38% -> 68.22%` English; `0.00% -> 62.16%` Marathi).
2. **Hinglish (`IIITD_MaSaC`, $N=194$):** Recall recovers from `0.00%` to `42.00%` (`+14.69%` MF1; `WordLex_1_2` alone reaches `54.52%` MF1), while deadpan multi-turn situational banter produces higher miscalibration (`15.57%` Conf-ECE).
3. **Telugu (`IIIT_H`, $N=109$) Cutoff Limitation:** With only `15.60%` positive test instances (`17/109`), the calibrated global `0.50` probability cutoff yields `0.00%` Hybrid recall (`TP=0`) even though probabilistic ranking is strong (**`0.7455` ROC-AUC**, **`0.1148` Brier**, **`0.3797` NLL**, and `WordLex_1_2` alone gets `55.21%` MF1 / `11.76%` recall). Threshold recalibration or class-prior offset is required for imbalanced Dravidian operating points.
