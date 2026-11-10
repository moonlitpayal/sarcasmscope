# SARcasmScope — Stage 9 Robustness Stress Testing & Error Taxonomy

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9H — Controlled 9-Perturbation Stress Tests (`eval_results/stage9/robustness_results.csv`)

---

## 1. Controlled 9-Perturbation Stress Benchmark (`testRecords`, $N = 100$, Seed `42`)

| Perturbation Test | Targeted Level | Tested $N$ | Empirical Stability (%) | Mean $|\Delta p|$ | Prediction Flips | Honest Diagnostic Observation |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **1. Character Transposition** | Spelling / Subword | 100 | **95.0%** | 0.022 | 5 / 100 | `CharSub_3_5` partially retains surrounding shingles under adjacent swaps. |
| **2. Repeated Vowel Elongation** | Phonetic / Orthographic | 100 | **75.0%** | 0.086 | 25 / 100 | Heavy trailing triple-letter inflation (`w + c.repeat(3)`) alters subword 3–5 grams more than `ph:` lookup recovers. |
| **3. Emoji Removal** | Pragmatic Markers | 100 | **99.0%** | 0.001 | 1 / 100 | Statistical & lexical signals maintain verdict stability when emojis are dropped. |
| **4. Punctuation Removal** | Hyperbole / Syntax | 100 | **98.0%** | 0.016 | 2 / 100 | Word and subword weights outweigh surface `!`/`?` cues. |
| **5. Romanized Spelling Variation** | Dialect / Transliteration | 100 | **100.0%** | 0.000 | 0 / 100 | `ph:` phonetic keys align `bahut/bohot`, `accha/achha`, `bohot/bahut`, etc. |
| **6. Code-Mixing Alternation** | Language Switching | 100 | **98.0%** | 0.001 | 2 / 100 | Mixed English–Hinglish training vocab absorbs lexical code-switches. |
| **7. Dialogue Context Removal** | Contextual Grounding | 100 | **70.0%** | 0.154 | 30 / 100 | **Genuine Context Dependence:** Removing parent context shifts 30% of sampled verdicts (`mean |dp|=0.154`), consistent with the `-4.23%` MF1 context ablation. |
| **8. Unseen Vocabulary Insertion** | Open Vocabulary | 100 | **97.0%** | 0.010 | 3 / 100 | Out-of-vocabulary tokens safely abstain from sparse lookup without runtime errors. |
| **9. Dialectal Morphological Shift** | Agglutinative Shift | 100 | **100.0%** | 0.000 | 0 / 100 | Devanagari/Telugu suffix normalization matches shared root stems (`st:`). |

---

## 2. Calibration Reliability & Selective Abstention (`calibration_results.csv`, `risk_coverage.csv`)

On frozen `test.jsonl` ($N = 1,123$), **Conf-ECE = `1.36%`** across 5 monotonic confidence bins (`|diff| <= 0.026` everywhere: `[0.5-0.6]` 0.556 vs 0.549, `[0.6-0.7]` 0.624 vs 0.647, `[0.7-0.8]` 0.760 vs 0.747, `[0.8-0.9]` 0.852 vs 0.847, `[0.9-1.0]` 0.956 vs 0.930). At default selective gate $\tau^* = 0.35$, accepted accuracy rises from `68.03%` to **`80.31%`** (`78.92%` Macro-F1; risk drops `31.97% -> 19.69%`) over **`45.68%` coverage** (`513/1,123` accepted; `610` marked `"Uncertain"`).
