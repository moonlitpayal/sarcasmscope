# SARcasmScope — Dataset Provenance, License Governance & Leakage Audit
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)

---

## 1. Overview & Ethical Governance

SARcasmScope compiles **5,600 unique records** across five publicly available, peer-reviewed corpora spanning English, Hindi-English code-mixed (`IIITD MaSaC` Hinglish + Devanagari Hindi), Marathi (`L3Cube MahaEmotions`), and Telugu (`IIIT Hyderabad`). All records adhere to source license terms with Twitter handles normalized (`@USER`). Separate closed-loop internal suites (`BENCHMARK_CORPUS` $N=30$ and `UNSEEN_TEST_CASES` $N=15$) are strictly segregated as regression diagnostics and never mixed with benchmark claims.

---

## 2. Dataset Inventory & Exact Provenance (`eval_results/stage9/dataset_integrity.csv`)

| Dataset Identifier | Language / Script | Total Unique | Train (`train.jsonl`) | Val (`val.jsonl`) | Test (`test.jsonl`) | Citation & Venue | License | Near-Dups ($\ge 0.80$) |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---:|
| `ACL_2020_Twitter_Sarcasm` | English | 1,716 | 1,029 | 343 | 344 | Ghosh, Vajpayee, & Muresan (2020), *FigLang / ACL* | Research-Only (Twitter Terms) | 0 total / 0 Tr-Te |
| `Misra_Arora_News_Headlines` | English | 1,999 | 1,199 | 399 | 401 | Misra & Arora (2023), *AI Open*, 4, 13–18 | Public Domain / CC0 | 0 total / 0 Tr-Te |
| `IIITD_MaSaC_CodeMixed` | Hinglish + Devanagari Hindi | 974 | 599 | 181 | 194 | Bedi, Kumar, Akhtar, & Chakraborty (2023/2021), *IEEE TAFFC* ($\kappa = 0.81$) | Research Purpose Only | 13 total / 5 Tr-Te |
| `L3Cube_MahaEmotions` | Marathi (Devanagari) | 369 | 221 | 73 | 75 | Kowtal & Joshi (2025), *PACLIC 39*; MahaSent base: Kulkarni et al. (2021) | CC-BY-4.0 | 0 total / 0 Tr-Te |
| `IIIT_Hyderabad_Telugu_Sarcasm` | Telugu (Telugu script) | 542 | 325 | 108 | 109 | Marreddy, Oota, Vakada, Chinni, & Mamidi (2023), *ACM TALLIP* | Research Only | 0 total / 0 Tr-Te |
| **Total Unified Corpus** | **Multilingual** | **5,600** | **3,373 (60.23%)** | **1,104 (19.71%)** | **1,123 (20.06%)** | **5 Peer-Reviewed Corpora** | **Per-Corpus Terms** | **13 total / 5 Tr-Te** |

### Dataset Notes & Research-Validity Qualifications (Schema Verified Across All `5,600` Records)
- **ACL 2020 Twitter (`HasContext = true`, `1,716/1,716`):** Only corpus in the benchmark with populated preceding dialogue text (`r.context` non-empty across all `1,029` train / `343` val / `344` test records); uses native `SARCASM` (`1`) vs `NOT_SARCASM` (`0`). Limitation: relies on pragmatic world knowledge beyond short parent replies.
- **Misra-Arora News Headlines (`HasContext = false`, `0/1,999`):** Uses native publisher-based `1`/`0` ground truth (*The Onion* satire vs *HuffPost* journalism, `context=""`). Tests **editorial journalistic satire**, not interactive social-media sarcasm.
- **IIITD MaSaC (`HasContext = false`, `0/974`):** Uses native human binary `1`/`0` annotations ($\kappa=0.81$). Crucially, `Episodewise_dialoguelabels.txt` supplies `conversation_id` exclusively to enforce atomic episode-level train/val/test grouping (`0` cross-split dialogue overlap); in `split_independent_datasets.ts`, **preceding turn text is NOT populated (`context=""` on all 974 records)** and audio/video sitcom prosody are omitted, leaving `47.33%` Macro-F1 and `15.57%` Conf-ECE on isolated Hinglish lines.
- **L3Cube MahaEmotions (`HasContext = false`, `0/369`):** Restricted strictly to the corpus's explicit **`Sarcasm`** (`1`, $n=185$) and **`Neutral`** (`0`, $n=184$) labels—deliberately **excluding** all other 9 emotion classes (`Anger/Disgust`, `Sadness`, `Fear`, etc.) so angry/negative literal tweets are not conflated with sarcasm. Two key qualifications remain: (1) omitting non-neutral literal emotions narrows the control distribution; (2) per Kowtal & Joshi (*PACLIC 39*, 2025), L3Cube-MahaEmotions `train.csv` used GPT-4/Llama3 CoTR prompting while `test.csv` is manual gold—because ingestion merged both and capped `Neutral` (`<250`) early in `train.csv`, all `184` Neutral controls and ~93% of `Sarcasm` rows derive from CoTR-annotated rows (`test` includes 3 manual test rows). Thus Marathi serves strictly as a restricted proxy subset.
- **IIIT-H Telugu Sarcasm (`HasContext = false`, `0/542`):** Native binary `yes` (`1`, $n=85$) / `no` (`0`, $n=457$), retaining real `15.7%` sarcastic prevalence (`17/109 = 15.60%` in `test.jsonl`). Under minority prior calibration, predictions achieve `0.7455` ROC-AUC / `0.1148` Brier yet stay below `0.50` (`0%` Hybrid recall without prior-adjusted threshold tuning).
- **Transformations & Synchronized Audit Files:** Applies `@USER` handle anonymization, UTF-8 NFKC normalization, and exact normalized deduplication (`5,712` -> `5,600` unique rows). Both `eval_results/stage9/` and synchronized `eval_results/independent/` CSVs match exact `test.jsonl` counts (`344 + 401 + 194 + 75 + 109 = 1,123`).

---

## 3. Exhaustive Leakage & Near-Duplicate Audit (`eval/run_stage9_master_execution.ts`)

Running exact normalized matching and pairwise word 3-shingle Jaccard similarity ($\ge 0.80$) over all $\binom{5600}{2}$ pairs in `train.jsonl`, `val.jsonl`, and `test.jsonl` produces:
1. **Exact Normalized Text Overlap:** **0 pairs** (`train-val = 0`, `train-test = 0`, `val-test = 0`).
2. **`conversation_id` Overlap:** **0 shared IDs** across any partition boundary.
3. **Word 3-Shingle Near-Duplicates ($\text{Jaccard} \ge 0.80$):**
   - **Total corpus-wide pairs:** **13** (all within `IIITD_MaSaC_CodeMixed` short recurring sitcom utterances such as brief interjections).
   - **Cross-split near-duplicate pairs:** **8** total (`train-test = 5`, `train-val = 1`, `val-test = 2`).
   - **Label-conflicting near-duplicate pairs:** **8** (identical short utterances taking context-dependent opposite sarcasm labels across different sitcom episodes).

To preserve split immutability once `test.jsonl` was sealed, `test.jsonl` was not re-partitioned after Stage 1; instead, these exact 13/8/5 counts are reported transparently alongside zero exact overlap.
