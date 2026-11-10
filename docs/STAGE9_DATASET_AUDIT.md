# SARcasmScope — Stage 9 Full-Scale Dataset Audit & Integrity Verification

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Audit Phase:** 9B — Dataset Validation, Licensing & Exhaustive Near-Duplicate Scan (`eval_results/stage9/dataset_integrity.csv`)

---

## 1. Multi-Corpus Inventory & Stratified Partitioning (`5,600` Records, Seed `42`)

| Dataset Identifier | Language / Script | Total Unique | Train (`train.jsonl`) | Val (`val.jsonl`) | Test (`test.jsonl`) | Citation | License | Total Near-Dups ($\ge 0.80$) | Train-Test Near-Dups |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|:---:|:---:|
| `ACL_2020_Twitter_Sarcasm` | English | 1,716 | 1,029 | 343 | 344 | Ghosh et al. (2020), *FigLang / ACL* | Research-Only (Twitter Terms) | 0 | 0 |
| `Misra_Arora_News_Headlines` | English | 1,999 | 1,199 | 399 | 401 | Misra & Arora (2023), *AI Open* | Public Domain / CC0 | 0 | 0 |
| `IIITD_MaSaC_CodeMixed` | Hinglish + Devanagari Hindi | 974 | 599 | 181 | 194 | Bedi et al. (2023/2021), *IEEE TAFFC* | Research Purpose Only | **13** | **5** |
| `L3Cube_MahaEmotions` | Marathi (Devanagari) | 369 | 221 | 73 | 75 | Kowtal & Joshi (2025), *PACLIC 39*; MahaSent base: Kulkarni et al. (2021) | CC-BY-4.0 | 0 | 0 |
| `IIIT_Hyderabad_Telugu_Sarcasm` | Telugu (Telugu script) | 542 | 325 | 108 | 109 | Marreddy et al. (2023), *ACM TALLIP* | Research Only | 0 | 0 |
| **Total Unified Corpus** | **Multilingual** | **5,600** | **3,373 (60.23%)** | **1,104 (19.71%)** | **1,123 (20.06%)** | **5 Established Public Corpora** | **Per-Corpus Terms** | **13** | **5** |

---

## 2. Corpus Integrity & Exhaustive Near-Duplicate Findings

Running `eval/run_stage9_master_execution.ts` over all $\binom{5600}{2}$ pairs in `train.jsonl`, `val.jsonl`, and `test.jsonl`:
- **Exact Normalized Text Overlap (`train-val`, `train-test`, `val-test`):** **`0`** pairs (`0.00%`).
- **`conversation_id` Overlap Across Splits (`IIITD_MaSaC` atomic grouping):** **`0`** shared conversation IDs (`0.00%`).
- **Word 3-Shingle Near-Duplicate Pairs ($\text{Jaccard} \ge 0.80$):**
  - **Total pairs corpus-wide:** **`13`** (all occurring in `IIITD_MaSaC_CodeMixed`).
  - **Cross-split near-duplicate pairs:** **`8`** (`train-test = 5`, `train-val = 1`, `val-test = 2`).
  - **Label-conflicting near-duplicate pairs:** **`8`** (short colloquial turns recurring across episodes with context-dependent opposite labels).
- **Telugu Class Balance:** `IIIT_Hyderabad_Telugu_Sarcasm` preserves natural real-world skew (`15.7%` sarcastic corpus-wide; `17/109 = 15.60%` sarcastic in `test.jsonl`).
- **Context Population Audit:** Only `ACL_2020_Twitter_Sarcasm` populates `r.context` (`1,716/1,716` records = `100%`). All other four corpora have `r.context === ""` (`0%`).

---

## 3. Domain, Label-Mapping & Annotation Validity Boundaries

1. **`L3Cube_MahaEmotions` Label Justification & CoTR Provenance Limitation:**
   - **Why strictly `Sarcasm` ($n=185$) vs `Neutral` ($n=184$):** `Sarcasm` is a native annotated tag inside MahaEmotions' 11-class schema; we deliberately **excluded** non-neutral literal emotions (`Anger/Disgust`, `Sadness`, `Fear`, etc.) rather than merging negative emotions with non-sarcasm, preventing label contamination where sarcastic frustration resembles anger.
   - **Dual Limitations Disclosed:** First, omitting emotional non-sarcastic tweets narrows the literal control class strictly to neutral posts. Second, in Kowtal & Joshi (*PACLIC 39*, 2025), L3Cube-MahaEmotions `train.csv` used GPT-4/Llama3 CoTR synthetic annotations while `test.csv` used human annotations; because our ingestion read `[...train, ...test]` and stopped `Neutral` collection below 250 rows in `train.csv`, **100% of `Neutral` controls and ~93% of `Sarcasm` rows stem from L3Cube's CoTR-annotated training split** (with only `12` manual-split `Sarcasm` items). Marathi is therefore framed strictly as an exploratory subset requiring full human-gold binary sarcasm evaluation.
2. **`Misra_Arora_News_Headlines` Genre Distinction:** Ground-truth (`1`/`0`) derives from publisher provenance (*The Onion* vs *HuffPost*, $N=1,999$, $401$ in `test`, `context=""`). Evaluates **editorial journalistic satire**, distinct from conversational social-media sarcasm.
3. **`IIITD_MaSaC_CodeMixed` Isolated-Turn & Modality Constraint:** Although `Episodewise_dialoguelabels.txt` provides episode `conversation_id` to guarantee zero dialogue-group leakage across splits, **`split_independent_datasets.ts` sets `context=""` on all 974 MaSaC utterances** (`HasContext = false`), and omits audio prosody/visual facial cues from the original sitcom video (*Sarabhai vs Sarabhai*). Evaluating isolated hinglish lines without surrounding turns or A/V cues explains `47.33%` Macro-F1 and `15.57%` Conf-ECE.
4. **`ACL_2020_Twitter_Sarcasm` & `IIIT-H Telugu` Boundaries:** ACL Twitter (`SARCASM`/`NOT_SARCASM`, $n=344$ test) provides preceding parent tweet `context` (`100%` populated) yet omits broader thread/world knowledge; IIIT-H Telugu (`yes`/`no`, $n=109$ test) retains `15.60%` natural minority prevalence causing `0%` Hybrid recall at `0.50` despite `0.7455` ROC-AUC.
5. **Source Transformations & Compliance:** Normalized `@USER` handles, UTF-8 NFKC normalization, `112` exact normalized duplicate removals (`5,712` -> `5,600`), and strict adherence to Research-Only / Twitter Terms / CC0 / CC-BY-4.0 dataset licenses.
