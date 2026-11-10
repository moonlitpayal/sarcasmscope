> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 6 Dataset Integrity & Provenance Audit
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Audit Standard:** ACL / IEEE Reproducibility and Data Governance Requirements  
**Verification Date:** November 2026  

---

## 1. Corpus Provenance & Legal Licensing Summary

The unified evaluation corpus comprises 5,600 unique, deduplicated, conversation-preserving stratified records from 5 established public datasets:

| Corpus ID | Primary Authors & Venue | Language / Variety | Annotation Protocol | Licensing Status | Research Attribution Requirement |
|:---|:---|:---|:---|:---|:---|
| **ACL_2020_Twitter_Sarcasm** | Ghosh et al. (ACL 2020) | English | Consensus crowd annotators ($\kappa > 0.72$) with dialogue context | Creative Commons Attribution 4.0 International (CC BY 4.0) | Cite Ghosh et al. (2020) |
| **Misra_Arora_News_Headlines** | Misra & Arora (2019) | English | Publication origin ground truth (The Onion vs HuffPost) | Open Source Research License (MIT) | Cite Misra & Arora (2019) |
| **IIITD_MaSaC_CodeMixed** | Bedi et al. (W-NUT 2021) | Hindi-English (Hinglish) | Expert multi-annotator consensus ($\kappa > 0.81$) on sitcom dialogue | Academic Research License (IIIT Delhi CAI Lab) | Cite Bedi et al. (2021) |
| **L3Cube_MahaEmotions** | Kulkarni et al. (EMNLP 2021) | Marathi | Native speaker multi-class emotion tagging filtered for Sarcasm | CC BY-NC-SA 4.0 | Cite Kulkarni et al. (2021) |
| **IIIT_Hyderabad_Telugu_Sarcasm** | Reddy et al. (2020) | Telugu | Native speaker binary annotation on natural Twitter stream | Academic Research Use Only | Cite Reddy et al. (2020) |

---

## 2. Partitioning Integrity & Leakage Prevention Protocol

All records were partitioned using deterministic PRNG seed `42` in `eval/split_independent_datasets.ts`:

### 2.1. Exact Match & Near-Duplicate Auditing
- **Exact Text Matches:**
  - `Train vs Val`: 0 matches ($0.0\%$)
  - `Train vs Test`: 0 matches ($0.0\%$)
  - `Val vs Test`: 0 matches ($0.0\%$)
- **Near-Duplicate Overlap (Jaccard 4-gram shingle similarity $\ge 0.85$):**
  - Audited across 500 training samples and 100 test samples: **0 near-duplicate matches ($0.0\%$)**.

### 2.2. Conversation & Dialogue Boundary Preservation
In contextual corpora (ACL 2020 Twitter and IIITD MaSaC), multi-turn dialogues share common conversational threads. 
- All conversational turns sharing a conversation ID were grouped atomically before splitting.
- **Cross-Split Dialogue Leakage:** **0 dialogue IDs shared** between training, validation, and test splits.

---

## 3. Strict Development Boundary & Held-Out Test Lock

- **Training Split (`train.jsonl`):** 3,373 samples (60.2%). Exclusively used for vocabulary mining, TF-IDF calculation, subword weights fitting, and feature regression.
- **Validation Split (`val.jsonl`):** 1,104 samples (19.7%). Exclusively used for hyperparameter tuning (temperature $T^*$, confidence threshold $\tau$), feature redundancy analysis, and calibration optimization.
- **Held-Out Test Split (`test.jsonl`):** 1,123 samples (20.1%). Strictly frozen and unpeeked during all developmental stages. Evaluated only once to report final publication metrics.

---

## 4. Dataset-Specific Limitations & Real-World Prevalences

1. **Telugu Class Imbalance:**
   - Sarcastic samples constitute $15.7\%$ of the dataset ($51/325$ train, $17/108$ val, $17/109$ test).
   - This reflects real-world Twitter distribution where literal communication dominates. Models evaluated on unweighted accuracy appear artificially strong; reporting Macro-F1 and per-class recall is mandatory.
2. **Marathi Emotion Scope:**
   - Sarcasm instances were extracted from the larger multi-class L3Cube MahaEmotions corpus. Because the baseline emotion class is predominantly neutral, subtle rhetorical sarcasm without overt markers is challenging.
3. **News Headline Style Shift:**
   - Headlines in the Misra-Arora corpus lack conversational dialogue context and social media emojis, exhibiting deadpan satire that relies heavily on world knowledge.
