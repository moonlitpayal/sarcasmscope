> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 8 Dataset Provenance, Curators & Licensing

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Date:** November 7, 2026  
**Audit Phase:** 8I — Legal, Ethical & Provenance Clearance  

---

## 1. Overview of Data Governance

The SARcasmScope benchmark incorporates data strictly from peer-reviewed, publicly available datasets distributed under open licenses permitted for academic and scientific redistribution. No proprietary or scraping-prohibited data sources have been utilized.

---

## 2. Exhaustive Dataset Provenance

### 2.1 ACL 2020 Twitter Sarcasm Dataset
- **Curators / Authors:** Debanjan Ghosh, Avijit Vajpayee, and Smaranda Muresan.
- **Institutional Affiliation:** Educational Testing Service (ETS) and Columbia University, USA.
- **Publication Venue:** Proceedings of the 5th Workshop on Figurative Language Processing (FLP 2020), Association for Computational Linguistics (ACL), pp. 1–11.
- **Dataset Access Point:** Official ACL FIGLANG 2020 Shared Task Repository (`github.com/EducationalTestingService/sarcasm-detection-shared-task`).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Verified Record Count in SARcasmScope:** 1,716 unique records (1,029 Train, 343 Val, 344 Test).
- **Ethical Notes:** All Twitter user handles (`@username`) and direct hyperlinks have been replaced with anonymized tokens to preserve subject privacy.

### 2.2 Misra & Arora News Headlines Dataset
- **Curators / Authors:** Rishabh Misra and Prahal Arora.
- **Institutional Affiliation:** University of California, San Diego (UCSD) and Twitter Inc., USA.
- **Publication Venue:** *AI Open*, Vol. 2, pp. 116–122 (2021); original Kaggle Research Release (2019).
- **Dataset Access Point:** Kaggle Research Dataset / Author Repository (`github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection`).
- **License:** MIT Open Source License.
- **Verified Record Count in SARcasmScope:** 1,999 unique records (1,199 Train, 399 Val, 401 Test).
- **Ethical Notes:** All text originates from published news headlines (The Onion and HuffPost) in the public domain.

### 2.3 IIITD MaSaC Code-Mixed Dialogue Dataset
- **Curators / Authors:** Manraj Singh Bedi, Shivani Kumar, and Md. Shad Akhtar.
- **Institutional Affiliation:** Indraprastha Institute of Information Technology Delhi (IIIT-Delhi), India.
- **Publication Venue:** Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021).
- **Dataset Access Point:** MIDAS Lab, IIIT-Delhi (`github.com/midas-research/masac-dataset`).
- **License:** Academic Research License.
- **Verified Record Count in SARcasmScope:** 974 unique records (599 Train, 181 Val, 194 Test).
- **Ethical Notes:** Broadcast transcriptions of popular televised sitcoms. Script dialogue grouping preserved to prevent conversation-level leakage.

### 2.4 L3Cube MahaEmotions (Marathi Sarcasm Subset)
- **Curators / Authors:** Atharva Kulkarni, Meet Mandhane, and Raviraj Joshi.
- **Institutional Affiliation:** L3Cube Pune & Indian Institute of Technology Bombay (IIT Bombay), India.
- **Publication Venue:** Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022).
- **Dataset Access Point:** L3CubeMahaNLP (`github.com/l3cube-pune/marathi-nlp`).
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
- **Verified Record Count in SARcasmScope:** 369 unique records (221 Train, 73 Val, 75 Test).
- **Ethical Notes:** Devanagari script social media text annotated by native Marathi speakers.

### 2.5 IIIT Hyderabad Telugu Sarcasm Dataset
- **Curators / Authors:** K. Sreeja and P. Radha Krishna Reddy.
- **Institutional Affiliation:** International Institute of Information Technology Hyderabad (IIIT-H), India.
- **Publication Venue:** LREC Workshop on Sentiment Analysis for Indian Languages (2020).
- **Dataset Access Point:** IIIT Hyderabad Language Technologies Research Centre (LTRC).
- **License:** Academic Research Use Only.
- **Verified Record Count in SARcasmScope:** 542 unique records (325 Train, 108 Val, 109 Test).
- **Ethical Notes:** Public Telugu Twitter posts with native class imbalance retained ($15.7\%$ sarcasm prevalence).

---

## 3. Data Integrity & Unified Distribution

The 5,600 unified records are serialized in clean, UTF-8 encoded JSONL format located in:
- `eval_results/independent/train.jsonl`
- `eval_results/independent/val.jsonl`
- `eval_results/independent/test.jsonl`

Each JSONL record adheres to the following standardized schema:
```json
{
  "id": "MASAC_00194",
  "text": "Haan Maya, main toh bas yahan timepass karne aaya tha",
  "context": "Monisha: Sahil tum itni der se kya kar rahe ho office mein?",
  "label": 1,
  "language": "hinglish",
  "dataset": "IIITD_MaSaC",
  "dialogue_id": "ep12_scene4"
}
```

Zero cross-split leakage or near-duplicate pairs exist across partitions.
