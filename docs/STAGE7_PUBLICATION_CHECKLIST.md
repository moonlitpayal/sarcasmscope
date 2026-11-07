# SARcasmScope — Stage 7 Pre-Submission Publication Checklist
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** Vellore Institute of Technology  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Submission Status:** CERTIFIED SUBMISSION-READY (Scopus Q1 / Q2 Venues)  
**Date:** November 2026  

---

## 1. Editorial Compliance Checklist

### Manuscript & Scientific Requirements
- [x] **Title Accuracy:** Verified consistent: *SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text*
- [x] **Author Attribution:** Payal Mehta (`payal.12620246@vit.edu`), School of Computer Science and Engineering, Vellore Institute of Technology
- [x] **21-Section Structure:** Verified complete in `docs/STAGE7_FINAL_MANUSCRIPT.md`
- [x] **Citation Integrity:** 18 citations verified against Google Scholar/DBLP; 0 fabricated DOIs or references
- [x] **Abstract Word Count:** 248 words (within 250-word journal limits)
- [x] **Keywords:** Sarcasm Detection, Multilingual NLP, Indic Languages, Hinglish Code-Mixing, Calibrated Feature Fusion, Selective Prediction

### Data Governance & Open Science
- [x] **Data Provenance:** 5 public corpora documented with full licensing in `docs/DATASET_PROVENANCE_AND_LICENSES.md`
- [x] **Reproducibility Guarantee:** Deterministic seed `42` documented in `docs/STAGE7_REPRODUCIBILITY.md`
- [x] **Zero-Leakage Certification:** 0 exact matches, 0 near-duplicates, 0 conversation grouping leaks certified in `eval_results/stage7/dataset_validation_report.csv`
- [x] **Open Source Code:** Cleanly structured under MIT License in `https://github.com/payal-mehta/SARcasmScope`

### Statistical Rigor
- [x] **Bootstrap Confidence Intervals:** 1,000 iterations for global and per-language metrics in `eval_results/stage7/statistical_validation.csv`
- [x] **McNemar Significance Disentanglement:**
  - Sarcasm Recall recovery ($3.08\% \to 57.23\%$): $\chi^2 = 277.03, p < 0.0001$ (**Significant**, Odds Ratio = 282)
  - Overall Accuracy ($53.78\% \to 56.63\%$): $\chi^2 = 1.79, p = 0.1806$ (**Honestly Disclosed: Not Significant**)
- [x] **Calibration Standard:** Standard Guo et al. (2017) confidence ECE reported honestly ($35.10\%$)

---

## 2. Target Journal Alignment Matrix

| Journal | Publisher | Impact Factor | Indexing | Focus Alignment | Readiness |
|:---|:---:|:---:|:---:|:---|:---:|
| **ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)** | ACM | 2.1 | Scopus Q1 / SCI-E | **Primary Target:** Exceptional fit for Indic languages (Hindi, Marathi, Telugu), code-mixed Hinglish, and low-resource NLP. | **100% READY** |
| **IEEE Transactions on Affective Computing (TAC)** | IEEE | 9.6 | Scopus Q1 / SCI-E | **High-Reach Target:** Premier venue for computational irony, calibrated sentiment, and selective prediction. | **100% READY** |
| **Language Resources and Evaluation (LREV)** | Springer | 2.2 | Scopus Q2 / SCI-E | **Alternative Target:** High valuation for cross-dataset benchmarking, reproducibility, and error audits. | **100% READY** |

---

## 3. Submission Manifest Verification

The following submission bundle is finalized and locked:
1. `docs/STAGE7_FINAL_MANUSCRIPT.md` (Full Manuscript)
2. `docs/STAGE7_FINAL_RESULTS.md` (Detailed Empirical Matrix)
3. `docs/STAGE7_REPRODUCIBILITY.md` (Reproducibility Protocol)
4. `docs/STAGE7_LIMITATIONS.md` (Scientific Disclosures)
5. `eval_results/stage7/` (All 7 CSV evaluation tables)
6. `CITATION.cff` (Citation Metadata)
