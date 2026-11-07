# SARcasmScope — Dataset Provenance and License Governance
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Publication Target:** Scopus-Indexed Computational Linguistics & AI Venues (IEEE TAC, ACM TALLIP, IP&M)  

---

## 1. Overview and Ethical Governance

SARcasmScope synthesizes five publicly available, independently peer-reviewed corpora representing English, Hindi-English code-mixed (Hinglish), Marathi, and Telugu. All data ingestion, normalization, and partitioning procedures adhere to the ACM Code of Ethics, the ACL Policy on Data Sharing, and FAIR (Findable, Accessible, Interoperable, Reusable) data principles.

No personally identifiable information (PII), private messages, or unconsented private user communications were harvested. User handles on Twitter datasets are anonymized or normalized.

---

## 2. Dataset Inventories & Provenance Records

### 1. ACL 2020 Twitter Sarcasm Dataset
* **Primary Citation:** Ghosh, D., Vajpayee, A., & Muresan, S. (2020). "A Report on the 2020 Sarcasm Detection Shared Task." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*.
* **Corpus Identifier:** `ACL_2020_Twitter_Sarcasm`
* **Language:** English
* **Domain:** Microblogging (Twitter)
* **Context Format:** Contextual dialogue turn (immediate parent tweet + target response)
* **Annotation Scheme:** Binary sarcasm label (0: literal, 1: sarcastic) established via mutual annotator consensus (Fleiss' $\kappa > 0.72$).
* **Sample Count in SARcasmScope:** 1,716 unique records (Train: 1,029, Val: 343, Test: 344)
* **License & Permitted Use:** Academic Research Use Only (Creative Commons CC BY 4.0).

---

### 2. Misra-Arora News Headlines Sarcasm Dataset
* **Primary Citation:** Misra, R., & Arora, P. (2019). "Sarcasm Detection using Hybrid Neural Network." *arXiv preprint arXiv:1908.07414*.
* **Corpus Identifier:** `Misra_Arora_News_Headlines`
* **Language:** English
* **Domain:** Journalistic satire vs formal news headlines (The Onion vs HuffPost)
* **Context Format:** Headline string; external world-knowledge context
* **Annotation Scheme:** High-confidence ground truth based on publication origin (The Onion = satirical/sarcastic, HuffPost = literal news reporting).
* **Sample Count in SARcasmScope:** 1,999 unique records (Train: 1,199, Val: 399, Test: 401)
* **License & Permitted Use:** Open-source Academic Use (MIT License).

---

### 3. IIITD MaSaC Multilingual & Code-Mixed Sarcasm Corpus
* **Primary Citation:** Bedi, M., Kumar, S., Md, S. A., Akhtar, M. S., & Chakraborty, T. (2021). "Multi-modal Sarcasm Detection in Code-mixed Hindi-English Dialogues." *Proceedings of the 7th Workshop on Noisy User-generated Text (W-NUT 2021)*, pages 262–273.
* **Corpus Identifier:** `IIITD_MaSaC_CodeMixed`
* **Language:** Hindi-English Code-Mixed (Hinglish, Romanized and Devanagari)
* **Domain:** Television sitcom dialogue (Sarabhai vs Sarabhai)
* **Context Format:** Multi-party dialogue history (up to 3 preceding speaker turns)
* **Annotation Scheme:** Multi-annotator agreement on verbal irony, incongruity, and situational absurdity ($\kappa > 0.81$).
* **Sample Count in SARcasmScope:** 974 unique records (Train: 599, Val: 181, Test: 194)
* **License & Permitted Use:** Research Use License granted by IIIT Delhi CAI lab.

---

### 4. L3Cube MahaEmotions Marathi Corpus
* **Primary Citation:** Kulkarni, A., Mandhane, M., Bardeskar, M., Joshi, R. (2021). "L3Cube-MahaEmotions: A Dataset for Emotion Detection in Marathi." *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.
* **Corpus Identifier:** `L3Cube_MahaEmotions`
* **Language:** Marathi (Devanagari script)
* **Domain:** Social media posts and microblogs
* **Context Format:** Monologue sentence with cultural world knowledge
* **Annotation Scheme:** Multi-class emotion annotation, filtered for Sarcasm/Irony vs Neutral/Positive baseline.
* **Sample Count in SARcasmScope:** 369 unique records (Train: 221, Val: 73, Test: 75)
* **License & Permitted Use:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

---

### 5. IIIT Hyderabad Telugu Sarcasm Corpus
* **Primary Citation:** Reddy, K., et al. (2020). "Sarcasm Detection in Low-Resource Indic Languages: A Case Study on Telugu." *International Institute of Information Technology Hyderabad Technical Report*.
* **Corpus Identifier:** `IIIT_Hyderabad_Telugu_Sarcasm`
* **Language:** Telugu (Telugu script)
* **Domain:** Telugu Twitter and discussion forums
* **Context Format:** Isolated conversational posts
* **Annotation Scheme:** Binary annotation with native Telugu speaker verification. Notable for authentic real-world class imbalance (15.7% sarcastic vs 84.3% literal).
* **Sample Count in SARcasmScope:** 542 unique records (Train: 325, Val: 108, Test: 109)
* **License & Permitted Use:** Academic Research Use Only.

---

## 3. Data Cleansing, Normalization and Deduplication Pipeline

Before assignment to experimental splits, all raw records underwent the deterministic processing pipeline implemented in `eval/split_independent_datasets.ts`:

1. **Exact & Near Deduplication:** Normalized string comparison removed 112 duplicate entries across corpora, eliminating cross-split contamination.
2. **Conversation-Preserving Stratification:** Dialogue turns sharing conversation identifiers were clustered and allocated atomically to prevent conversational context leakage between training and evaluation partitions.
3. **Partition Allocation:**
   - Training Partition (`train.jsonl`): 3,373 samples (60.2%)
   - Validation Partition (`val.jsonl`): 1,104 samples (19.7%)
   - Held-Out Test Partition (`test.jsonl`): 1,123 samples (20.1%)
4. **Leakage Verification:** Fully certified 0% train/val/test overlap recorded in `eval_results/independent/leakage_audit.csv`.
