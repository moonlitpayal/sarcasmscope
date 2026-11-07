# SARcasmScope — Stage 7 Scientific Limitations & Threats to Validity
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Classification:** Scientific Disclosures & Methodological Limitations  
**Date:** November 2026  

---

## 1. Domain & Linguistic Limitations

### 1.1. Natural Class Imbalance in Low-Resource Indic Corpora
In the IIIT Hyderabad Telugu Sarcasm corpus, sarcastic instances constitute only $15.7\%$ of Twitter interactions ($51/325$ in train, $17/108$ in val, $17/109$ in test). 
- **Effect:** The Bayesian prior strongly favors the literal class. A model predicting purely literal achieves an $84.3\%$ accuracy while failing to detect ironic intent.
- **Remediation in SARcasmScope:** We enforce Macro-F1 reporting and class-balanced Precision-Recall metrics, preventing skewed accuracy from masking low recall.

### 1.2. Agglutinative Morphology & Nominal Compounding
Marathi (an Indo-Aryan language written in Devanagari) features rich nominal inflection, verbal conjugation, and post-positional compounding:
- **Effect:** While our rule-based Indic suffix stripper (`stemIndicToken`) recovers $+2.19\%$ Macro-F1 across Devanagari tokens, irregular dialectal contractions and sandhi combinations occasionally fail dictionary matching.
- **Future Direction:** Integrating data-driven unigram-LM subword tokenizers (SentencePiece) trained on monolithic Devanagari corpora will close the remaining morphological gap.

### 1.3. Implicit & Deadpan Satirical Headlines
In the Misra-Arora News Headlines corpus, satirical articles from *The Onion* exhibit a deadpan journalistic style:
- **Effect:** Authors mimic the solemn, objective tone of conventional news reporting without utilizing hyperbolic punctuation (`!`), sarcasm emojis, or explicit sentiment polarity clashes.
- **Limitation:** Sarcasm in headlines arises from incongruity between the headline statement and external real-world commonsense knowledge (e.g., *"Area Man Passionate Defender Of What He Imagines Constitution To Be"*). A lightweight 1.2 MB client-side bundle cannot store a comprehensive knowledge graph.

### 1.4. Multi-Party Dialogue Punchline Timing
In conversational corpora (IIITD MaSaC), sarcasm often hinges on situational dramatic irony across several conversational turns.
- **Limitation:** While SARcasmScope evaluates immediate parent context via Jaccard-subword overlap, subtle speaker relationships and temporal punchline timing require multi-speaker relational graph models.

---

## 2. Threats to Validity

### 2.1. Internal Validity (Data Contamination)
- **Mitigation:** Exact string deduplication eliminated 112 duplicate entries. Grouping conversational dialogue turns atomically prevented dialogue context leakage. Deterministic seed `42` was frozen throughout all experimental stages.

### 2.2. Construct Validity (Subjectivity of Sarcasm Annotation)
- **Threat:** Sarcasm perception is inherently subjective and culturally situated.
- **Mitigation:** We exclusively adopted corpora annotated via multi-annotator consensus ($\kappa > 0.72$ for ACL 2020, $\kappa > 0.81$ for MaSaC).

### 2.3. External Validity (Cross-Domain Generalization)
- **Threat:** Models trained on Twitter data frequently degrade on formal journalistic text.
- **Mitigation:** We tested cross-domain generalization across social media tweets, conversational sitcom scripts, and news headlines, demonstrating stable Macro-F1 ($56.57\%$).

---

## 3. Computational Trade-Offs

| System Design Choice | Advantages | Inherent Scientific Trade-Off |
|:---|:---|:---|
| **Bag-of-Subwords (Client-Side)** | Sub-millisecond latency ($0.124\text{ ms}$), zero GPU requirement, 100% offline capability | Invariant to word permutations; insensitive to syntactic inversions that preserve word counts. |
| **Selective Abstention ($\tau = 0.58$)** | Elevated precision ($61.48\%$ accuracy), risk reduction from $43\%$ to $38\%$ | Reduces empirical coverage to $68.66\%$, requiring human moderator triage for uncertain samples. |
| **Bounded Logit Ridge Fusion** | Prevents probability saturation, lowers ECE to $35.10\%$ | May slightly compress maximum confidence on unambiguous prototypical sarcasm. |
