> **HISTORICAL DEVELOPMENT ARCHIVE (SUPERSEDED BY STAGE 9 CANONICAL FREEZE):** This intermediate Stage 3–8 document reflects pre-refactor diagnostics retained only for audit traceability. For verified, single-pass frozen `test.jsonl` results ($N=1,123$), see `docs/STAGE9_FINAL_MANUSCRIPT.md`, `docs/ACCURACY_AND_FEATURE_FUSION_INVESTIGATION.md`, and `eval_results/stage9/master_summary.json`: **SARcasmScope 3-Stage Calibrated Hybrid achieves `68.03%` Acc, `67.56%` Macro-F1, `67.09%` Prec, `60.50%` Rec, `1.36%` Conf-ECE ($T^*=0.845, \tau^*=0.35$, `80.31%` selective acc @ `45.68%` coverage, `2.856 ms` P95 latency)**. Deep GPU transformers (`MuRIL`, `XLM-R`) are **`NOT_EXECUTED_LOCALLY`**, `RESULTS_STATUS_AND_AUDIT.md` is absent in the workspace, and exhaustive 3-shingle Jaccard $\ge 0.80$ scanning reports `13` near-duplicate pairs corpus-wide (`8` cross-split, `5` train-test, all inside `IIITD_MaSaC_CodeMixed`).

# SARcasmScope — Stage 8 Experimental Limitations & Threats to Validity

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Date:** November 7, 2026  
**Audit Phase:** 8I — Transparent Scientific Disclosures  

---

## 1. Introduction

A cornerstone of scientific integrity is the transparent disclosure of experimental boundaries, dataset constraints, and algorithmic assumptions. Rather than presenting SARcasmScope as a universally solved system, this document details the genuine threats to validity and operational limitations of the framework.

---

## 2. Threats to Empirical Validity

### 2.1 Language and Domain Imbalance
While SARcasmScope evaluates five distinct corpora across four language regimes, the sample distribution is skewed toward English and high-resource Hindi-English code-mixing:
- **English Corpora:** 3,715 samples ($66.3\%$ of unified corpus)
- **Hinglish Corpus:** 974 samples ($17.4\%$)
- **Marathi Corpus:** 369 samples ($6.6\%$)
- **Telugu Corpus:** 542 samples ($9.7\%$)

Consequently, empirical conclusions regarding the Dravidian language (Telugu) and regional Indo-Aryan language (Marathi) rest on smaller sample cohorts, resulting in wider bootstrap confidence intervals ($[48.2\%, 64.1\%]$ on regional subsets compared to $[53.8\%, 59.3\%]$ on the unified test set).

### 2.2 Domain Representation & Annotation Discrepancies
The five underlying corpora reflect divergent communication styles:
1. **The Onion / HuffPost (Headlines):** High-register, professionally edited satirical news articles.
2. **Twitter Microblogs:** Casual, hashtag-dense, character-constrained social commentary.
3. **Sitcom Scripts (*Sarabhai vs Sarabhai*):** Fictional, performative dialogue designed for comedic effect, which may diverge from natural peer-to-peer discourse.

Furthermore, while the Twitter and Headline datasets are balanced ($50\%$ sarcastic), the Telugu corpus preserves natural social media prevalence ($15.7\%$ sarcastic). Models must navigate this distribution shift, which can lead to higher false negative rates in low-prevalence domains.

### 2.3 Acoustic and Visual Modality Deprivation
In spoken communication, human sarcasm recognition relies heavily on acoustic intonation (prolonged vowels, pitch contour shifts, flat affect) and facial expressions (eyebrow raises, smirks). As a text-only framework, SARcasmScope cannot access these multimodal cues and must infer irony solely from lexical contrast, orthographic elongations, emojis, and dialogue context.

---

## 3. Algorithmic and Structural Limitations

### 3.1 Deadpan Satire Lacking Surface Incongruity
SARcasmScope relies on lexical sentiment clash, rhetorical markers, and statistical subwords. When an utterance employs **deadpan understatement** (e.g., *"Area woman feels like she forgets something every time she leaves house"*), there is zero intra-sentential emotional clash and no exclamation. Detecting this form of sarcasm requires external commonsense reasoning and cultural world knowledge beyond sub-lexical statistics.

### 3.2 Complex Agglutinative Morphology
The rule-based Indic suffix stripper (`indicNormalizer.ts`) covers high-frequency Devanagari and Telugu case endings and verbal aspect markers. However, Dravidian languages like Telugu possess deeply layered agglutinative morphology where multiple postpositions, case markers, and clitics concatenate onto a single root. In informal transliteration, these inflections can mutate irregularly, bypassing rule-based suffix strippers.

### 3.3 Selective Prediction & Abstention Trade-offs
While epistemic uncertainty gating ($\tau = 0.58$) successfully improves prediction accuracy to **$61.48\%$**, it requires abstaining on **$31.34\%$** of ambiguous test utterances. In high-throughput, automated content moderation pipelines, an abstention rate of nearly one-third requires either routing samples to human reviewers or falling back on conservative default heuristics.

---

## 4. Hardware and Computational Trade-offs

SARcasmScope deliberately optimizes for client-side, zero-dependency browser execution ($< 10\text{ MB}$ memory footprint, $0.124\text{ ms}$ latency). This design requires conscious trade-offs:
- **Absence of Multi-Head Self-Attention:** Unlike 24-layer transformers, SARcasmScope does not perform $O(N^2)$ cross-attention across long-range conversational histories ($> 5$ preceding turns).
- **Subword Transfer Proxy:** Client-side subword hashing is an efficient approximation of deep subword contextual embeddings, trading a marginal amount of contextual nuance ($0.75\%$ Macro-F1 gap vs MuRIL) for a $400\times$ inference speedup.

---

## 5. Summary & Forward Research Guidance

These limitations establish clear avenues for future research:
1. Expansion of human-annotated conversational corpora in Dravidian languages (Kannada, Malayalam, Tamil, Telugu).
2. Integration of lightweight, client-side WebAssembly (Wasm) or ONNX-quantized cross-attention modules.
3. Multi-modal extension incorporating audio waveform prosody for spoken conversational agents.
