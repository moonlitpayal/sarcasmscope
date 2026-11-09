# SARcasmScope — Stage 9 Robustness Stress Testing & Error Taxonomy

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Evaluation Standard:** Adversarial Robustness & Pragmatic Error Analysis  
**Audit Date:** November 9, 2026  
**Audit Phase:** 9H — Controlled Perturbation Stress Testing  

---

## 1. Controlled 9-Perturbation Robustness Benchmark

To verify that SARcasmScope is resilient against real-world digital noise, 100 randomly sampled test instances were subjected to nine controlled perturbations (`eval_results/stage9/robustness_results.csv`):

| Perturbation Test | Representation Level Targeted | Sample Count | Stability (%) | Mean $|\Delta p|$ | Prediction Flips | Scientific Explanation |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **1. Character Transposition** | Spelling / Morphology | 100 | **100.0%** | 0.005 | 0 / 100 | Subword character $n$-grams (lengths 3-6) preserve partial matches, preventing total token breakdown. |
| **2. Repeated Vowel Elongation** | Phonetic / Orthographic | 100 | **98.0%** | 0.022 | 2 / 100 | Romanized phonetic folder collapses repeated graphemes ($[c]{3,} \to c$), neutralizing elongation. |
| **3. Emoji Removal** | Pragmatic Markers | 100 | **100.0%** | 0.000 | 0 / 100 | Core sentiment clash and subword scores maintain prediction even without visual markers. |
| **4. Punctuation Removal** | Hyperbole / Syntax | 100 | **100.0%** | 0.005 | 0 / 100 | Lexical polarity clash and subword scores dominate punctuation hyperbole. |
| **5. Romanized Spelling Variation**| Dialect / Transliteration | 100 | **99.0%** | 0.008 | 1 / 100 | Phonetic folding maps common transliteration variants to uniform phonetic keys. |
| **6. Code-Mixing Alternation** | Language Switching | 100 | **98.0%** | 0.015 | 2 / 100 | Multilingual subwords and bilingual sentiment dictionaries handle code-switched terms. |
| **7. Context Removal** | Contextual Grounding | 100 | **99.0%** | 0.007 | 1 / 100 | Utterances relying strictly on dialogue clash revert to literal prior when context is detached. |
| **8. Unseen Vocabulary Insertion** | Open Vocabulary | 100 | **99.0%** | 0.006 | 1 / 100 | Subword hashing absorbs unseen terms gracefully without causing runtime exceptions. |
| **9. Dialectal Morphological Shift** | Agglutinative Shift | 100 | **98.0%** | 0.012 | 2 / 100 | Indic suffix stripper matches canonical case families, neutralizing dialectal variation. |

---

## 2. Multi-Lingual Qualitative Error Taxonomy

Qualitative failure analysis on held-out errors revealed three persistent boundary conditions:
1. **Deadpan Satire:** News headlines lacking explicit sentiment markers (*The Onion*) require external human commonsense reasoning.
2. **Context-Dependent Sitcom Banter:** Hinglish dialogue irony often hinges on multi-speaker interpersonal dynamics rather than surface text.
3. **Dravidian Class Imbalance:** Low-resource Dravidian datasets (Telugu) exhibiting natural $84.3\%$ literal skew heavily penalize false alarms, depressing sarcastic recall.

### Epistemic Uncertainty Selective Abstention
When encountering boundary cases ($p \approx 0.50$), the calibrated uncertainty gate ($\tau = 0.58$) correctly marks the sample as **Uncertain**, achieving **$61.48\%$ selective accuracy** across **$68.66\%$ coverage**.
