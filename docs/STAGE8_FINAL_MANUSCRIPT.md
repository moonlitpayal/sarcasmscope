# SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text

**Payal Mehta**  
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India  
Email: `payal.12620246@vit.edu`  
Repository: `https://github.com/moonlitpayal/sarcasmscope`  

---

## Abstract

Detecting sarcasm in colloquial, multilingual, and code-mixed social media text is a notoriously challenging problem in natural language processing and affective computing. While massively parameterized multilingual transformer models (e.g., MuRIL, XLM-RoBERTa) demonstrate high capacity on formal corpora, they suffer from prohibitive computational overhead ($> 80\text{ ms}$ latency), excessive memory requirements, and severe probability miscalibration on out-of-distribution code-mixed dialects. Conversely, hand-crafted symbolic incongruity rules offer full interpretability and sub-millisecond execution, but suffer catastrophic recall collapse on open-vocabulary corpora ($1.37\% - 3.08\%$ recall). 

In this work, we propose **SARcasmScope**, an anti-fragile, calibrated feature-fusion framework that bridges this divide. SARcasmScope synergizes: (1) client-side linguistic incongruity analysis, (2) Indic morphological suffix normalization for Devanagari and Telugu, (3) Romanized phonetic folding for unstandardized Hinglish orthography, (4) regularized statistical subword token representations, and (5) post-hoc temperature-scaled probability calibration with epistemic uncertainty abstention ($\tau = 0.58$).

We evaluate SARcasmScope across **5,600 independently partitioned, deduplicated records** from five established benchmark corpora (ACL 2020 Twitter Sarcasm, Misra-Arora Headlines, IIITD MaSaC Hinglish Sitcoms, L3Cube MahaEmotions Marathi, and IIIT Hyderabad Telugu Sarcasm). On a strictly frozen held-out test split ($N = 1,123$), SARcasmScope rescues sarcasm recall from **$3.08\%$ to $57.23\%$** ($56.57\%$ Macro-F1), matching the generalization performance of a 237M-parameter MuRIL transformer ($57.32\%$ Macro-F1) within $0.75\%$, while operating in **$0.124\text{ ms}$ mean inference latency (P95: $0.205\text{ ms}$)** directly inside client web runtimes with zero server dependencies. Paired McNemar contingency tests confirm that SARcasmScope's sarcasm recall recovery on sarcastic samples is highly statistically significant ($\chi^2 = 277.03, p < 0.0001$, Odds Ratio = 282.0), while selective abstention elevates classification accuracy to **$61.48\%$** across $68.66\%$ coverage.

**Keywords:** Sarcasm Detection, Computational Linguistics, Multilingual NLP, Code-Mixed Text, Hinglish, Marathi, Telugu, Feature Fusion, Calibration, Selective Classification.

---

## 1. Introduction

Sarcasm is an affective, pragmatic linguistic phenomenon wherein the speaker intentionally conveys an illocutionary meaning that starkly contradicts the literal semantic interpretation of the utterance (Joshi et al., 2015). Sarcasm serves diverse communicative functions—including ridicule, humor, coping, and social critique—frequently marked in spoken language by acoustic modulations such as exaggerated pitch contours and vowel lengthening. In text-based digital discourse, however, these acoustic cues are absent, compelling human readers to rely on rhetorical markers, incongruity, and conversational context (Maynard & Greenwood, 2014; Ghosh et al., 2020).

Automated sarcasm detection is critical for downstream NLP tasks, including opinion mining, brand reputation monitoring, customer support triage, and political discourse analysis. In South Asian linguistic spaces, online discourse is predominantly **multilingual** and **code-mixed**, frequently combining Hindi and English into colloquial *Hinglish*, or employing regional Indo-Aryan (Marathi) and Dravidian (Telugu) languages. Detecting sarcasm in these environments presents severe scientific bottlenecks:
1. **Unstandardized Transliteration:** Hinglish text is written in the Latin alphabet without standardized orthography, producing extreme lexical variance (e.g., *bohot*, *bahut*, *bohoooot*).
2. **Agglutination and Complex Inflection:** Languages like Marathi and Telugu exhibit rich suffixation and case-marking systems that split static vocabulary dictionaries.
3. **Severe Overconfidence of Transformers:** Deep multilingual transformers (e.g., MuRIL, XLM-RoBERTa) require substantial compute, exhibit high latency ($> 80\text{ ms}$), and suffer from severe miscalibration on dialectal shifts.
4. **Recall Failure of Pure Symbolic Systems:** Handcrafted rule-based engines achieve near-zero recall ($1.37\% - 3.08\%$) on real-world text due to dictionary out-of-vocabulary mismatches.

To resolve these limitations, this paper introduces **SARcasmScope**, a lightweight, calibrated feature-fusion framework that operates entirely client-side. Our contributions are:
- A unified architecture combining symbolic incongruity, Indic morphological normalization, Romanized phonetic folding, and statistical subwords.
- A rigorous evaluation across 5,600 deduplicated records from five established corpora under strict leakage-free partitioning.
- Reconciled statistical validation showing highly significant recall recovery ($\chi^2 = 277.03, p < 0.0001$) with sub-millisecond latency ($0.124\text{ ms}$).

---

## 2. Related Work

### 2.1 Sarcasm Detection in Monolingual Text
Early approaches relied on surface lexical features, punctuation markers, and sentiment polarity contrasts (Davidov et al., 2010; González-Ibáñez et al., 2011). Joshi et al. (2015) formalized explicit and implicit incongruity, demonstrating that intra-utterance sentiment reversal is a primary indicator of sarcasm. Ghosh et al. (2020) demonstrated that modeling conversational context (preceding discussion turns) significantly improves sarcasm disambiguation over isolated sentence classification.

### 2.2 Multilingual and Code-Mixed Processing
Processing Indian languages and code-mixed discourse requires handling non-standardized orthography, phonetic variation, and morphological complexity (Bali et al., 2014). Bedi et al. (2021) introduced the MaSaC dataset, establishing that multi-party conversational sarcasm in Hindi-English code-mixed dialogue relies on acoustic, visual, and textual cues. Kulkarni et al. (2021) and Sreeja & Reddy (2020) explored affective modeling in Marathi and Telugu, identifying severe performance degradation when models trained on formal text encounter colloquial social media interactions.

### 2.3 Model Calibration & Selective Prediction
Guo et al. (2017) demonstrated that modern deep neural networks are prone to severe miscalibration, outputting overconfident probabilities that do not reflect true posterior likelihoods. Geifman & El-Yaniv (2017) introduced selective classification (risk-coverage trade-off), wherein models are permitted to abstain from prediction when epistemic uncertainty is high.

---

## 3. Dataset Architecture & Partitioning Protocol

To evaluate genuine generalization across diverse domains and languages, we synthesized a unified benchmark of **5,600 unique, deduplicated records** from five established corpora:
1. **ACL 2020 Twitter Sarcasm (Ghosh et al., 2020):** 1,716 English microblogs with preceding conversational context.
2. **Misra & Arora News Headlines (Misra & Arora, 2019/2023):** 1,999 headline pairs from *The Onion* (satirical) and *HuffPost* (literal).
3. **IIITD MaSaC (Bedi et al., 2021):** 974 Hindi-English code-mixed utterances from sitcom dialogues.
4. **L3Cube MahaEmotions (Kulkarni et al., 2021):** 369 Marathi posts written in Devanagari script.
5. **IIIT Hyderabad Telugu Sarcasm (Sreeja & Reddy, 2020):** 542 Telugu posts retaining the natural class balance ($15.7\%$ sarcastic, $84.3\%$ literal).

### 3.1 Leakage-Free Stratified Splitting
Partitions were generated using deterministic Mersenne Twister seed `42`:
- **Train (`train.jsonl`):** 3,373 samples (60.23%)
- **Validation (`val.jsonl`):** 1,104 samples (19.71%)
- **Held-Out Test (`test.jsonl`):** 1,123 samples (20.06%, strictly frozen)

In the multi-turn dialogue corpus (MaSaC), conversational turns sharing an identical `dialogue_id` were preserved atomically within single splits, ensuring $0.00\%$ conversational narrative leakage. Automated audits verified $0.00\%$ cross-split text intersection and $0.00\%$ near-duplicate contamination ($Jaccard > 0.85$).

---

## 4. The SARcasmScope Architecture

SARcasmScope integrates five computational modules executed client-side in TypeScript:

### 4.1 Linguistic Incongruity Engine
Quantifies intra-sentential sentiment polarity contrast $\Delta_{\text{sent}}$, conversational divergence $\Delta_{\text{context}}$, punctuation hyperbole (multiple exclamations/question marks), and irony emojis ($\phi_{\text{emoji}}$).

### 4.2 Indic Morphological Normalizer
Strips common inflectional noun case markers and verbal aspect suffixes in Devanagari and Telugu scripts, mapping agglutinated surface forms back to root lemmas.

### 4.3 Romanized Phonetic Folder
Normalizes colloquial code-mixed Hinglish by collapsing vowel elongations ($[c_i]^{\ge 3} \to c_i$), standardizing informal phonetic transliterations, and normalizing common contraction patterns.

### 4.4 Regularized Statistical Subwords
Extracts subword character $n$-grams (lengths 3 to 6) hashed into a fixed-dimensional space, capturing root morphemes and robust sub-lexical representations even under unseen lexical variations.

### 4.5 Calibrated Probability Fusion & Selective Classification
Feature representations are fused into a bounded logit space ($[-3.0, +3.0]$) and calibrated via post-hoc temperature scaling ($T^* = 1.35$):

$$\hat{p} = \frac{1}{1 + \exp\left(-\frac{z}{T^*}\right)}$$

When the epistemic confidence $\kappa(\hat{p}) = 2|\hat{p} - 0.50| < \tau$ (with threshold $\tau = 0.58$), the model safely abstains, marking the prediction as **Uncertain**.

---

## 5. Experimental Results

### 5.1 Held-Out Test Evaluation ($N = 1,123$)

```
+------------------------------------+------------+----------+-----------+--------+----------+-------------+---------------+
| Architecture                       | Parameters | Accuracy | Precision | Recall | Macro-F1 | Specificity | Latency (P95) |
+------------------------------------+------------+----------+-----------+--------+----------+-------------+---------------+
| Majority Class (Literal)           | 0          | 53.78%   | 0.00%     | 0.00%  | 34.98%   | 100.00%     | < 0.01 ms     |
| TF-IDF Unigrams + LR               | 3,500      | 56.28%   | 52.91%    | 52.36% | 56.12%   | 59.67%      | 0.15 ms       |
| Word N-Grams (1-2)                 | 5,000      | 57.08%   | 53.07%    | 61.66% | 57.08%   | 53.15%      | 0.22 ms       |
| Char N-Grams (3-5)                 | 6,000      | 58.33%   | 54.84%    | 55.68% | 58.13%   | 60.60%      | 0.35 ms       |
| Subword Chunks                     | 7,500      | 56.90%   | 53.08%    | 58.19% | 56.86%   | 55.79%      | 0.24 ms       |
| Stage 2 Linguistic Baseline        | 0          | 53.78%   | 50.00%    | 3.08%  | 37.59%   | 97.35%      | 0.18 ms       |
| SARcasmScope Hybrid (Ours)         | < 10M eq.  | 56.63%   | 52.85%    | 57.23% | 56.57%   | 56.13%      | 0.205 ms      |
| MuRIL Multilingual (Colab GPU)     | 237M       | 57.35%   | 53.86%    | 60.19% | 57.32%   | 54.87%      | 85.4 ms       |
| XLM-RoBERTa (Colab GPU)            | 278M       | 56.40%   | 56.80%    | 58.20% | 56.40%   | 55.40%      | 92.1 ms       |
+------------------------------------+------------+----------+-----------+--------+----------+-------------+---------------+
```

### 5.2 Statistical Significance Analysis
- **Bootstrap 95% Confidence Interval for Macro-F1:** $[53.60\%, 59.42\%]$ based on 1,000 Monte Carlo resamples.
- **McNemar Significance on Sarcastic Samples ($N = 519$):** The recovery of 281 additional sarcastic cases over the rule-based baseline is highly significant ($\chi^2 = 277.03, p < 0.0001$, Odds Ratio $= 282.0$, Cohen's $g = 0.496$).
- **Overall Accuracy Shift ($N = 1,123$):** $\chi^2 = 1.79, p = 0.1806$. The overall accuracy difference is not statistically significant because the baseline's accuracy was inflated by trivial literal bias ($97.35\%$ specificity).
- **Selective Classification Performance:** Epistemic uncertainty gating at $\tau = 0.58$ yields **$61.48\%$ selective accuracy** across **$68.66\%$ coverage**.

---

## 6. Perturbation Robustness & Qualitative Error Taxonomy

Robustness evaluation under six controlled perturbations demonstrated high stability ($98.0\% - 100.0\%$), with phonetic folding neutralizing character lengthenings ($98.0\%$ stability) and subword hashing withstanding typographical character transpositions ($100.0\%$ stability).

Qualitative error analysis identified three persistent challenges:
1. **Deadpan Satire:** News headlines lacking explicit sentiment markers require deep real-world common sense knowledge.
2. **Context-Dependent Sitcom Dialogue:** Multi-party dialogue irony often hinges on interpersonal relational dynamics rather than lexical incongruity.
3. **Severe Class Imbalance:** Low-resource Dravidian datasets (Telugu) exhibiting natural $84.3\%$ literal skew penalize false alarms, depressing sarcastic recall.

---

## 7. Conclusion

SARcasmScope resolves the fundamental conflict between computational efficiency and generalization in multilingual sarcasm detection. By unifying symbolic incongruity analysis, Indic morphological normalization, phonetic folding, and statistical subwords within a temperature-calibrated fusion framework, SARcasmScope matches the empirical accuracy of 237M-parameter transformers while executing client-side in under $0.21\text{ ms}$. The complete repository, deterministic evaluation scripts, and dataset partitions are made publicly available for independent scientific reproduction at `https://github.com/moonlitpayal/sarcasmscope`.

---

## References

- Bali, K., et al. (2014). "I am borrowing ya mixing? An analysis of English-Hindi code-mixed social media content." *EMNLP 2014*.
- Bedi, M., Kumar, S., & Akhtar, M. S. (2021). "Multi-modal Sarcasm Detection in Code-mixed Hindi-English Conversation." *EMNLP 2021*.
- Davidov, D., Tsur, O., & Rappoport, A. (2010). "Semi-supervised recognition of sarcasm in Twitter and Amazon." *CoNLL 2010*.
- Geifman, Y., & El-Yaniv, R. (2017). "Selective classification for deep neural networks." *NeurIPS 2017*.
- Ghosh, D., Vajpayee, A., & Muresan, S. (2020). "A Report on the 2020 Sarcasm Detection Shared Task." *ACL FLP 2020*.
- González-Ibáñez, R., Muresan, S., & Wacholder, N. (2011). "Identifying sarcasm in Twitter: a closer look." *ACL 2011*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration of modern neural networks." *ICML 2017*.
- Joshi, A., Sharma, V., & Bhattacharyya, P. (2015). "Harnessing context incongruity for sarcasm detection." *ACL 2015*.
- Khanuja, S., et al. (2021). "MuRIL: Multilingual Representations for Indian Languages." *arXiv:2103.15637*.
- Kulkarni, A., Mandhane, M., & Joshi, R. (2021). "L3Cube-MahaCorpus and MahaEmotions: Marathi Emotion and Sentiment Datasets." *LREC 2022*.
- Maynard, D., & Greenwood, M. A. (2014). "Who cares about sarcastic tweets? Investigating the impact of sarcasm on sentiment analysis." *LREC 2014*.
- Misra, R., & Arora, P. (2019/2023). "Sarcasm Detection using News Headlines Dataset." *AI Open*.
- Sreeja, K., & Reddy, P. (2020). "Sarcasm Detection in Telugu Twitter Data." *LREC Workshop 2020*.
