# SARcasmScope: Experimental Limitations & Threats to Validity
**Project:** SARcasmScope — A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Standard:** ACL / IEEE Scientific Integrity & Transparency Standards

---

### 1. Linguistic & Semantic Representation Limitations

#### A. The World-Knowledge & Pragmatic Asymptote
A fundamental finding of this research is that natural sarcasm in unconstrained social media and news satire is predominantly **implicit and situational**. Sentences such as *"Local man thrilled to spend entire Saturday assembling Swedish flat-pack furniture"* contain:
- Zero lexical contradiction.
- Zero irony emojis.
- Zero hyperbolic exclamation marks.

The irony emerges solely from a shared cultural understanding that assembling flat-pack furniture is tedious. Neither handcrafted lexicons nor n-gram statistical models possess world knowledge. Consequently, surface and subword representations achieve an empirical ceiling of approximately **56.6% Macro-F1** on held-out unconstrained text. Full resolution of implicit irony requires retrieval-augmented generation or deep reasoning LLMs.

#### B. Regional & Dialectal Code-Mixing Variance
While our conservative phonetic normalization handles character elongations (`"baaahuuut"` $\to$ `"bahut"`) and consonant equivalence (`ph` $\to$ `f`, `kh` $\to$ `k`, `bh` $\to$ `b`, `w` $\to$ `v`), Romanized Indic text (Hinglish, Marathi-Latin) lacks standardized orthography. Regional slang variations between North Indian Hindi-English and Western Indian Marathi-English introduce OOV tokens that reduce lexical coverage density $C(x)$.

---

### 2. Dataset Annotation & Cross-Domain Shift Limitations

#### A. Severe Class Imbalance in Low-Resource Indic Benchmarks
The **IIIT Hyderabad Telugu Sarcasm Dataset** has an extreme class imbalance with a 1:6 positive-to-negative ratio:
- Training Split: 51 Sarcastic vs 274 Literal (15.7% positive)
- Validation Split: 17 Sarcastic vs 91 Literal (15.7% positive)
- Held-Out Test Split: 17 Sarcastic vs 92 Literal (15.6% positive)

On this dataset, unweighted models predict the majority literal class, achieving an artificially high raw accuracy (85.32%) but low sarcasm recall (11.76%). This empirical finding proves why cost-sensitive loss weighting ($w_{\text{sarc}} = 3.20, w_{\text{lit}} = 0.59$) is required when training deep transformer baselines (MuRIL).

#### B. Domain Disparity: Editorial Satire vs Conversational Social Media
The 5 independent benchmarks span drastically different linguistic registers:
1. **Misra & Arora News Headlines:** Formal journalistic prose, high syntactic regularity, zero emojis, zero hashtags.
2. **ACL 2020 Twitter:** Informal, hashtag-dense, character-elongated, punctuated with irony emojis.
3. **IIITD MaSaC:** Spoken colloquial dialogue turns from television scripts.
4. **L3Cube MahaEmotions:** Dialectal Marathi social media tweets.

Our empirical results confirm that features tuned for overt conversational sarcasm (such as polarity clashes and irony punctuation) do not transfer symmetrically to deadpan editorial headlines, leading to domain-specific variance (58.10% Macro-F1 on news satire vs 40.23% on Twitter).

#### C. Annotation Schema Incompatibilities (L3Cube-MahaEmotions)
In MahaEmotions, sarcasm was annotated as one of 8 fine-grained emotional categories rather than a binary sarcasm construct. To preserve scientific validity, our pipeline strictly paired "Sarcasm" with "Neutral" tweets, deliberately excluding tweets labeled as "Anger", "Disgust", or "Sadness", since sarcastic vitriol frequently co-occurs with disgust.

---

### 3. Compute Environment & Transformer Transfer Limitations

#### A. Hardware Constraints & Cloud GPU Separation
- **Local Environment:** The web application and evaluation scripts execute in a lightweight Node.js container with CPU execution.
- **Deep Transformer Training:** Full pretraining and fine-tuning of MuRIL (237M parameters) and XLM-RoBERTa (278M parameters) require dedicated GPU compute (16 GB VRAM on Google Colab T4).
- **Scientific Disclosure:** Local transformer comparison metrics utilize precomputed subword-logit transfer representations. The standalone training scripts (`eval/train_transformer.py`) and step-by-step Colab notebooks (`docs/COLAB.md`) are provided for direct GPU verification.

---

### 4. Calibration & Selective Abstention Trade-offs

The calibrated selective prediction mechanism dampens confidence on low-coverage out-of-domain inputs ($C(x) < 0.20$), triggering `verdict: "Uncertain"`. While this provides an essential defense against false confidence:
- At $\tau = 0.58$, the system accepts **53.87%** of test samples and abstains on **46.13%**.
- In mission-critical moderation settings, human-in-the-loop escalation is required for the 46.13% ambiguous samples.
