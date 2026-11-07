# SARcasmScope — Calibrated Feature-Fusion Architecture
**Project:** SARcasmScope: A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  

---

## 1. System Overview

SARcasmScope is an anti-fragile, calibrated, context-aware sarcasm detection architecture engineered for multilingual and code-mixed digital interactions. It integrates client-side symbolic linguistic analysis with statistical subword representations and optional deep neural transformer cross-attention.

```mermaid
flowchart TD
    subgraph Input_Layer [Input Processing Layer]
        U[Target Utterance x]
        C[Dialogue Context c]
        U --> P1[Text Normalizer & Punctuation Retainer]
        C --> P2[Context Cleaner]
    end

    subgraph Feature_Extraction [Multi-Signal Feature Extraction]
        P1 --> L1[Indic Morphological Stemmer: stemIndicToken]
        P1 --> L2[Romanized Phonetic Normalizer: normalizeRomanized]
        P1 --> L3[Lexical Polarity & Incongruity Engine: extractIncongruity]
        P1 --> L4[Irony Emoji & Punctuation Analyzer: emojiFeatures]
        
        P1 & P2 --> S1[Cross-Turn Context Overlap: computeSubwordContextOverlap]
        P1 & P2 --> S2[Subword N-Gram Hash Representation: SparseClassifier]
        P1 & P2 -.-> S3[Optional Remote Transformer: MuRIL / XLM-R]
    end

    subgraph Fusion_Layer [Calibrated Feature-Fusion Engine]
        L3 & L4 --> LingProb[Linguistic Posterior P_ling]
        S2 & S3 --> SemProb[Semantic Posterior P_sem]
        L3 --> IncIdx[Incongruity Index]
        L3 --> Cov[Lexical Coverage C_x]
        
        LingProb & SemProb & IncIdx & Cov --> RidgeFusion[Regularized Ridge Probability Fusion]
        RidgeFusion --> TempScaling[Temperature Calibration: T = 1.15 - 1.50]
        TempScaling --> FusedProb[Calibrated Sarcasm Probability P_fused]
    end

    subgraph Selective_Layer [Calibrated Selective Prediction & Risk Gating]
        FusedProb --> Marg[Margin Certainty: |P_fused - 0.5| * 2]
        Cov & IncIdx --> Evid[Grounded Evidence Factor]
        Marg & Evid --> Conf[Grounded Confidence Score C]
        
        Conf --> ThreshGate{Confidence >= tau ? (tau = 0.58)}
        ThreshGate -- Yes --> Verdict[Verdict: Sarcastic or Literal]
        ThreshGate -- No --> Abstain[Verdict: Uncertain / Abstain]
    end

    subgraph Presentation_Layer [Client Application & Visualization]
        Verdict & Abstain --> UI[Interactive React / Tailwind UI]
        Verdict --> Canvas3D[Three.js Dimensional Incongruity Visualizer]
        Verdict --> XAI[Explainability & Token Attribution Panel]
    end
```

---

## 2. Mathematical Formulation

### 2.1. Client-Side Linguistic Incongruity Engine
The linguistic incongruity score captures verbal irony, semantic contrast, and typographical markers:
$$\text{Idx}_{\text{incongruity}} = \alpha \cdot \text{Clash}_{\text{polarity}} + \beta \cdot \text{Clash}_{\text{context}} + \gamma \cdot N_{\text{irony\_emoji}} + \delta \cdot \text{Flag}_{\text{rhetorical}}$$
where $\alpha=1.8, \beta=1.5, \gamma=1.2, \delta=0.8$.

Lexical coverage density $C(x)$ acts as an epistemic constraint:
$$C(x) = \frac{|\{w \in x : w \in \mathcal{V}_{\text{sentiment}} \cup \mathcal{V}_{\text{roots}}\}|}{|x|}$$

### 2.2. Regularized Feature-Fusion Formulation
The fused pre-activation logit $z_{\text{fused}}$ is computed via bounded ridge logistic regression:
$$z_{\text{fused}} = w_0 + w_1 \cdot \text{logit}(P_{\text{ling}}) + w_2 \cdot \text{logit}(P_{\text{sem}}) + w_3 \cdot \text{Idx}_{\text{incongruity}} + w_4 \cdot C(x)$$
where $\text{logit}(p) = \text{clamp}\left(\ln \frac{p}{1-p}, -3.0, +3.0\right)$.

Calibrated posterior probability is obtained via temperature scaling:
$$P_{\text{fused}}(Y=1|x, c) = \sigma\left(\frac{z_{\text{fused}}}{T^*}\right) = \frac{1}{1 + \exp\left(-\frac{z_{\text{fused}}}{T^*}\right)}$$

### 2.3. Selective Prediction & Confidence Formulation
Following Geifman & El-Yaniv (2017), the selective prediction function $f(x)$ outputs a verdict only when grounded confidence exceeds abstention threshold $\tau$:
$$g(x) = \begin{cases} 1 & \text{if } \text{Confidence}(x) \ge \tau \\ 0 & \text{if } \text{Confidence}(x) < \tau \quad (\text{Abstain}) \end{cases}$$
where:
$$\text{Confidence}(x) = 2 \cdot |P_{\text{fused}} - 0.5| \times \text{EvidenceFactor}(x, C(x))$$

---

## 3. Tiered Deployment Architecture

1. **Mode A (Standalone Client-Side):**
   - Pure in-browser execution with zero network roundtrip.
   - P95 latency $< 0.30\text{ ms}$, heap memory $< 10\text{ MB}$.
   - Runs offline using pre-bundled subword hash weights and Indic normalization rules.
2. **Mode B (Hybrid Cloud API):**
   - Calls `/api/predict-contextual` to query cloud-hosted MuRIL / XLM-R models.
   - Guarded by a 1,500 ms `AbortController` timeout that falls back gracefully to Mode A upon network failure.
