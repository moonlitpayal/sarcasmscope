# SARcasmScope (React Edition)

A high-fidelity, calibrated, and explainable **sarcasm and incongruity detection suite** for multilingual and code-mixed Indian-language text (Hindi, Marathi, Hinglish, Telugu, Tamil, Bengali, and English).

Rewritten as a modern React application with TypeScript, Vite, Tailwind CSS, Three.js 3D ambient visualization, and client-side cognitive NLP intelligence.

---

## Key Features

1. **Cognitive Studio Analyzer**
   - Multilingual detection across English, Hindi, Hinglish, and Marathi text.
   - Dual-input support: **Target Statement** and optional **Conversational Context** (discourse anchoring).
   - Incongruity Engine:
     - Intra-sentence opposition (positive surface cues vs negative reality cues).
     - Contextual clash (surface praise opposing preceding failure discourse).
     - Hyperbolic amplifier / intensifier detection.
     - Contrast marker detection.
     - Quantitative Incongruity Index.
   - Script & Code-Mixing Analysis (CMI):
     - Unicode script identification (Devanagari, Latin, Bengali, Gurmukhi, Tamil, Telugu, etc.).
     - Script Code-Mix Index (CMI %) calculation.
     - Romanized Indic language heuristic (Hinglish, Marathi-English, Telugu-English).
   - Emoji Sentiment Breakdown:
     - Emoji extraction and classification (irony emojis: 🙄, 😏, 😒, 🙃, 🤡, 😑, 😬; positive and negative emojis).

2. **Calibrated Confidence & Selective Abstention**
   - Post-hoc Platt / Temperature Scaling ($T \in [0.2, 2.5]$).
   - Selective Abstention Threshold ($\tau \in [0.0, 0.95]$): instead of hallucinating a binary prediction when confidence is low, the engine outputs an explicit **"Uncertain"** verdict for human review.

3. **Explainability via Occlusion (Leave-One-Token-Out)**
   - Model-agnostic token attributions: measures the prediction shift ($\Delta = P - P_{-i}$) when each individual word is omitted.
   - Interactive visual highlights: Rose/Red badges push toward sarcasm ($\Delta > 0$), Sky/Blue badges push toward literal ($\Delta < 0$).
   - Token inspector showing exact contribution values and cue classifications.

4. **Controlled Perturbation & Noise Robustness Suite**
   - Controlled noise benchmarks testing resilience against social media anomalies:
     - Indic grapheme-cluster preserving character swap/duplication/drop.
     - Letter elongation (e.g., *soooo*, *wahhh*).
     - Emoji stripping.
     - Punctuation removal.
     - Hashtag and mention stripping.
     - Sentence truncation.
     - Word order shuffling.
     - Uppercase transformation.
   - Real-time Stability Score (%) and verdict flip detection.

5. **Batch Corpus Evaluator**
   - Multi-sentence batch evaluation (up to 200 lines).
   - Summary statistics (Sarcastic %, Literal %, Uncertain %, Average Confidence).
   - Filter by verdict, search by keyword, and export results directly to CSV.

6. **Interactive 3D Ambient Visualizer**
   - Three.js WebGL canvas featuring interactive orbital quantum rings and celestial particles that dynamically adapt to analysis verdicts and cursor interaction.

---

## Running Locally

```bash
# Install dependencies
npm install

# Start the dev server on port 3000
npm run dev

# Build for production
npm run build
```
