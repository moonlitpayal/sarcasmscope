# SARcasmScope: Technology Stack, Academic References & Corpus Documentation

**Author:** Payal Mehta (`payal.12620246@vit.edu`)  
**Framework:** React 19 + TypeScript 5.7 + Vite 6 + Tailwind CSS v4 + Three.js r174

---

## 1. Software Stack & Actual Modules

- **UI Layer (`src/components/`):** `StudioView.tsx`, `BatchView.tsx`, `RobustnessView.tsx`, `ProtocolView.tsx`, `QuantumOrb.tsx` mounted in `src/App.tsx`.
- **Feature-Fusion Engine (`src/engine/`):**
  - `sarcasmEngine.ts`: `stemIndicToken` (Devanagari/Telugu), `normalizeRomanized` (Hinglish folding), `analyzeCodeMixing` (CMI), `extractIncongruity` (`0.40*clash + 0.35*ctxClash + 0.15*min(contrast,2) + 0.10*min(inten,2)`), `prevPredict`, `predict`.
  - `textFeatures.ts`: Bit-identical extractors (`extractWordFeatures`, `extractCharFeatures`), `boundedLogit`, `stableSigmoid`, Mulberry32 PRNG.
  - `statisticalModel.ts`: Level-1 sparse experts (`WordLex_1_2` $V=8,000$; `CharSub_3_5` $V=8,000$).
  - `trainedModelData.ts`: Serialized `v2.0.0-oof-calibrated` weights (`405.36 KB`, `16,006` params).
  - `hybridFusion.ts`: 3-Stage Calibrated Stacked Feature-Fusion (`T* = 0.845`, `tau* = 0.35`, `predictHybrid`, `computeHybridOcclusion`, `perturbHybridAll`, `calculateECE`).
  - `contextualEngine.ts`: Mode A offline execution + Mode B `/api/predict-contextual` `1,500 ms` fallback adapter.

---

## 2. Scientific Methodology & Citations

- **Stacked Generalization & Fusion:** Wolpert (1992); Snoek et al. (ACM MM 2005). 5-fold group-stratified train OOF MAP Ridge logistic meta-learner (`w0=0.3483, w1_word=0.54817, w2_char=0.54648, w3_linguistic=0.26486, w4_incongruity=0.19418, w5_coverage=0.20441`).
- **Post-Hoc Calibration:** Guo et al. (ICML 2017) scalar temperature scaling $P_{\text{fused}} = \sigma(z_{\text{raw}}/T^*)$ with $T^* = 0.845$ fitted strictly on `val.jsonl` ($N=1,104$), reducing test Conf-ECE to **`1.36%`**.
- **Selective Prediction:** Geifman & El-Yaniv (NeurIPS 2017) margin confidence $2|P_{\text{fused}}-0.5|$ gated at $\tau^* = 0.35$ (`80.31%` selective acc across `45.68%` coverage).
- **LOTO Attribution & CMI:** Zeiler & Fergus (2014); Ribeiro et al. (KDD 2016); Gambäck & Das (LREC 2014/2016).
- **External Transformer Scripts (Google Colab GPU Protocol):** Khanuja et al. (2021) MuRIL; Conneau et al. (2020) XLM-RoBERTa (`eval/train_transformer.py`, `NOT_EXECUTED_LOCALLY`).

---

## 3. Authenticated Corpora (`5,600` Records, `3,373 / 1,104 / 1,123`)

1. `ACL_2020_Twitter_Sarcasm` ($1,716$) — Ghosh et al. (2020), *FigLang / ACL* — **Research-Only (Twitter Terms)**
2. `Misra_Arora_News_Headlines` ($1,999$) — Misra & Arora (2019/2023), *AI Open* — **Public Domain / CC0**
3. `IIITD_MaSaC_CodeMixed` ($974$) — Bedi et al. (2021), *IEEE TAFFC* ($\kappa=0.81$) — **Research Purpose Only** (`13` near-dup pairs / `5` train-test)
4. `L3Cube_MahaEmotions` ($369$) — Kulkarni et al. (2023); Kowtal & Joshi (2025), *PACLIC 39* — **CC-BY-4.0**
5. `IIIT_Hyderabad_Telugu_Sarcasm` ($542$) — Marreddy et al. (2022), *LREC / ACM TALLIP* — **Research Only** (`15.7%` sarcastic natural skew)
