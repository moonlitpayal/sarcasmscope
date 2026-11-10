# SARcasmScope: Technology Stack, Academic References & Dataset Documentation
**Author / Maintainer:** Payal Mehta (`payal.12620246@vit.edu`)  
**Framework Version:** React 19 + TypeScript 5.7 + Vite 6 + Tailwind CSS v4

---

## 1. Software Technologies & Source Layout

| Layer / File | Stack / Technology | Role in SARcasmScope |
|:---|:---|:---|
| `src/App.tsx` | React `^19.0.0`, Lucide Icons | Root navigation across Studio, Robustness, Batch, and Protocol views with live $T$ and $\tau$ state. |
| `src/components/StudioView.tsx` | React + Tailwind CSS `^4.0.0` | Interactive sarcasm & context analyzer, calibrated bar, verdict badge, and LOTO occlusion breakdown. |
| `src/components/RobustnessView.tsx` | React + Tailwind | Interactive 8-perturbation client stress tester (`perturbHybridAll`). |
| `src/components/BatchView.tsx` | React + Tailwind | Multi-utterance batch processor with RFC 4180 CSV export. |
| `src/components/ProtocolView.tsx` | React + Tailwind | Live temperature ($T$) and abstention ($\tau$) controls + lexicon/protocol viewer. |
| `src/components/QuantumOrb.tsx` | Three.js `^0.174.0` | Interactive 3D orbital canvas responder. |
| `src/engine/sarcasmEngine.ts` | TypeScript | Stage 2 symbolic engine (`stemIndicToken`, `normalizeRomanized`, `analyzeCodeMixing`, `extractIncongruity`, `predict`, `prevPredict`). |
| `src/engine/textFeatures.ts` | TypeScript | Shared extractors (`extractWordFeatures`, `extractCharFeatures`), `boundedLogit`, `stableSigmoid`, Mulberry32 PRNG. |
| `src/engine/statisticalModel.ts` | TypeScript | Level-1 sparse TF-IDF logistic experts (`WordLex_1_2`, `CharSub_3_5`). |
| `src/engine/trainedModelData.ts` | TypeScript (`405.36 KB`) | Serialized `v2.0.0-oof-calibrated` weights, IDFs, MAP Ridge meta-weights, $T^*=0.845, \tau^*=0.35$. |
| `src/engine/hybridFusion.ts` | TypeScript | 3-stage calibrated feature-fusion (`predictHybrid`, `computeHybridOcclusion`, `calculateECE`). |
| `src/engine/contextualEngine.ts` | TypeScript | Mode A offline client engine + Mode B `/api/predict-contextual` fallback adapter. |

---

## 2. Verified Scientific & Mathematical References

### 2.1 Feature Fusion, Calibration, Selective Prediction, Explainability & Code-Mixing Metrics
1. **Wolpert, D. H. (1992).** *Stacked Generalization.* **Neural Networks**, 5(2), 241–259. DOI: `10.1016/S0893-6080(05)80023-1`. *(Level-2 MAP Ridge out-of-fold meta-stacking foundation.)*
2. **Snoek, C. G. M., Worring, M., & Smeulders, A. W. M. (2005).** *Early versus Late Fusion in Semantic Video Analysis.* **Proceedings of the 13th Annual ACM International Conference on Multimedia (ACM MM 2005)**, pp. 399–402. DOI: `10.1145/1101149.1101236`.
3. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* **Proceedings of the 34th International Conference on Machine Learning (ICML 2017)**, PMLR Vol. 70, pp. 1321–1330. *(Fits $T^* = 0.845$ strictly on `val.jsonl`, $N=1,104$; cuts sealed test Conf-ECE `4.99% -> 1.36%`.)*
4. **Geifman, Y., & El-Yaniv, R. (2017).** *Selective Classification for Deep Neural Networks.* **Advances in Neural Information Processing Systems 30 (NIPS / NeurIPS 2017)**, pp. 4878–4887. *(Validation-chosen $\tau^* = 0.35$ risk-coverage abstention yielding `80.31%` selective accuracy @ `45.68%` coverage on sealed test.)*
5. **Joshi, A., Sharma, V., & Bhattacharyya, P. (2015).** *Harnessing Context Incongruity for Sarcasm Detection.* **Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL 2015)**, Short Papers, pp. 757–762. ACL Anthology: `P15-2124`.
6. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* **Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)**, pp. 1135–1144. DOI: `10.1145/2939672.2939778`. *(Leave-One-Token-Out attribution motivation alongside Zeiler & Fergus.)*
7. **Zeiler, M. D., & Fergus, R. (2014).** *Visualizing and Understanding Convolutional Networks.* **Computer Vision – ECCV 2014**, Lecture Notes in Computer Science (LNCS), Vol. 8689, pp. 818–833. *(Occlusion sensitivity $\Delta_i = P(S) - P(S_{\setminus i})$.)*
8. **Gambäck, B., & Das, A. (2014 / 2016).** *On Measuring the Complexity of Code-Mixing.* **Proceedings of the 11th International Conference on Natural Language Processing (ICON 2014)**; extended in **Gambäck, B., & Das, A. (2016).** *Comparing the Level of Code-Switching in Corpora.* **Proceedings of LREC 2016**, pp. 1850–1855. ACL Anthology: `L16-1292`. *(Code-Mixing Index, CMI.)*
9. **Cohen, J. (1960).** *A Coefficient of Agreement for Nominal Scales.* **Educational and Psychological Measurement**, 20(1), 37–46. DOI: `10.1177/001316446002000104`. *(Inter-annotator $\kappa$.)*
10. **Krippendorff, K. (2011).** *Computing Krippendorff's Alpha-Reliability.* University of Pennsylvania ScholarlyCommons, Technical Report (`https://repository.upenn.edu/asc_papers/43`).
11. **Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., Dey, A., Gopalan, B., Margam, D. K., Aggarwal, P., Nagipogu, R. T., Dave, S., Gupta, S., Gali, S. C. B., Subramanian, V., & Talukdar, P. (2021).** *MuRIL: Multilingual Representations for Indian Languages.* arXiv preprint `arXiv:2103.10730`. *(External GPU script reference only; `NOT_EXECUTED_LOCALLY`.)*
12. **Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020).** *Unsupervised Cross-lingual Representation Learning at Scale.* **Proceedings of ACL 2020**, pp. 8440–8451. ACL Anthology: `2020.acl-main.747`. *(External GPU script reference only; `NOT_EXECUTED_LOCALLY`.)*

---

## 3. Verified Dataset References (`5,600` Records)
1. **ACL 2020 Twitter Sarcasm (`1,716`):** **Ghosh, D., Vajpayee, A., & Muresan, S. (2020).** *A Report on the 2020 Sarcasm Detection Shared Task.* **Proceedings of the Second Workshop on Figurative Language Processing (FigLang 2020 at ACL 2020)**, pp. 1–11. DOI: `10.18653/v1/2020.figlang-1.1` (ACL Anthology: `2020.figlang-1.1`). License: Research-Only (Twitter Terms).
2. **Misra-Arora News Headlines (`1,999`):** **Misra, R., & Arora, P. (2023).** *Sarcasm detection using news headlines dataset.* **AI Open**, Vol. 4, pp. 13–18. DOI: `10.1016/j.aiopen.2023.01.001` (v1.0 dataset released 2019). License: Public Domain / CC0.
3. **IIITD MaSaC Code-Mixed (`974`):** **Bedi, M., Kumar, S., Akhtar, M. S., & Chakraborty, T. (2023).** *Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations.* **IEEE Transactions on Affective Computing**, Vol. 14, No. 2, pp. 1363–1375 (Online Early Access 2021). DOI: `10.1109/TAFFC.2021.3083522`. License: Research Purpose Only.
4. **L3Cube MahaEmotions (`369`):** **Kowtal, N., & Joshi, R. (2025).** *L3Cube-MahaEmotions: A Marathi Emotion Recognition Dataset with Synthetic Annotations using CoTR prompting and Large Language Models.* **Proceedings of the 39th Pacific Asia Conference on Language, Information and Computation (PACLIC 39)**, pp. 419–427. ACL Anthology: `2025.paclic-1.38` (`arXiv:2506.00863`, DOI: `10.48550/arXiv.2506.00863`). Built on **MahaSent:** **Kulkarni, A., Mandhane, M., Likhitkar, M., Kshirsagar, G., & Joshi, R. (2021).** *L3CubeMahaSent: A Marathi Tweet-based Sentiment Analysis Dataset.* **Proceedings of WASSA 2021**, pp. 213–220 (`2021.wassa-1.23`, DOI: `10.48550/arXiv.2103.11408`). License: CC-BY-4.0.
5. **IIIT Hyderabad Telugu Sarcasm (`542`):** **Marreddy, M., Oota, S. R., Vakada, L. S., Chinni, V. C., & Mamidi, R. (2023).** *Am I a Resource-Poor Language? Data Sets, Embeddings, Models and Analysis for four different NLP Tasks in Telugu Language.* **ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)**, Vol. 22, Issue 1, Article 18, pp. 1–34. DOI: `10.1145/3531535` (dataset release `mounikaiiith/Telugu-Sarcasm`). License: Research Only.

---

## 4. Authorship & Permissions Reconciliation Sign-Off Template (Pre-Submission Blocker)

*Journal upload requires confirming exact order, CRediT roles, corresponding author, and written approval across all five manuscript co-authors:*

| Author Order | Full Name & Affiliation | CRediT Contribution Roles | Corresponding Author? | Written Approval / Permission |
|:---:|:---|:---|:---:|:---:|
| 1 | *[Confirm Position: e.g., Payal Mehta, VIT Vellore]* | Conceptualization, Software, Investigation, Formal Analysis, Validation, Writing | *TBD* | Pending co-author confirmation |
| 2 | *[Co-Author 2 — From original `.docx`]* | *[Confirm CRediT roles]* | *TBD* | Pending |
| 3 | *[Co-Author 3 — From original `.docx`]* | *[Confirm CRediT roles]* | *TBD* | Pending |
| 4 | *[Co-Author 4 — From original `.docx`]* | *[Confirm CRediT roles]* | *TBD* | Pending |
| 5 | *[Faculty Advisor / Collaborator — Original Corresponding Author]* | Supervision, Methodology, Validation, Review & Editing | *Original Draft Designee* | Pending |
