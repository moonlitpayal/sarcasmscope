# SARcasmScope: Experimental Limitations & Threats to Validity
**Project:** SARcasmScope — A Calibrated Feature-Fusion Framework for Context-Aware Sarcasm Detection in Multilingual and Code-Mixed Text  
**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)

---

### 1. Telugu Class-Imbalance & Global Decision Threshold Limitation
- On `IIIT_Hyderabad_Telugu_Sarcasm` ($N=109$ in `test.jsonl`, natural prevalence `15.60%` sarcastic = 17/109), the calibrated hybrid achieves strong probabilistic discrimination (**`0.7455` ROC-AUC**, **`0.1148` Brier Score**, **`0.3797` NLL**, **`5.77%` Conf-ECE**) yet records **`0.00%` sarcasm recall** (`84.40%` accuracy, `45.77%` Macro-F1) at the global `0.50` classification cutoff.
- Because the model is well-calibrated under an 84.4% literal prior, true sarcastic Telugu posts receive elevated relative probabilities (top quartile ranking) without exceeding `P >= 0.50`. Notably, standalone `WordLex_1_2` triggers `11.76%` recall (`55.21%` Macro-F1) on Telugu, indicating that class-prior logit offsetting or language-specific operating thresholds are needed for skewed Dravidian deployments.

---

### 2. Hindi-English Code-Mixed Performance & Cross-Split Near-Duplicates
- On `IIITD_MaSaC_CodeMixed` ($N=194$ in `test.jsonl`), overall test Macro-F1 rises from `32.64%` (`v2`) to **`47.33%`** (Hybrid) and **`54.52%`** (`WordLex_1_2` alone), leaving an open generalization gap driven by deadpan TV banter (`15.57%` Conf-ECE on `IIITD_MaSaC`).
- Exhaustive pairwise word 3-shingle Jaccard ($\ge 0.80$) auditing confirms **0** exact text overlaps and **0** `conversation_id` overlaps corpus-wide, but uncovers **13 near-duplicate pairs corpus-wide** (`8` cross-split, `5` train-test, `8` label-conflicting)—all confined to brief recurring interjections inside `IIITD_MaSaC_CodeMixed` (such as short one-line replies carrying context-dependent opposite labels across different sitcom episodes).

---

### 3. Vowel Elongation & Context Sensitivity in Stress Tests
- On the deterministic 100-sample `testRecords` stress suite (`seed=42`, `eval_results/stage9/robustness_results.csv`):
  - **Repeated Vowel Elongation:** Stability is **`75.0%`** (25/100 flips, mean shift `0.086`) because heavy synthetic triple-letter suffix inflation alters character 3–5 n-grams in `CharSub_3_5` even after Romanized phonetic folding.
  - **Dialogue Context Removal:** Stability is **`70.0%`** (30/100 flips, mean shift `0.154`), reflecting genuine predictive dependence on preceding turns (confirmed independently by the **`-4.23%` Macro-F1 drop** when context is ablated across all 1,123 test records).

---

### 4. Local Compute & Deep Transformer Execution Scope
- While `eval/train_transformer.py`, `eval/evaluate_transformers.py`, and `eval/export_onnx.py` pass Python compilation and `--dry_run` checks locally, **fine-tuning MuRIL (`237M`) and XLM-RoBERTa (`278M`) was `NOT_EXECUTED_LOCALLY`** due to the absence of `torch` and CUDA GPU hardware in this CPU container.
- Consequently, all comparative claims in the frozen test tables restrict themselves honestly to measured client-side baselines (`Majority`, `v1`, `v2`, `WordLex`, `CharSub`, `Statistical OOF`, `Calibrated Hybrid`), while GPU runs remain reproducibly documented via `docs/COLAB.md`.

---

### 5. Selective Abstention Coverage Trade-Off & Morphological Stemming Ablation
- At the validation-selected gate $\tau^* = 0.35$, selective accuracy rises from `68.03%` to **`80.31%`** (`78.92%` Macro-F1; risk reduced `31.97% -> 19.69%`), at the cost of abstaining on **`54.32%`** (`610/1,123`) of boundary test instances (`45.68%` coverage).
- Runtime ablations show removing the Stage-1 morphological stemmer or phonetic folder shifts overall 5-corpus Macro-F1 by `0.00%` / `+0.15%` when word unigrams and character 3–5 n-grams remain enabled because dense `WordLex`/`CharSub` n-grams subsume frequent surface stems in the combined 1,123-item split (`WordLex` alone gives `63.90%` MF1 on Marathi and `55.21%` on Telugu).

---

### 6. Dataset Construct Validity & Cross-Domain Narrowing
- **Emotion-to-Binary Proxy (`L3Cube_MahaEmotions`):** Retaining only native `Sarcasm` ($n=185$) vs `Neutral` ($n=184$) avoids falsely labeling emotional literal tweets (`Anger`, `Sadness`) as non-sarcastic, yet narrows the control distribution. Furthermore, L3Cube-MahaEmotions `train.csv` used LLM CoTR synthetic labels (Kowtal & Joshi, 2025) while only `test.csv` used manual gold annotation; hence all `184` Neutral controls and ~93% of `Sarcasm` items draw from CoTR-annotated records, making Marathi strictly an exploratory proxy.
- **Editorial Satire vs Social Sarcasm (`Misra_Arora_News_Headlines`):** Satirical headlines (*The Onion* vs *HuffPost*, `context=""`) lack multi-turn dialogue context and evaluate journalistic irony rather than conversational banter.
- **Isolated Text Turns of Multimodal Dialogue (`IIITD_MaSaC`):** While `conversation_id` isolates episodes cleanly across splits (`0` conversation overlap), utterance `r.context` is empty (`""` on all `974` items), omitting both surrounding conversational turns and original sitcom audio/visual cues—capping text-only Macro-F1 at `47.33%` and driving `15.57%` Hinglish Conf-ECE.
