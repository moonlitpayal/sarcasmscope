# SARcasmScope — Stage 9 Preemptive Peer-Reviewer Response Preparation

**Principal Investigator:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Target Repository:** `https://github.com/moonlitpayal/sarcasmscope`  
**Target Journals:** ACM TALLIP / IEEE Transactions on Affective Computing  
**Audit Date:** November 9, 2026  
**Audit Phase:** 9J — Reviewer Rebuttal & Defense Formulation  

---

## Preemptive Defense & Detailed Responses to Critical Peer-Review Queries

### Query 1: Transformer Comparison & Efficiency Trade-off
**Reviewer Comment:**  
*"MuRIL achieves 57.32% Macro-F1 compared to SARcasmScope's 56.57%. Since MuRIL achieves higher performance, why should practitioners deploy SARcasmScope rather than a fine-tuned transformer?"*

**Author Response:**  
We thank the reviewer for raising this fundamental deployment consideration. While MuRIL achieves a marginal $+0.75\%$ Macro-F1 advantage on this noisy corpus, it incurs massive operational costs:
1. **Computational Latency:** MuRIL requires $85.4\text{ ms}$ on an NVIDIA T4 GPU and $> 800\text{ ms}$ on CPU hardware. SARcasmScope evaluates an utterance in **$0.205\text{ ms}$ (P95)** on CPU, providing a **$> 400\times$ speedup** on GPU and **$> 4,000\times$ speedup** on CPU.
2. **Resource Footprint:** MuRIL contains 237 million parameters ($> 900\text{ MB}$ weights) and requires a Python PyTorch runtime with multi-gigabyte memory overhead. SARcasmScope requires **$< 10\text{ MB}$ RAM** and runs natively in standard browser JavaScript engines with zero server dependencies.
3. **Data Privacy:** Because SARcasmScope executes client-side, sensitive user communications never leave the user's device, ensuring complete privacy compliance.

---

### Query 2: Statistical Significance Disentanglement
**Reviewer Comment:**  
*"The manuscript claims statistically significant improvement over baseline, yet the McNemar test on overall accuracy yields p = 0.1806. How is this justified?"*

**Author Response:**  
We appreciate the reviewer's precision. We have explicitly disentangled the statistical claims:
- The baseline rule system was an ultra-conservative literal predictor ($97.35\%$ specificity, $588/604$ true negatives) that suffered catastrophic collapse on sarcasm recall ($3.08\%$, detecting only $16/519$ sarcastic items).
- On the target phenomenon—detecting genuine sarcasm—SARcasmScope rescued **281 additional sarcastic cases** ($57.23\%$ recall). A paired McNemar test on sarcastic samples confirms this recovery is **exceptionally significant ($\chi^2 = 277.03, p = 3.7 \times 10^{-62} \ll 0.0001$, Odds Ratio = 282.0)**.
- The overall accuracy difference ($53.78\% \to 56.63\%$) is not statistically significant ($p = 0.1806$) precisely because SARcasmScope traded off artificial specificity ($56.13\%$) to achieve functional sarcasm recall. This trade-off is now transparently documented in Section 7 and Table 4 of the manuscript.

---

### Query 3: Multi-Turn Dialogue Integrity & Leakage Prevention
**Reviewer Comment:**  
*"In conversational datasets such as IIITD MaSaC, multi-turn utterances from the same dialogue could contaminate train and test splits. How did you ensure zero leakage?"*

**Author Response:**  
To prevent narrative leakage, our partitioning algorithm (`eval/partition_independent_datasets.ts`) enforces an **atomic dialogue-grouping constraint**:
- All conversational turns sharing an identical `dialogue_id` were restricted to reside exclusively within one partition (Train, Validation, or Test).
- Automated auditing verified that cross-split dialogue ID intersection is **$0.00\%$** ($0$ shared dialogue IDs). Cross-split text intersection is also verified at **$0.00\%$**.

---

### Query 4: Retaining Natural Class Imbalance in Telugu
**Reviewer Comment:**  
*"Why did you preserve the natural 15.7% sarcasm class balance in the Telugu dataset rather than synthetically balancing it like Twitter and Headlines?"*

**Author Response:**  
Preserving natural class distribution in real-world social media is essential for evaluating **false discovery risk**:
- In deployed applications, sarcasm is a low-prevalence phenomenon. Synthetically balanced benchmarks mask false positive vulnerabilities.
- By evaluating on Telugu's natural $15.7\%$ prevalence, we demonstrate that SARcasmScope maintains stable precision ($51.4\%$) without generating excessive false sarcasm alarms on Dravidian text.

---

### Query 5: Practical Utility of Selective Classification
**Reviewer Comment:**  
*"The selective prediction mechanism achieves 61.48% accuracy but abstains on 31.34% of samples. How is this useful in practice?"*

**Author Response:**  
In high-stakes sentiment analysis (e.g., brand triage, automated moderation), making an overconfident incorrect decision is far more damaging than flagging an utterance as ambiguous. By abstaining on borderline probabilities ($|p - 0.50| < 0.29$), SARcasmScope allows production pipelines to automatically route high-confidence predictions ($68.66\%$ of traffic) while redirecting ambiguous cases to human moderators or conservative fallback rules.
