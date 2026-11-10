# SARcasmScope — Stage 9 Publication Readiness Assessment

**Codebase & Audit Lead:** Payal Mehta (`payal.12620246@vit.edu`)  
**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology, India  
**Submission Gate Status:** **TECHNICAL VERIFICATION COMPLETE — NOT READY FOR JOURNAL UPLOAD UNTIL CO-AUTHOR SIGN-OFF (5 AUTHORS), AUTHORSHIP ORDER / CORRESPONDING AUTHOR RESOLUTION, AND OPTIONAL COLAB GPU BASELINE POLICY ARE FINALIZED**  
**Target Venues:** ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), IEEE Transactions on Affective Computing (TAC), Language Resources and Evaluation (LREV)

---

## 1. Evidence & Artifact Verification Summary

| Readiness Criterion | Evidence Status | Verified Details |
|:---|:---|:---|
| **Nov 7 vs Stage 9 Reconciliation** | **VERIFIED** | `eval/verify_comparison.ts` reproduces Nov 7 `v1` on $N=30$ (`66.67%` Acc, `33.33%` Rec, `TP=5,FP=0,TN=15,FN=10`) alongside `v2` (`96.67%`), `StatOOF` (`56.67%`), and `Hybrid` (`86.67%`), contrasted strictly with frozen `test.jsonl` ($N=1,123$). `RESULTS_STATUS_AND_AUDIT.md` does not exist in workspace. |
| **Deployed vs Trained Integration** | **VERIFIED** | `src/engine/hybridFusion.ts` directly loads `src/engine/trainedModelData.ts` (`v2.0.0-oof-calibrated`, `405.36 KB`) and `src/engine/statisticalModel.ts` (`"statistical_oof_stack"`), replacing earlier unlinked fallback weights and fixed `1.15` temperature. |
| **Non-Circular 3-Stage Calibration** | **VERIFIED** | Level-1 `WordLex_1_2` ($V=8,000$) + `CharSub_3_5` ($V=8,000$) fitted on `train.jsonl` ($3,373$); Level-2 MAP Ridge fitted on 5-fold group OOF; Level-3 $T^*=0.845, \tau^*=0.35$ fitted by NLL minimization & risk-coverage strictly on `val.jsonl` ($1,104$). |
| **Frozen Held-Out Performance & CSV Sync** | **VERIFIED** | `68.03%` Acc, `67.56%` MF1, `67.09%` Prec, `60.50%` Rec, `0.7535` ROC-AUC, **`1.36%` Conf-ECE**, `0.5826` NLL (`80.31%` selective acc @ `45.68%` coverage). `eval/verify_independent_recomputation.ts` confirms `1123/1123` CSV row parity (`max|Δp|=0`) and independently recomputes all metrics across `eval_results/stage9/` and `eval_results/independent/` (`344+401+194+75+109`). |
| **Uncertainty & Calibration Qualification** | **VERIFIED (SCOPED)** | Exports both `val.jsonl` (`NLL 0.5798->0.5777`, `SelAcc 81.06% @ 49.73% cov`) and sealed `test.jsonl` (`NLL 0.5856->0.5826`, `Conf-ECE 2.28%->1.36%`, `Class-ECE 4.40%->2.80%`, Bin 5 `68` items / `0.026` error); distinguishes $P_{\text{fused}}$, Margin Confidence, $\tau^*$, and Selective Acc; notes Hinglish `15.57%` Conf-ECE & Telugu `0%` Rec @ `0.50`. |
| **Dataset Validity Boundaries** | **VERIFIED (QUALIFIED)** | `0` exact/`conversation_id` overlaps, `13` 3-shingle near-dups (`8` cross-split, `5` train-test in `MaSaC`); verified schema (`HasContext=true` ONLY on `ACL 2020`). Explicitly scopes `MahaEmotions` (`Sarcasm` vs `Neutral` only + CoTR train provenance; Kowtal & Joshi 2025), `Misra` (editorial satire), and `MaSaC` (`context=""`, no A/V prosody). |
| **GPU Transformer Baselines** | **SCOPED (`NOT_EXECUTED_LOCALLY`)** | `eval/*.py` syntax-checked & dry-run verified; no MuRIL/XLM-R parity claim made. Empirical T4 GPU execution remains external (`docs/COLAB.md`). |
| **Five-Author, Permissions & References** | **PENDING BLOCKER** | Verified DOIs/pages/venues for all references (`docs/TECHNOLOGY_AND_REFERENCES.md`); original draft lists **five co-authors with supervisor/collaborator as corresponding author**. **Mandatory co-author order, CRediT & written approval required before submission.** |

---

## 2. Mandatory Pre-Submission Actions
1. **Resolve Authorship & Corresponding Author:** Confirm exact 5-author ordering, CRediT roles, corresponding-author contact, and written co-author approval of the Stage 9 edits across manuscript, `CITATION.cff`, and `README.md`.
2. **Decide External GPU Baselines:** Either retain honest CPU-only client baselines (removing neural parity claims completely) or execute `docs/COLAB.md` once on T4 GPU to generate empirical MuRIL/XLM-RoBERTa prediction files and checkpoints.
