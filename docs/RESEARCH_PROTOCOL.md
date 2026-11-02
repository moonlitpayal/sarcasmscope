# Research protocol (write this BEFORE running the final experiments)

Fill every `<...>`, date it, commit it. Changing it later is allowed, but each change must be listed under
"Deviations" with the reason. This is what separates a study from a search for good-looking numbers.

## 1. Title and claim boundary
Working title: *<...>*
We will claim only what Section 7 allows. We will NOT claim "first", "state of the art" or "novel" unless the
literature matrix (Sarcasm_Research_Workbook.xlsx) shows, after reading the papers, that no earlier work does it.

## 2. Research questions and hypotheses
| ID | Question | Hypothesis (directional) | Minimal meaningful effect |
|---|---|---|---|
| RQ1 | Are reported scores inflated by duplicates, shortcuts or splits? | Macro-F1 is lower on the leakage-safe split than on a naive random split | 0.02 macro-F1 |
| RQ2 | Does performance transfer across languages, families and scripts? | In-language > same-family > cross-family; Roman-script text is hardest | 0.05 macro-F1 between adjacent levels |
| RQ3 | Can calibration + abstention make predictions more reliable? | Selective accuracy at the validated threshold exceeds full-coverage accuracy; ECE drops after temperature scaling | ECE reduction >= 0.02 |
| RQ4 | How robust are models to noisy social-media text? | Drops are larger for Roman-script / Dravidian text and for mBERT than for MuRIL | 0.03 macro-F1 |
| RQ5 (optional) | Does conversational context help? | Context improves macro-F1 on conversational data | 0.02 macro-F1 |

## 3. Data
| Dataset | Languages | Rows | Licence checked? | Label provenance checked? | Role |
|---|---|---|---|---|---|
| <D1> | <...> | <...> | <yes/no> | <human/automatic> | <train/eval> |

Inclusion/exclusion rules (fixed now): drop empty text, drop texts with conflicting labels, drop exact duplicates,
merge near-duplicates (Jaccard >= 0.7) into one cluster. Languages with fewer than <100> rows are reported as
"indicative only".

## 4. Models and settings (fixed now)
TF-IDF + LR (baseline); MuRIL, XLM-R base, mBERT fine-tuned with the settings in the config; seeds: 5 for the final run.
No tuning on the test split. Hyper-parameter search, if any, uses validation only and is listed here: <...>

## 5. Metrics and statistics
Primary metric: macro-F1. Also: accuracy, precision/recall/F1 of the sarcastic class, ECE, AURC, selective accuracy.
Uncertainty: 95% bootstrap CI; mean and std over seeds. Model comparison: exact McNemar and paired bootstrap on the
same test items. With several comparisons, apply Holm correction and say so.

## 6. Test-set discipline
The test split is evaluated once with `sarcasmscope run --final`. `FINAL_LOCK.json` is archived with the paper.
If the design changes after the final run, results must be reported as exploratory.

## 7. What each outcome allows us to say
| Outcome | Allowed statement |
|---|---|
| Controlled-split score is lower than naive-split score and the CI excludes 0 | "Scores on <dataset> are inflated by <cause> by <x> macro-F1." |
| No significant difference | "We found no evidence of inflation on <dataset>." (this is a valid result) |
| Model A beats B with p < 0.05 after correction on >= 3 seeds | "A outperforms B on <dataset> under this protocol." |
| A beats B on one seed only | "Suggestive; not confirmed across seeds." |
| Abstention improves selective accuracy | Report coverage together with accuracy, never accuracy alone |

## 8. Threats to validity (expand with your own)
Label noise and annotation provenance; domain shift between platforms; translated text; language-ID errors;
small per-language samples; Roman-script spelling variation; sarcasm that needs world knowledge.

## 9. Deviations log
| Date | Change | Reason |
|---|---|---|
