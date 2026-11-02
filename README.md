# SarcasmScope

A reliability-first toolkit and web app for **sarcasm detection in code-mixed Indian-language text**
(Hindi, Marathi, Telugu, Tamil, Bengali, Hinglish and other Indian languages mixed with English).

Most sarcasm papers report one score on one random split. SarcasmScope is built to answer the
harder questions a reviewer asks:

| Question | Feature |
|---|---|
| Is the dataset clean? | **Dataset audit**: duplicates, near-duplicates, conflicting labels, language/label confounds, label-correlated cues (emoji, hashtags, length...) |
| Is the score inflated by leakage? | **Leakage-safe splits**: whole duplicate/near-duplicate/conversation clusters stay in one split; stratified by label x language; seeded |
| Is it really reading the sentence? | **Shortcut baselines**: emoji-only, length-only, punctuation-only, language-tag-only, source-only |
| Is the result statistically meaningful? | Multi-seed runs, bootstrap confidence intervals, exact McNemar test, paired bootstrap |
| Does it know when it is unsure? | **Temperature scaling, ECE, risk-coverage, abstention threshold** |
| Does it survive messy social-media text? | **Robustness suite**: spelling noise, repeated letters, emoji removal, punctuation removal, truncation, shuffling... |
| Does it work across languages? | **Cross-language transfer matrix** with Indo-Aryan / Dravidian family summary |
| Why did it say that? | **Token explanations** (leave-one-token-out occlusion), shown in the web app |
| Is the test set still untouched? | **Test-set lock**: the test split is only evaluated with `--final`, once |

## Status (be honest in your paper)

* Everything in the toolkit is covered by **47 automated tests** (`pytest`), all passing.
* All tests and the demo use **machine-written TOY sentences**. They are not a research dataset and no result on
  them may be reported.
* The transformer fine-tuning module (MuRIL / XLM-R / mBERT) could not be executed where the toolkit was written
  (no GPU, no model downloads). Run `scripts/smoke_transformer.py` on Colab first; see `docs/COLAB.md`.
* No real dataset is bundled. Download one, **audit it**, and check its licence before training.

## Quick start (Windows PowerShell; Mac/Linux use `source env/bin/activate`)

```powershell
python -m venv env
env\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest -q                                   # 47 passed
sarcasmscope demo                           # toy data -> full pipeline -> report (checks the install)
sarcasmscope demo-data --out data/toy_demo.csv
sarcasmscope run --config configs/toy_demo.yaml
sarcasmscope serve                          # web app at http://127.0.0.1:8000  (DEMO model banner shown)
```

### Real sarcasm data that is publicly downloadable: MaSaC (Hindi-English)

```powershell
git clone https://github.com/LCS2-IIITD/MSH-COMICS ..\MSH-COMICS
sarcasmscope prepare-masac --repo ..\MSH-COMICS --out data\raw\masac.csv
sarcasmscope audit --config configs\masac.yaml
sarcasmscope run --config configs\masac.yaml
```
MaSaC has human sarcasm labels, dialogue ids (used as split groups) and previous turns (used as context). The repository
ships no licence file; check the terms with the authors before redistributing anything derived from it.
**Check every dataset before use:** the "Code-Mixed Indic Languages with Emoticons for Sarcasm Detection" archive on
Mendeley contains sentiment labels only (see Known limitations), so it cannot be used as sarcasm ground truth.

### Building the Marathi-English labelled set (human annotation)

```powershell
sarcasmscope annotate-sample --source data\raw\mecorpus.txt --annotators Payal Divanshi Aniket Balaji
```
This writes `annotation\pilot\<Name>_pilot.xlsx` and `annotation\final\<Name>_final.xlsx` (one blind workbook per person with a
drop-down label) and `annotation\master_key.csv` (keep private). After labelling:
```powershell
sarcasmscope annotate-agreement --dir annotation\final_filled --key annotation\master_key.csv
```
Source corpus: L3Cube-MeCorpus (CC BY-NC-SA 4.0, research use). A derived labelled set must keep that licence.
Guidelines: `docs/ANNOTATION_GUIDELINES.md`.

### Using a real dataset

1. `sarcasmscope init-config --out configs/my_multilingual.yaml` (or copy `configs/example_multilingual.yaml`).
2. Put the file in `data/raw/`, open it, and fill the `mapping:` section with its real column names.
3. `sarcasmscope audit --config configs/my_multilingual.yaml` and read the flags **before** training anything.
4. `sarcasmscope run --config configs/my_multilingual.yaml` writes `runs/<name>/report.md` and `tables/*.csv`.
5. When the design is frozen: `sarcasmscope run --config ... --final` (once). A second `--final` is refused.
6. `sarcasmscope serve --bundle runs/<name>/bundles/tfidf_seed42.joblib` serves your trained model.

## Outputs of a run (`runs/<run_name>/`)

`audit.md` | `splits/seed_*.csv` | `predictions/` | `bundles/` | `tables/main_results.csv`, `per_language.csv`,
`reliability.csv`, `significance.csv`, `shortcuts.csv`, `robustness.csv`, `transfer_*.csv` | `report.md` |
`FINAL_LOCK.json` (after the final run)

## Web app / API

`GET /` UI, `GET /health`, `POST /predict`, `POST /predict_batch`, `POST /explain`, `POST /robustness`
(interactive docs at `/docs`). The UI shows the verdict, calibrated probability, an **"uncertain"** state when
confidence is below the validated threshold, detected script / language guess / code-mix index, emoji, per-token
influence and a robustness table. Inputs are length-limited and the service stores nothing.

## Project layout

```
src/sarcasmscope/
  text/      scripts, langid (heuristic), emoji, normalize (label-hashtag removal)
  data/      schema, loaders, audit (MinHash near-duplicates), splits, toy (tests/demo only)
  models/    tfidf, transformer, registry, calibration
  eval/      metrics, stats, robustness, shortcuts, transfer
  explain/   occlusion
  bundle.py  pipeline.py  cli.py  app/ (FastAPI + web UI)
tests/       47 test cases (TC01-TC47), see docs/REQUIREMENTS.md
docs/        REQUIREMENTS, ARCHITECTURE, RESEARCH_PROTOCOL, PAPER_MAP, PATENT_NOTES, COLAB
configs/     toy_demo.yaml, example_multilingual.yaml
scripts/     smoke_transformer.py, commit_by_feature.ps1 / .sh
```

## Known limitations

* Duplicate handling differs by data type: `split.dedup: drop` (tweets/comments) or `keep` (dialogues). On MaSaC the
  default `drop` would have discarded 24% of rows, mostly short context-dependent lines, so use `keep` there.

* Language identification is a heuristic for display. Experiments use the dataset's gold language column.
* The code-mix index is script-based, so Roman-script mixing (Hinglish) shows as 0; use a real language-ID model
  for token-level analysis.
* TF-IDF features are script-specific, so cross-script transfer for that baseline is near chance by construction.
* Occlusion explanations show sensitivity of the model, not human-verified reasons.
* Temperature scaling assumes validation and test come from the same distribution.

No licence file is included on purpose: decide on a licence only after discussing IP and patent options with your
institute (publishing code early can affect patentability). Keep the repository private until then.
