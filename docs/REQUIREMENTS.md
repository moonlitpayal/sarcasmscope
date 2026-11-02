# Requirements and test cases

## Functional requirements

| ID | Requirement | Module | Test cases |
|---|---|---|---|
| FR1 | Detect the script of Indian-language text and a script-based code-mix index | text/scripts | TC01, TC02, TC03 |
| FR2 | Normalise text without destroying sarcasm cues; remove label-revealing hashtags; keep ZWJ/ZWNJ | text/normalize | TC04, TC05, TC08 |
| FR3 | Extract emoji including ZWJ sequences | text/emoji | TC06 |
| FR4 | Heuristic language guess for display | text/langid | TC07 |
| FR5 | Load any CSV/TSV/JSON/XLSX dataset through a column mapping and map labels to 0/1 | data/schema, loaders | TC09, TC10 |
| FR6 | Audit a dataset: duplicates, conflicts, near-duplicates, confounds, cues | data/audit | TC11, TC12 |
| FR7 | Produce leakage-safe, stratified, seeded, group-aware splits | data/splits | TC13, TC14, TC15, TC16 |
| FR8 | Train a TF-IDF baseline and (optionally) fine-tune multilingual transformers | models | TC19 (+ smoke script for transformers) |
| FR9 | Report accuracy, precision, recall, F1, macro-F1, bootstrap CI, per-language scores | eval/metrics | TC17, TC27 |
| FR10 | Compare models with exact McNemar and paired bootstrap | eval/stats | TC18 |
| FR11 | Calibrate (temperature scaling), report ECE, risk-coverage, choose an abstention threshold | models/calibration | TC20, TC21 |
| FR12 | Evaluate robustness to controlled perturbations | eval/robustness | TC22, TC23 |
| FR13 | Detect shortcut features with shortcut-only baselines | eval/shortcuts | TC24 |
| FR14 | Cross-language transfer matrix with family summary | eval/transfer | TC25 |
| FR15 | Explain a prediction with token attributions | explain/occlusion | TC26 |
| FR16 | Run everything end to end and write a report; protect the test set | pipeline | TC28, TC29 |
| FR17 | Save/load a model with its temperature and threshold | bundle | TC30, TC31 |
| FR18 | Merge a YAML config with defaults | pipeline | TC32 |
| FR19 | Serve predictions, explanations, robustness checks and a web UI | app/api | TC33, TC34, TC35, TC36 |
| FR20 | Command-line interface | cli | TC37 |
| FR21 | Optional conversational context (previous turn) for TF-IDF and transformers | pipeline | TC38 |
| FR22 | Duplicate policy for dialogue data (keep repeats, keep dialogues together, report cross-split text overlap) | data/splits | TC39 |
| FR23 | Convert the MaSaC release into one clean CSV with verified alignment; refuse unsafe pickles | data/prepare | TC40, TC41 |
| FR24 | Shortcut baselines for extra columns such as speaker | eval/shortcuts | TC42 |
| FR25 | Sample pilot / random / enriched sentences from a large corpus and write one blind labelling workbook per annotator | annotation | TC43, TC44, TC47 |
| FR26 | Compute inter-annotator agreement (Cohen, Fleiss), majority labels and a discussion list | annotation | TC45, TC46 |

## Non-functional requirements

| ID | Requirement | How it is met |
|---|---|---|
| NFR1 | Reproducibility | Fixed seeds, saved split files, config-driven runs, deterministic perturbations |
| NFR2 | Honest evaluation | Test-set lock, multi-seed means, confidence intervals, shortcut baselines |
| NFR3 | Privacy and abuse resistance | API stores no data; input length limits (2000 chars, 200 batch); empty-after-cleaning rejected |
| NFR4 | Portability | Pure Python; TF-IDF path needs no GPU; transformers optional |
| NFR5 | Transparency | Demo-model banner; heuristic outputs are labelled as heuristics |

## Test cases (all automated, `pytest -q`)

| TC | What is checked | Expected result |
|---|---|---|
| TC01 | Script detection for 10 Indian scripts + Latin | Correct script per sample word |
| TC02 | Tokenizer keeps Indic words whole | `["वाह","क्या","बात","है"]` |
| TC03 | Code-mix index | >0 for mixed script, 0 for single script |
| TC04 | Label hashtags removed, URLs/mentions masked, repeats squeezed, `!!!` kept | As stated |
| TC05 | Emoji kept/removed on request; ZWJ preserved | As stated |
| TC06 | Emoji with ZWJ sequences extracted; lone ZWJ in Indic text is not an emoji | As stated |
| TC07 | Language guess for Telugu, Hindi, Marathi, Roman Hinglish | te / hi / mr / hi (romanized) |
| TC08 | Duplicate key ignores case, emoji, punctuation | Equal keys |
| TC09 | Label mapping and bad-row dropping with a report | Labels [1,0,1]; 2 rows dropped; "maybe" reported |
| TC10 | Missing columns give a clear error | ValueError "Missing required column" |
| TC11 | Near-duplicate detection | Only the planted pair found |
| TC12 | Audit flags duplicates, conflicting labels, planted emoji cue | Flags raised |
| TC13 | Split has no text overlap, correct ratios, stratified | No leakage; ratios within 6 points |
| TC14 | Split determinism and seed dependence | Same seed same split; different seed differs |
| TC15 | Conversation groups stay together | One split per group |
| TC16 | Near-duplicates never cross train/test | One split per cluster |
| TC17 | Metrics vs hand computation | Exact |
| TC18 | McNemar vs analytic value | p = 2 x 0.5^8 |
| TC19 | TF-IDF learns toy data | Macro-F1 > 0.70 |
| TC20 | Temperature scaling reduces ECE on an over-confident model | T > 1, ECE lower |
| TC21 | Selective prediction raises accuracy; threshold hits target | Monotone gain |
| TC22 | Perturbations deterministic and safe | Identical outputs per seed |
| TC23 | Robustness table shape; shuffling does not help | 9 rows |
| TC24 | Planted emoji shortcut flagged; clean data not flagged | Flag only when planted |
| TC25 | Transfer matrix and family summary | In-language > 0.6 |
| TC26 | Occlusion finds a planted trigger token | Trigger ranked first |
| TC27 | Per-language metrics | Hand-computed values |
| TC28 | Pipeline writes every artifact | All files exist |
| TC29 | Second final run refused | RuntimeError; force overrides |
| TC30 | Bundle round trip and decision record | Valid probability and fields |
| TC31 | Demo bundle flagged as demo | `demo=True` |
| TC32 | Config merges with defaults | Defaults preserved |
| TC33 | /health and UI served | 200 |
| TC34 | /predict returns decision, language, emoji | Valid |
| TC35 | /explain and /robustness | 5 attributions, 8 perturbations |
| TC36 | Batch and input validation | 422 for empty, too long, too many |
| TC37 | CLI init-config and demo-data | Files written |
| TC38 | Pipeline with `use_context` on; clear error when no context exists | Runs; ValueError otherwise |
| TC39 | `dedup: keep` keeps all rows and dialogues together, reports repeated texts across splits | No rows dropped; overlap reported |
| TC40 | MaSaC preparer checks alignment, builds per-dialogue context, refuses misaligned files | Context only from earlier turns of the same dialogue |
| TC41 | Unsafe pickle is refused and not executed | UnpicklingError; no side effect |
| TC42 | Planted speaker shortcut flagged | `speaker_tag_only` flagged |
| TC43 | Sampler subsets disjoint, reproducible, enriched items carry cues, clear warnings and errors | As stated |
| TC44 | Annotator workbook has a drop-down and hides subset and cues | As stated |
| TC45 | Fleiss kappa reproduces the published worked example (0.210) | Within 0.001 |
| TC46 | End to end: sample, label (simulated), agreement report, adjudication, discussed overrides | Files written; kappa in range |
| TC47 | Reservoir sampling is reproducible; CSV with and without header | As stated |
