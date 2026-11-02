# Where each paper section gets its content

Structure follows the IEEE-style paper you shared (Abstract, Introduction, Related Work with comparison table,
Requirements, Architecture, Methodology, Implementation, Testing, Results, Limitations, Future Work, References).

| Paper section | Source in this repository | Note |
|---|---|---|
| Abstract | `runs/<final>/report.md` main table + audit flags | Write last; numbers copied from `tables/` only |
| Introduction | Research protocol Section 1-2 | Motivation + gap, stated within the claim boundary |
| Related Work + comparison table | `Sarcasm_Research_Workbook.xlsx` (Literature_Matrix) | Only papers you actually read |
| Problem definition and requirements | `docs/REQUIREMENTS.md` (FR1-FR21, NFR1-NFR5) | Tables can be reused almost directly |
| System architecture | `docs/ARCHITECTURE.md` (mermaid diagram) | Redraw as a figure |
| Methodology | Normalisation, cluster-based splitting, calibration, robustness suite, shortcut baselines, transfer, occlusion | One subsection each |
| Datasets | `runs/<final>/audit.md`, `tables/per_language.csv` | Include the audit flags honestly |
| Implementation | README, `pyproject.toml`, config file | Library versions from `pip freeze` |
| Testing | `docs/REQUIREMENTS.md` test-case table (TC01-TC38) + pytest output | Mirrors the black-box test table in the reference paper |
| Results | `tables/main_results.csv`, `significance.csv`, `reliability.csv`, `robustness.csv`, `transfer_*.csv`, `shortcuts.csv` | Report mean +- std over seeds and CIs |
| Error analysis | `predictions/` + 30 hand-checked mistakes per language group | Needs native-speaker review |
| Limitations | README "Known limitations" + protocol Section 8 | Be specific |
| Reproducibility statement | Seeds, split files in `splits/`, `FINAL_LOCK.json` | Archive the run folder |
