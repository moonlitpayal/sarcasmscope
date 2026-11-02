#!/usr/bin/env bash
set -e
commit_feature() { msg="$1"; shift; git add -- "$@"; if ! git diff --cached --quiet; then git commit -q -m "$msg"; echo "committed: $msg"; else echo "nothing new for: $msg"; fi; }
commit_feature "feat(annotation): corpus sampler with pilot, random and enriched subsets" "src/sarcasmscope/annotation/__init__.py" "src/sarcasmscope/annotation/sample.py"
commit_feature "feat(annotation): blind per-annotator Excel workbooks with drop-down labels" "src/sarcasmscope/annotation/workbook.py" "docs/ANNOTATION_GUIDELINES.md"
commit_feature "feat(annotation): Cohen and Fleiss kappa, majority labels and discussion list" "src/sarcasmscope/annotation/agreement.py"
commit_feature "feat(cli): annotate-sample and annotate-agreement commands, openpyxl dependency" "src/sarcasmscope/cli.py" "pyproject.toml" "requirements.txt"
commit_feature "docs: labelling instructions in README and requirements; tests TC43-TC47" "README.md" "docs/REQUIREMENTS.md" "tests/test_annotation.py" "scripts/commit_update_annotation.ps1" "scripts/commit_update_annotation.sh"
echo; git status --short; git log --oneline -6
