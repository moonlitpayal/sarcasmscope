#!/usr/bin/env bash
set -e
commit_feature() { msg="$1"; shift; git add -- "$@"; if ! git diff --cached --quiet; then git commit -q -m "$msg"; echo "committed: $msg"; else echo "nothing new for: $msg"; fi; }
commit_feature "feat(data): dedup policy for dialogue data and pipeline options (dedup, shortcut_columns)" "src/sarcasmscope/data/audit.py" "src/sarcasmscope/data/splits.py" "src/sarcasmscope/pipeline.py"
commit_feature "feat(data): carry extra columns through the schema and add tag-only shortcut baselines for them" "src/sarcasmscope/data/schema.py" "src/sarcasmscope/data/loaders.py" "src/sarcasmscope/eval/shortcuts.py"
commit_feature "feat(data): safe MaSaC preparer (refuses unsafe pickles) and prepare-masac command" "src/sarcasmscope/data/prepare.py" "src/sarcasmscope/cli.py"
commit_feature "docs: MaSaC config, dataset peek helper, README and requirements; tests TC39-TC42" "configs/masac.yaml" "scripts/peek_dataset.py" "README.md" "docs/REQUIREMENTS.md" "tests/test_masac.py" "scripts/commit_update_masac.ps1" "scripts/commit_update_masac.sh"
echo; git status --short; git log --oneline -6
