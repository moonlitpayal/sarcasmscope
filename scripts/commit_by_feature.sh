#!/usr/bin/env bash
# Creates ONE commit per feature, in dependency order, authored by YOU. Run from the project root.
set -e
if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then echo "Set git user.name and user.email first"; exit 1; fi
[ -d .git ] || { git init -q; git branch -M main; }
commit_feature() { msg="$1"; shift; git add -- "$@"; if ! git diff --cached --quiet; then git commit -q -m "$msg"; echo "committed: $msg"; else echo "nothing new for: $msg"; fi; }
commit_feature "chore: project skeleton (packaging, requirements, gitignore)" "pyproject.toml" "requirements.txt" ".gitignore" "src/sarcasmscope/__init__.py" "src/sarcasmscope/text/__init__.py" "src/sarcasmscope/data/__init__.py" "src/sarcasmscope/models/__init__.py" "src/sarcasmscope/eval/__init__.py" "src/sarcasmscope/explain/__init__.py" "src/sarcasmscope/app/__init__.py"
commit_feature "feat(text): Unicode script detection and code-mix index for 10 Indian scripts" "src/sarcasmscope/text/scripts.py"
commit_feature "feat(text): heuristic language and script guess for display" "src/sarcasmscope/text/langid.py"
commit_feature "feat(text): emoji extraction and emoji features" "src/sarcasmscope/text/emoji.py"
commit_feature "feat(text): cue-preserving normalisation that removes label hashtags" "src/sarcasmscope/text/normalize.py" "tests/test_text.py"
commit_feature "feat(data): unified schema, label mapping and multi-format loaders" "src/sarcasmscope/data/schema.py" "src/sarcasmscope/data/loaders.py"
commit_feature "feat(data): dataset audit with MinHash near-duplicate detection and cue analysis" "src/sarcasmscope/data/audit.py"
commit_feature "feat(data): leakage-safe cluster-based stratified splits" "src/sarcasmscope/data/splits.py"
commit_feature "feat(data): toy data generator for tests and demo only, with data tests" "src/sarcasmscope/data/toy.py" "tests/test_data.py"
commit_feature "feat(models): TF-IDF character+word logistic regression baseline and model registry" "src/sarcasmscope/models/tfidf.py" "src/sarcasmscope/models/registry.py"
commit_feature "feat(models): multilingual transformer fine-tuning wrapper with smoke test" "src/sarcasmscope/models/transformer.py" "scripts/smoke_transformer.py"
commit_feature "feat(models): temperature scaling, ECE, risk-coverage and abstention" "src/sarcasmscope/models/calibration.py"
commit_feature "feat(eval): metrics, bootstrap confidence intervals, McNemar and paired bootstrap" "src/sarcasmscope/eval/metrics.py" "src/sarcasmscope/eval/stats.py"
commit_feature "feat(eval): controlled robustness perturbation suite" "src/sarcasmscope/eval/robustness.py"
commit_feature "feat(eval): shortcut-only baselines" "src/sarcasmscope/eval/shortcuts.py"
commit_feature "feat(eval): cross-language transfer matrix with family summary" "src/sarcasmscope/eval/transfer.py"
commit_feature "feat(explain): occlusion token attributions, with model and evaluation tests" "src/sarcasmscope/explain/occlusion.py" "tests/test_models_eval.py"
commit_feature "feat(pipeline): model bundle and end-to-end pipeline with test-set lock and context option" "src/sarcasmscope/bundle.py" "src/sarcasmscope/pipeline.py" "tests/test_pipeline.py"
commit_feature "feat(app): FastAPI service and web interface with uncertainty, explanations and robustness" "src/sarcasmscope/app/api.py" "src/sarcasmscope/app/web/index.html" "tests/test_api_cli.py"
commit_feature "feat(cli): command-line interface (init-config, demo-data, audit, run, serve, demo)" "src/sarcasmscope/cli.py"
commit_feature "docs: README, requirements and test cases, architecture, protocol, paper map, patent notes, Colab guide, configs" "README.md" "docs/REQUIREMENTS.md" "docs/ARCHITECTURE.md" "docs/RESEARCH_PROTOCOL.md" "docs/PAPER_MAP.md" "docs/PATENT_NOTES.md" "docs/COLAB.md" "configs/toy_demo.yaml" "configs/example_multilingual.yaml" "scripts/commit_by_feature.ps1" "scripts/commit_by_feature.sh"
echo; git log --oneline
