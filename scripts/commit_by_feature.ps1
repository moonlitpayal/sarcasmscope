# Creates ONE commit per feature, in dependency order, authored by YOU.
# Run once from the project root:   powershell -ExecutionPolicy Bypass -File scripts\commit_by_feature.ps1
$ErrorActionPreference = "Stop"
if (-not (git config user.name) -or -not (git config user.email)) { Write-Host "Set your git name/email first: git config --global user.name ""Payal""; git config --global user.email ""you@example.com"""; exit 1 }
if (-not (Test-Path .git)) { git init | Out-Null; git branch -M main }
function Commit-Feature($msg, $files) {
  git add -- $files
  git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { git commit -m $msg | Out-Null; Write-Host "committed: $msg" } else { Write-Host "nothing new for: $msg" }
}
Commit-Feature "chore: project skeleton (packaging, requirements, gitignore)" @("pyproject.toml", "requirements.txt", ".gitignore", "src/sarcasmscope/__init__.py", "src/sarcasmscope/text/__init__.py", "src/sarcasmscope/data/__init__.py", "src/sarcasmscope/models/__init__.py", "src/sarcasmscope/eval/__init__.py", "src/sarcasmscope/explain/__init__.py", "src/sarcasmscope/app/__init__.py")
Commit-Feature "feat(text): Unicode script detection and code-mix index for 10 Indian scripts" @("src/sarcasmscope/text/scripts.py")
Commit-Feature "feat(text): heuristic language and script guess for display" @("src/sarcasmscope/text/langid.py")
Commit-Feature "feat(text): emoji extraction and emoji features" @("src/sarcasmscope/text/emoji.py")
Commit-Feature "feat(text): cue-preserving normalisation that removes label hashtags" @("src/sarcasmscope/text/normalize.py", "tests/test_text.py")
Commit-Feature "feat(data): unified schema, label mapping and multi-format loaders" @("src/sarcasmscope/data/schema.py", "src/sarcasmscope/data/loaders.py")
Commit-Feature "feat(data): dataset audit with MinHash near-duplicate detection and cue analysis" @("src/sarcasmscope/data/audit.py")
Commit-Feature "feat(data): leakage-safe cluster-based stratified splits" @("src/sarcasmscope/data/splits.py")
Commit-Feature "feat(data): toy data generator for tests and demo only, with data tests" @("src/sarcasmscope/data/toy.py", "tests/test_data.py")
Commit-Feature "feat(models): TF-IDF character+word logistic regression baseline and model registry" @("src/sarcasmscope/models/tfidf.py", "src/sarcasmscope/models/registry.py")
Commit-Feature "feat(models): multilingual transformer fine-tuning wrapper with smoke test" @("src/sarcasmscope/models/transformer.py", "scripts/smoke_transformer.py")
Commit-Feature "feat(models): temperature scaling, ECE, risk-coverage and abstention" @("src/sarcasmscope/models/calibration.py")
Commit-Feature "feat(eval): metrics, bootstrap confidence intervals, McNemar and paired bootstrap" @("src/sarcasmscope/eval/metrics.py", "src/sarcasmscope/eval/stats.py")
Commit-Feature "feat(eval): controlled robustness perturbation suite" @("src/sarcasmscope/eval/robustness.py")
Commit-Feature "feat(eval): shortcut-only baselines" @("src/sarcasmscope/eval/shortcuts.py")
Commit-Feature "feat(eval): cross-language transfer matrix with family summary" @("src/sarcasmscope/eval/transfer.py")
Commit-Feature "feat(explain): occlusion token attributions, with model and evaluation tests" @("src/sarcasmscope/explain/occlusion.py", "tests/test_models_eval.py")
Commit-Feature "feat(pipeline): model bundle and end-to-end pipeline with test-set lock and context option" @("src/sarcasmscope/bundle.py", "src/sarcasmscope/pipeline.py", "tests/test_pipeline.py")
Commit-Feature "feat(app): FastAPI service and web interface with uncertainty, explanations and robustness" @("src/sarcasmscope/app/api.py", "src/sarcasmscope/app/web/index.html", "tests/test_api_cli.py")
Commit-Feature "feat(cli): command-line interface (init-config, demo-data, audit, run, serve, demo)" @("src/sarcasmscope/cli.py")
Commit-Feature "docs: README, requirements and test cases, architecture, protocol, paper map, patent notes, Colab guide, configs" @("README.md", "docs/REQUIREMENTS.md", "docs/ARCHITECTURE.md", "docs/RESEARCH_PROTOCOL.md", "docs/PAPER_MAP.md", "docs/PATENT_NOTES.md", "docs/COLAB.md", "configs/toy_demo.yaml", "configs/example_multilingual.yaml", "scripts/commit_by_feature.ps1", "scripts/commit_by_feature.sh")
Write-Host ""; git log --oneline
