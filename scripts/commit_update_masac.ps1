# One commit per feature for the MaSaC update. Run once from the project root after extracting the new zip over your folder:
#   powershell -ExecutionPolicy Bypass -File scripts\commit_update_masac.ps1
$ErrorActionPreference = "Stop"
function Commit-Feature($msg, $files) {
  git add -- $files
  git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { git commit -m $msg | Out-Null; Write-Host "committed: $msg" } else { Write-Host "nothing new for: $msg" }
}
Commit-Feature "feat(data): dedup policy for dialogue data and pipeline options (dedup, shortcut_columns)" @("src/sarcasmscope/data/audit.py", "src/sarcasmscope/data/splits.py", "src/sarcasmscope/pipeline.py")
Commit-Feature "feat(data): carry extra columns through the schema and add tag-only shortcut baselines for them" @("src/sarcasmscope/data/schema.py", "src/sarcasmscope/data/loaders.py", "src/sarcasmscope/eval/shortcuts.py")
Commit-Feature "feat(data): safe MaSaC preparer (refuses unsafe pickles) and prepare-masac command" @("src/sarcasmscope/data/prepare.py", "src/sarcasmscope/cli.py")
Commit-Feature "docs: MaSaC config, dataset peek helper, README and requirements; tests TC39-TC42" @("configs/masac.yaml", "scripts/peek_dataset.py", "README.md", "docs/REQUIREMENTS.md", "tests/test_masac.py", "scripts/commit_update_masac.ps1", "scripts/commit_update_masac.sh")
Write-Host ""; git status --short; git log --oneline -6
