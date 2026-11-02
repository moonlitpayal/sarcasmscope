# One commit per feature for the labelling kit. Run once from the project root after extracting the new zip over your folder:
#   powershell -ExecutionPolicy Bypass -File scripts\commit_update_annotation.ps1
$ErrorActionPreference = "Stop"
function Commit-Feature($msg, $files) {
  git add -- $files
  git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { git commit -m $msg | Out-Null; Write-Host "committed: $msg" } else { Write-Host "nothing new for: $msg" }
}
Commit-Feature "feat(annotation): corpus sampler with pilot, random and enriched subsets" @("src/sarcasmscope/annotation/__init__.py", "src/sarcasmscope/annotation/sample.py")
Commit-Feature "feat(annotation): blind per-annotator Excel workbooks with drop-down labels" @("src/sarcasmscope/annotation/workbook.py", "docs/ANNOTATION_GUIDELINES.md")
Commit-Feature "feat(annotation): Cohen and Fleiss kappa, majority labels and discussion list" @("src/sarcasmscope/annotation/agreement.py")
Commit-Feature "feat(cli): annotate-sample and annotate-agreement commands, openpyxl dependency" @("src/sarcasmscope/cli.py", "pyproject.toml", "requirements.txt")
Commit-Feature "docs: labelling instructions in README and requirements; tests TC43-TC47" @("README.md", "docs/REQUIREMENTS.md", "tests/test_annotation.py", "scripts/commit_update_annotation.ps1", "scripts/commit_update_annotation.sh")
Write-Host ""; git status --short; git log --oneline -6
