# Bootstrap a fresh clone. Idempotent -- safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not on PATH. Install it from https://docs.astral.sh/uv/"
}

Write-Host "==> Installing dependencies (torch cu128 is a ~3 GB download)" -ForegroundColor Cyan
uv sync --extra dev

# uv builds the local package from whatever was on disk when the sync began. If
# the tree was empty at that moment the wheel is empty too, and `import ftlab`
# fails with a green install. Re-installing editable is cheap and settles it.
Write-Host "==> Installing ftlab in editable mode" -ForegroundColor Cyan
uv pip install -e . --no-deps

Write-Host "==> Checking the environment" -ForegroundColor Cyan
uv run ftlab doctor
if ($LASTEXITCODE -ne 0) { Write-Error "Environment checks failed -- see the table above." }

Write-Host "==> Running tests" -ForegroundColor Cyan
uv run pytest -q

Write-Host ""
Write-Host "Ready. Next:" -ForegroundColor Green
Write-Host "  uv run ftlab train -c smoke.yaml      # ~1 min pipeline proof"
Write-Host "  uv run ftlab check-data -c smoke.yaml # inspect the loss mask"
