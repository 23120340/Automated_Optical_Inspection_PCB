$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python 3.12 environment in .venv..."
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.12 virtual environment (exit code $LASTEXITCODE)."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip (exit code $LASTEXITCODE)."
}
& $venvPython -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    throw "Could not install project dependencies (exit code $LASTEXITCODE)."
}

Write-Host ""
Write-Host "Setup complete. Run the app with:"
Write-Host "  .\scripts\run_app.ps1"
