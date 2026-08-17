$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app\streamlit_app.py"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

Set-Location -LiteralPath $projectRoot
& $venvPython -m streamlit run $appPath --server.address 127.0.0.1
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit exited with code $LASTEXITCODE."
}
