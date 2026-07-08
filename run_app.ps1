$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$env:YOLO_CONFIG_DIR = Join-Path $projectRoot "Ultralytics"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".matplotlib"
$env:FLASK_APP = "app.py"

& (Join-Path $projectRoot ".venv\Scripts\python.exe") -m flask run --host 127.0.0.1 --port 8000
