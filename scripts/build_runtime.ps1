# Build the portable Python runtime used by the packaged desktop app.
#
#   desktop/runtime/
#     python/          embeddable Python + pip-installed deps (torch cpu, fastapi, ...)
#     tools/           llama.cpp binaries (llama-server, llama-quantize, ...)
#     launcher.py      backend entry (copied from repo root)
#
# The embed python has NO venv redirector, so the packaged app runs it in-place.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\build_runtime.ps1

param(
    [string]$PythonVersion = "3.12.10",
    [string]$Arch = "amd64"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root "desktop\runtime"
$embedDir = Join-Path $runtime "python"
$toolsDir = Join-Path $runtime "tools"
$temp = Join-Path $env:TEMP "ocd_runtime_build"

$pyUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-$Arch.zip"
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"

Write-Host "==> runtime dir: $runtime"
New-Item -ItemType Directory -Path $temp -Force | Out-Null

Write-Host "==> download embed python $PythonVersion"
$zip = Join-Path $temp "python-embed.zip"
if (-not (Test-Path $zip)) { Invoke-WebRequest -Uri $pyUrl -OutFile $zip -UseBasicParsing }
if (Test-Path $embedDir) { Remove-Item $embedDir -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $embedDir -Force
$exe = Join-Path $embedDir "python.exe"
if (-not (Test-Path $exe)) { throw "embed python not extracted: $exe" }

Write-Host "==> enable site-packages (uncomment 'import site')"
$pth = Get-ChildItem $embedDir -Filter "python*._pth" | Select-Object -First 1
$content = Get-Content $pth.FullName
$content = $content | ForEach-Object { if ($_ -eq "#import site") { "import site" } else { $_ } }
Set-Content -Path $pth.FullName -Value $content -Encoding ASCII

Write-Host "==> install pip"
$getPip = Join-Path $temp "get-pip.py"
if (-not (Test-Path $getPip)) { Invoke-WebRequest -Uri $getPipUrl -OutFile $getPip -UseBasicParsing }
& $exe $getPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip failed ($LASTEXITCODE)" }

Write-Host "==> install deps (torch cpu first, then the rest)"
& $exe -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "torch install failed ($LASTEXITCODE)" }
& $exe -m pip install --no-warn-script-location `
    transformers huggingface-hub safetensors fastapi uvicorn websockets pydantic `
    "PyYAML>=6.0" requests psutil mcp gguf sentencepiece
if ($LASTEXITCODE -ne 0) { throw "deps install failed ($LASTEXITCODE)" }

Write-Host "==> install oneclick-distill itself"
& $exe -m pip install --no-deps --no-warn-script-location $root
if ($LASTEXITCODE -ne 0) { throw "package install failed ($LASTEXITCODE)" }

Write-Host "==> bundle llama.cpp tools"
New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
$toolsSrc = Join-Path $root "oneclick_distill\tools"
if (Test-Path $toolsSrc) {
    Copy-Item (Join-Path $toolsSrc "*") $toolsDir -Recurse -Force -ErrorAction Stop
} else {
    Write-Host "==> tools not present locally; download into runtime via ensure_tools()"
    $env:OCD_TOOLS_DIR = $toolsDir
    & $exe -c "import oneclick_distill.quantize.llama_cpp as q; print('tools:', q.ensure_tools())"
    if ($LASTEXITCODE -ne 0) { throw "ensure_tools failed ($LASTEXITCODE)" }
    Remove-Item Env:OCD_TOOLS_DIR
}

Write-Host "==> write launcher"
Copy-Item (Join-Path $root "server_launcher.py") (Join-Path $runtime "launcher.py") -Force

# strip pycache / __pycache__ for size
Get-ChildItem $runtime -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$size = (Get-ChildItem $runtime -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host "==> done. runtime size: $([math]::Round($size,1)) MB"
