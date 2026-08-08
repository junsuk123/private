<#
.SYNOPSIS
  One command to make a fresh machine able to run this project.

.DESCRIPTION
  Creates .venv, installs the project and its dependencies, and verifies the
  result actually imports. Safe to re-run: every step is idempotent.

  Optional accelerator and local-LLM extras are installed only when asked for,
  because they are large and the project runs correctly without them - device
  placement falls back to CPU on its own (see app/realtime/device_plan.py).

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -WithNpu
  .\setup.ps1 -All
#>
param(
  # Install the OpenVINO extra so GPU/NPU placement becomes available.
  [switch]$WithNpu,
  # Install the local-LLM extra (large download).
  [switch]$WithLocalLlm,
  # Everything optional.
  [switch]$All,
  # Rebuild .venv from scratch.
  [switch]$Recreate
)

$ErrorActionPreference = "Stop"
# Every path below is derived from this script's own location, so the project can
# be cloned or renamed to anything without editing a line.
$root = $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"

function Write-Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Write-Ok($message) { Write-Host "    $message" -ForegroundColor Green }

if ($Recreate -and (Test-Path $venv)) {
  Write-Step "Removing existing .venv (-Recreate)"
  Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $python)) {
  Write-Step "Creating virtual environment"
  $base = $null
  foreach ($candidate in @("py", "python", "python3")) {
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved) { $base = $resolved.Source; break }
  }
  if (-not $base) {
    throw "No Python interpreter found on PATH. Install Python 3.11+ and re-run."
  }
  if ((Split-Path $base -Leaf) -eq "py.exe") {
    & $base -3 -m venv $venv
  } else {
    & $base -m venv $venv
  }
  Write-Ok "created $venv"
} else {
  Write-Ok ".venv already present"
}

if (-not (Test-Path $python)) {
  throw "Virtual environment did not produce $python"
}

$version = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Ok "python $version"
$tooNew = & $python -c "import sys; print('1' if sys.version_info < (3, 11) else '0')"
if ($tooNew -eq "1") {
  throw "This project requires Python 3.11 or newer; .venv has $version"
}

Write-Step "Upgrading installer tooling"
& $python -m pip install --quiet --upgrade pip setuptools wheel
Write-Ok "pip/setuptools/wheel current"

# Extras are additive, so build the install target once and install once.
$extras = @()
if ($WithNpu -or $All) { $extras += "npu" }
if ($WithLocalLlm -or $All) { $extras += "local-llm" }
$target = if ($extras.Count -gt 0) { ".[" + ($extras -join ",") + "]" } else { "." }

Write-Step "Installing project ($target)"
& $python -m pip install --editable $target
Write-Ok "dependencies installed"

if ($extras.Count -eq 0) {
  Write-Host "    (optional extras skipped; .\setup.ps1 -All installs OpenVINO + local LLM)" -ForegroundColor DarkGray
}

Write-Step "Verifying the install"
$env:PYTHONPATH = "src"
& $python -c @"
import sys
sys.path.insert(0, 'src')
import app.web  # the server entrypoint must import cleanly
from app.realtime.device_plan import probe_devices
inventory = probe_devices()
print('    import OK')
print(f'    compute devices: {\", \".join(inventory.available)}')
if inventory.probe_error:
    print(f'    note: {inventory.probe_error} (CPU fallback is normal and supported)')
"@
if ($LASTEXITCODE -ne 0) { throw "Verification failed - the project does not import cleanly." }

Write-Host ""
Write-Host "Setup complete. Start the server with:  .\run.ps1" -ForegroundColor Green
