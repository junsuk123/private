#!/usr/bin/env pwsh
# The shebang makes `./setup.ps1` work from a Linux shell. PowerShell treats it as
# an ordinary comment, so Windows is unaffected, and a comment is allowed before
# the help block.
<#
.SYNOPSIS
  One command to make a fresh machine able to run OBAITS.

.DESCRIPTION
  Creates the virtual environment for THIS operating system, installs the project
  and its dependencies, and verifies the result actually imports. Safe to re-run:
  every step is idempotent.

  Windows and Linux get separate environments (.venv and .venv-linux) because this
  project directory is synchronised between machines and a venv is portable across
  neither the OS nor the interpreter that built it.

  Accelerator extras are selected from this machine's architecture and visible
  devices. Use -BaseOnly for a deliberately minimal CPU installation. Runtime
  placement still falls back to CPU (see app/realtime/device_plan.py).

.EXAMPLE
  ./setup.ps1
  ./setup.ps1 -WithNpu
  ./setup.ps1 -All
#>
param(
  # Install the OpenVINO extra so Intel GPU/NPU placement becomes available.
  [switch]$WithNpu,
  # Install the local-LLM extra (large download).
  [switch]$WithLocalLlm,
  # Everything optional.
  [switch]$All,
  # Install only the portable base dependencies; skip automatic accelerator extras.
  [switch]$BaseOnly,
  # Rebuild the virtual environment from scratch.
  [switch]$Recreate,
  # Build the local-LLM extra against a specific CUDA wheel index, e.g. "cu128".
  # Only meaningful on Linux with an NVIDIA GPU; see the note further down about
  # why the default PyPI wheel is not always the right one.
  [string]$CudaWheels = ""
)

$ErrorActionPreference = "Stop"
# Every path below is derived from this script's own location, so the project can
# be cloned or renamed to anything without editing a line.
$root = $PSScriptRoot
Set-Location $root

# Windows PowerShell 5.1 does not define $IsWindows; PowerShell 7 defines all
# three. An undefined variable therefore means 5.1, which only ever runs on Windows.
$onWindows = if ($null -eq $IsWindows) { $true } else { [bool]$IsWindows }

# uv, when present, is used both to provision an interpreter and to install. It is
# resolved once here because the venv-creation step and the install step both need
# to know whether it exists.
$uv = Get-Command uv -ErrorAction SilentlyContinue

if ($onWindows) {
  $venv = Join-Path $root ".venv"
  $python = Join-Path (Join-Path $venv "Scripts") "python.exe"
} else {
  $venv = Join-Path $root ".venv-linux"
  $python = Join-Path (Join-Path $venv "bin") "python"
}

function Write-Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Write-Ok($message) { Write-Host "    $message" -ForegroundColor Green }

if ($Recreate -and (Test-Path $venv)) {
  Write-Step "Removing existing $(Split-Path $venv -Leaf) (-Recreate)"
  Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $python)) {
  Write-Step "Creating virtual environment at $venv"
  # The interpreter has to be 3.11+ (app/schemas/domain.py uses enum.StrEnum), and
  # on a distro whose "python3" is older that interpreter may not be on PATH at
  # all. uv can fetch a suitable one, so it is tried first where it exists;
  # everything after that is the ordinary stdlib venv path.
  $created = $false
  if ($uv) {
    Write-Ok "using uv to provision Python 3.12"
    # --seed installs pip into the new environment. Without it uv creates a
    # pip-less venv, which is fine for uv itself but leaves "python -m pip" -
    # the fallback path below, and what anyone typing into this venv by hand will
    # reach for - reporting "No module named pip".
    & $uv.Source venv --seed --python 3.12 $venv
    if ($LASTEXITCODE -eq 0 -and (Test-Path $python)) { $created = $true }
  }
  if (-not $created) {
    $base = $null
    # Newest-first: a bare "python3" on Ubuntu 22.04 is 3.10, which this project
    # cannot use, so the versioned names are tried before the generic ones.
    $candidates = if ($onWindows) {
      @("py", "python", "python3")
    } else {
      @("python3.13", "python3.12", "python3.11", "python3", "python")
    }
    foreach ($candidate in $candidates) {
      $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
      if (-not $resolved) { continue }
      $base = $resolved.Source
      break
    }
    if (-not $base) {
      throw "No Python interpreter found on PATH. Install Python 3.11+ (or uv) and re-run."
    }
    if ((Split-Path $base -Leaf) -eq "py.exe") {
      & $base -3 -m venv $venv
    } else {
      & $base -m venv $venv
    }
  }
  Write-Ok "created $venv"
} else {
  Write-Ok "$(Split-Path $venv -Leaf) already present"
}

if (-not (Test-Path $python)) {
  throw "Virtual environment did not produce $python"
}

$version = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Ok "python $version"
$tooOld = & $python -c "import sys; print('1' if sys.version_info < (3, 11) else '0')"
if ($tooOld -eq "1") {
  throw "This project requires Python 3.11 or newer; $venv has $version. Re-run with -Recreate after installing a newer interpreter (or uv)."
}

function Install-Packages {
    <#
      Install into $venv with whichever installer this machine has.

      A venv created by `uv venv` without --seed contains no pip at all, so
      "python -m pip" is not a safe assumption even inside a working environment.
      Preferring uv also avoids re-resolving what uv already has cached.
    #>
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PackageArguments)
    if ($uv) {
        & $uv.Source pip install --python $python @PackageArguments
    } else {
        & $python -m pip install @PackageArguments
    }
}

if (-not $uv) {
    Write-Step "Upgrading installer tooling"
    & $python -m pip install --quiet --upgrade pip setuptools wheel
    Write-Ok "pip/setuptools/wheel current"
} else {
    Write-Ok "installing with uv"
}

# Extras are additive. On ordinary x64 PCs OpenVINO is installed automatically so
# Intel CPU/GPU/NPU discovery is available. A visible NVIDIA GPU also enables the
# small-model PyTorch extra; language models are installed only when requested.
$extras = @()
$isX64 = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -eq "X64"
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$autoNpu = (-not $BaseOnly) -and $isX64
$autoCuda = (-not $BaseOnly) -and [bool]$nvidia
if ($WithNpu -or $All -or $autoNpu) { $extras += "npu" }
if ($autoCuda -or $All) { $extras += "cuda" }
if ($WithLocalLlm -or $All) { $extras += "local-llm" }
$extras = @($extras | Select-Object -Unique)
$target = if ($extras.Count -gt 0) { ".[" + ($extras -join ",") + "]" } else { "." }

Write-Step "Installing project ($target)"
Install-Packages --editable $target
if ($LASTEXITCODE -ne 0) { throw "Install failed for $target" }
Write-Ok "dependencies installed"

if ($extras.Count -eq 0) {
  Write-Host "    (accelerator extras skipped; remove -BaseOnly or use -All to enable them)" -ForegroundColor DarkGray
} elseif ($autoNpu -or $autoCuda) {
  Write-Ok "auto-selected extras for this machine: $($extras -join ', ')"
}

# --- CUDA wheel selection -----------------------------------------------------
# PyPI's default torch wheel is built against the newest CUDA runtime, which a
# machine on an older driver cannot load: torch imports fine and then reports
# cuda_available=False with a "driver is too old" warning, so the GPU silently
# does nothing while everything still appears to work. -CudaWheels pins the wheel
# index to the CUDA version the installed driver actually supports.
if ($CudaWheels -and (($extras -contains "cuda") -or ($extras -contains "local-llm"))) {
  Write-Step "Reinstalling torch from the $CudaWheels wheel index"
  Install-Packages --upgrade --index-url "https://download.pytorch.org/whl/$CudaWheels" torch
  if ($LASTEXITCODE -ne 0) { throw "Could not install torch from the $CudaWheels index." }
  Write-Ok "torch pinned to $CudaWheels"
}

Write-Step "Verifying the install"
$env:PYTHONPATH = "src"
# A LITERAL here-string (@' … '@). The expandable form would have PowerShell
# interpret $-names and escapes inside Python source; the previous version used it
# and its f-string separator arrived at Python as a stray backslash, so this whole
# verification step died with a SyntaxError instead of checking anything.
$verification = @'
import sys
sys.path.insert(0, "src")
import app.web  # the server entrypoint must import cleanly
from app.realtime.device_plan import probe_devices

inventory = probe_devices()
print("    import OK")
print("    OpenVINO devices: " + ", ".join(inventory.available))
if inventory.probe_error:
    print(f"    note: {inventory.probe_error} (CPU fallback is normal and supported)")
try:
    import torch
except ImportError:
    print("    torch: not installed (local-LLM extra not selected)")
else:
    if torch.cuda.is_available():
        print(f"    torch {torch.__version__} CUDA: {torch.cuda.get_device_name(0)}")
    else:
        # Not fatal, but it IS the difference between using the GPU and not, so it
        # is reported rather than left for someone to notice in a latency graph.
        print(f"    torch {torch.__version__}: CUDA unavailable (running on CPU)")
'@
$verificationPath = Join-Path ([System.IO.Path]::GetTempPath()) "obaits-setup-verify-$PID.py"
try {
  # Windows PowerShell 5.1 strips embedded quotes while marshalling a multi-line
  # native -c argument. A temporary UTF-8 script keeps Python source byte-for-byte.
  [System.IO.File]::WriteAllText(
    $verificationPath,
    $verification,
    (New-Object System.Text.UTF8Encoding($false))
  )
  & $python $verificationPath
  if ($LASTEXITCODE -ne 0) { throw "Verification failed - the project does not import cleanly." }
} finally {
  Remove-Item -LiteralPath $verificationPath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
$startCommand = if ($onWindows) { ".\run.bat" } else { "./run.ps1" }
Write-Host "Setup complete. Start the server with:  $startCommand" -ForegroundColor Green
