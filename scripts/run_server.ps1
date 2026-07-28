$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$envFile = Join-Path $workspace "config\secrets\kis_api_keys.env"
if (Test-Path -LiteralPath $envFile) {
  foreach ($rawLine in Get-Content -LiteralPath $envFile) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
    $pair = $line.Split("=", 2)
    $name = $pair[0].Trim()
    $value = $pair[1].Trim().Trim('"').Trim("'")
    if ($name) { [Environment]::SetEnvironmentVariable($name, $value, "Process") }
  }
}

$listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
  $command = if ($owner) { $owner.CommandLine } else { "unknown" }
  throw "Port 8010 is already in use by PID $($listener.OwningProcess): $command"
}

& (Join-Path $workspace ".venv\Scripts\python.exe") -m uvicorn app.web:app --app-dir src --host 127.0.0.1 --port 8010
