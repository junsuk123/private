param(
  [string]$TaskName = "GPT Work Report Receiver - private",
  [int]$IntervalMinutes = 2
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $workspace ".venv\Scripts\python.exe"
$worker = Join-Path $workspace ".codex-monitor\work_report_receiver.py"

if (-not (Test-Path -LiteralPath $python)) {
  throw "Workspace Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $worker)) {
  throw "GPT Work receiver was not found: $worker"
}
if ($IntervalMinutes -lt 1) {
  throw "IntervalMinutes must be at least 1."
}

$action = New-ScheduledTaskAction `
  -Execute $python `
  -Argument ('"{0}" --watch-once' -f $worker) `
  -WorkingDirectory $workspace
$trigger = New-ScheduledTaskTrigger `
  -Once `
  -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
  -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$principal = New-ScheduledTaskPrincipal `
  -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
  -LogonType Interactive `
  -RunLevel Limited

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Principal $principal `
  -Description "Validate, process and audit ChatGPT Work analysis reports in an explicit Codex session; deployment remains disabled." `
  -Force | Out-Null

foreach ($directory in @("incoming", "accepted", "processed", "failed", "results", "locks")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $workspace ".codex-monitor\$directory") | Out-Null
}
Write-Host "Installed scheduled task: $TaskName"
Write-Host "Drop completed schema-v1 reports into .codex-monitor\incoming as *.ready.json"
Write-Host "Deployment/restart remains disabled; use safe_deploy_gate.py for dry-run review only."
