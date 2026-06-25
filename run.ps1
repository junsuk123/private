$ErrorActionPreference = "Stop"

function Set-DefaultEnv($Name, $Value) {
  if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
  }
}

function Stop-ExistingLocalAppServers {
  $processIdsToStop = New-Object 'System.Collections.Generic.HashSet[int]'
  $connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -in @("127.0.0.1", "0.0.0.0") -and $_.LocalPort -ge 8000 -and $_.LocalPort -le 8050 }
  foreach ($connection in $connections) {
    $ownerProcessId = $connection.OwningProcess
    if (-not $ownerProcessId) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerProcessId" -ErrorAction SilentlyContinue
    if (-not $process -or -not $process.CommandLine) { continue }
    $command = $process.CommandLine.ToLowerInvariant()
    $isPython = $command.Contains("python.exe") -or $command.Contains("python ")
    $isLocalApp = $command.Contains("run.py")
    if ($isPython -and $isLocalApp) {
      Write-Host "Stopping existing local app server on port $($connection.LocalPort) (PID $ownerProcessId)"
      [void]$processIdsToStop.Add([int]$ownerProcessId)
      if ($process.ParentProcessId) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ParentProcessId)" -ErrorAction SilentlyContinue
        if ($parent -and $parent.CommandLine) {
          $parentCommand = $parent.CommandLine.ToLowerInvariant()
          $parentIsPython = $parentCommand.Contains("python.exe") -or $parentCommand.Contains("python ")
          $parentIsLocalApp = $parentCommand.Contains("run.py")
          if ($parentIsPython -and $parentIsLocalApp) {
            [void]$processIdsToStop.Add([int]$parent.ProcessId)
          }
        }
      }
    }
  }

  foreach ($processIdToStop in $processIdsToStop) {
    Stop-Process -Id $processIdToStop -Force -ErrorAction SilentlyContinue
  }

  for ($attempt = 0; $attempt -lt 20; $attempt++) {
    $stillListening = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
      Where-Object { $_.LocalAddress -in @("127.0.0.1", "0.0.0.0") -and $_.LocalPort -ge 8000 -and $_.LocalPort -le 8050 }
    if (-not $stillListening) { return }
    Start-Sleep -Milliseconds 250
  }
}

function Stop-ProcessTree {
  param([int]$RootProcessId)

  if (-not $RootProcessId) { return }
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    Stop-ProcessTree -RootProcessId ([int]$child.ProcessId)
  }
  Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

function Find-BrowserExecutable {
  $candidates = @(
    (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"),
    (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe")
  )
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path $candidate)) { return $candidate }
  }
  return $null
}

function Wait-LocalAppReady {
  param(
    [string]$Url,
    [System.Diagnostics.Process]$ServerProcess,
    [int]$TimeoutSeconds = 60
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    if ($ServerProcess.HasExited) {
      throw "Local app server exited before it became ready. Check logs\run-server.err.log."
    }
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  throw "Local app server did not become ready within $TimeoutSeconds seconds."
}

Stop-ExistingLocalAppServers

Set-DefaultEnv "PYTHONPATH" "src"
Set-DefaultEnv "APP_ENV" "local"
Set-DefaultEnv "APP_PORT" "8010"
Set-DefaultEnv "DATA_ENV" "realtime"
Set-DefaultEnv "TRADING_MODE" "learning"
Set-DefaultEnv "LIVE_TRADING_ENABLED" "false"
Set-DefaultEnv "ONTOLOGY_ACCELERATOR" "NPU"
Set-DefaultEnv "REALTIME_LATENCY_PROFILE" "low_latency"
Set-DefaultEnv "OPENVINO_DEVICE" "NPU"
Set-DefaultEnv "OPENVINO_HINT_PERFORMANCE_MODE" "LATENCY"
Set-DefaultEnv "OPENVINO_ENABLE_CPU_PINNING" "YES"
Set-DefaultEnv "OPENVINO_CACHE_DIR" (Join-Path $PSScriptRoot "data\runtime\openvino_cache")
Set-DefaultEnv "LLM_EVENT_INFERENCE_BACKEND" "openvino"
Set-DefaultEnv "LLM_EVENT_DEVICE" "NPU"

$embeddedModelPath = Join-Path $PSScriptRoot "models\local-llm\event-classifier"
if (-not [Environment]::GetEnvironmentVariable("LLM_EVENT_PROVIDER", "Process")) {
  if (Test-Path $embeddedModelPath) {
    [Environment]::SetEnvironmentVariable("LLM_EVENT_PROVIDER", "embedded", "Process")
    [Environment]::SetEnvironmentVariable("LLM_EVENT_MODEL", $embeddedModelPath, "Process")
    Set-DefaultEnv "LLM_EVENT_MODEL_CACHE_DIR" (Join-Path $PSScriptRoot "models\local-llm\cache")
    Set-DefaultEnv "LLM_EVENT_LOCAL_FILES_ONLY" "true"
    Set-DefaultEnv "LLM_EVENT_DEVICE" "NPU"
  } else {
    [Environment]::SetEnvironmentVariable("LLM_EVENT_PROVIDER", "local", "Process")
    Set-DefaultEnv "LLM_EVENT_MODEL" "qwen2.5:1.5b-instruct"
    Set-DefaultEnv "LLM_EVENT_LOCAL_ENDPOINT" "http://127.0.0.1:11434/v1/chat/completions"
  }
}
if (-not [Environment]::GetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "Process")) {
  $provider = [Environment]::GetEnvironmentVariable("LLM_EVENT_PROVIDER", "Process")
  if (($provider -eq "embedded" -or $provider -eq "inprocess" -or $provider -eq "transformers" -or $provider -eq "multimodal") -and (Test-Path ([Environment]::GetEnvironmentVariable("LLM_EVENT_MODEL", "Process")))) {
    [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "true", "Process")
  } elseif ($provider -eq "local" -or $provider -eq "ollama") {
    try {
      $localLlm = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:11434/api/tags" -TimeoutSec 1
      if ($localLlm.StatusCode -eq 200) {
        [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "true", "Process")
      } else {
        [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "false", "Process")
      }
    } catch {
      [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "false", "Process")
    }
  } else {
    if ([Environment]::GetEnvironmentVariable("LLM_EVENT_MODEL", "Process")) {
      [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "true", "Process")
    } else {
      [Environment]::SetEnvironmentVariable("LLM_EVENT_CLASSIFIER_ENABLED", "false", "Process")
    }
  }
}
Set-DefaultEnv "LIVE_REFRESH_SECONDS" "15"
Set-DefaultEnv "LEARNING_COLLECTION_INTERVAL_SECONDS" "60"
Set-DefaultEnv "AUTO_START_LIVE_WORKER" "false"
Set-DefaultEnv "RESEARCH_RETENTION_DAYS" "30"
Set-DefaultEnv "ANALYSIS_MARKET_LIMIT" "300"
Set-DefaultEnv "ONTOLOGY_NPU_BATCH_SIZE" "2048"
Set-DefaultEnv "SIM_STRATEGY_CANDIDATES" "1800"
Set-DefaultEnv "SIM_STREAMING_UNIVERSE_LIMIT" "300"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_SOURCE" "1"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_RUN" "1"
Set-DefaultEnv "LLM_EVENT_KNOWN_TICKER_PROMPT_LIMIT" "80"
Set-DefaultEnv "LLM_EVENT_RESPONSE_MAX_TOKENS" "180"
Set-DefaultEnv "LLM_EVENT_TIMEOUT_SECONDS" "12"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

$logsDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$serverOutLog = Join-Path $logsDir "run-server.out.log"
$serverErrLog = Join-Path $logsDir "run-server.err.log"
$port = [int]([Environment]::GetEnvironmentVariable("APP_PORT", "Process"))
$url = "http://127.0.0.1:$port"
$server = $null
$browser = $null

try {
  $server = Start-Process `
    -FilePath $python `
    -ArgumentList @(".\run.py", "--skip-startup-checks", "--port", "$port", "--strict-port") `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $serverOutLog `
    -RedirectStandardError $serverErrLog `
    -PassThru `
    -WindowStyle Hidden

  Write-Host "Starting local app server (PID $($server.Id))..."
  Wait-LocalAppReady -Url $url -ServerProcess $server
  Write-Host "Web UI: $url"
  Write-Host "Server logs: $serverOutLog"

  $browserExe = Find-BrowserExecutable
  if ($browserExe) {
    $browserProfile = Join-Path $PSScriptRoot "data\runtime\managed-browser-profile"
    New-Item -ItemType Directory -Force -Path $browserProfile | Out-Null
    $browser = Start-Process `
      -FilePath $browserExe `
      -ArgumentList @(
        "--app=$url",
        "--user-data-dir=$browserProfile",
        "--no-first-run",
        "--disable-extensions"
      ) `
      -PassThru
    Write-Host "Opened managed browser window (PID $($browser.Id))."
    Write-Host "Close that browser window to stop the server, or press Ctrl+C here to close both."
  } else {
    Start-Process $url | Out-Null
    Write-Host "No Chrome or Edge executable was found for managed mode."
    Write-Host "Press Ctrl+C here to stop the server."
  }

  while ($true) {
    if ($server.HasExited) {
      Write-Host "Server process exited."
      break
    }
    if ($browser -and $browser.HasExited) {
      Write-Host "Managed browser window closed. Stopping server..."
      break
    }
    Start-Sleep -Milliseconds 500
  }
} finally {
  if ($browser -and -not $browser.HasExited) {
    Stop-ProcessTree -RootProcessId ([int]$browser.Id)
  }
  if ($server -and -not $server.HasExited) {
    Stop-ProcessTree -RootProcessId ([int]$server.Id)
  }
  Stop-ExistingLocalAppServers
  Write-Host "Local app stopped."
}
