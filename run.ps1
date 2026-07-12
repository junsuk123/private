$ErrorActionPreference = "Stop"

function Set-DefaultEnv($Name, $Value) {
  if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
  }
}

function Set-RunEnv($Name, $Value) {
  [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
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

function Stop-WorkspaceRunPyProcesses {
  $workspacePath = (Resolve-Path -LiteralPath $PSScriptRoot).Path.ToLowerInvariant()
  $currentProcessId = $PID
  $processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue
  foreach ($process in $processes) {
    if (-not $process.CommandLine) { continue }
    if ([int]$process.ProcessId -eq [int]$currentProcessId) { continue }
    $command = $process.CommandLine.ToLowerInvariant()
    $isWorkspaceRunPy = $command.Contains("run.py") -and (
      $command.Contains($workspacePath.ToLowerInvariant()) -or
      $command.Contains(".\run.py") -or
      $command.Contains("./run.py")
    )
    if ($isWorkspaceRunPy) {
      Write-Host "Stopping existing workspace run.py process (PID $($process.ProcessId))"
      Stop-ProcessTree -RootProcessId ([int]$process.ProcessId)
    }
  }
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

function Test-ManagedBrowserProfileRunning {
  param([string]$BrowserProfilePath)

  if (-not $BrowserProfilePath) { return $false }
  $resolvedProfilePath = (Resolve-Path -LiteralPath $BrowserProfilePath -ErrorAction SilentlyContinue)
  if ($resolvedProfilePath) {
    $profileNeedle = $resolvedProfilePath.Path.ToLowerInvariant()
  } else {
    $profileNeedle = $BrowserProfilePath.ToLowerInvariant()
  }

  $browserProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      ($_.Name -ieq "chrome.exe" -or $_.Name -ieq "msedge.exe") -and
      $_.CommandLine.ToLowerInvariant().Contains($profileNeedle)
    }
  return [bool]$browserProcesses
}

Stop-ExistingLocalAppServers
Stop-WorkspaceRunPyProcesses

Set-DefaultEnv "PYTHONPATH" "src"
Set-DefaultEnv "APP_ENV" "local"
Set-DefaultEnv "APP_PORT" "8010"
Set-DefaultEnv "DATA_ENV" "realtime"
Set-RunEnv "TRADING_MODE" "live_trading"
Set-RunEnv "LIVE_TRADING_ENABLED" "true"
Set-RunEnv "KIS_LIVE_ENABLED" "true"
Set-RunEnv "KIS_PAPER_TRADING" "false"
Set-RunEnv "LIVE_ORDER_SUBMIT_ENABLED" "true"
Set-DefaultEnv "KIS_ACCOUNT_CACHE_SECONDS" "3"
Set-DefaultEnv "REALTIME_SMALL_ACCOUNT_MODE" "true"
Set-DefaultEnv "REALTIME_SMALL_ACCOUNT_EQUITY_KRW" "300000"
Set-DefaultEnv "REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT" "1.25"
# --- Day-trading (단타) exit discipline (2026-07-07) --------------------------------
# Research-grounded (Van Tharp expectancy/R-multiple; Odean 1998 & Barber-Lee-Liu-Odean
# on the disposition effect; StockCharts ATR stops; PwC/EY KR 0.20% sell tax). The prior
# "investment mode" HELD losers (BLOCK_ONE_SHARE_LOSS_REDUCE + no loss exit) until the
# 3% hard stop — the classic 물림 that let small losses ride to -3%. Day trading requires
# cutting losses fast and realizing meaningful profits, so:
#  - allow loss exit + never block a 1-share stop or a below-breakeven stop
#  - tight net stop ~0.8% (≈0.5% gross) so each loss stays small
#  - take-profit net 1.4% (quick) / 0.8% (routine); NO tiny won-amount take-profit
#  - hard stop 2% capital backstop; emergency 5%
# 2026-07-13: the values below were previously wired the OPPOSITE of this comment
# (ALLOW_LOSS_EXIT=false, STOP_LOSS_NET=0.0, HARD_STOP=0.03) which disabled every
# routine stop and let losers ride to the 3% hard stop — the 물림 this block set out
# to fix. Corrected to match the documented discipline. Ordering: net 0.8% < hard 2%
# < emergency 5%. Validated by TradingPolicySnapshot.conflicts() (no STOP_LOSS_DISABLED).
# NOTE: research's top caveat — small-account day trading is structurally negative-
# expectancy (round-trip cost amplifies losses); this discipline MINIMIZES losses, it
# does not guarantee profit. Tune from realized PnL.
Set-DefaultEnv "REALTIME_ALLOW_LOSS_EXIT" "true"
Set-DefaultEnv "REALTIME_HARD_STOP_LOSS" "0.02"
Set-DefaultEnv "REALTIME_BLOCK_SELL_BELOW_BREAKEVEN" "false"
Set-DefaultEnv "REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE" "false"
Set-DefaultEnv "REALTIME_LOSS_EXIT_REDUCE_FRACTION" "0.5"
Set-DefaultEnv "REALTIME_EMERGENCY_STOP_LOSS" "0.05"
Set-DefaultEnv "REALTIME_DAILY_REALIZED_LOSS_LIMIT_KRW" "1500"
Set-DefaultEnv "REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW" "1000"
Set-DefaultEnv "REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE" "0.004"
Set-DefaultEnv "REALTIME_QUICK_TAKE_PROFIT_NET" "0.014"
Set-DefaultEnv "REALTIME_MIN_NET_PROFIT_EXIT" "0.008"
Set-DefaultEnv "REALTIME_STOP_LOSS_NET" "0.008"
Set-DefaultEnv "REALTIME_TAKE_PROFIT_AMOUNT_KRW" "0"
Set-DefaultEnv "REALTIME_PROFIT_LOCK_ARM_NET" "0.012"
Set-DefaultEnv "REALTIME_PROFIT_LOCK_GIVEBACK" "0.30"
# Small-account buy tuning synced to the Raspberry Pi node:
# keep the net-edge floor permissive enough for live fills, widen candidate discovery,
# and reduce quote delay so the local launcher behaves like the deployed service.
Set-DefaultEnv "REALTIME_MIN_BUY_NET_RETURN_KR" "0.0005"
Set-DefaultEnv "REALTIME_MIN_BUY_NET_RETURN_US" "0.0005"
Set-DefaultEnv "REALTIME_SMALL_ACCOUNT_EXTRA_NET" "0.0"
Set-DefaultEnv "REALTIME_ONE_SHARE_CASH_BUFFER" "1.03"
Set-DefaultEnv "REALTIME_FALLBACK_EDGE_BPS_PER_SCORE" "220"
Set-DefaultEnv "REALTIME_US_EXCLUDE_SYMBOL_SUFFIXES" "U,WS,WT,W,R,P"
Set-DefaultEnv "REALTIME_LOSS_REENTRY_COOLDOWN_SEC" "7200"
Set-DefaultEnv "REALTIME_LOSS_REBUY_COOLDOWN_SEC" "86400"
Set-DefaultEnv "REALTIME_LOSS_REBUY_RETURN_THRESHOLD" "-0.004"
Set-DefaultEnv "REALTIME_MAX_BUY_ORDERS_PER_CYCLE" "2"
Set-DefaultEnv "REALTIME_BUY_WEIGHT" "0.4"
Set-DefaultEnv "REALTIME_MIN_NET_PROFIT_BUFFER_RATE" "0.0"
Set-DefaultEnv "REALTIME_COLLECTOR_MAX_SYMBOLS" "40"
Set-DefaultEnv "REALTIME_BUY_CANDIDATE_LIMIT" "360"
Set-DefaultEnv "REALTIME_BUY_CANDIDATE_MAX_AGE_SEC" "180"
Set-DefaultEnv "REALTIME_MAX_BUY_EVALUATIONS_PER_CYCLE" "40"
Set-DefaultEnv "REALTIME_US_DISCOVERY_CANDIDATE_LIMIT" "16"
Set-DefaultEnv "REALTIME_KRX_DISCOVERY_CANDIDATE_LIMIT" "12"
Set-DefaultEnv "REALTIME_AFFORDABLE_CANDIDATE_TTL_SEC" "120"
Set-DefaultEnv "REALTIME_BROKER_QUOTE_DELAY_SEC" "0.15"
Set-DefaultEnv "REALTIME_VOLUME_SURGE_LIMIT" "16"
Set-DefaultEnv "REALTIME_US_FEATURE_WARM_LIMIT" "10"
Set-DefaultEnv "REALTIME_SYMBOL_VOLATILITY_WINDOW_SEC" "300"
Set-DefaultEnv "REALTIME_MARKET_VOLATILITY_WINDOW_SEC" "300"
Set-DefaultEnv "REALTIME_MAX_SYMBOL_VOLATILITY_BUY" "0.015"
Set-DefaultEnv "REALTIME_MAX_MARKET_VOLATILITY_BUY" "0.008"
Set-DefaultEnv "REALTIME_SYMBOL_VOLATILITY_REFERENCE" "0.006"
Set-DefaultEnv "REALTIME_MARKET_VOLATILITY_REFERENCE" "0.004"
Set-DefaultEnv "REALTIME_DOMESTIC_DRAWDOWN_BUY_TIGHTEN_TRIGGER" "0.005"
Set-DefaultEnv "REALTIME_DOMESTIC_DRAWDOWN_BUY_BONUS_MULTIPLIER" "6.0"
Set-DefaultEnv "REALTIME_DOMESTIC_DRAWDOWN_BUY_MAX_BONUS" "0.18"
Set-DefaultEnv "REALTIME_RUNTIME_PROBE_BUY_ENABLED" "true"
Set-DefaultEnv "REALTIME_RUNTIME_PROBE_BUY_MARGIN" "0.18"
Set-DefaultEnv "REALTIME_RUNTIME_PROBE_BUY_WEIGHT" "0.003"
Set-DefaultEnv "REALTIME_BUY_ENABLED" "true"
Set-DefaultEnv "REALTIME_IGNORE_SYMBOLS" "LCFYW"
Set-DefaultEnv "REALTIME_MODEL_AUXILIARY_ONLY" "true"
Set-DefaultEnv "REALTIME_DOMESTIC_BUY_CORE_SESSION_ONLY" "false"
Set-DefaultEnv "REALTIME_DOMESTIC_DRAWDOWN_REDUCE_TRIGGER" "0.015"
Set-DefaultEnv "REALTIME_DOMESTIC_EMERGENCY_EXIT_TRIGGER" "0.03"
Set-DefaultEnv "REALTIME_DOMESTIC_CONCENTRATION_REDUCE_WEIGHT" "0.60"
Set-DefaultEnv "ONTOLOGY_ACCELERATOR" "NPU"
Set-DefaultEnv "REALTIME_LATENCY_PROFILE" "low_latency"
Set-DefaultEnv "OPENVINO_DEVICE" "NPU"
Set-DefaultEnv "OPENVINO_HINT_PERFORMANCE_MODE" "LATENCY"
Set-DefaultEnv "OPENVINO_ENABLE_CPU_PINNING" "YES"
Set-DefaultEnv "OPENVINO_CACHE_DIR" (Join-Path $PSScriptRoot "data\runtime\openvino_cache")
Set-DefaultEnv "LLM_EVENT_INFERENCE_BACKEND" "openvino"
Set-DefaultEnv "LLM_EVENT_DEVICE" "NPU"

# Shared local-LLM config (config/local_llm.env): the single place to set the
# news/event sentiment model for both Windows and Raspberry Pi. Applied as
# defaults (only if not already set), so it runs before the provider logic below
# and lets one file pick the model for every machine.
$localLlmConfig = Join-Path $PSScriptRoot "config\local_llm.env"
if (Test-Path $localLlmConfig) {
  foreach ($line in Get-Content -LiteralPath $localLlmConfig) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#") -or ($trimmed -notmatch "=")) { continue }
    $pair = $trimmed.Split("=", 2)
    $key = $pair[0].Trim()
    $value = $pair[1].Trim().Trim('"').Trim("'")
    if ($key) { Set-DefaultEnv $key $value }
  }
}

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
Set-DefaultEnv "AUTO_START_REALTIME_TRADING" "true"
# 데이터 수집은 실시간(KIS 수집기+트레이딩 평가 저널링), 학습은 주기적으로 백그라운드 재학습.
Set-DefaultEnv "AUTO_START_LIVE_TRAINING" "true"
Set-DefaultEnv "LIVE_TRAINING_INTERVAL_SECONDS" "60"
Set-DefaultEnv "LIVE_SIGNAL_MODEL_INFERENCE_ENABLED" "false"
Set-DefaultEnv "RESEARCH_RETENTION_DAYS" "30"
Set-DefaultEnv "ANALYSIS_MARKET_LIMIT" "300"
Set-DefaultEnv "ONTOLOGY_NPU_BATCH_SIZE" "4096"
Set-DefaultEnv "ONTOLOGY_FILTER1_TARGET_COUNT" "80"
Set-DefaultEnv "SIM_STRATEGY_CANDIDATES" "160"
Set-DefaultEnv "SIM_STREAMING_UNIVERSE_LIMIT" "160"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_SOURCE" "1"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_RUN" "1"
Set-DefaultEnv "LLM_EVENT_KNOWN_TICKER_PROMPT_LIMIT" "80"
Set-DefaultEnv "LLM_EVENT_RESPONSE_MAX_TOKENS" "180"
Set-DefaultEnv "LLM_EVENT_TIMEOUT_SECONDS" "12"

# --- Execution-safety strict defaults (live) --------------------------------
# A BUY must have a fresh order book (last_price is a reference, not an executable
# price); a missing/stale book blocks the buy instead of pricing it at zero spread.
Set-DefaultEnv "EXEC_REQUIRE_ORDERBOOK_FOR_BUY" "true"
Set-DefaultEnv "EXEC_REQUIRE_FRESH_ORDERBOOK_FOR_BUY" "true"
Set-DefaultEnv "EXEC_MAX_ORDERBOOK_AGE_SEC" "3.0"
Set-DefaultEnv "EXEC_UNKNOWN_SPREAD_PENALTY_RATE" "0.006"
Set-DefaultEnv "EXEC_BUY_MAX_CHASE_BPS" "20"
# Urgent stops still exit without a book (discounted reference price); non-urgent
# no-book sells are held.
Set-DefaultEnv "EXEC_ALLOW_NO_ORDERBOOK_EMERGENCY_SELL" "true"
Set-DefaultEnv "EXEC_SELL_EMERGENCY_OFFSET_TICKS" "1"
Set-DefaultEnv "EXEC_SELL_STOP_OFFSET_TICKS" "1"
Set-DefaultEnv "EXEC_SELL_EMERGENCY_FALLBACK_OFFSET_RATE" "0.003"
# Never silently route an unknown US BUY to NASD in live mode.
Set-DefaultEnv "KIS_US_EXCHANGE_STRICT" "true"
Set-DefaultEnv "KIS_ALLOW_DEFAULT_US_EXCHANGE_IN_LIVE" "false"

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
$launchUrl = "$url/account"
$server = $null
$browser = $null
$browserProfile = $null
$browserStartedAt = $null

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
  Write-Host "Account dashboard: $launchUrl"
  Write-Host "Server logs: $serverOutLog"

  $browserExe = Find-BrowserExecutable
  if ($browserExe) {
    $browserProfile = Join-Path $PSScriptRoot "data\runtime\managed-browser-profile"
    New-Item -ItemType Directory -Force -Path $browserProfile | Out-Null
    $browserStartedAt = Get-Date
    $browser = Start-Process `
      -FilePath $browserExe `
      -ArgumentList @(
        "--app=$launchUrl",
        "--user-data-dir=$browserProfile",
        "--no-first-run",
        "--disable-extensions"
      ) `
      -PassThru
    Write-Host "Opened managed browser window (PID $($browser.Id))."
    Write-Host "Close that browser window to stop the server, or press Ctrl+C here to close both."
  } else {
    Start-Process $launchUrl | Out-Null
    Write-Host "No Chrome or Edge executable was found for managed mode."
    Write-Host "Press Ctrl+C here to stop the server."
  }

  while ($true) {
    if ($server.HasExited) {
      Write-Host "Server process exited."
      break
    }
    if ($browserProfile) {
      $browserStartupGraceElapsed = $browserStartedAt -and ((Get-Date) -gt $browserStartedAt.AddSeconds(5))
      if ($browserStartupGraceElapsed -and -not (Test-ManagedBrowserProfileRunning -BrowserProfilePath $browserProfile)) {
        Write-Host "Managed browser window closed. Stopping server..."
        break
      }
    } elseif ($browser -and $browser.HasExited) {
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
