param(
  [switch]$Headless,
  # Stop a running server even when it is holding a position. A restart discards
  # the in-memory stop/target/trailing state for an open trade, so this is opt-in.
  [switch]$ForceRestart,
  # Skip the graceful request entirely and kill immediately (last resort, e.g. the
  # server is hung and not answering HTTP).
  [switch]$HardKill
)

$ErrorActionPreference = "Stop"

function Set-DefaultEnv($Name, $Value) {
  if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
  }
}

function Set-RunEnv($Name, $Value) {
  [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Get-LocalAppServerListeners {
  $listeners = @()
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
      $listeners += [pscustomobject]@{
        Port      = [int]$connection.LocalPort
        ProcessId = [int]$ownerProcessId
        ParentId  = if ($process.ParentProcessId) { [int]$process.ParentProcessId } else { 0 }
      }
    }
  }
  return $listeners
}

function Test-PortRangeFree {
  $stillListening = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -in @("127.0.0.1", "0.0.0.0") -and $_.LocalPort -ge 8000 -and $_.LocalPort -le 8050 }
  return (-not $stillListening)
}

function Stop-LocalAppServerProcessTree {
  param([pscustomobject]$Listener)

  $processIdsToStop = New-Object 'System.Collections.Generic.HashSet[int]'
  [void]$processIdsToStop.Add($Listener.ProcessId)
  # run.py spawns a child that owns the socket; killing only one leaves an orphan
  # holding the port, which then looks like "the new server failed to bind".
  if ($Listener.ParentId) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($Listener.ParentId)" -ErrorAction SilentlyContinue
    if ($parent -and $parent.CommandLine) {
      $parentCommand = $parent.CommandLine.ToLowerInvariant()
      $parentIsPython = $parentCommand.Contains("python.exe") -or $parentCommand.Contains("python ")
      if ($parentIsPython -and $parentCommand.Contains("run.py")) {
        [void]$processIdsToStop.Add([int]$parent.ProcessId)
      }
    }
  }
  foreach ($processIdToStop in $processIdsToStop) {
    Stop-Process -Id $processIdToStop -Force -ErrorAction SilentlyContinue
  }
}

function Stop-ExistingLocalAppServers {
  <#
    Replace a running server WITHOUT force-killing it.

    A force kill terminates the realtime trading engine mid-cycle and the SQLite
    writers mid-operation. Worse, it silently abandons the in-memory exit state of
    an open position -- the armed stop, target, trailing high-watermark and holding
    clock. The broker keeps the position; nothing is left managing it.

    So: ask the server whether stopping is safe, ask it to stop itself, wait, and
    only force-kill if it will not go. Force-kill remains available because a hung
    server that cannot answer HTTP must still be replaceable.
  #>
  $listeners = Get-LocalAppServerListeners
  if (-not $listeners) { return $true }

  foreach ($listener in $listeners) {
    $base = "http://127.0.0.1:$($listener.Port)"
    Write-Host "Found existing local app server on port $($listener.Port) (PID $($listener.ProcessId))"

    if ($HardKill) {
      Write-Host "  -HardKill: skipping the graceful request and terminating now."
      Stop-LocalAppServerProcessTree -Listener $listener
      continue
    }

    # 1. Is stopping safe? The server fails this check closed, so an unreadable
    #    account reports unsafe rather than "probably fine".
    $safe = $false
    $safetyKnown = $false
    try {
      $safety = Invoke-RestMethod -Uri "$base/api/system/restart-safety" -TimeoutSec 20
      $safetyKnown = $true
      $safe = [bool]$safety.safe
      if (-not $safe) {
        Write-Host "  Restart is NOT safe: $($safety.reasons -join ', ')"
        if ($null -ne $safety.holdings_count) {
          Write-Host "  Holdings: $($safety.holdings_count) $(if ($safety.positions) { '(' + ($safety.positions -join ', ') + ')' })"
        }
      }
    } catch {
      Write-Host "  Could not read restart safety ($($_.Exception.Message))."
    }

    if ($safetyKnown -and -not $safe -and -not $ForceRestart) {
      Write-Host ""
      Write-Host "ABORTING: the running server is holding managed state." -ForegroundColor Yellow
      Write-Host "Restarting now would leave an open position with no stop, target or" -ForegroundColor Yellow
      Write-Host "trailing logic watching it. Choose one:" -ForegroundColor Yellow
      Write-Host "  * flatten first, then rerun .\run.ps1" -ForegroundColor Yellow
      Write-Host "  * keep the current server running (it is already healthy)" -ForegroundColor Yellow
      Write-Host "  * .\run.ps1 -ForceRestart   (accepts abandoning that state)" -ForegroundColor Yellow
      return $false
    }

    # 2. Ask it to stop itself, so workers unwind in order.
    $graceful = $false
    try {
      $query = if ($ForceRestart) { "?force=true" } else { "" }
      $response = Invoke-RestMethod -Method Post -Uri "$base/api/system/graceful-shutdown$query" -TimeoutSec 30
      if ($response.ok) {
        Write-Host "  Graceful shutdown accepted; waiting for the process to exit."
        $graceful = $true
      } else {
        Write-Host "  Graceful shutdown refused: $($response.message)"
      }
    } catch {
      Write-Host "  Graceful shutdown request failed ($($_.Exception.Message))."
    }

    if ($graceful) {
      $deadline = (Get-Date).AddSeconds(30)
      while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $listener.ProcessId -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 400
      }
    }

    # 3. Force-kill only what refused to leave.
    if (Get-Process -Id $listener.ProcessId -ErrorAction SilentlyContinue) {
      Write-Host "  Process did not exit in time; terminating it."
      Stop-LocalAppServerProcessTree -Listener $listener
    } else {
      Write-Host "  Previous server stopped cleanly."
    }
  }

  # 4. The port must actually be released before the new server tries to bind.
  for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if (Test-PortRangeFree) { return $true }
    Start-Sleep -Milliseconds 250
  }
  Write-Host "WARNING: a listener is still bound in 8000-8050 after waiting." -ForegroundColor Yellow
  return $true
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
    # Rebuilding the live feature/GNN state from a large realtime store can
    # legitimately take longer than one minute.  Do not tear down a healthy
    # boot while the application is still restoring that state.
    [int]$TimeoutSeconds = 180
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

if (-not (Stop-ExistingLocalAppServers)) {
  # The running server is holding managed state and -ForceRestart was not given.
  # Leaving it alone is the safe outcome, so exit without starting a second one.
  exit 2
}
# Safety net for a run.py that is not (yet) listening — mid-boot or hung — and so
# was invisible to the graceful path above.
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
# Real orders auto-submit with NO manual arming file (operator's explicit choice).
# To run WITHOUT placing real orders, set this to "true" (then arm via
# scripts/arm_live_trading.py) or set LIVE_ORDER_SUBMIT_ENABLED "false".
Set-RunEnv "REQUIRE_MANUAL_ARMING" "false"
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
Set-DefaultEnv "REALTIME_ENABLE_ROUTINE_LOSS_SELL" "true"
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
Set-DefaultEnv "REALTIME_TAKE_PROFIT" "0.008"
Set-DefaultEnv "AUTO_RELIABILITY_US_WARM_SYMBOLS" "6"
Set-DefaultEnv "KIS_OVERSEAS_REALTIME_MAX_SYMBOLS" "6"
Set-DefaultEnv "REALTIME_US_ROTATION_POOL_MULTIPLIER" "3"
# Keep each window for one full model-label horizon before rotating it.
Set-DefaultEnv "REALTIME_US_WATCHLIST_RECHECK_SEC" "600"
Set-DefaultEnv "GNN_TRUST_HORIZON_SECONDS" "1800"
Set-DefaultEnv "STRATEGY_SESSION_MIN_NET_TARGET_BPS" "25"
Set-DefaultEnv "REALTIME_MIN_NET_PROFIT_BUFFER_RATE" "0.0"
Set-DefaultEnv "REALTIME_COLLECTOR_MAX_SYMBOLS" "40"
# 세션 앵커: 로테이션에서 제외하고 장 내내 유지하는 소수 종목.
# 세션 구조 전략(market_intraday_momentum)은 같은 종목의 09:00-09:30과 14:50-15:20을
# 동시에 필요로 하는데, 수집기는 300초마다 종목을 교체한다. 실측: 저장된 KRX
# 360 심볼-일 중 두 구간을 모두 가진 것이 단 2건이어서 평가 자체가 불가능했다.
Set-DefaultEnv "REALTIME_SESSION_ANCHOR_SYMBOLS" "005930,000660"
Set-DefaultEnv "REALTIME_SESSION_ANCHOR_MAX" "2"
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
Set-RunEnv "REALTIME_BUY_ENABLED" "true"
Set-RunEnv "LIVE_TERMINATION_SELL_ONLY_ON_START" "false"
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
Set-DefaultEnv "LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS" "180"
Set-DefaultEnv "AUTO_START_LIVE_WORKER" "true"
Set-DefaultEnv "AUTO_START_REALTIME_TRADING" "true"
# 데이터 수집은 실시간(KIS 수집기+트레이딩 평가 저널링), 학습은 주기적으로 백그라운드 재학습.
Set-DefaultEnv "AUTO_START_LIVE_TRAINING" "true"
Set-DefaultEnv "LIVE_TRAINING_INTERVAL_SECONDS" "60"
# 투자자별 매매동향(개인/외국인/기관 순매수) 일일 갱신. KIS는 이 값을 영업일 단위로만
# 제공하고, residual_relative_strength는 이 정보를 필수 조건으로 쓴다. 갱신이 멈추면
# 저장된 30영업일 창이 밀려나면서 해당 전략이 조용히 평가 불가 상태로 돌아간다.
# 6시간 주기인 이유: 당일 수치는 장중 계속 변하므로 24시간 주기면 정작 필요한
# 당일 데이터가 하루 대부분 낡은 상태로 남는다. 읽기 전용 조회만 사용한다.
Set-DefaultEnv "AUTO_START_INVESTOR_FLOW_REFRESH" "true"
Set-DefaultEnv "INVESTOR_FLOW_REFRESH_SECONDS" "21600"
Set-DefaultEnv "INVESTOR_FLOW_MINIMUM_BARS" "100"
Set-DefaultEnv "LIVE_SIGNAL_MODEL_INFERENCE_ENABLED" "true"
Set-DefaultEnv "RESEARCH_RETENTION_DAYS" "30"
Set-DefaultEnv "ANALYSIS_MARKET_LIMIT" "300"
Set-DefaultEnv "ONTOLOGY_NPU_BATCH_SIZE" "4096"
Set-DefaultEnv "ONTOLOGY_FILTER1_TARGET_COUNT" "80"
Set-DefaultEnv "SIM_STRATEGY_CANDIDATES" "160"
Set-DefaultEnv "SIM_STREAMING_UNIVERSE_LIMIT" "160"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_SOURCE" "1"
Set-DefaultEnv "LLM_EVENT_MAX_ITEMS_PER_RUN" "3"
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
# --- 진입 체결 방식: 스프레드를 지불하지 않고 게시한다 -------------------------
# 기존에는 매수를 best_ask에, 매도를 best_bid에 넣어 왕복 스프레드 전액을 지불했다.
# 실측 KRX 스프레드는 13~50bps(005930 약 19bps)인데 비용 모델의 spread_rate는 0이고
# 학습 라벨은 신호봉 종가 체결을 가정한다 — 즉 모델이 채점한 체결가를 실행이 한 번도
# 시도하지 않고 있었다. 전략 플랜 자체도 passive_limit을 선언하고 있었다.
#
# 진입은 미체결이면 그냥 거래를 안 하는 것이므로 비용이 0이다. gross 엣지가 ~0인
# 상황에서는 -19bps로 거래하는 것보다 거래하지 않는 편이 낫다. 그래서 기본 ON.
Set-DefaultEnv "EXEC_PASSIVE_ENTRY" "true"
Set-DefaultEnv "EXEC_PASSIVE_ENTRY_OFFSET_TICKS" "0"
# 익절은 다르다. 미체결이면 계속 보유하는 것이고 열린 수익을 반납할 위험이 실재한다.
# 그래서 기본 OFF이며, 켤 경우 트레일링/시간청산이 대기 리스크의 상한이 된다.
Set-DefaultEnv "EXEC_PASSIVE_TAKE_PROFIT" "false"
# 손절/하드스톱/긴급 매도는 절대 passive로 바뀌지 않는다(미체결 손절 = 무한 손실).
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

  $browserExe = if ($Headless) { $null } else { Find-BrowserExecutable }
  if ($Headless) {
    Write-Host "Headless mode: server will keep running without a managed browser."
  } elseif ($browserExe) {
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
    # Our own server gets the same courtesy as one we are replacing: ask it to
    # unwind its trading engine and feeds before the process dies. This path runs
    # when the managed browser closes or on Ctrl+C, which is the ordinary way this
    # server is stopped -- so it must not be the one path that kills it hard.
    #
    # force=true here on purpose: the operator has already decided to stop, and a
    # refusal at this point would leave the process running after "Local app
    # stopped." was printed, which is worse than an acknowledged unsafe stop.
    try {
      Invoke-RestMethod -Method Post -Uri "$url/api/system/graceful-shutdown?force=true" -TimeoutSec 20 | Out-Null
      $deadline = (Get-Date).AddSeconds(25)
      while ((Get-Date) -lt $deadline -and -not $server.HasExited) {
        Start-Sleep -Milliseconds 400
      }
    } catch {
      Write-Host "Graceful stop request failed ($($_.Exception.Message)); terminating."
    }
    if (-not $server.HasExited) {
      Stop-ProcessTree -RootProcessId ([int]$server.Id)
    }
  }
  # Whatever is still bound after that is not ours to negotiate with.
  if (-not (Test-PortRangeFree)) {
    foreach ($listener in Get-LocalAppServerListeners) {
      Stop-LocalAppServerProcessTree -Listener $listener
    }
  }
  Write-Host "Local app stopped."
}
