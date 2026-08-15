#!/usr/bin/env pwsh
# The shebang makes `./run.ps1` work from a Linux shell. PowerShell treats it as
# an ordinary comment, so Windows is unaffected, and a comment is allowed before
# param().
param(
  [switch]$Headless,
  # Stop a running server even when it is holding a position. A restart discards
  # the in-memory stop/target/trailing state for an open trade, so this is opt-in.
  [switch]$ForceRestart,
  # Skip the graceful request entirely and kill immediately (last resort, e.g. the
  # server is hung and not answering HTTP).
  [switch]$HardKill,
  # Bind to every interface so a phone or the Pi can open the SAME dashboard this
  # machine sees. Off by default: this server has no authentication of its own and
  # accepts live-trading and shutdown requests, so reachability and a token are one
  # decision. A token is generated and printed if APP_ACCESS_TOKEN is not already set.
  [switch]$External
)

$ErrorActionPreference = "Stop"

# --- Platform -----------------------------------------------------------------
# This launcher runs on two machines with the same script: the Windows notebook it
# was written on and a Linux workstation. Everything platform-specific below goes
# through the three helpers in this section, so the operational logic - when it is
# safe to restart, how a server is asked to stop, what the finally block guarantees
# - is written once and is identical on both.
#
# Windows PowerShell 5.1 does not define $IsWindows; PowerShell 7 defines all
# three. An undefined variable therefore means 5.1, which only ever runs on
# Windows.
$script:OnWindows = if ($null -eq $IsWindows) { $true } else { [bool]$IsWindows }

function Get-ProcessTable {
  <#
    pid / ppid / name / full command line for every visible process, on either OS.

    Every process-matching decision in this script (is that listener ours? is that
    an orphaned launcher? is the managed browser still open?) is made on the
    command line, so both branches must return it in full. `ps -ww` is required for
    that: without it ps truncates args at the terminal width and the "run.py" this
    script matches on disappears from a long venv path.
  #>
  if ($script:OnWindows) {
    return Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | ForEach-Object {
      [pscustomobject]@{
        ProcessId = [int]$_.ProcessId
        ParentId  = if ($_.ParentProcessId) { [int]$_.ParentProcessId } else { 0 }
        Name      = [string]$_.Name
        Command   = [string]$_.CommandLine
      }
    }
  }
  $rows = @()
  foreach ($line in (& ps -eww -o pid=,ppid=,comm=,args= 2>$null)) {
    $parts = $line.Trim() -split '\s+', 4
    if ($parts.Count -lt 4) { continue }
    $rows += [pscustomobject]@{
      ProcessId = [int]$parts[0]
      ParentId  = [int]$parts[1]
      Name      = [string]$parts[2]
      Command   = [string]$parts[3]
    }
  }
  return $rows
}

function Get-ListeningEntries {
  <#
    Port + owning pid for every TCP socket in LISTEN state bound to loopback or to
    every interface. Ports outside the app's range are filtered by the callers.
  #>
  if ($script:OnWindows) {
    return Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
      Where-Object { $_.LocalAddress -in @("127.0.0.1", "0.0.0.0") } |
      ForEach-Object {
        [pscustomobject]@{ Port = [int]$_.LocalPort; ProcessId = [int]$_.OwningProcess }
      }
  }
  $entries = @()
  # -H drops the header, -p attaches the owning process. ss only reveals the pid
  # for sockets this user owns, which is exactly the set this script may act on.
  foreach ($line in (& ss -ltnpH 2>$null)) {
    if ($line -notmatch '(?:^|\s)(\S+):(\d+)\s') { continue }
    $address = $Matches[1]
    $port = [int]$Matches[2]
    # "*" is how ss renders the any-address; [::] is its IPv6 form, and a v6 any
    # socket is reachable on 127.0.0.1 too, so it counts as a local listener.
    if ($address -notin @("127.0.0.1", "0.0.0.0", "*", "[::]", "[::1]")) { continue }
    $processId = 0
    if ($line -match 'pid=(\d+)') { $processId = [int]$Matches[1] }
    $entries += [pscustomobject]@{ Port = $port; ProcessId = $processId }
  }
  return $entries
}

function Test-ProcessAlive {
  param([int]$ProcessId)
  if (-not $ProcessId) { return $false }
  return [bool](Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Join-ProjectPath {
  <#
    Build a path under the project root from separate segments.

    Join-Path with an embedded "a\b" is a Windows-only spelling: on Linux the
    backslash is an ordinary filename character, so "data\runtime\openvino_cache"
    became one directory with backslashes in its name instead of three nested ones.
    Passing the segments separately lets each OS insert its own separator.
    (PowerShell 7's multi-argument Join-Path would do this too, but Windows
    PowerShell 5.1 has no such parameter and this script still has to run there.)
  #>
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Segments)
  $path = $PSScriptRoot
  foreach ($segment in $Segments) { $path = Join-Path $path $segment }
  return $path
}

# A separate venv per OS on purpose: this project directory is synchronised
# between the Windows notebook and the Linux workstation, and a venv is portable
# across neither the OS nor the interpreter it was built from. Sharing one .venv
# would mean each machine silently breaking the other's interpreter.
$script:VenvPython = if ($script:OnWindows) {
  Join-ProjectPath ".venv" "Scripts" "python.exe"
} else {
  Join-ProjectPath ".venv-linux" "bin" "python"
}

function Set-DefaultEnv($Name, $Value) {
  if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
  }
}

function Set-RunEnv($Name, $Value) {
  [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Get-LocalAppServerListeners {
  $processes = @{}
  foreach ($row in Get-ProcessTable) { $processes[$row.ProcessId] = $row }

  $listeners = @()
  $seen = New-Object 'System.Collections.Generic.HashSet[string]'
  foreach ($entry in Get-ListeningEntries) {
    if ($entry.Port -lt 8000 -or $entry.Port -gt 8050) { continue }
    if (-not $entry.ProcessId) { continue }
    $process = $processes[$entry.ProcessId]
    if (-not $process -or -not $process.Command) { continue }
    $command = $process.Command.ToLowerInvariant()
    # "python " also matches a POSIX "/…/bin/python ./run.py"; "python.exe" is the
    # Windows form. Both interpreters are named in the command line, never inferred.
    $isPython = $command.Contains("python.exe") -or $command.Contains("python ")
    $isLocalApp = $command.Contains("run.py")
    if ($isPython -and $isLocalApp) {
      # A dual-stack server answers on both a v4 and a v6 LISTEN row; without this
      # the same pid would be stopped twice and the second attempt would look like
      # a failure.
      $key = "$($entry.Port)/$($entry.ProcessId)"
      if (-not $seen.Add($key)) { continue }
      $listeners += [pscustomobject]@{
        Port      = [int]$entry.Port
        ProcessId = [int]$entry.ProcessId
        ParentId  = [int]$process.ParentId
      }
    }
  }
  return $listeners
}

function Test-PortRangeFree {
  $stillListening = Get-ListeningEntries |
    Where-Object { $_.Port -ge 8000 -and $_.Port -le 8050 }
  return (-not $stillListening)
}

function Stop-LocalAppServerProcessTree {
  param([pscustomobject]$Listener)

  $processIdsToStop = New-Object 'System.Collections.Generic.HashSet[int]'
  [void]$processIdsToStop.Add($Listener.ProcessId)
  # run.py spawns a child that owns the socket; killing only one leaves an orphan
  # holding the port, which then looks like "the new server failed to bind".
  if ($Listener.ParentId) {
    $parent = Get-ProcessTable | Where-Object { $_.ProcessId -eq [int]$Listener.ParentId } | Select-Object -First 1
    if ($parent -and $parent.Command) {
      $parentCommand = $parent.Command.ToLowerInvariant()
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

function Stop-OrphanedSupervisors {
  <#
    Kill run.ps1 supervisors left over from previous launches - every one except
    this process.

    Each launch starts a supervisor that owns the server and, in its finally block,
    asks the server to shut down. An orphan from an earlier run is therefore a
    process holding a shutdown trigger for a server it no longer owns: when it
    eventually notices its child is gone, it runs that finally and stops whichever
    server is listening on the port by then - which will be the NEW one. That is a
    server dying minutes after a clean start for no visible reason.

    Stop-Process is used deliberately instead of a graceful stop: running the
    orphan's finally is precisely the behaviour being prevented.
  #>
  $self = $PID
  $table = Get-ProcessTable
  $parentById = @{}
  foreach ($row in $table) { $parentById[$row.ProcessId] = $row.ParentId }

  # Never treat this launcher's own ancestry as an orphan. On Linux the ordinary
  # invocation is `pwsh ./run.ps1` from an interactive pwsh or a wrapper, so a
  # parent shell can carry "run.ps1" on its command line and would otherwise be
  # killed by its own child - taking the terminal with it.
  $ancestors = New-Object 'System.Collections.Generic.HashSet[int]'
  [void]$ancestors.Add($self)
  $walk = $parentById[$self]
  while ($walk -and $walk -gt 1 -and $ancestors.Add([int]$walk)) {
    $walk = $parentById[[int]$walk]
  }

  $orphans = $table | Where-Object {
    -not $ancestors.Contains([int]$_.ProcessId) -and
    $_.Command -and
    ($_.Name -in @("powershell.exe", "pwsh.exe", "powershell", "pwsh")) -and
    $_.Command -like '*run.ps1*'
  }
  foreach ($orphan in $orphans) {
    Write-Host "  Stopping orphaned launcher (PID $($orphan.ProcessId))."
    Stop-Process -Id $orphan.ProcessId -Force -ErrorAction SilentlyContinue
  }
  if ($orphans) { Start-Sleep -Milliseconds 400 }
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
  Stop-OrphanedSupervisors
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
  param(
    [int]$RootProcessId,
    # Snapshot taken once by the caller. Re-reading the table at every recursion
    # level races against the kills already issued, which on Linux showed up as
    # children being missed because their parent had died first and reparented them.
    [System.Collections.IEnumerable]$ProcessTable
  )

  if (-not $RootProcessId) { return }
  if ($RootProcessId -eq $PID) { return }
  if (-not $ProcessTable) { $ProcessTable = Get-ProcessTable }
  foreach ($child in @($ProcessTable | Where-Object { $_.ParentId -eq $RootProcessId })) {
    Stop-ProcessTree -RootProcessId ([int]$child.ProcessId) -ProcessTable $ProcessTable
  }
  Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-WorkspaceRunPyProcesses {
  $workspacePath = (Resolve-Path -LiteralPath $PSScriptRoot).Path.ToLowerInvariant()
  $currentProcessId = $PID
  $table = Get-ProcessTable
  foreach ($process in $table) {
    if (-not $process.Command) { continue }
    if ([int]$process.ProcessId -eq [int]$currentProcessId) { continue }
    $name = ([string]$process.Name).ToLowerInvariant()
    # Linux interpreters are python3, python3.12, or just the venv's "python";
    # Windows is python.exe. Matching the prefix covers all of them, and the
    # workspace check below is what actually narrows this to our own server.
    if (-not $name.StartsWith("python")) { continue }
    $command = $process.Command.ToLowerInvariant()
    $isWorkspaceRunPy = $command.Contains("run.py") -and (
      $command.Contains($workspacePath) -or
      $command.Contains(".\run.py") -or
      $command.Contains("./run.py")
    )
    if ($isWorkspaceRunPy) {
      Write-Host "Stopping existing workspace run.py process (PID $($process.ProcessId))"
      Stop-ProcessTree -RootProcessId ([int]$process.ProcessId) -ProcessTable $table
    }
  }
}

function Find-BrowserExecutable {
  <#
    A Chromium-family browser, because managed mode needs --app and
    --user-data-dir: the dashboard opens as its own window and closing that window
    is the documented way to stop the server. Firefox supports neither, so a
    Firefox-only machine correctly falls through to the plain-open path below
    rather than getting a window whose close does nothing.
  #>
  if ($script:OnWindows) {
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
  foreach ($name in @("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "microsoft-edge-stable")) {
    $resolved = Get-Command $name -ErrorAction SilentlyContinue
    if ($resolved) { return $resolved.Source }
  }
  return $null
}

function Open-DefaultBrowser {
  param([string]$Url)
  if ($script:OnWindows) {
    Start-Process $Url | Out-Null
    return
  }
  # Start-Process cannot hand a URL to the desktop on Linux; xdg-open is what does
  # that. A headless box has neither, and that is not a failure worth stopping for.
  $opener = Get-Command xdg-open -ErrorAction SilentlyContinue
  if ($opener) {
    Start-Process -FilePath $opener.Source -ArgumentList @($Url) | Out-Null
  } else {
    Write-Host "No xdg-open available; open $Url manually."
  }
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

  # Matching on the profile directory rather than on the pid is deliberate and is
  # what makes this work on Linux at all: `google-chrome` is a shell wrapper that
  # execs the real binary, so the pid Start-Process returns exits immediately and
  # would read as "the window was closed" the moment the browser finished starting.
  $browserNames = @("chrome.exe", "msedge.exe", "chrome", "chromium", "chromium-browser", "msedge", "microsoft-edge")
  $browserProcesses = Get-ProcessTable | Where-Object {
    $_.Command -and
    ($browserNames -contains ([string]$_.Name).ToLowerInvariant()) -and
    $_.Command.ToLowerInvariant().Contains($profileNeedle)
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

# PYTHONPATH must CONTAIN src, which is not the same as being unset.
#
# This was Set-DefaultEnv, i.e. "src" only when nothing was set at all. On a
# workstation that sources another toolchain's setup script (ROS 2 exports its
# python3.10 site-packages this way) PYTHONPATH is already populated, so the
# launcher quietly set nothing and left a different interpreter's packages on the
# import path of this project's 3.12 venv. Prepending keeps a deliberately
# configured PYTHONPATH working while guaranteeing our own source wins.
$existingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$pathSeparator = [System.IO.Path]::PathSeparator
$pythonPathParts = @("src")
if ($existingPythonPath) {
  foreach ($part in $existingPythonPath.Split($pathSeparator)) {
    if ($part -and $part -ne "src" -and ($pythonPathParts -notcontains $part)) {
      $pythonPathParts += $part
    }
  }
}
Set-RunEnv "PYTHONPATH" ($pythonPathParts -join $pathSeparator)
# Foreign entries stay on the path (removing them could break a deliberate setup)
# but they are named, because "module not found" or a version-mismatched import in
# a venv that clearly has the package is otherwise a long thing to track down.
$foreignPythonPaths = @($pythonPathParts | Where-Object { $_ -match 'python3\.\d+' -and $_ -notmatch '\.venv' })
if ($foreignPythonPaths) {
  Write-Host "NOTE: PYTHONPATH carries another interpreter's packages:" -ForegroundColor DarkYellow
  foreach ($entry in $foreignPythonPaths) { Write-Host "  $entry" -ForegroundColor DarkGray }
  Write-Host "  src is first, so this project's modules win. Unset PYTHONPATH if imports misbehave." -ForegroundColor DarkGray
}
Set-DefaultEnv "APP_ENV" "local"
Set-DefaultEnv "APP_PORT" "8010"

# --- External access (opt-in) -------------------------------------------------
# Without -External nothing here runs and the server stays loopback-only, which
# is the posture every previous launch had.
if ($External) {
  Set-RunEnv "APP_HOST" "0.0.0.0"
  $accessToken = [Environment]::GetEnvironmentVariable("APP_ACCESS_TOKEN", "Process")
  if (-not $accessToken -or $accessToken.Length -lt 16) {
    # Generated per launch rather than persisted: a token written to disk in a
    # OneDrive-synced repo is a token that leaves this machine.
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $accessToken = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
    Set-RunEnv "APP_ACCESS_TOKEN" $accessToken
    Write-Host "Generated a one-session access token." -ForegroundColor Cyan
  }
  Write-Host ""
  Write-Host "EXTERNAL ACCESS IS ON." -ForegroundColor Yellow
  Write-Host "  This server submits REAL orders and accepts shutdown requests." -ForegroundColor Yellow
  Write-Host "  Anything that can reach this port needs only the token below." -ForegroundColor Yellow
  Write-Host ""
}
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
# --- GNN-direct election (operator posture, 2026-08-08) ------------------------------
# 운영자 결정: GNN 이 고른 전략이 그대로 채택되고 곧바로 실거래로 간다.
# 이 한 변수가 두 곳을 동시에 바꾼다(플래그 두 개가 어긋나는 상태를 만들지 않으려고
# 일부러 같은 이름을 쓴다):
#   * StrategySessionManager — 밴딧의 비관적 하한과 NO_TRADE 선택지를 건너뛰고
#     GNN 예측 net edge 1등을 그대로 ARM 한다.
#   * SharedDecisionEngine  — 채택된 전략(strategy_locked)에 한해 ProfitabilityGate 의
#     거부권을 자문으로 강등한다. 게이트는 계속 돌고 판정도 기록되지만 막지 않는다.
#     (진단 필드: profitability_gate_bypassed / _overruled_reasons)
#
# 켜기 전 실측을 남겨 둔다. 이 posture 가 받아들이는 숫자다:
#   * GNN 선택의 라이브 전방검증 107건 — positive_net_rate 0.0, 평균 실현 net -62.08bps
#   * success 헤드 체결 408셀 정확도 61.8% vs 상수 예측기 기준선 84.6%
#   * counterfactual 16전략 중 14개 net 음수 (KRX intraday_momentum +9.8,
#     cross_sectional_relative_strength +8.5 만 양수. US 는 전 전략 -33~-66)
#
# 그대로 두는 것: RiskManager(주문 자체를 만드는 주체라 우회 대상이 아니다),
# 숏 SHADOW 사다리, 세션/유동성 판정.
# 전역 (종목, 전략, 방향) 공동 순위와 NO_TRADE를 사용한다. 종목별 GNN
# 1등을 먼저 확정하면 최종 밴딧과 다른 목적함수로 전역 우승 조합을 버릴 수 있다.
Set-RunEnv "STRATEGY_SESSION_GNN_DIRECT_ELECTION" "false"
# --- Event-driven market-data ingestion (2026-08-04) ---------------------------------
# 국내·미국 실시간 수집을 event bus + 영속화 워커 경로로 돌린다. 켜면:
#   * 분 bar 를 in-memory 집계기(IncrementalMinuteBarBuilder)로 만든다 — 메시지마다
#     저장소를 재조회하지 않으므로 6GB 규모 tick 테이블의 lock 경합이 사라진다.
#   * DB 작업이 WebSocket 콜백 밖으로 나간다.
#   * 국내와 미국이 **같은** 수집 경로를 쓴다.
#
# 이 값이 false 였던 동안 두 시장 모두 non-sink 경로로 돌았고, 거기서 매 메시지
# build_latest_minute_bar 가 호출되며 lock 경합 예외로 수집기가 죽어 분 bar 가 결손됐다.
# 그 결손이 macro 의 MACRO_INSUFFICIENT_DATA -> NO_TRADE_MARKET -> 신규매수 전면차단으로
# 이어졌다. 자세한 배경: docs/realtime_session_gap_analysis.md
#
# config/refactor_profile.json 의 flags 는 진단·비교 화면의 posture 선언이고, 실제 코드
# 경로를 여는 것은 이 환경변수다. 둘이 어긋나 있으면 JSON 이 아니라 이 값이 이긴다.
Set-DefaultEnv "REFACTOR_WEBSOCKET_MARKET_DATA" "true"
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
# Run the new context/utility selector beside the legacy authority. This records
# Starts in SHADOW and gathers coverage/counterfactual outcomes. The persisted controller
# may later grant V2 LIVE_PROBE/LIVE selection authority; order construction and every
# downstream profitability/risk/broker gate remain outside V2.
Set-DefaultEnv "STRATEGY_SELECTOR_V2_ENABLED" "true"
Set-DefaultEnv "STRATEGY_SELECTOR_V2_SHADOW_ONLY" "true"
Set-DefaultEnv "STRATEGY_SELECTOR_V2_AUTO_PROMOTE" "true"
Set-DefaultEnv "STRATEGY_COUNTERFACTUAL_ENABLED" "true"
Set-DefaultEnv "STRATEGY_NO_TRADE_ENABLED" "true"
Set-DefaultEnv "STRATEGY_ONTOLOGY_MASK_V2_ENABLED" "true"
Set-DefaultEnv "STRATEGY_SESSION_MIN_NET_TARGET_BPS" "25"
Set-DefaultEnv "REALTIME_MIN_NET_PROFIT_BUFFER_RATE" "0.0"
Set-DefaultEnv "REALTIME_COLLECTOR_MAX_SYMBOLS" "40"
# 세션 앵커: 로테이션에서 제외하고 장 내내 유지하는 소수 종목.
# 세션 구조 전략(market_intraday_momentum)은 같은 종목의 09:00-09:30과 14:50-15:20을
# 동시에 필요로 하는데, 수집기는 300초마다 종목을 교체한다.
# 비워 두면 보유/관찰/거래대금 랭킹/상장 유니버스 순으로 세션 앵커를 동적 선정한다.
#
# 이 값은 2 였다. 코드 기본값은 8 이고(_realtime_session_anchor_symbols) 그 docstring 은
# 2가 macro 에 "demonstrably not enough" 라고 명시하는데, Set-DefaultEnv 는 미설정 시에만
# 적용하므로 런처의 2가 코드의 8을 조용히 되돌리고 있었다. 코드만 읽으면 8로 보인다.
#
# 실측(2026-08-08, 저장 분봉 40일): 09:00-09:30 을 가진 KR 심볼-일 182, 14:50-15:20 을
# 가진 것 303, **둘 다 가진 것 49**. 게다가 최근이 더 나쁘다 — 08-06 은 개장 구간이 0건,
# 08-07 은 open 28 / close 73 인데 **겹침 0**. 그래서 market_intraday_momentum 과
# _short 는 40일 학습 구간에서 단 한 번도 발화하지 못했다
# (STRUCTURALLY_UNREACHABLE:CONTEXT_UNAVAILABLE:FIRST_HALF_HOUR).
#
# 앵커 1개당 등록 1건이고(호가+체결 쌍이 아니다) 수집 상한은 40 이므로 8은 여유 안이다.
Set-DefaultEnv "REALTIME_SESSION_ANCHOR_SYMBOLS" ""
Set-DefaultEnv "REALTIME_SESSION_ANCHOR_MAX" "8"
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
# --- Compute placement --------------------------------------------------------
# This block used to pin ONTOLOGY_ACCELERATOR / OPENVINO_DEVICE / LLM_EVENT_DEVICE
# to "NPU" unconditionally, which was right for exactly one machine: the Intel
# notebook this project was written on. It is not a harmless over-request. The
# names are read via os.environ.setdefault in
# RealtimeAccelerationPolicy.apply_process_hints, so a value set here WINS over
# the per-workload plan in app/realtime/device_plan.py — the launcher silently
# overrode the code that probes what the machine actually has, and every consumer
# then took a failed-compile fallback path instead of being told to use the
# hardware that is present.
#
# So the rule is: only pin a device on a machine that is known to have it, and
# otherwise say nothing and let probe_devices()/plan_devices() decide. Leaving a
# variable unset is the instruction to plan; setting it is an override.
Set-DefaultEnv "REALTIME_LATENCY_PROFILE" "low_latency"
Set-DefaultEnv "OPENVINO_HINT_PERFORMANCE_MODE" "LATENCY"
Set-DefaultEnv "OPENVINO_ENABLE_CPU_PINNING" "YES"
Set-DefaultEnv "OPENVINO_CACHE_DIR" (Join-ProjectPath "data" "runtime" "openvino_cache")
if ($script:OnWindows) {
  # The Core Ultra notebook: OpenVINO enumerates NPU (AI Boost) and an Intel iGPU,
  # and the short-horizon path has been running on the NPU since it was written.
  Set-DefaultEnv "ONTOLOGY_ACCELERATOR" "NPU"
  Set-DefaultEnv "OPENVINO_DEVICE" "NPU"
  Set-DefaultEnv "LLM_EVENT_INFERENCE_BACKEND" "openvino"
  Set-DefaultEnv "LLM_EVENT_DEVICE" "NPU"
} else {
  # The Linux workstation has no NPU. It has an NVIDIA GPU, which OpenVINO cannot
  # target at all — OpenVINO's "GPU" plugin means the Intel iGPU — so the two
  # accelerators are reached by two different runtimes and must be configured
  # separately rather than through one "device" string:
  #
  #   * ONTOLOGY_ACCELERATOR / OPENVINO_DEVICE stay UNSET so device_plan places
  #     each workload on whatever OpenVINO reports here (Intel iGPU if the driver
  #     is loaded, otherwise CPU). Note that the decision-carrying workloads are
  #     pinned to CPU in device_plan.py regardless, and that is not relaxed here.
  #   * LLM_EVENT_* goes to the transformers backend with device "auto", which is
  #     the path that reaches CUDA: llm_classifier passes device_map="auto" to
  #     transformers, which places the model on the NVIDIA GPU when torch sees it
  #     and on CPU when it does not. Asking for "openvino"/"NPU" here would instead
  #     select a runtime that cannot use this machine's only real accelerator.
  Set-DefaultEnv "LLM_EVENT_INFERENCE_BACKEND" "transformers"
  Set-DefaultEnv "LLM_EVENT_DEVICE" "auto"
}

# Shared local-LLM config (config/local_llm.env): the single place to set the
# news/event sentiment model for both Windows and Raspberry Pi. Applied as
# defaults (only if not already set), so it runs before the provider logic below
# and lets one file pick the model for every machine.
$localLlmConfig = Join-ProjectPath "config" "local_llm.env"
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

$embeddedModelPath = Join-ProjectPath "models" "local-llm" "event-classifier"
if (-not [Environment]::GetEnvironmentVariable("LLM_EVENT_PROVIDER", "Process")) {
  if (Test-Path $embeddedModelPath) {
    [Environment]::SetEnvironmentVariable("LLM_EVENT_PROVIDER", "embedded", "Process")
    [Environment]::SetEnvironmentVariable("LLM_EVENT_MODEL", $embeddedModelPath, "Process")
    Set-DefaultEnv "LLM_EVENT_MODEL_CACHE_DIR" (Join-ProjectPath "models" "local-llm" "cache")
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
Set-DefaultEnv "LEARNING_COLLECTION_INTERVAL_SECONDS" "300"
Set-DefaultEnv "LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS" "180"
Set-DefaultEnv "LIVE_EVENT_EVIDENCE_TTL_HOURS" "24"
Set-DefaultEnv "LIVE_MACRO_EVENT_EVIDENCE_TTL_HOURS" "96"
Set-DefaultEnv "AUTO_START_LIVE_WORKER" "true"
Set-DefaultEnv "AUTO_START_REALTIME_TRADING" "true"
# 데이터 수집은 실시간(KIS 수집기+트레이딩 평가 저널링), 학습은 주기적으로 백그라운드 재학습.
Set-DefaultEnv "AUTO_START_LIVE_TRAINING" "true"
Set-DefaultEnv "LIVE_TRAINING_INTERVAL_SECONDS" "300"
Set-DefaultEnv "LIVE_TRAINING_STARTUP_DELAY_SECONDS" "90"
# 투자자별 매매동향(개인/외국인/기관 순매수) 일일 갱신. KIS는 이 값을 영업일 단위로만
# 제공하고, residual_relative_strength는 이 정보를 필수 조건으로 쓴다. 갱신이 멈추면
# 저장된 30영업일 창이 밀려나면서 해당 전략이 조용히 평가 불가 상태로 돌아간다.
# 6시간 주기인 이유: 당일 수치는 장중 계속 변하므로 24시간 주기면 정작 필요한
# 당일 데이터가 하루 대부분 낡은 상태로 남는다. 읽기 전용 조회만 사용한다.
# 주말 리서치: KRX 금요일 마감 ~ 월요일 개장 사이에는 양 시장 모두 정지하므로
# 연산 여유가 있고 방해할 거래도 없다. 이 구간에 거시·이벤트를 집계해 월요일 개장
# 갭에 대한 "검증 가능한 사전 예측"을 남기고, 개장 후 실제 갭과 대조해 채점한다.
# 채점하지 않는 주말 분석은 틀린 분석과 구별되지 않는다.
Set-DefaultEnv "AUTO_START_WEEKEND_BRIEF" "true"
Set-DefaultEnv "WEEKEND_BRIEF_INTERVAL_SECONDS" "3600"
Set-DefaultEnv "AUTO_START_INVESTOR_FLOW_REFRESH" "true"
Set-DefaultEnv "INVESTOR_FLOW_REFRESH_SECONDS" "21600"
Set-DefaultEnv "INVESTOR_FLOW_MINIMUM_BARS" "100"
Set-DefaultEnv "LIVE_SIGNAL_MODEL_INFERENCE_ENABLED" "true"
Set-DefaultEnv "RESEARCH_RETENTION_DAYS" "30"
Set-DefaultEnv "REALTIME_RAW_RETENTION_HOURS" "168"
Set-DefaultEnv "REALTIME_MINUTE_BAR_RETENTION_DAYS" "45"
Set-DefaultEnv "REALTIME_PRUNE_BATCH_SIZE" "25000"
Set-DefaultEnv "RESEARCH_TYPED_MARKET_RETENTION_DAYS" "7"
Set-DefaultEnv "RESEARCH_TYPED_QUOTE_RETENTION_DAYS" "7"
Set-DefaultEnv "RESEARCH_TYPED_BAR_RETENTION_DAYS" "45"
Set-DefaultEnv "RESEARCH_TYPED_SCORE_RETENTION_DAYS" "30"
Set-DefaultEnv "RESEARCH_PRUNE_BATCH_SIZE" "10000"
Set-DefaultEnv "ACCOUNT_DASHBOARD_RETENTION_DAYS" "90"
Set-DefaultEnv "ACCOUNT_TRADE_RETENTION_DAYS" "365"
Set-DefaultEnv "ACCOUNT_DASHBOARD_RAW_PAYLOAD_KEEP" "10"
Set-DefaultEnv "ACCOUNT_DASHBOARD_PRUNE_BATCH_SIZE" "250"
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
# 실측 KRX 스프레드는 종목별로 크게 다른데 비용 모델의 spread_rate는 0이고
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

$python = $script:VenvPython
if (-not (Test-Path $python)) {
  # No venv for THIS OS. Falling back to a bare interpreter is deliberate — the
  # project imports fine from any 3.11+ environment — but say which one is missing,
  # because "python" resolving to a 3.10 system interpreter is the failure this
  # message exists to shorten (app.schemas.domain uses enum.StrEnum, 3.11+).
  Write-Host "No virtual environment at $python; falling back to the interpreter on PATH." -ForegroundColor Yellow
  Write-Host "  Create one with:  ./setup.ps1" -ForegroundColor DarkGray
  $python = if ($script:OnWindows) { "python" } else { "python3" }
}

$logsDir = Join-ProjectPath "logs"
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
  $serverArguments = @{
    FilePath               = $python
    # "run.py" without a leading .\ or ./ : -WorkingDirectory already puts the
    # process in the project root, and the two prefixes are not interchangeable
    # across the two shells.
    ArgumentList           = @("run.py", "--skip-startup-checks", "--port", "$port", "--strict-port")
    WorkingDirectory       = $PSScriptRoot
    RedirectStandardOutput = $serverOutLog
    RedirectStandardError  = $serverErrLog
    PassThru               = $true
  }
  # -WindowStyle describes a Win32 window and throws PSNotSupportedException on
  # Linux, where a redirected child has no window to style in the first place.
  if ($script:OnWindows) { $serverArguments.WindowStyle = "Hidden" }
  $server = Start-Process @serverArguments

  Write-Host "Starting local app server (PID $($server.Id))..."
  Wait-LocalAppReady -Url $url -ServerProcess $server
  Write-Host "Web UI: $url"
  Write-Host "Account dashboard: $launchUrl"
  Write-Host "Server logs: $serverOutLog"

  if ($External) {
    # The address a phone can actually open — "0.0.0.0" is not one. Printed with
    # the token attached because the first request from a browser cannot set a
    # header; the server turns that one query parameter into a cookie.
    $lanAddress = if ($script:OnWindows) {
      Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
          $_.IPAddress -notlike "127.*" -and
          $_.IPAddress -notlike "169.254.*" -and
          $_.PrefixOrigin -in @("Dhcp", "Manual")
        } |
        Select-Object -First 1 -ExpandProperty IPAddress
    } else {
      # Same three exclusions as the Windows branch: loopback, link-local, and
      # (via `scope global`) anything that is not routable off this host.
      $found = $null
      foreach ($line in (& ip -4 -o addr show scope global 2>$null)) {
        if ($line -match 'inet\s+(\d+\.\d+\.\d+\.\d+)') {
          $candidate = $Matches[1]
          if ($candidate -like "127.*" -or $candidate -like "169.254.*") { continue }
          $found = $candidate
          break
        }
      }
      $found
    }
    if ($lanAddress) {
      $token = [Environment]::GetEnvironmentVariable("APP_ACCESS_TOKEN", "Process")
      Write-Host ""
      Write-Host "Open this on the other device (same dashboard as here):" -ForegroundColor Cyan
      Write-Host "  http://${lanAddress}:${port}/account?token=$token" -ForegroundColor Cyan
      Write-Host ""
      if ($script:OnWindows) {
        Write-Host "If it does not load, Windows Firewall is blocking the port. Allow it once:" -ForegroundColor DarkGray
        Write-Host "  New-NetFirewallRule -DisplayName 'OBAITS app $port' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port -Profile Private" -ForegroundColor DarkGray
      } else {
        Write-Host "If it does not load, the host firewall is blocking the port. Allow it once:" -ForegroundColor DarkGray
        Write-Host "  sudo ufw allow $port/tcp" -ForegroundColor DarkGray
      }
    } else {
      Write-Host "Could not determine a LAN address for this machine." -ForegroundColor Yellow
    }
  }

  $browserExe = if ($Headless) { $null } else { Find-BrowserExecutable }
  if ($Headless) {
    Write-Host "Headless mode: server will keep running without a managed browser."
  } elseif ($browserExe) {
    $browserProfile = Join-ProjectPath "data" "runtime" "managed-browser-profile"
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
    Open-DefaultBrowser -Url $launchUrl
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
