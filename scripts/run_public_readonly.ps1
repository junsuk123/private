#!/usr/bin/env pwsh
<#
.SYNOPSIS
  A second OBAITS instance intended to sit behind Tailscale Funnel: read-only,
  token-guarded, and unable to submit an order.

.DESCRIPTION
  run.ps1 starts a LIVE instance. It forces TRADING_MODE=live_trading,
  LIVE_ORDER_SUBMIT_ENABLED=true and REQUIRE_MANUAL_ARMING=false with Set-RunEnv,
  which means those values cannot be overridden from outside - by design. So a
  publicly reachable instance cannot be a variant of run.ps1; it has to be its own
  process with its own posture, which is what this script starts.

  Three decisions here exist because this port may be published to the internet:

  1. NO ORDER PATH. TRADING_MODE=read_only, LIVE_TRADING_ENABLED=false,
     LIVE_ORDER_SUBMIT_ENABLED=false, REQUIRE_MANUAL_ARMING=true, and every
     AUTO_START_* worker off. The 22 state-changing POST endpoints still EXIST -
     this is not a hardened build - but the ones that reach a broker have nothing
     live behind them.

  2. BINDS THE TAILNET ADDRESS, NOT LOOPBACK. This is the part that is easy to get
     wrong and expensive to get wrong. AccessGuardMiddleware decides whether to
     demand a token from request.client.host alone; it does not read
     X-Forwarded-For. Tailscale Funnel proxies inbound requests, so if this server
     listened on 127.0.0.1 and Funnel pointed there, every request from the public
     internet would arrive as a loopback client and the guard would wave it
     through - a public, unauthenticated trading dashboard. Pointing Funnel at the
     tailnet address instead makes request.client.host non-loopback, so the token
     is enforced on exactly the traffic that needs it. Do not "simplify" this to
     127.0.0.1.

  3. PORT OUTSIDE 8000-8050. run.ps1 stops every python run.py listener in that
     range when it starts, so a second instance inside it would be killed by the
     next ordinary launch of the live server.

  The token is printed once and not persisted. Restarting issues a new one.

.EXAMPLE
  ./scripts/run_public_readonly.ps1
  ./scripts/run_public_readonly.ps1 -Port 8110
#>
param(
  # Kept out of 8000-8050 on purpose; see (3) above.
  [int]$Port = 8110,
  # Reuse a token instead of generating one, e.g. to keep a published link valid
  # across a restart. Must be at least 16 characters or the server refuses to bind.
  [string]$Token = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$onWindows = if ($null -eq $IsWindows) { $true } else { [bool]$IsWindows }
$python = if ($onWindows) {
  Join-Path $root ".venv/Scripts/python.exe"
} else {
  Join-Path $root ".venv-linux/bin/python"
}
if (-not (Test-Path $python)) { throw "No virtual environment at $python. Run ./setup.ps1 first." }

# --- The address Funnel must point at -----------------------------------------
$tailscale = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $tailscale) { throw "tailscale is not installed; this script exists to serve a tailnet/Funnel address." }
$bindAddress = $null
foreach ($line in (& $tailscale.Source ip -4 2>$null)) {
  $candidate = ([string]$line).Trim()
  if ($candidate -match '^100\.\d{1,3}\.\d{1,3}\.\d{1,3}$') { $bindAddress = $candidate; break }
}
if (-not $bindAddress) { throw "Tailscale is not up (no tailnet IPv4 address). Start it, then re-run." }

$dnsName = $bindAddress
try {
  $status = & $tailscale.Source status --json 2>$null | ConvertFrom-Json
  if ($status.Self.DNSName) { $dnsName = ([string]$status.Self.DNSName).TrimEnd('.') }
} catch { }

# --- Token --------------------------------------------------------------------
if ($Token) {
  if ($Token.Length -lt 16) { throw "-Token must be at least 16 characters (the server enforces this too)." }
  $accessToken = $Token
} else {
  $bytes = New-Object byte[] 24
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $accessToken = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
}

# --- Posture ------------------------------------------------------------------
# Set-, not SetDefault-: every one of these is a value this instance must have
# regardless of what is already in the environment.
$environment = @{
  APP_ENV                            = "local"
  APP_HOST                           = $bindAddress
  APP_PORT                           = "$Port"
  APP_ACCESS_TOKEN                   = $accessToken
  PYTHONPATH                         = "src"

  # No order can be constructed, let alone submitted.
  TRADING_MODE                       = "read_only"
  LIVE_TRADING_ENABLED               = "false"
  LIVE_ORDER_SUBMIT_ENABLED          = "false"
  KIS_PAPER_TRADING                  = "false"
  REQUIRE_MANUAL_ARMING              = "true"
  REALTIME_BUY_ENABLED               = "false"

  # Account reads stay on: a dashboard with no balances is not the thing being
  # published. This is the deliberate trade - holdings and cash ARE visible to
  # anyone holding the token.
  KIS_LIVE_ENABLED                   = "true"
  DATA_ENV                           = "realtime"

  # Nothing that writes. The live instance owns the stores; this one reads them,
  # and two processes running the same writers against the same SQLite files is a
  # corruption risk taken for no benefit.
  AUTO_START_REALTIME_TRADING        = "false"
  AUTO_START_LIVE_WORKER             = "false"
  AUTO_START_LIVE_TRAINING           = "false"
  AUTO_START_WEEKEND_BRIEF           = "false"
  AUTO_START_INVESTOR_FLOW_REFRESH   = "false"
  LIVE_SIGNAL_MODEL_INFERENCE_ENABLED = "false"
  LLM_EVENT_CLASSIFIER_ENABLED       = "false"
}
foreach ($pair in $environment.GetEnumerator()) {
  [Environment]::SetEnvironmentVariable($pair.Key, $pair.Value, "Process")
}

Write-Host ""
Write-Host "READ-ONLY INSTANCE" -ForegroundColor Cyan
Write-Host "  bind        : ${bindAddress}:${Port}  (tailnet address, so the token is enforced)" -ForegroundColor Cyan
Write-Host "  trading     : read_only, order submission disabled, workers off" -ForegroundColor Cyan
Write-Host "  account data: VISIBLE to anyone holding the token below" -ForegroundColor Yellow
Write-Host ""
Write-Host "On the tailnet:" -ForegroundColor Cyan
Write-Host "  http://${dnsName}:${Port}/account?token=$accessToken" -ForegroundColor Cyan
Write-Host ""
Write-Host "To publish it on the public internet, as root (Funnel needs it):" -ForegroundColor DarkGray
Write-Host "  sudo tailscale funnel --bg --https=443 http://${bindAddress}:${Port}" -ForegroundColor DarkGray
Write-Host "  # then: https://${dnsName}/account?token=$accessToken" -ForegroundColor DarkGray
Write-Host "  # point it at ${bindAddress}, NEVER 127.0.0.1 - see this script's header." -ForegroundColor DarkGray
Write-Host "  # off again: sudo tailscale funnel --https=443 off" -ForegroundColor DarkGray
Write-Host ""

& $python run.py --skip-startup-checks --host $bindAddress --port $Port --strict-port
