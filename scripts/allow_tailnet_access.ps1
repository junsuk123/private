#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Open the app's port to tailnet peers only. Run once, elevated.

.DESCRIPTION
  `./run.ps1 -External` binds the server to every interface so tailnet peers reach it
  and WebAccessGuard can challenge them for the token (a request arriving over
  loopback is never challenged, which is why `tailscale serve` is the wrong tool
  here -- it proxies from 127.0.0.1 and every tailnet device would arrive
  pre-trusted on a server that submits real orders).

  Binding wide is not the same as being reachable, and two things stand in the way:

  1. Windows answered a firewall prompt for python.exe with Cancel at some point,
     which leaves "Query User" rules that BLOCK all inbound TCP/UDP to that image on
     the Private and Public profiles. Block wins over Allow, so no amount of allowing
     the port helps while they exist. They are removed here.

  2. The default inbound action is Block, so the port needs an explicit Allow.

  The Allow is scoped by REMOTE ADDRESS to 100.64.0.0/10 -- the CGNAT range Tailscale
  assigns -- not by program. Scoping by program would not work: a Windows venv's
  python.exe is a redirector, so the process actually holding the socket reports the
  BASE interpreter as its image, and that is the same image every other Python on this
  machine uses. Allowing that image would open far more than this app. Restricting by
  remote address is also what keeps the local LAN out, which a bare port rule would not:
  the server is bound to 0.0.0.0 and answers anyone who has the token.

.EXAMPLE
  # From an elevated PowerShell:
  ./scripts/allow_tailnet_access.ps1
  ./scripts/allow_tailnet_access.ps1 -Port 8010
  ./scripts/allow_tailnet_access.ps1 -Remove      # undo
#>
param(
  # Must match APP_PORT (run.ps1 defaults it to 8010).
  [int]$Port = 8010,
  # Delete the rule this script created and stop.
  [switch]$Remove
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$elevated = (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $elevated) {
  Write-Host "This needs an elevated PowerShell (firewall rules are machine state)." -ForegroundColor Yellow
  Write-Host "  Start-Process powershell -Verb RunAs" -ForegroundColor DarkGray
  exit 1
}

$ruleName = "OBAITS app $Port (tailnet only)"
# Tailscale hands out addresses from the 100.64.0.0/10 CGNAT block.
$tailnet = "100.64.0.0/10"

if ($Remove) {
  Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
  Write-Host "Removed: $ruleName"
  exit 0
}

# --- 1. Clear the Cancel-the-prompt block rules -------------------------------
# Matched by application filter rather than by name: the name is a generated
# "TCP Query User{GUID}..." string, so it differs on every machine and after every
# prompt. Only Block+Inbound rules for a python image are touched.
$cleared = 0
foreach ($rule in (Get-NetFirewallRule -Direction Inbound -Action Block -ErrorAction SilentlyContinue)) {
  $program = ($rule | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue).Program
  if ($program -and $program -like "*python*.exe") {
    Write-Host "Removing inbound block rule for $program"
    $rule | Remove-NetFirewallRule
    $cleared++
  }
}
if ($cleared -eq 0) {
  Write-Host "No python inbound block rules present."
}

# --- 2. Allow the port from the tailnet only ----------------------------------
# Recreated rather than edited so a re-run always converges on the same rule.
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule `
  -DisplayName $ruleName `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort $Port `
  -RemoteAddress $tailnet `
  -Profile Private `
  -Description "OBAITS dashboard, reachable from tailnet peers only. Token still required (WebAccessGuard)." | Out-Null

Write-Host ""
Write-Host "Allowed inbound TCP $Port from $tailnet (Private profile)." -ForegroundColor Green
Write-Host "The token is still required: only loopback goes unchallenged." -ForegroundColor DarkGray
Write-Host ""
Get-NetFirewallRule -DisplayName $ruleName |
  Select-Object DisplayName, Enabled, Direction, Action, Profile |
  Format-List
