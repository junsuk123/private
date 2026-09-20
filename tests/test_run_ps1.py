"""Exercise launcher parsing and isolated functions, never a live application.

The PowerShell harness loads AST function definitions only. Process enumeration,
termination, HTTP and sleep are replaced before any launcher function is called.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32/WindowsPowerShell/v1.0/powershell.exe"
)
POWERSHELL = str(LEGACY_POWERSHELL) if LEGACY_POWERSHELL.is_file() else shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not POWERSHELL, reason="PowerShell is unavailable")


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _function_harness(tmp_path: Path, body: str, *, extra_files: dict[str, str] | None = None) -> dict:
    workspace = tmp_path / "한글 launcher space"
    workspace.mkdir()
    for name, contents in (extra_files or {}).items():
        (workspace / name).write_text(contents, encoding="utf-8-sig")
    harness = workspace / "function-test.ps1"
    result_path = tmp_path / "result.json"
    harness.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$tokens = $null; $errors = $null\n"
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        + _ps_literal(ROOT / "run.ps1")
        + ", [ref]$tokens, [ref]$errors)\n"
        "if ($errors.Count) { throw ($errors | Out-String) }\n"
        "$definitions = foreach ($statement in $ast.EndBlock.Statements) {\n"
        "  if ($statement -is [System.Management.Automation.Language.FunctionDefinitionAst]) {\n"
        "    $statement.Extent.Text\n"
        "  }\n"
        "}\n"
        "$functionFile = Join-Path $PSScriptRoot 'isolated-launcher-functions.ps1'\n"
        "($definitions -join [Environment]::NewLine) | Set-Content -Encoding UTF8 -LiteralPath $functionFile\n"
        ". $functionFile\n"
        "$script:OnWindows = $true\n"
        "$script:ProjectRoot = $PSScriptRoot\n"
        "$script:WorkspaceRoot = $PSScriptRoot\n"
        "$script:Stopped = @()\n"
        "function Get-ProcessTable { return $script:TestProcesses }\n"
        "function Get-ListeningEntries { return $script:TestListeners }\n"
        "function Stop-Process { param($Id, [switch]$Force, $ErrorAction) $script:Stopped += $Id }\n"
        "function Invoke-RestMethod { throw 'HTTP is forbidden in the launcher test harness' }\n"
        "function Start-Process { throw 'Process launch is forbidden in the launcher test harness' }\n"
        "function Start-Sleep { param($Milliseconds, $Seconds) }\n"
        + body
        + "\n$result | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 -LiteralPath "
        + _ps_literal(result_path)
        + "\n",
        encoding="utf-8-sig",
    )
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8-sig"))


def test_launcher_parses_with_windows_powershell_compatibility(tmp_path):
    payload = _function_harness(tmp_path, "$result = @{ parsed = $true }")
    assert payload["parsed"] is True


def test_workspace_match_requires_the_actual_entrypoint_in_this_workspace(tmp_path):
    payload = _function_harness(tmp_path, r'''
$ownRun = Join-Path $PSScriptRoot 'run.py'
$ownSupervisor = Join-Path $PSScriptRoot 'run.ps1'
$ownSource = Join-Path $PSScriptRoot 'src'
$ownPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$otherRun = Join-Path ($PSScriptRoot + '-other') 'run.py'
$result = @{
  own = Test-WorkspaceCommand -Command ('python.exe "' + $ownRun + '" --port 8010') -EntryPoint 'run.py'
  ownSupervisor = Test-WorkspaceCommand -Command ('powershell.exe -File "' + $ownSupervisor + '"') -EntryPoint 'run.ps1'
  ownRelative = Test-WorkspaceCommand -Command ('"' + $ownPython + '" .\run.py --port 8010') -EntryPoint 'run.py'
  module = Test-WorkspaceCommand -Command ('python.exe -m uvicorn app.web:app --app-dir "' + $ownSource + '"') -EntryPoint 'run.py'
  sibling = Test-WorkspaceCommand -Command ('python.exe "' + $otherRun + '" --port 8010') -EntryPoint 'run.py'
  unscoped = Test-WorkspaceCommand -Command 'python.exe .\run.py --port 8010' -EntryPoint 'run.py'
  unrelated = Test-WorkspaceCommand -Command ('python.exe unrelated.py --output "' + $PSScriptRoot + '"') -EntryPoint 'run.py'
  incidental = Test-WorkspaceCommand -Command ('python.exe unrelated.py --output "' + $ownRun + '"') -EntryPoint 'run.py'
}
''')
    assert payload == {
        "own": True,
        "ownSupervisor": True,
        "ownRelative": True,
        "module": True,
        "sibling": False,
        "unscoped": False,
        "unrelated": False,
        "incidental": False,
    }


def test_listener_filter_deduplicates_and_preserves_other_workspaces(tmp_path):
    payload = _function_harness(tmp_path, r'''
$ownRun = Join-Path $PSScriptRoot 'run.py'
$otherRun = Join-Path ($PSScriptRoot + '-other') 'run.py'
$script:TestProcesses = @(
  [pscustomobject]@{ ProcessId=101; ParentId=1; Name='python3'; Command=('python3 "' + $ownRun + '" --port 8010') },
  [pscustomobject]@{ ProcessId=102; ParentId=1; Name='python.exe'; Command=('python.exe "' + $otherRun + '" --port 8011') },
  [pscustomobject]@{ ProcessId=103; ParentId=1; Name='python.exe'; Command='python.exe .\run.py --port 8012' },
  [pscustomobject]@{ ProcessId=104; ParentId=1; Name='python.exe'; Command=('python.exe "' + $ownRun + '" --port 8110') }
)
$script:TestListeners = @(
  [pscustomobject]@{ ProcessId=101; Port=8010; Address='127.0.0.1' },
  [pscustomobject]@{ ProcessId=101; Port=8010; Address='::' },
  [pscustomobject]@{ ProcessId=102; Port=8011; Address='127.0.0.1' },
  [pscustomobject]@{ ProcessId=103; Port=8012; Address='127.0.0.1' },
  [pscustomobject]@{ ProcessId=104; Port=8110; Address='127.0.0.1' }
)
$listeners = @(Get-LocalAppServerListeners)
$result = @{ processes=@($listeners | ForEach-Object { $_.ProcessId }); ports=@($listeners | ForEach-Object { $_.Port }); stopped=@($script:Stopped) }
''')
    assert payload == {"processes": [101], "ports": [8010], "stopped": []}


def test_requested_port_owned_by_another_program_prevents_startup_without_killing(tmp_path):
    payload = _function_harness(tmp_path, r'''
$env:APP_PORT = '8010'
$script:TestProcesses = @([pscustomobject]@{ ProcessId=301; ParentId=1; Name='python.exe'; Command='python.exe another-app.py' })
$script:TestListeners = @([pscustomobject]@{ ProcessId=301; Port=8010; Address='127.0.0.1' })
$result = @{ allowed=(Stop-ExistingLocalAppServers); stopped=@($script:Stopped) }
''')
    assert payload == {"allowed": False, "stopped": []}


def test_explicit_app_port_outside_legacy_range_remains_managed(tmp_path):
    payload = _function_harness(tmp_path, r'''
$env:APP_PORT = '9123'
$ownRun = Join-Path $PSScriptRoot 'run.py'
$script:TestProcesses = @([pscustomobject]@{ ProcessId=301; ParentId=1; Name='python.exe'; Command=('python.exe "' + $ownRun + '" --port 9123') })
$script:TestListeners = @([pscustomobject]@{ ProcessId=301; Port=9123; Address='127.0.0.1' })
$result = @{ ports=@(Get-LocalAppServerListeners | ForEach-Object { $_.Port }); stopped=@($script:Stopped) }
''')
    assert payload == {"ports": [9123], "stopped": []}


def test_orphan_cleanup_never_terminates_foreign_or_ancestor_launchers(tmp_path):
    payload = _function_harness(tmp_path, r'''
$ownSupervisor = Join-Path $PSScriptRoot 'run.ps1'
$otherSupervisor = Join-Path ($PSScriptRoot + '-other') 'run.ps1'
$script:TestProcesses = @(
  [pscustomobject]@{ ProcessId=$PID; ParentId=211; Name='powershell.exe'; Command='powershell.exe function-test.ps1' },
  [pscustomobject]@{ ProcessId=211; ParentId=1; Name='powershell.exe'; Command=('powershell.exe -File "' + $ownSupervisor + '"') },
  [pscustomobject]@{ ProcessId=212; ParentId=1; Name='powershell.exe'; Command=('powershell.exe -File "' + $otherSupervisor + '"') },
  [pscustomobject]@{ ProcessId=213; ParentId=1; Name='powershell.exe'; Command='powershell.exe -File .\run.ps1' }
)
Stop-OrphanedSupervisors
$result = @{ stopped=@($script:Stopped) }
''')
    assert payload["stopped"] == []


@pytest.mark.parametrize("safety_known", [True, False])
def test_restart_refuses_unsafe_or_unreadable_state_without_explicit_force(tmp_path, safety_known):
    payload = _function_harness(tmp_path, r'''
$ForceRestart = $false
$HardKill = $false
$script:Requests = @()
function Get-LocalAppServerListeners {
  return [pscustomobject]@{ ProcessId=301; ParentId=1; Port=8010; Address='127.0.0.1' }
}
function Stop-OrphanedSupervisors { }
function Test-PortRangeFree { return $true }
function Test-ProcessAlive { return $false }
function Invoke-RestMethod {
  param($Uri, $Method, $TimeoutSec, $Headers)
  $script:Requests += [string]$Uri
  if ($Uri -notlike '*/restart-safety') { throw 'Shutdown must not be requested' }
''' + ("  return @{ safe=$false; reasons=@('OPEN_POSITION'); holdings_count=1 }\n" if safety_known
       else "  throw 'Unreachable server'\n") + r'''
}
$allowed = Stop-ExistingLocalAppServers
$result = @{ allowed=[bool]$allowed; stopped=@($script:Stopped); requests=@($script:Requests) }
''')
    assert payload["allowed"] is False
    assert payload["stopped"] == []
    assert payload["requests"] == ["http://127.0.0.1:8010/api/system/restart-safety"]


@pytest.mark.parametrize("shutdown_responds", [True, False])
def test_browser_cleanup_preserves_running_server_if_shutdown_is_refused_or_unknown(tmp_path, shutdown_responds):
    payload = _function_harness(tmp_path, r'''
$ForceRestart = $false
$HardKill = $false
$browser = [pscustomobject]@{ Id=401; HasExited=$false }
$server = [pscustomobject]@{ Id=402; HasExited=$false }
$publicSite = [pscustomobject]@{ Id=403; HasExited=$false }
$serverLeftRunning = $false
$launcherExitCode = 0
$url = 'http://127.0.0.1:8010'
$launchUrl = $url + '/account'
$script:Requests = @()
function Stop-ProcessTree { param($RootProcessId) $script:Stopped += $RootProcessId }
function Invoke-RestMethod {
  param($Uri, $Method, $TimeoutSec, $Headers)
  $script:Requests += [string]$Uri
''' + ("  return @{ ok=$false; message='Managed position remains' }\n" if shutdown_responds
       else "  throw 'Unreachable server'\n") + r'''
}
# Execute only the final cleanup body. The server-start try body is never run.
$topLevelTry = @($ast.EndBlock.Statements | Where-Object {
  $_ -is [System.Management.Automation.Language.TryStatementAst] -and $_.Finally
})[-1]
if (-not $topLevelTry) { throw 'Launcher cleanup AST is missing' }
$cleanup = $topLevelTry.Finally.Extent.Text
$cleanupFile = Join-Path $PSScriptRoot 'isolated-launcher-cleanup.ps1'
$cleanup.Substring(1, $cleanup.Length - 2) | Set-Content -Encoding UTF8 -LiteralPath $cleanupFile
. $cleanupFile
$result = @{ leftRunning=$serverLeftRunning; exitCode=$launcherExitCode; stopped=@($script:Stopped); requests=@($script:Requests) }
''')
    assert payload["leftRunning"] is True
    assert payload["exitCode"] != 0
    assert payload["stopped"] == [401]
    assert payload["requests"] == ["http://127.0.0.1:8010/api/system/graceful-shutdown"]


@pytest.mark.parametrize("setup_exit_code", [0, 42])
def test_initialize_provisions_then_rechecks_or_stops_on_setup_failure(tmp_path, setup_exit_code):
    payload = _function_harness(tmp_path, r'''
$CheckOnly = $false
$SkipSetup = $false
$script:Checks = 0
function Get-Process {
  param($Id)
  if ($Id -ne $PID) { throw 'Only the test harness host process may be inspected' }
  Microsoft.PowerShell.Management\Get-Process -Id $PID
}
function Invoke-LauncherPreflight {
  $script:Checks++
  if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'setup-marker.txt'))) {
    return [pscustomobject]@{ ok=$false; dependency_errors=@('fake-missing'); errors=@('fake-missing'); warnings=@() }
  }
  return [pscustomobject]@{
    ok=$true; dependency_errors=@(); errors=@(); warnings=@(); environment_defaults=[pscustomobject]@{}
    python=[pscustomobject]@{ version='3.13.0'; executable='inert-test-python' }
    devices=[pscustomobject]@{
      openvino=[pscustomobject]@{ available_devices=@('CPU') }
      torch=[pscustomobject]@{ cuda_available=$false }
    }
  }
}
$initialized = $false
$failure = $null
try { Initialize-Launcher; $initialized = $true } catch { $failure = $_.Exception.Message }
$result = @{
  initialized=$initialized; checks=$script:Checks; failure=$failure; stopped=@($script:Stopped)
  setupExecuted=(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'setup-marker.txt'))
}
''', extra_files={
        "setup.ps1": "'setup-ran' | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot 'setup-marker.txt')\n"
        + f"exit {setup_exit_code}\n",
    })
    assert payload["setupExecuted"] is True
    assert payload["stopped"] == []
    if setup_exit_code == 0:
        assert payload["initialized"] is True
        assert payload["checks"] == 2
        assert payload["failure"] is None
    else:
        assert payload["initialized"] is False
        assert payload["checks"] == 1
        assert "Setup failed" in payload["failure"]


@pytest.mark.parametrize("preflight_ok,check_only", [(True, True), (False, True), (False, False)])
def test_preflight_only_or_failed_check_never_touches_running_servers(tmp_path, preflight_ok, check_only):
    """A copied launcher sees only a fake preflight and blocked control cmdlets."""
    workspace = tmp_path / "한글 check only workspace"
    workspace.mkdir()
    shutil.copyfile(ROOT / "run.ps1", workspace / "run.ps1")
    venv_path = workspace / (".venv" if os.name == "nt" else ".venv-linux")
    venv.EnvBuilder(with_pip=False).create(venv_path)
    scripts = workspace / "scripts"
    scripts.mkdir()
    marker = tmp_path / "forbidden-action.txt"
    preflight_marker = tmp_path / "preflight.json"
    report = {
        "ok": preflight_ok,
        "errors": [] if preflight_ok else ["FAKE_MISSING_DEPENDENCY"],
        "dependency_errors": [] if preflight_ok else ["FAKE_MISSING_DEPENDENCY"],
        "warnings": [],
        "python": {"version": "3.13.0", "executable": "inert-test-python"},
        "devices": {"available": ["CPU"]},
        "environment_defaults": {},
    }
    (scripts / "launcher_preflight.py").write_text(
        "import json, os, pathlib\n"
        "pathlib.Path(os.environ['OBAITS_PREFLIGHT_TEST_MARKER']).write_text("
        "json.dumps({'cwd': os.getcwd(), 'live_order': os.getenv('LIVE_ORDER_SUBMIT_ENABLED'), "
        "'manual_arming': os.getenv('REQUIRE_MANUAL_ARMING')}), encoding='utf-8')\n"
        + "print(" + repr(json.dumps(report)) + ")\n"
        + "raise SystemExit(" + ("0" if preflight_ok else "1") + ")\n",
        encoding="utf-8",
    )
    (workspace / "setup.ps1").write_text(
        "'setup' | Set-Content -LiteralPath $env:OBAITS_FORBIDDEN_TEST_MARKER\n"
        "throw 'Setup is forbidden in CheckOnly'\n",
        encoding="utf-8-sig",
    )
    (workspace / "run.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['OBAITS_FORBIDDEN_TEST_MARKER']).write_text('server')\n"
        "raise SystemExit('Server is forbidden in CheckOnly')\n",
        encoding="utf-8",
    )
    harness = tmp_path / "check-only-harness.ps1"
    harness.write_text(
        "$ErrorActionPreference='Stop'\n"
        "foreach ($name in @('Get-CimInstance', 'Get-NetTCPConnection', 'Get-Process', "
        "'Stop-Process', 'Start-Process', 'Invoke-RestMethod', 'Invoke-WebRequest')) {\n"
        "  Set-Item -Path ('Function:global:' + $name) -Value {\n"
        "    'process-or-network' | Set-Content -LiteralPath $env:OBAITS_FORBIDDEN_TEST_MARKER\n"
        "    throw 'Control or network operation is forbidden in CheckOnly'\n"
        "  }\n"
        "}\n"
        + "& " + _ps_literal(workspace / "run.ps1") + (" -CheckOnly" if check_only else "") + " -SkipSetup\n"
        + "exit $LASTEXITCODE\n",
        encoding="utf-8-sig",
    )
    env = os.environ.copy()
    env.update(
        OBAITS_FORBIDDEN_TEST_MARKER=str(marker),
        OBAITS_PREFLIGHT_TEST_MARKER=str(preflight_marker),
        LIVE_ORDER_SUBMIT_ENABLED="false",
        REQUIRE_MANUAL_ARMING="true",
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert not marker.exists(), marker.read_text() if marker.exists() else ""
    assert preflight_marker.is_file(), result.stdout + result.stderr
    invocation = json.loads(preflight_marker.read_text(encoding="utf-8"))
    assert Path(invocation["cwd"]) == workspace
    assert invocation["live_order"] == "false"
    assert invocation["manual_arming"] == "true"
    assert (result.returncode == 0) is preflight_ok, result.stdout + result.stderr
    if not preflight_ok:
        assert "FAKE_MISSING_DEPENDENCY" in result.stdout + result.stderr
