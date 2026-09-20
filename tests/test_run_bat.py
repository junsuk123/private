"""Execute the Windows wrapper against a harmless PowerShell script only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="cmd.exe wrapper runs on Windows")
ROOT = Path(__file__).resolve().parents[1]


def _harness(tmp_path, *, exit_code=0, path=None, with_script=True):
    workspace = tmp_path / "한글 작업 (launcher) ! &"
    workspace.mkdir()
    wrapper = workspace / "run.bat"
    shutil.copyfile(ROOT / "run.bat", wrapper)
    output = tmp_path / "result.json"
    if with_script:
        (workspace / "run.ps1").write_text(
            "$ErrorActionPreference = 'Stop'\n"
            "$result = [ordered]@{ arguments = @($args); directory = $PWD.Path; "
            "script = $PSScriptRoot; edition = $PSVersionTable.PSEdition }\n"
            "$result | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $env:OBAITS_BAT_TEST_OUTPUT\n"
            "exit ([int]$env:OBAITS_BAT_TEST_EXIT)\n",
            encoding="utf-8-sig",
        )
    env = os.environ.copy()
    env.update(OBAITS_BAT_TEST_OUTPUT=str(output), OBAITS_BAT_TEST_EXIT=str(exit_code))
    if path is not None:
        env["PATH"] = path
    return wrapper, output, env


def _run(wrapper, env, *args, input=""):
    # Execute a copied wrapper and stub script, never the repository's run.ps1.
    quoted = " ".join(f'"{value}"' for value in (str(wrapper), *args))
    command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c "{quoted}"'
    return subprocess.run(command, cwd=wrapper.parent.parent, env=env, input=input,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)


def test_forwards_flags_and_quoted_arguments_from_unicode_workspace(tmp_path):
    wrapper, output, env = _harness(tmp_path)
    arguments = ("-CheckOnly", "-SkipSetup", "-Headless", "-External", "-ExternalSitePort", "8119",
                 "-Example", "한글 value ! & spaces")
    result = _run(wrapper, env, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8-sig"))
    assert payload["arguments"] == list(arguments)
    assert Path(payload["directory"]) == wrapper.parent
    assert Path(payload["script"]) == wrapper.parent


def test_preserves_failure_exit_after_cleanup_without_pausing_flagged_command(tmp_path):
    wrapper, output, env = _harness(tmp_path, exit_code=37)
    result = _run(wrapper, env, "-CheckOnly")
    assert result.returncode == 37
    assert output.is_file()
    assert "code 37" in result.stderr


def test_system_powershell_fallback_works_with_empty_path(tmp_path):
    legacy = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not legacy.is_file():
        pytest.skip("Windows PowerShell 5.1 is unavailable")
    wrapper, output, env = _harness(tmp_path, path="")
    result = _run(wrapper, env, "-CheckOnly", "-SkipSetup")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text(encoding="utf-8-sig"))["edition"] == "Desktop"


def test_prefers_powershell_seven_when_available_on_path(tmp_path):
    pwsh = shutil.which("pwsh.exe")
    if pwsh is None:
        pytest.skip("PowerShell 7 is unavailable")
    wrapper, output, env = _harness(tmp_path, path=str(Path(pwsh).parent))
    result = _run(wrapper, env, "-CheckOnly")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text(encoding="utf-8-sig"))["edition"] == "Core"


def test_missing_script_returns_failure_instead_of_launching_any_application(tmp_path):
    wrapper, output, env = _harness(tmp_path, with_script=False)
    result = _run(wrapper, env, "-CheckOnly")
    assert result.returncode != 0
    assert not output.exists()
    assert "run.ps1 is missing" in result.stderr


def test_failed_no_argument_launch_keeps_message_visible_and_exit_code(tmp_path):
    wrapper, _, env = _harness(tmp_path, exit_code=23)
    result = _run(wrapper, env, input="\n")
    assert result.returncode == 23
    assert "code 23" in result.stderr
    assert result.stdout.strip(), "The failed double-click path should display its pause prompt"
