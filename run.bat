@echo off
rem Keep every startup, preflight and restart-safety decision in run.ps1.
rem Disable delayed expansion so exclamation marks in paths/arguments survive.
setlocal EnableExtensions DisableDelayedExpansion
set "OBAITS_EXIT_CODE=1"
set "OBAITS_DIRECTORY_PUSHED="
set "OBAITS_POWERSHELL="

rem Unlike cd /d, pushd also supports a Synology UNC share and restores the
rem caller's directory (and temporary drive mapping) when the launcher returns.
pushd "%~dp0" >nul 2>&1
if errorlevel 1 goto directory_error
set "OBAITS_DIRECTORY_PUSHED=1"
if not exist "%~dp0run.ps1" goto script_error

rem Prefer PowerShell 7 when installed; Windows PowerShell 5.1 remains supported.
for %%I in (pwsh.exe) do set "OBAITS_POWERSHELL=%%~$PATH:I"
if defined OBAITS_POWERSHELL goto launch
for %%I in (powershell.exe) do set "OBAITS_POWERSHELL=%%~$PATH:I"
if defined OBAITS_POWERSHELL goto launch
if not exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" goto powershell_error
set "OBAITS_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

:launch
rem Bypass is process-scoped. Forward all flags, including -CheckOnly and
rem -SkipSetup, without changing authentication, network or trading defaults.
"%OBAITS_POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set "OBAITS_EXIT_CODE=%ERRORLEVEL%"
goto finish

:directory_error
echo [ERROR] Could not open the launcher directory: "%~dp0" 1>&2
goto finish

:script_error
echo [ERROR] run.ps1 is missing from "%~dp0". Restore the complete repository. 1>&2
goto finish

:powershell_error
echo [ERROR] PowerShell was not found. Install PowerShell 7 or enable Windows PowerShell. 1>&2

:finish
if defined OBAITS_DIRECTORY_PUSHED popd
if "%OBAITS_EXIT_CODE%"=="0" goto return
echo [ERROR] OBAITS launcher exited with code %OBAITS_EXIT_CODE%. 1>&2
rem Keep a failed double-click launch visible; flagged CLI/preflight invocations
rem never pause, so automation receives the original failure code immediately.
if "%~1"=="" pause
:return
endlocal & exit /b %OBAITS_EXIT_CODE%
