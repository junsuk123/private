@echo off
rem Windows entry point. setup.ps1 prints ".\run.bat" as the start command, so this
rem must be the SAME launch as ./run.ps1 on Linux -- not a second, thinner one.
rem
rem It used to call run.py directly with a handful of env defaults. That skipped
rem run.ps1 entirely, and with it the /api/system/restart-safety check that aborts a
rem restart while a position is open with no stop, target or trailing logic watching
rem it. run.ps1 also sets every variable that version set, and more.
rem
rem -ExecutionPolicy Bypass is scoped to this one process: a default Windows install
rem refuses ".\run.ps1" outright, and a double-click has no shell to relax it in.
rem
rem Arguments are forwarded, so ".\run.bat -ForceRestart" and ".\run.bat -Headless"
rem reach run.ps1 unchanged.
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
