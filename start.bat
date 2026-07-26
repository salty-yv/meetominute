@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto environment_ready

echo [MeetOminute] Project virtual environment was not found.
echo Run: py -3.11 -m venv .venv
echo Then: .venv\Scripts\python.exe -m pip install -e ".[dev]"
pause
exit /b 1

:environment_ready
if /i "%~1"=="--check" goto check_only

".venv\Scripts\python.exe" -m app.launcher
if not errorlevel 1 exit /b 0

echo [MeetOminute] The application exited with an error.
pause
exit /b 1

:check_only
".venv\Scripts\python.exe" -c "from app.config import Settings; s=Settings.from_env(); print('[MeetOminute] Startup check passed:', s.host, s.port)"
exit /b %errorlevel%
