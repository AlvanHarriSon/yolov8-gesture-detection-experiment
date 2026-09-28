@echo off
REM Create a project-local Python environment on Windows.
cd /d "%~dp0"

echo [1/3] Checking Python
python --version >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo Select "Add Python to PATH" during installation, then run this file again.
  pause
  exit /b 1
)
python --version

if not exist .venv\Scripts\python.exe (
  echo [2/3] Creating .venv
  python -m venv .venv
  if errorlevel 1 exit /b 1
) else (
  echo [2/3] Reusing existing .venv
)

echo [3/3] Installing dependencies
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Run:
echo     .venv\Scripts\python.exe inference.py
echo.
pause
