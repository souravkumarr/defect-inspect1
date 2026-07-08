@echo off
TITLE Aircraft Fault Detection & AI Inspection System Launcher
COLOR 0A
echo ====================================================================
echo        AIRCRAFT SURFACE FAULT & DAMAGE DETECTION AI SYSTEM
echo ====================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.9 - 3.12 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: Check if virtual environment exists, create if not
if not exist ".venv" (
    echo [1/3] Creating Python Virtual Environment (.venv)...
    python -m venv .venv
)

:: Activate virtual environment and install dependencies
echo [2/3] Checking and installing required Python packages...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

echo.
echo ====================================================================
echo [3/3] Starting AI Web Application Server...
echo The application will open in your web browser at http://127.0.0.1:5000
echo Keep this terminal window open while using the application.
echo ====================================================================
echo.

:: Start the Flask app
python app.py
pause
