@echo off
REM Data Comparison Test Runner for Windows
REM This script sets up the environment and runs the data comparison test

echo ==========================================
echo Data Comparison Test Setup
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH. Please install Python first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

echo ==========================================
echo Starting Data Comparison Test
echo ==========================================

REM Run the comparison test
python data_comparison_test.py

echo ==========================================
echo Test Execution Complete
echo ==========================================

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo Check the generated report files for results.
pause