@echo off
echo Setting up the AI Crypto Trading System development environment...

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Install the project in development mode
echo Installing project in development mode...
pip install -e .

:: Special handling for TA-Lib on Windows
echo.
echo Note about TA-Lib:
echo If you encounter issues installing TA-Lib from requirements.txt,
echo please download the appropriate wheel file from:
echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo Then install it manually with: pip install ^<downloaded_file^>

echo.
echo Setup complete!
echo You can now run the project with: python -m src.main

pause 