@echo off
echo Starting backend server...
c:\Users\vp199\Documents\Projects\GitHub\AI-Trading-Agent\.venv\Scripts\python.exe -m uvicorn ai_trading_agent.api.main:app --host 0.0.0.0 --port 8000
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Backend server failed to start with error code %ERRORLEVEL%
  echo Press any key to close this window...
  pause > nul
)
