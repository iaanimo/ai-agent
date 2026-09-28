@echo off  
setlocal  
cd /d %~dp0  
if not exist .venv\Scripts\python.exe goto :no_venv  
if not exist logs mkdir logs  
  
echo.  
echo  ==============================================  
echo    J.A.R.V.I.S. - AI Agent Server  
echo  ==============================================  
echo    Web UI:  http://127.0.0.1:8000  
echo    Stop:    close this window or press Ctrl+C  
echo.  
.venv\Scripts\python.exe server.py  
goto :eof  
:no_venv  
echo [ERROR] .venv not found. Run:  
echo   python -m venv .venv  
echo   .venv\Scripts\pip install -r requirements.txt  
pause  
