@echo off
echo Starting Exam Monitoring Backend Server...
cd /d "d:\CODING\fastapi-opencv"
call venv\Scripts\activate.bat
echo Virtual environment activated
echo Installing/checking dependencies...
pip install -r requirements.txt
echo Starting FastAPI server...
python main.py
pause
