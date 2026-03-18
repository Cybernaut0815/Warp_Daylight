@echo off
echo Starting Annual Daylight Analysis API...
echo Server will be available at http://localhost:8000
echo API docs at http://localhost:8000/docs
echo.
call WarpEnv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
