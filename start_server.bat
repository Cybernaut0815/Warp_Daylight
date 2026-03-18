@echo off
echo Starting Annual Daylight Analysis API...
echo Configuration is read from .env (copy .env.example to get started)
echo.
call WarpEnv\Scripts\activate
python -m app.main
