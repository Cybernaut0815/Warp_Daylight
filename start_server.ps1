Write-Host "Starting Annual Daylight Analysis API..."
Write-Host "Server will be available at http://localhost:8000"
Write-Host "API docs at http://localhost:8000/docs"
Write-Host ""
& .\WarpEnv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
