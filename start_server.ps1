Write-Host "Starting Annual Daylight Analysis API..."
Write-Host "Configuration is read from .env (copy .env.example to get started)"
Write-Host ""
& .\WarpEnv\Scripts\python.exe -m app.main
