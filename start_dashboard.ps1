# PowerShell script to run the modern dashboard

Write-Host "Starting AI Trading Agent Modern Dashboard..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Login Credentials:" -ForegroundColor Green
Write-Host "  - Username: admin Password: admin123" -ForegroundColor Yellow
Write-Host "  - Username: operator Password: operator123" -ForegroundColor Yellow
Write-Host "  - Username: viewer Password: viewer123" -ForegroundColor Yellow
Write-Host ""

# Environment variables to correctly find templates
$env:FLASK_APP = "run_modern_dashboard.py"
$env:FLASK_DEBUG = 1
$env:FLASK_ENV = "development"
$env:FLASK_TEMPLATE_FOLDER = (Get-Location).Path + "\templates"
$env:FLASK_STATIC_FOLDER = (Get-Location).Path + "\static"

# Run the dashboard
python run_modern_dashboard.py --debug
