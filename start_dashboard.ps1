# PowerShell script to run the fixed dashboard

Write-Host "Starting fixed modern dashboard..." -ForegroundColor Cyan
Write-Host "This version fixes template recursion and login issues" -ForegroundColor Yellow
Write-Host ""
Write-Host "Login Credentials:" -ForegroundColor Green
Write-Host "  - Username: admin, Password: admin123" -ForegroundColor Yellow
Write-Host "  - Username: operator, Password: operator123" -ForegroundColor Yellow
Write-Host "  - Username: viewer, Password: viewer123" -ForegroundColor Yellow
Write-Host ""

# Environment variables to correctly find templates
$env:FLASK_APP = "fixed_dashboard.py"
$env:FLASK_DEBUG = 1
$env:FLASK_ENV = "development"
$env:FLASK_TEMPLATE_FOLDER = (Get-Location).Path + "\templates"
$env:FLASK_STATIC_FOLDER = (Get-Location).Path + "\static"

# Make tooltips.html safe by directly fixing it
$tooltipPath = Join-Path (Get-Location).Path "templates\tooltip.html"
if (Test-Path $tooltipPath) {
    Write-Host "Checking tooltip.html for recursive includes..." -ForegroundColor Cyan
    $content = Get-Content $tooltipPath -Raw
    if ($content -match "\{\%\s*include\s*'tooltip.html'\s*\%\}") {
        Write-Host "Found recursive include in tooltip.html. Fixing it..." -ForegroundColor Yellow
        $content = $content -replace "\{\%\s*include\s*'tooltip.html'\s*\%\}", "<!-- WARNING: Do not include this template recursively -->"
        Set-Content -Path $tooltipPath -Value $content
        Write-Host "Fixed tooltip.html successfully" -ForegroundColor Green
    } else {
        Write-Host "tooltip.html looks good" -ForegroundColor Green
    }
}

# Run the dashboard
python fixed_dashboard.py --debug