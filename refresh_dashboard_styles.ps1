# PowerShell script to refresh dashboard styles

Write-Host "Refreshing dashboard styles..." -ForegroundColor Cyan

# Get current timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

# Update API keys panel CSS with timestamp
$sourceCssPath = Join-Path (Get-Location).Path "static\css\api_keys_panel.css"
$destCssPath = Join-Path (Get-Location).Path "static\css\api_keys_panel_$timestamp.css"
Copy-Item -Path $sourceCssPath -Destination $destCssPath

# Create a launcher script
$launcherPath = Join-Path (Get-Location).Path "run_fixed_dashboard_temp.ps1"
$launcherContent = @"
# PowerShell script to run the fixed dashboard with refreshed styles

Write-Host "Starting fixed modern dashboard with refreshed styles..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Login Credentials:" -ForegroundColor Green
Write-Host "  - Username: admin, Password: admin123" -ForegroundColor Yellow
Write-Host "  - Username: operator, Password: operator123" -ForegroundColor Yellow
Write-Host "  - Username: viewer, Password: viewer123" -ForegroundColor Yellow
Write-Host ""

# Environment variables to correctly find templates
`$env:FLASK_APP = "fixed_dashboard.py"
`$env:FLASK_DEBUG = 1
`$env:FLASK_ENV = "development"
`$env:FLASK_TEMPLATE_FOLDER = (Get-Location).Path + "\templates"
`$env:FLASK_STATIC_FOLDER = (Get-Location).Path + "\static"

# Run the dashboard
python fixed_dashboard.py --debug
"@

Set-Content -Path $launcherPath -Value $launcherContent

# Update HTML template to use the new CSS file
$dropdownFixPath = Join-Path (Get-Location).Path "templates\dropdown_fix.html"
$content = Get-Content $dropdownFixPath -Raw
$content = $content -replace "api_keys_panel.css", "api_keys_panel_$timestamp.css"
Set-Content -Path $dropdownFixPath -Value $content

Write-Host "Dashboard styles refreshed. Run .\run_fixed_dashboard_temp.ps1 to start the dashboard." -ForegroundColor Green