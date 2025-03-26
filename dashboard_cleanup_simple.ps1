# Simple PowerShell script to clean up temporary dashboard files
Write-Host "Dashboard Cleanup Script" -ForegroundColor Cyan
Write-Host "------------------------" -ForegroundColor Cyan
Write-Host ""

# Files to delete (temporary/debug files)
$filesToDelete = @(
    "bypass_login.html",
    "debug_login.log",
    "debug_login.ps1",
    "debug_login.py",
    "fix_dashboard_login.ps1",
    "fix_dashboard_login.py",
    "direct_login.html"
)

Write-Host "Cleaning up temporary files:" -ForegroundColor Yellow
foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  Deleted: $file" -ForegroundColor Green
    } else {
        Write-Host "  File not found: $file" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Consolidating dashboard files:" -ForegroundColor Yellow

# Rename fixed dashboard to permanent solution
if (Test-Path "fixed_dashboard.py") {
    # Backup existing run_modern_dashboard.py if it exists
    if (Test-Path "run_modern_dashboard.py") {
        Copy-Item "run_modern_dashboard.py" "run_modern_dashboard.py.old" -Force
        Write-Host "  Backed up: run_modern_dashboard.py -> run_modern_dashboard.py.old" -ForegroundColor Green
    }
    
    # Rename fixed dashboard
    Copy-Item "fixed_dashboard.py" "run_modern_dashboard.py" -Force
    Write-Host "  Copied: fixed_dashboard.py -> run_modern_dashboard.py" -ForegroundColor Green
}

# Replace start script
if (Test-Path "run_fixed_dashboard.ps1") {
    if (Test-Path "start_dashboard.ps1") {
        Copy-Item "start_dashboard.ps1" "start_dashboard.ps1.old" -Force
        Write-Host "  Backed up: start_dashboard.ps1 -> start_dashboard.ps1.old" -ForegroundColor Green
    }
    
    Copy-Item "run_fixed_dashboard.ps1" "start_dashboard.ps1" -Force
    Write-Host "  Copied: run_fixed_dashboard.ps1 -> start_dashboard.ps1" -ForegroundColor Green
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the dashboard, use:" -ForegroundColor Cyan
Write-Host "  .\start_dashboard.ps1" -ForegroundColor White
Write-Host ""
Write-Host "For more details, see dashboard_cleanup.md" -ForegroundColor Cyan
