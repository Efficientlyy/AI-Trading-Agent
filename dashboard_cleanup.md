# Dashboard Cleanup Plan

This document outlines files that can be safely deleted or should be kept as part of the dashboard cleanup.

## Safe to Delete (Temporary/Debug Files)

These files were created for debugging and fixing the login issues and are no longer needed:

1. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/bypass_login.html`
2. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/debug_login.log`
3. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/debug_login.ps1`
4. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/debug_login.py`
5. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/fix_dashboard_login.ps1`
6. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/fix_dashboard_login.py`
7. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/direct_login.html`

## Keep (But Consolidate)

These files are part of the fixed solution and should be kept, but could be renamed for clarity:

1. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/fixed_dashboard.py` -> Rename to `run_modern_dashboard.py`
2. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/run_fixed_dashboard.ps1` -> Rename to `start_dashboard.ps1` (replace existing)

## Backups (Safe to Delete After Verification)

These backup files can be deleted after verifying the system works correctly:

1. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/src/dashboard/modern_dashboard.py.bak`

## Redundant Dashboard Scripts (Potentially Safe to Delete)

These scripts might be older or redundant versions, but further investigation is needed before deletion:

1. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/simple_dashboard.py`
2. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/minimal_dashboard.py`
3. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/working_dashboard.py`
4. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/start_dashboard.sh` (if you're on Windows)

## Keep (Essential Files)

These files are part of the main dashboard or specialized versions and should be kept:

1. All files in `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/src/dashboard/`
2. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/templates/` (all dashboard templates)
3. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/static/css/` (all CSS files)
4. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/run_dashboard.py` (main runner script)
5. `/mnt/c/Users/vp199/documents/projects/github/ai-trading-agent/run_dashboard_tests.py` (test runner)
6. Specialized dashboards (sentiment, performance, etc.)

## Clean Up Command

```powershell
# PowerShell script to clean up temporary dashboard files
$filesToDelete = @(
    "bypass_login.html",
    "debug_login.log",
    "debug_login.ps1",
    "debug_login.py",
    "fix_dashboard_login.ps1",
    "fix_dashboard_login.py",
    "direct_login.html"
)

foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "Deleted: $file" -ForegroundColor Green
    } else {
        Write-Host "File not found: $file" -ForegroundColor Yellow
    }
}

# Optional: Delete backup after verification
# Remove-Item "src/dashboard/modern_dashboard.py.bak" -Force

# Rename fixed dashboard to permanent solution
if (Test-Path "fixed_dashboard.py") {
    # Backup existing run_modern_dashboard.py if it exists
    if (Test-Path "run_modern_dashboard.py") {
        Move-Item "run_modern_dashboard.py" "run_modern_dashboard.py.old" -Force
        Write-Host "Backed up: run_modern_dashboard.py -> run_modern_dashboard.py.old" -ForegroundColor Yellow
    }
    
    # Rename fixed dashboard
    Move-Item "fixed_dashboard.py" "run_modern_dashboard.py" -Force
    Write-Host "Renamed: fixed_dashboard.py -> run_modern_dashboard.py" -ForegroundColor Green
}

# Replace start script
if (Test-Path "run_fixed_dashboard.ps1") {
    if (Test-Path "start_dashboard.ps1") {
        Move-Item "start_dashboard.ps1" "start_dashboard.ps1.old" -Force
        Write-Host "Backed up: start_dashboard.ps1 -> start_dashboard.ps1.old" -ForegroundColor Yellow
    }
    
    Move-Item "run_fixed_dashboard.ps1" "start_dashboard.ps1" -Force
    Write-Host "Renamed: run_fixed_dashboard.ps1 -> start_dashboard.ps1" -ForegroundColor Green
}

Write-Host "Cleanup complete!" -ForegroundColor Cyan
```

## Recommended Dashboard Launch Command

After cleanup, use this command to start the dashboard:

```
.\start_dashboard.ps1
```