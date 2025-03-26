# PowerShell script to run the fixed dashboard with improved dropdown menu

Write-Host "Starting fixed modern dashboard with improved dropdown menu..." -ForegroundColor Cyan
Write-Host "This version fixes template recursion, login issues, and the user dropdown menu" -ForegroundColor Yellow
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

# Force browser to reload CSS by adding timestamp to query string
Write-Host "Adding version timestamp to CSS files to force cache refresh..." -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$cssPath = Join-Path (Get-Location).Path "static\css\modern_dashboard.css"
$cssContent = Get-Content $cssPath -Raw
$cssContent = $cssContent + "`n/* Cache-busting timestamp: $timestamp */`n"
Set-Content -Path $cssPath -Value $cssContent

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

# Verify the new user_menu.js file exists
$userMenuPath = Join-Path (Get-Location).Path "static\js\user_menu.js"
if (Test-Path $userMenuPath) {
    Write-Host "Found user_menu.js file" -ForegroundColor Green
} else {
    Write-Host "user_menu.js file not found! Creating it..." -ForegroundColor Red
    # Create minimal version if file is missing
    $userMenuContent = @'
/**
 * User Menu and Navigation Functionality - Minimal Version
 */
document.addEventListener('DOMContentLoaded', function() {
    const userProfile = document.querySelector('.user-profile');
    const userDropdown = document.querySelector('.user-dropdown');
    
    if (userProfile && userDropdown) {
        // Direct style modifications for dropdown menu
        userDropdown.style.position = "absolute";
        userDropdown.style.top = "100%";
        userDropdown.style.right = "0";
        userDropdown.style.width = "200px";
        userDropdown.style.backgroundColor = "white";
        userDropdown.style.borderRadius = "8px";
        userDropdown.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.15)";
        userDropdown.style.zIndex = "1000";
        userDropdown.style.marginTop = "8px";
        userDropdown.style.overflow = "hidden";
        
        userProfile.addEventListener('mouseenter', function() {
            userDropdown.style.display = "block";
        });
        
        userProfile.addEventListener('mouseleave', function() {
            setTimeout(() => {
                if (!userDropdown.matches(':hover')) {
                    userDropdown.style.display = "none";
                }
            }, 200);
        });
        
        userDropdown.addEventListener('mouseleave', function() {
            userDropdown.style.display = "none";
        });
        
        // Also handle clicks for touch devices
        userProfile.addEventListener('click', function(event) {
            if (userDropdown.style.display === "none" || userDropdown.style.display === "") {
                event.preventDefault();
                userDropdown.style.display = "block";
            }
        });
        
        document.addEventListener('click', function(event) {
            if (!userProfile.contains(event.target) && !userDropdown.contains(event.target)) {
                userDropdown.style.display = "none";
            }
        });
        
        console.log("User menu dropdown enhancements applied");
    } else {
        console.warn("User profile or dropdown elements not found");
    }
});
'@
    Set-Content -Path $userMenuPath -Value $userMenuContent
    Write-Host "Created minimal user_menu.js file" -ForegroundColor Green
}

# Verify the API keys panel is included in the modern_dashboard.html
$dashboardPath = Join-Path (Get-Location).Path "templates\modern_dashboard.html"
$dashboardContent = Get-Content $dashboardPath -Raw
if ($dashboardContent -notmatch "include 'api_keys_panel.html'") {
    Write-Host "API keys panel not included in modern_dashboard.html. Adding it..." -ForegroundColor Yellow
    # Find the closing body tag and add the include right before it
    $dashboardContent = $dashboardContent -replace "</body>", "    {% include 'api_keys_panel.html' %}`n</body>"
    Set-Content -Path $dashboardPath -Value $dashboardContent
    Write-Host "Added API keys panel to modern_dashboard.html" -ForegroundColor Green
}

# Check if the user_menu.js is included in the HTML
if ($dashboardContent -notmatch "user_menu.js") {
    Write-Host "user_menu.js not included in modern_dashboard.html. Adding it..." -ForegroundColor Yellow
    # Find the dashboard_optimizer.js script tag and add user_menu.js after it
    $dashboardContent = $dashboardContent -replace "dashboard_optimizer.js", "dashboard_optimizer.js`"></script>`n    <!-- Include user menu functionality -->`n    <script src=""{{ url_for('static', filename='js/user_menu.js')"
    Set-Content -Path $dashboardPath -Value $dashboardContent
    Write-Host "Added user_menu.js to modern_dashboard.html" -ForegroundColor Green
}

# Run the dashboard
Write-Host "Starting dashboard with dropdown menu fixes..." -ForegroundColor Green
python fixed_dashboard.py --debug