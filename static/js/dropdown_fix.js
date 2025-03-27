/**
 * Dropdown menu fix
 * 
 * This script applies fixes to ensure dropdown menus work correctly
 * on all browsers and environments.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Apply fix to all dropdowns on the page
    fixAllDropdowns();
    
    // Force a reload of the stylesheets to overcome caching issues
    forceStyleReload();
});

/**
 * Force a reload of stylesheets by adding a timestamp
 */
function forceStyleReload() {
    const links = document.querySelectorAll('link[rel="stylesheet"]');
    const timestamp = new Date().getTime();
    
    links.forEach(function(link) {
        const url = link.getAttribute('href');
        if (url && url.includes('/static/')) {
            link.setAttribute('href', url + '?v=' + timestamp);
        }
    });
}

/**
 * Fix all dropdown menus on the page
 */
function fixAllDropdowns() {
    // Find all dropdown toggle buttons
    const dropdownToggles = document.querySelectorAll('[data-toggle="dropdown"]');
    
    // Apply fix to each one
    dropdownToggles.forEach(function(toggle) {
        fixDropdown(toggle);
    });
    
    // Also look for user dropdown specifically
    const userMenu = document.getElementById('user-menu');
    if (userMenu) {
        fixDropdown(userMenu);
    }
}

/**
 * Fix a specific dropdown toggle button
 */
function fixDropdown(toggle) {
    // Find the associated dropdown menu
    let dropdown;
    
    // Try various methods to find the associated dropdown
    const target = toggle.getAttribute('data-target');
    if (target) {
        dropdown = document.querySelector(target);
    }
    
    if (!dropdown) {
        // Look for next sibling that's a dropdown
        let sibling = toggle.nextElementSibling;
        while (sibling) {
            if (sibling.classList.contains('dropdown-menu')) {
                dropdown = sibling;
                break;
            }
            sibling = sibling.nextElementSibling;
        }
    }
    
    if (!dropdown) {
        // Find dropdown within the same parent
        const parent = toggle.closest('.dropdown');
        if (parent) {
            dropdown = parent.querySelector('.dropdown-menu');
        }
    }
    
    // Exit if no dropdown found
    if (!dropdown) {
        return;
    }
    
    // Make sure the dropdown has high z-index
    dropdown.style.zIndex = '9999';
    
    // Apply the fix
    toggle.addEventListener('click', function(event) {
        event.preventDefault();
        event.stopPropagation();
        
        // Force toggle with !important-like approach
        const isVisible = dropdown.classList.contains('show');
        
        // Close all other dropdowns first
        document.querySelectorAll('.dropdown-menu.show').forEach(function(menu) {
            if (menu !== dropdown) {
                menu.classList.remove('show');
                menu.style.display = '';
            }
        });
        
        if (isVisible) {
            dropdown.classList.remove('show');
            dropdown.style.display = '';
            toggle.setAttribute('aria-expanded', 'false');
        } else {
            dropdown.classList.add('show');
            dropdown.style.display = 'block';
            toggle.setAttribute('aria-expanded', 'true');
            
            // Position the dropdown correctly
            positionDropdown(toggle, dropdown);
        }
    });
    
    // Close when clicking outside
    document.addEventListener('click', function(event) {
        if (!toggle.contains(event.target) && !dropdown.contains(event.target)) {
            dropdown.classList.remove('show');
            dropdown.style.display = '';
            toggle.setAttribute('aria-expanded', 'false');
        }
    });
}

/**
 * Position a dropdown to ensure it's visible
 */
function positionDropdown(toggle, dropdown) {
    // Get positions
    const toggleRect = toggle.getBoundingClientRect();
    const dropdownRect = dropdown.getBoundingClientRect();
    
    // Calculate if dropdown goes off-screen
    const rightEdge = toggleRect.right;
    const bottomEdge = toggleRect.bottom;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    
    // Position horizontally
    if (rightEdge + dropdownRect.width > windowWidth) {
        dropdown.style.right = '0';
        dropdown.style.left = 'auto';
    } else {
        dropdown.style.left = '0';
        dropdown.style.right = 'auto';
    }
    
    // Position vertically
    if (bottomEdge + dropdownRect.height > windowHeight) {
        dropdown.style.bottom = toggleRect.height + 'px';
        dropdown.style.top = 'auto';
    } else {
        dropdown.style.top = '100%';
        dropdown.style.bottom = 'auto';
    }
}