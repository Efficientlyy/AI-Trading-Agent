/**
 * Dark Theme Visual Verification Script
 * 
 * This script performs a visual check of dark theme styling across all dashboard components.
 * It logs the results to the console for manual verification.
 */

// List of components to check
const componentsToCheck = [
  {
    name: 'SessionManagementPanel',
    darkThemeElements: [
      { selector: '.MuiPaper-root', property: 'backgroundColor', expectedDark: true },
      { selector: '.MuiTableHead-root', property: 'backgroundColor', expectedDark: true },
      { selector: '.MuiTableCell-root', property: 'color', expectedLight: true }
    ]
  },
  {
    name: 'ActivityFeed',
    darkThemeElements: [
      { selector: '.activity-item', property: 'backgroundColor', expectedDark: true },
      { selector: '.activity-timestamp', property: 'color', expectedLight: true }
    ]
  },
  {
    name: 'SystemControlPanel',
    darkThemeElements: [
      { selector: '.system-status', property: 'backgroundColor', expectedDark: true },
      { selector: '.MuiButton-root', property: 'color', expectedLight: true }
    ]
  },
  {
    name: 'PerformanceMetricsPanel',
    darkThemeElements: [
      { selector: '.metrics-card', property: 'backgroundColor', expectedDark: true },
      { selector: '.metric-value', property: 'color', expectedLight: true }
    ]
  }
];

// Helper function to check if a color is dark
function isDarkColor(color) {
  // Convert color to RGB if it's in hex format
  let r, g, b;
  
  if (color.startsWith('#')) {
    // Hex color
    const hex = color.substring(1);
    r = parseInt(hex.substring(0, 2), 16);
    g = parseInt(hex.substring(2, 4), 16);
    b = parseInt(hex.substring(4, 6), 16);
  } else if (color.startsWith('rgb')) {
    // RGB color
    const matches = color.match(/\d+/g);
    if (matches && matches.length >= 3) {
      r = parseInt(matches[0]);
      g = parseInt(matches[1]);
      b = parseInt(matches[2]);
    }
  }
  
  if (r !== undefined && g !== undefined && b !== undefined) {
    // Calculate relative luminance
    // Dark colors have low luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance < 0.5;
  }
  
  return false;
}

// Function to check dark theme styling
function checkDarkThemeStyling() {
  console.log('🔍 Dark Theme Visual Verification');
  console.log('================================');
  
  // Check if dark mode is active
  const isDarkMode = document.documentElement.classList.contains('dark');
  console.log(`Dark mode active: ${isDarkMode ? '✅' : '❌'}`);
  
  if (!isDarkMode) {
    console.warn('Warning: Dark mode is not active. Results may not be accurate.');
  }
  
  // Check each component
  componentsToCheck.forEach(component => {
    console.log(`\nChecking ${component.name}...`);
    
    let passedChecks = 0;
    let totalChecks = component.darkThemeElements.length;
    
    component.darkThemeElements.forEach(element => {
      const elements = document.querySelectorAll(element.selector);
      
      if (elements.length === 0) {
        console.log(`  - ${element.selector}: ❓ Not found`);
        totalChecks--;
        return;
      }
      
      // Check the first element
      const style = window.getComputedStyle(elements[0]);
      const propertyValue = style[element.property];
      
      const isDark = isDarkColor(propertyValue);
      const pass = element.expectedDark ? isDark : !isDark;
      
      console.log(`  - ${element.selector} (${element.property}): ${propertyValue}`);
      console.log(`    Expected: ${element.expectedDark ? 'Dark' : 'Light'}, Actual: ${isDark ? 'Dark' : 'Light'} - ${pass ? '✅' : '❌'}`);
      
      if (pass) passedChecks++;
    });
    
    if (totalChecks > 0) {
      const passRate = Math.round((passedChecks / totalChecks) * 100);
      console.log(`\n${component.name} Dark Theme Compliance: ${passRate}% (${passedChecks}/${totalChecks})`);
    } else {
      console.log(`\n${component.name}: Unable to verify (component not found)`);
    }
  });
  
  console.log('\n================================');
  console.log('🏁 Dark Theme Visual Verification Complete');
}

// Make the function available globally
window.checkDarkThemeStyling = checkDarkThemeStyling;

console.log(`
To verify dark theme styling:
1. Make sure dark mode is enabled
2. Navigate to the dashboard
3. Open browser console
4. Run: checkDarkThemeStyling()
`);
