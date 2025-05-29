/**
 * Dark Theme Testing Script
 * 
 * This script helps verify that all dashboard components render correctly in dark mode.
 * Run this in the browser console when viewing the dashboard in dark mode.
 */

// Helper function to check contrast ratio
function getContrastRatio(color1, color2) {
  // Convert hex to rgb
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  // Calculate relative luminance
  const getLuminance = (rgb) => {
    const a = [rgb.r, rgb.g, rgb.b].map((v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
  };

  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);
  
  if (!rgb1 || !rgb2) return null;
  
  const l1 = getLuminance(rgb1);
  const l2 = getLuminance(rgb2);
  
  const ratio = l1 > l2 ? (l1 + 0.05) / (l2 + 0.05) : (l2 + 0.05) / (l1 + 0.05);
  return ratio.toFixed(2);
}

// Test dashboard components for dark theme compliance
function testDarkThemeCompliance() {
  console.log('🔍 Testing Dark Theme Compliance');
  console.log('================================');
  
  // 1. Check if dark mode is active
  const isDarkMode = document.documentElement.classList.contains('dark');
  console.log(`Dark mode active: ${isDarkMode ? '✅' : '❌'}`);
  
  if (!isDarkMode) {
    console.warn('Please enable dark mode to run this test');
    return;
  }
  
  // 2. Check background colors
  const components = [
    { name: 'SystemControlPanel', selector: '.system-control-panel' },
    { name: 'AgentStatusGrid', selector: '.agent-status-grid' },
    { name: 'SessionManagementPanel', selector: '.session-management-panel' },
    { name: 'ActivityFeed', selector: '.activity-feed' },
    { name: 'PerformanceMetricsPanel', selector: '.performance-metrics-panel' }
  ];
  
  components.forEach(component => {
    const element = document.querySelector(component.selector);
    if (!element) {
      console.log(`${component.name}: Component not found ❓`);
      return;
    }
    
    const bgColor = window.getComputedStyle(element).backgroundColor;
    const textColor = window.getComputedStyle(element).color;
    
    console.log(`${component.name}:`);
    console.log(`  - Background: ${bgColor}`);
    console.log(`  - Text color: ${textColor}`);
    
    // Check if children have proper contrast
    const textElements = element.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, button, a');
    let lowContrastElements = 0;
    
    textElements.forEach(el => {
      const elBg = window.getComputedStyle(el).backgroundColor;
      const elColor = window.getComputedStyle(el).color;
      
      // Skip elements with transparent background
      if (elBg === 'transparent' || elBg === 'rgba(0, 0, 0, 0)') return;
      
      // Convert rgb to hex for contrast calculation
      // This is a simplified conversion and might not work for all cases
      const rgbToHex = (rgb) => {
        const [r, g, b] = rgb.match(/\d+/g);
        return `#${Number(r).toString(16).padStart(2, '0')}${Number(g).toString(16).padStart(2, '0')}${Number(b).toString(16).padStart(2, '0')}`;
      };
      
      const contrast = getContrastRatio(rgbToHex(elBg), rgbToHex(elColor));
      if (contrast && contrast < 4.5) {
        lowContrastElements++;
        console.warn(`  - Low contrast element found: ${el.tagName} with text "${el.textContent.trim()}" (ratio: ${contrast})`);
      }
    });
    
    console.log(`  - Contrast issues: ${lowContrastElements === 0 ? '✅ None' : `❌ ${lowContrastElements} found`}`);
  });
  
  // 3. Check for any hardcoded light theme colors
  const allElements = document.querySelectorAll('*');
  const lightThemeColors = ['#ffffff', '#f8f9fa', '#e9ecef', '#dee2e6', '#fff'];
  let hardcodedLightColors = 0;
  
  allElements.forEach(el => {
    const bgColor = window.getComputedStyle(el).backgroundColor;
    if (lightThemeColors.includes(bgColor)) {
      hardcodedLightColors++;
      console.warn(`Hardcoded light color found in ${el.tagName} element`);
    }
  });
  
  console.log(`Hardcoded light colors: ${hardcodedLightColors === 0 ? '✅ None' : `❌ ${hardcodedLightColors} found`}`);
  
  console.log('================================');
  console.log('🏁 Dark Theme Compliance Test Complete');
}

// Export the test function
window.testDarkThemeCompliance = testDarkThemeCompliance;

// Instructions for use
console.log(`
To test dark theme compliance:
1. Make sure dark mode is enabled
2. Open browser console
3. Run: testDarkThemeCompliance()
`);
