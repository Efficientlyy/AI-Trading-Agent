# Dark Theme Testing Guide

This guide provides instructions for testing the dark theme implementation across all dashboard components in the AI Trading Agent application.

## Testing Objectives

1. Verify that all dashboard components render correctly in dark mode
2. Ensure text has sufficient contrast against backgrounds
3. Confirm that all interactive elements (buttons, links, etc.) are clearly visible
4. Check that status indicators maintain their meaning with appropriate colors

## Manual Testing Checklist

### System Control Panel

- [ ] Background color is dark and consistent with theme
- [ ] Text is clearly visible against the background
- [ ] Status indicators use appropriate colors (green for running, red for stopped, etc.)
- [ ] Buttons have sufficient contrast and are clearly visible
- [ ] Hover states are visible and provide clear feedback

### Agent Status Grid

- [ ] Agent cards have dark backgrounds with proper contrast
- [ ] Status indicators are clearly visible
- [ ] Text and metrics are readable
- [ ] Configuration dialog renders correctly in dark mode
- [ ] All form elements in dialogs are properly styled

### Session Management Panel

- [ ] Table headers and cells have appropriate dark styling
- [ ] Row hover states are visible
- [ ] Session status chips use appropriate colors
- [ ] Expanded details sections maintain dark theme styling
- [ ] Action buttons are clearly visible and use appropriate colors

### Activity Feed

- [ ] Activity items have proper dark backgrounds
- [ ] Text is readable with good contrast
- [ ] Icons use appropriate colors
- [ ] Timestamps and secondary text have proper styling

### Performance Metrics Panel

- [ ] Metric cards have dark backgrounds
- [ ] Values and labels are clearly visible
- [ ] Positive/negative indicators use appropriate colors
- [ ] Charts render correctly with dark theme colors

## Automated Testing

A test script has been created to help automate some of the dark theme testing. To use it:

1. Navigate to the dashboard in dark mode
2. Open the browser console (F12 or right-click > Inspect > Console)
3. Run the following command:

```javascript
import('./tests/darkThemeTest.js').then(module => {
  module.default.testDarkThemeCompliance();
});
```

The script will check for:
- Confirmation that dark mode is active
- Background and text colors for each component
- Contrast issues in text elements
- Hardcoded light theme colors

## Common Issues and Solutions

### Insufficient Contrast

If text is difficult to read against its background, update the text color using:
- For primary text: `color: darkText` or `sx={{ color: darkText }}`
- For secondary text: `color: darkSecondaryText` or `sx={{ color: darkSecondaryText }}`

### Inconsistent Background Colors

If a component's background doesn't match the dark theme:
- For paper/card backgrounds: `bgcolor: darkPaperBg` or `sx={{ bgcolor: darkPaperBg }}`
- For main backgrounds: `bgcolor: darkBg` or `sx={{ bgcolor: darkBg }}`

### Border Colors

If borders are too light or invisible:
- Use `borderColor: darkBorder` or `sx={{ borderColor: darkBorder }}`

## Accessibility Requirements

- Text should have a minimum contrast ratio of 4.5:1 against its background
- Interactive elements should have a minimum contrast ratio of 3:1
- Focus states should be clearly visible
- Color should not be the only means of conveying information

## Reporting Issues

When reporting dark theme issues, please include:
1. The component name
2. A screenshot showing the issue
3. Steps to reproduce
4. Browser and screen size information
