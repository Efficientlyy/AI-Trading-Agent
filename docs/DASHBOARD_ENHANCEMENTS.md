# Dashboard Enhancements

This document provides an overview of the enhancements made to the AI Trading Agent dashboard to improve its functionality, performance, and user experience.

## Table of Contents

1. [Data Source Toggle](#data-source-toggle)
2. [Enhanced Error Handling](#enhanced-error-handling)
3. [Global Loading Indicator](#global-loading-indicator)
4. [Persistent Cache System](#persistent-cache-system)
5. [Real Data Configuration](#real-data-configuration)
6. [Usage Instructions](#usage-instructions)
7. [Troubleshooting](#troubleshooting)

## Data Source Toggle

The data source toggle allows users to switch between mock data and real data sources in the dashboard. This is particularly useful for testing and development purposes.

### Features

- Visual toggle button in the dashboard interface
- Smooth transition between data sources
- Visual feedback on the current data source
- Error handling for unavailable data sources

### Implementation Details

The data source toggle is implemented using:

- CSS styles in `static/css/data_source_toggle.css`
- JavaScript functionality in `static/js/data_source_toggle.js`
- API endpoints in `src/dashboard/modern_dashboard.py`

## Enhanced Error Handling

The enhanced error handling system provides more detailed error messages and better user feedback when errors occur.

### Features

- Detailed error messages with possible solutions
- Visual error notifications
- Error logging and tracking
- Graceful degradation when services are unavailable

### Implementation Details

The error handling system is implemented in:

- JavaScript error handling in `static/js/dashboard_enhancements.js`
- CSS styles for error messages in `static/css/data_source_toggle.css`
- Server-side error handling in `src/dashboard/modern_dashboard.py`

## Global Loading Indicator

The global loading indicator provides visual feedback when the dashboard is loading data or performing operations.

### Features

- Subtle loading bar at the top of the page
- Automatic display during API requests
- Stacking of multiple requests (only hides when all requests are complete)

### Implementation Details

The loading indicator is implemented in:

- CSS styles in `static/css/data_source_toggle.css`
- JavaScript functionality in `static/js/dashboard_enhancements.js`

## Persistent Cache System

The persistent cache system improves dashboard performance by caching data locally and reducing the number of API requests.

### Features

- In-memory cache for fast access
- Persistent cache using localStorage for data that survives page reloads
- Type-specific cache expiration times
- Automatic cache invalidation
- Stale-while-revalidate pattern for improved perceived performance

### Implementation Details

The cache system is implemented in:

- JavaScript cache management in `static/js/dashboard_enhancements.js`
- Cache configuration in the `DashboardEnhancements` class

## Real Data Configuration

The real data configuration system allows for easy enabling and disabling of real data connections.

### Features

- Command-line interface for enabling/disabling real data
- Configuration file for detailed settings
- Environment variable support
- Automatic detection of available data sources

### Implementation Details

The real data configuration is implemented in:

- Configuration file at `config/real_data_config.json`
- Command-line tool in `enable_real_data.py`
- Integration with the dashboard in `src/dashboard/modern_dashboard.py`

## Usage Instructions

### Switching Between Mock and Real Data

1. Use the data source toggle in the dashboard interface:
   - Click the "Mock" button to use mock data
   - Click the "Real" button to use real data (if available)

2. Use the command-line tool:
   ```bash
   # Check current status
   python enable_real_data.py status

   # Enable real data
   python enable_real_data.py enable

   # Disable real data
   python enable_real_data.py disable
   ```

### Configuring Real Data Sources

Edit the `config/real_data_config.json` file to configure real data sources:

```json
{
    "enabled": true,
    "connections": {
        "exchange_api": {
            "enabled": true,
            "retry_attempts": 3,
            "timeout_seconds": 10,
            "cache_duration_seconds": 60
        },
        // Other connections...
    },
    // Other settings...
}
```

### Clearing the Cache

To clear the cache programmatically:

```javascript
// Clear all cache
window.dashboardEnhancements.clearCache();
```

## Troubleshooting

### Data Source Toggle Not Working

1. Check if real data is available:
   ```bash
   python enable_real_data.py status
   ```

2. Check the browser console for JavaScript errors

3. Verify that the configuration file exists and is valid:
   ```bash
   cat config/real_data_config.json
   ```

### Dashboard Performance Issues

1. Check the cache configuration in `static/js/dashboard_enhancements.js`

2. Consider increasing cache expiration times for frequently accessed data

3. Check browser memory usage and consider clearing the cache if it's too large

### Error Messages Not Displaying

1. Verify that the error container exists in the DOM

2. Check the browser console for JavaScript errors

3. Ensure that the CSS styles for error messages are loaded correctly