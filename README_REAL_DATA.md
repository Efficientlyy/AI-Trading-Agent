# Real Data Connections in AI Trading Agent Dashboard

## Overview

This document explains how the dashboard handles real data connections and provides guidance on enabling or disabling real data sources.

## Current Status

The dashboard is currently configured to use mock data by default. Real data connections are **disabled** in the system configuration.

## How Real Data Sources Work

The dashboard can operate in two data source modes:

1. **Mock Data**: Generates simulated data for all dashboard components. This is useful for development, testing, and demonstration purposes.

2. **Real Data**: Connects to actual data sources including:
   - Trading performance data from trade files
   - System monitoring data
   - Log data

## Configuration

Real data availability is controlled by the `REAL_DATA_AVAILABLE` flag in `src/dashboard/utils/data_service.py`. 

```python
# Flag to indicate if real data components are available
REAL_DATA_AVAILABLE = False  # Set to True to enable real data sources
```

The system also checks for the existence of real data files:
- Trade data in `data/trades/*.json`
- Log data in `logs/*.log*`

## Component Availability

Even when real data is enabled, individual components may fall back to mock data if their specific data sources are not available:

```python
COMPONENTS_AVAILABLE = {
    "system_health": True,  # SystemMonitor always available
    "logs_monitoring": HAS_LOG_DATA,
    "trading_performance": HAS_TRADE_DATA
}
```

## Enabling Real Data

To enable real data sources:

1. Ensure you have the necessary data files:
   - Trade data files in `data/trades/` (JSON format)
   - Log files in `logs/` directory

2. Set `REAL_DATA_AVAILABLE = True` in `src/dashboard/utils/data_service.py`

3. Restart the dashboard application

## Troubleshooting

If you encounter issues with real data connections:

1. Check the terminal output for error messages
2. Verify that data files exist in the expected locations
3. Ensure the `PerformanceTracker` and other data components are properly initialized
4. Check that the `REAL_DATA_AVAILABLE` flag is set to `True`

## Implementation Details

The real data handling is implemented in the `DataService` class in `src/dashboard/utils/data_service.py`. Key methods include:

- `set_data_source()`: Handles switching between mock and real data sources
- `_get_real_data()`: Fetches data from real sources
- `_get_real_performance_data()`: Gets performance data from trade files
- `_get_real_logs_data()`: Gets log data from log files
- `_get_real_system_health()`: Gets system health data from the system monitor

## Recent Fixes

### 2025-03-30: Fixed PerformanceTracker initialization

Fixed an issue in the `PerformanceTracker.__init__()` method where it was incorrectly handling the `data_dir` parameter when it was `None`. This was causing errors when trying to access trade files.

```python
# Before:
self.trades_dir = data_dir / "trades" if data_dir else TRADES_DIR

# After:
self.trades_dir = data_dir / "trades" if data_dir is not None else TRADES_DIR
```

### 2025-03-30: Disabled Real Data Sources

Set `REAL_DATA_AVAILABLE = False` to prevent errors when trying to access real data sources that may not be properly configured.
