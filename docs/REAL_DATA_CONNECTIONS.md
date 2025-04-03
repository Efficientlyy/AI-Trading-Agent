# Real Data Connections

## Overview

This document provides detailed information about the real data connections in the AI Trading Agent dashboard. It explains how the dashboard connects to real data sources, how the data is processed, and how to troubleshoot common issues.

## Architecture

The real data connections in the dashboard follow a layered architecture:

1. **Data Sources**: Raw data from trade files, log files, and system metrics
2. **Data Providers**: Classes that read and process data from specific sources
3. **Data Service**: A unified interface for accessing data from different providers
4. **Dashboard Components**: UI components that display the data

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Data Providers │────▶│   Data Service  │────▶│    Dashboard    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Data Sources

### Trade Data

Trade data is stored in JSON files in the `data/trades/` directory. Each file represents a single trade with the following structure:

```json
{
  "id": "T-1234567890-1234",
  "asset": "BTC/USD",
  "side": "buy",
  "quantity": 1.0,
  "entry_price": 50000.0,
  "exit_price": 55000.0,
  "entry_time": "2025-03-01T10:00:00Z",
  "exit_time": "2025-03-01T14:30:00Z",
  "profit_loss": 5000.0,
  "profit_loss_pct": 10.0,
  "strategy": "trend_following",
  "tags": ["breakout", "high_volume"],
  "notes": "Strong momentum trade"
}
```

### Log Data

Log data is stored in log files in the `logs/` directory. The dashboard uses the `LogQuery` class to read and process log files.

### System Metrics

System metrics are collected in real-time by the `SystemMonitor` class. These metrics include CPU usage, memory usage, disk usage, and network latency.

## Data Providers

### PerformanceTracker

The `PerformanceTracker` class is responsible for loading and processing trade data. It provides methods to calculate performance metrics such as win rate, profit factor, and drawdown.

```python
from src.common.performance import PerformanceTracker

# Create a PerformanceTracker instance
tracker = PerformanceTracker()

# Get performance summary
summary = tracker.get_performance_summary()

# Get detailed metrics
metrics = tracker.get_performance_metrics()
```

### SystemMonitor

The `SystemMonitor` class is responsible for collecting system metrics. It provides methods to get the current system health and resource usage.

```python
from src.common.system import SystemMonitor

# Create a SystemMonitor instance
monitor = SystemMonitor()

# Get system health
health = monitor.get_system_health()
```

### LogQuery

The `LogQuery` class is responsible for reading and processing log files. It provides methods to query logs by time range, level, and component.

```python
from src.common.log_query import LogQuery

# Create a LogQuery instance
query = LogQuery()

# Get recent logs
logs = query.get_logs(limit=100)
```

## Data Service

The `DataService` class provides a unified interface for accessing data from different providers. It supports both real and mock data sources, and can be configured to use either one.

```python
from src.dashboard.utils.data_service import DataService, DataSource

# Create a DataService instance with mock data
service = DataService(data_source=DataSource.MOCK)

# Switch to real data
service.set_data_source(DataSource.REAL)

# Get system health data
health = service.get_data("system_health")

# Get performance data
performance = service.get_data("trading_performance")
```

## Configuration

The real data connections can be enabled or disabled using the `deploy_real_data_connections.py` script:

```bash
# Enable real data connections
python deploy_real_data_connections.py enable

# Disable real data connections
python deploy_real_data_connections.py disable

# Check current status
python deploy_real_data_connections.py status
```

The script modifies the `REAL_DATA_AVAILABLE` flag in the `src/dashboard/utils/data_service.py` file.

## Troubleshooting

### Common Issues

1. **No trade data available**: Ensure that trade files exist in the `data/trades/` directory.
2. **No log data available**: Ensure that log files exist in the `logs/` directory.
3. **Cannot switch to real data**: Ensure that `REAL_DATA_AVAILABLE` is set to `True` in `data_service.py`.
4. **Timezone errors**: Ensure that trade files use ISO 8601 format with timezone information (e.g., `2025-03-01T10:00:00Z`).

### Debugging

To debug real data connections, you can use the test scripts:

```bash
# Test PerformanceTracker
python test_performance_tracker.py

# Test real data connections
python test_real_data_connections.py
```

## Best Practices

1. **Use ISO 8601 format for dates**: Always use ISO 8601 format with timezone information for dates in trade files.
2. **Validate trade data**: Ensure that trade files contain all required fields and use the correct data types.
3. **Handle timezone differences**: Be aware of timezone differences when comparing dates from different sources.
4. **Implement error handling**: Always handle errors gracefully and provide fallback options when real data is unavailable.
5. **Use caching**: Implement caching to improve performance when accessing real data.

## Future Improvements

1. **Real-time data updates**: Implement WebSocket connections for real-time data updates.
2. **Data validation**: Add validation for real data to ensure it is properly formatted.
3. **Data transformation**: Implement data transformation to normalize data from different sources.
4. **Data export**: Add functionality to export real data for further analysis.
5. **Configuration UI**: Create a UI for configuring real data connections.

## References

- [PerformanceTracker Documentation](../src/common/performance.py)
- [SystemMonitor Documentation](../src/common/system.py)
- [LogQuery Documentation](../src/common/log_query.py)
- [DataService Documentation](../src/dashboard/utils/data_service.py)
- [Real Data Implementation Plan](./REAL_DATA_IMPLEMENTATION_PLAN.md)
