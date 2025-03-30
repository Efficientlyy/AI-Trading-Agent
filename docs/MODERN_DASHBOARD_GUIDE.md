# AI Trading Agent Modern Dashboard Guide

## Overview

The AI Trading Agent Modern Dashboard provides a comprehensive web-based interface for monitoring and controlling the trading system. It serves as the primary user interface for interacting with the trading system through a browser.

## Key Features

- **System Control**: Start/stop the system and enable/disable trading
- **Real-time Monitoring**: View system status, component health, and performance metrics
- **Market Regime Analysis**: Analyze current market conditions and regime probabilities
- **Sentiment Analysis**: Track market sentiment from various sources
- **Risk Management**: Monitor risk metrics and portfolio exposure
- **Performance Analytics**: Analyze trading performance and strategy effectiveness
- **System Logs**: View and filter system logs for troubleshooting

## Getting Started

### Running the Dashboard

To start the dashboard, use one of the following methods:

**Windows:**
```powershell
.\start_dashboard.ps1
```

**Linux/macOS:**
```bash
./start_dashboard.sh
```

Or directly:
```bash
python run_modern_dashboard.py
```

### Accessing the Dashboard

Once started, the dashboard is available at:
- URL: http://127.0.0.1:8000 (or another available port if 8000 is taken)
- Default credentials:
  - Username: admin, Password: admin123
  - Username: operator, Password: operator123
  - Username: viewer, Password: viewer123

## Dashboard Sections

### 1. Overview
The main dashboard provides a high-level view of the system status, including:
- System power status (running/stopped)
- Trading status (enabled/disabled)
- System resource utilization (CPU, memory, disk, network)
- Component status

### 2. Market Regime
Analyzes and displays the current market regime:
- Dominant regime identification
- Regime confidence score
- Regime probabilities
- Historical regime transitions

### 3. Sentiment Analysis
Tracks market sentiment from various sources:
- Overall sentiment score
- Sentiment by source (social media, news, etc.)
- Sentiment trends over time
- Sentiment impact on price

### 4. Risk Management
Monitors risk metrics and portfolio exposure:
- Risk capacity utilization
- Risk tolerance level
- Key risk metrics (VaR, drawdown, etc.)
- Position sizing and exposure

### 5. Performance Analytics
Analyzes trading performance:
- Return metrics
- Win rate and profit factor
- Strategy comparison
- Drawdown analysis

### 6. System Logs
Provides access to system logs:
- Filterable log viewer
- Log level selection
- Component filtering
- Search functionality

## Configuration

The dashboard can be configured through command-line arguments:

```bash
python run_modern_dashboard.py --help
```

Common options:
- `--port`: Specify the port (default: 0 for auto-detection)
- `--host`: Specify the host (default: 127.0.0.1)
- `--debug`: Enable debug mode

You can also configure the dashboard through environment variables:
- `FLASK_TEMPLATE_FOLDER`: Path to the templates directory
- `FLASK_STATIC_FOLDER`: Path to the static files directory
- `FLASK_DEBUG`: Enable debug mode (1 for enabled, 0 for disabled)
- `FLASK_ENV`: Set the environment (development, testing, production)

## Troubleshooting

### Common Issues

1. **Dashboard Not Starting**
   - Check if port 8000 is already in use
   - Verify all dependencies are installed
   - Check logs for specific errors

2. **Login Issues**
   - Ensure you're using the correct credentials
   - Clear browser cookies and cache
   - Try a different browser

3. **Missing Data**
   - Verify the system is running (click "Start" button)
   - Check if trading is enabled
   - Ensure data source is properly configured

4. **Visualization Problems**
   - Check browser console for JavaScript errors
   - Try refreshing the page
   - Ensure browser is up to date

## Development

For developers working on the dashboard, refer to the following documentation:

- [DASHBOARD_ARCHITECTURE.md](DASHBOARD_ARCHITECTURE.md) - Technical architecture
- [DASHBOARD_IMPLEMENTATION.md](DASHBOARD_IMPLEMENTATION.md) - Implementation details
- [DASHBOARD_REDESIGN_PLAN.md](DASHBOARD_REDESIGN_PLAN.md) - Future enhancement plans
- [dashboard_testing_guide.md](dashboard_testing_guide.md) - Testing guidelines
- [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md) - Documentation guidelines

## Data Sources

The dashboard can use two types of data sources:

1. **Real Data**: Connects to the actual trading system and displays real-time data
2. **Mock Data**: Uses generated mock data for development and testing

You can switch between data sources using the data source toggle in the dashboard settings. Mock data is automatically used when real data is unavailable.

## User Roles

The dashboard supports three user roles:

1. **Admin**: Full access to all dashboard features and system controls
2. **Operator**: Can view all data and control trading operations
3. **Viewer**: Read-only access to dashboard data

Each role has different permissions and access levels to ensure proper security and control.
