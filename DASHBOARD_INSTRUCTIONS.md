# AI Trading Agent Dashboard Instructions

This document provides instructions for running and using the AI Trading Agent modern dashboard.

## Overview

The modern dashboard provides a unified interface with tabs for:

1. **Overview** - System status, performance metrics, component health
2. **Market Regime** - Market regime detection and visualization
3. **Sentiment Analysis** - Market sentiment from various sources
4. **Risk Management** - Risk utilization and portfolio metrics
5. **Performance Analytics** - Trading performance and strategy comparison
6. **Logs & Monitoring** - System logs and health monitoring

## Prerequisites

Before running the dashboard, ensure you have the following dependencies installed:

```bash
pip install flask flask-socketio plotly pandas numpy psutil
```

## Running the Dashboard

### Basic Usage

To run the modern dashboard:

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

The dashboard will start and be available at: `http://127.0.0.1:8000/` (or another available port if 8000 is taken)

### Login Credentials

The dashboard requires authentication. Use one of the following credentials:

- **Admin:** Username: `admin`, Password: `admin123`
- **Operator:** Username: `operator`, Password: `operator123`
- **Viewer:** Username: `viewer`, Password: `viewer123`

### Advanced Options

The dashboard runner supports several command-line options:

```bash
python run_modern_dashboard.py --host 0.0.0.0 --port 8080 --debug
```

Options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 0 for auto-detection)
- `--debug`: Enable debug mode for development

## Dashboard Features

### Overview

The Overview tab provides:

- System power status (running/stopped)
- Trading status (enabled/disabled)
- Real-time status of all system components
- Active orders and recent trades
- Execution metrics
- System alerts and warnings
- Performance statistics

### Market Regime

The Market Regime tab provides:

- Current market regime detection
- Regime confidence indicators
- Regime transition probabilities
- Strategy performance across different regimes
- Historical regime tracking

### Sentiment Analysis

The Sentiment tab shows:

- Overall market sentiment indicators
- Sentiment by source (social media, news, etc.)
- Sentiment trends over time
- Sentiment impact on price
- Sentiment extremes and contrarian signals
- Sentiment-price correlation analysis

### Risk Management

The Risk tab displays:

- Risk capacity utilization
- Risk tolerance level
- Key risk metrics (VaR, drawdown, etc.)
- Position sizing and exposure
- Strategy and asset risk visualization

### Performance Analytics

The Performance tab shows:

- Return metrics
- Win rate and profit factor
- Strategy comparison
- Drawdown analysis

### Logs & Monitoring

The Logs tab includes:

- Filterable log viewer
- Log level selection
- Component filtering
- Search functionality
- Resource usage monitoring (CPU, memory, disk)
- Error tracking and anomaly detection

## Real-time Updates

The modern dashboard uses WebSockets for real-time updates:

- Dashboard data refreshes automatically at configurable intervals
- System status changes are pushed immediately
- Trading status updates are pushed in real-time
- Component health changes are pushed as they occur

## Troubleshooting

If you encounter issues running the dashboard:

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

## Advanced Usage

### Custom Data Sources

The dashboard can switch between mock and real data:

1. Use the data source toggle in the dashboard settings
2. Mock data is automatically used when real data is unavailable
3. Data service provides seamless switching between sources

### Development Mode

For development and customization:

1. Enable debug mode: `python run_modern_dashboard.py --debug`
2. Edit templates in the `templates/` directory
3. Modify styles in `static/css/`
4. Add custom JavaScript in `static/js/`

## Technical Details

For more detailed technical information, see the implementation documentation:

- [Dashboard Architecture](docs/DASHBOARD_ARCHITECTURE.md)
- [Dashboard Implementation Guide](docs/DASHBOARD_IMPLEMENTATION.md)
- [Dashboard Redesign Plan](docs/DASHBOARD_REDESIGN_PLAN.md)
- [Modern Dashboard Guide](docs/MODERN_DASHBOARD_GUIDE.md)
- [Dashboard Testing Guide](docs/dashboard_testing_guide.md)

## Next Steps

Future dashboard enhancements include:

1. React-based frontend with Tailwind CSS
2. Enhanced interactive components
3. Customizable layouts and saved views
4. Advanced analytics capabilities
5. Mobile responsiveness
6. Expanded risk management features
