# AI Trading Agent Dashboard Instructions

This document provides instructions for running and using the integrated AI Trading Agent dashboard.

## Overview

The new integrated dashboard provides a unified interface with tabs for:

1. **System Monitoring Dashboard** - Overall system status, components, orders, and trades
2. **Sentiment Analysis Dashboard** - Market sentiment from various sources
3. **Risk Management Dashboard** - Risk utilization and portfolio metrics
4. **Logs & Monitoring Dashboard** - System logs and health monitoring
5. **Market Regime Analysis Dashboard** - Market regime detection and strategy performance

## Prerequisites

Before running the dashboard, ensure you have the following dependencies installed:

```bash
pip install flask plotly pandas numpy psutil structlog dash dash-bootstrap-components
```

## Running the Dashboard

### Basic Usage

To run the integrated dashboard:

```bash
python run_dashboard.py
```

The dashboard will start and be available at: `http://127.0.0.1:8050/`

### Advanced Options

The dashboard runner supports several command-line options:

```bash
python run_dashboard.py --host 0.0.0.0 --port 8080 --debug --log-level DEBUG
```

Options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8050)
- `--debug`: Enable debug mode for development
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Dashboard Features

### System Monitoring Dashboard

The System tab provides:

- Real-time status of all system components
- Active orders and recent trades
- Execution metrics
- System alerts and warnings
- Performance statistics

### Sentiment Analysis Dashboard

The Sentiment tab shows:

- Overall market sentiment indicators
- Fear & Greed Index with historical data
- News sentiment analysis with recent headlines
- Social media sentiment across platforms (Twitter/X, Reddit)
- On-chain metrics and sentiment indicators
- Sentiment extremes and contrarian signals
- Sentiment-price correlation analysis
- Historical sentiment data

### Risk Management Dashboard

The Risk tab displays:

- Risk utilization by strategy and asset
- Allocation pie charts
- Risk metrics (VaR, CVaR, Max Drawdown, Sharpe)
- Risk alerts
- Strategy and asset risk visualization

### Logs & Monitoring Dashboard

The Logs tab includes:

- System log browser with filtering
- Resource usage monitoring (CPU, memory, disk)
- Error tracking and anomaly detection
- Log pattern analysis
- Performance metrics

### Market Regime Analysis Dashboard

The Market Regime tab provides:

- Current market regime detection
- Regime transition probabilities
- Strategy performance across different regimes
- Historical regime tracking
- Regime confidence indicators

## Generating Test Data

For testing purposes, the dashboard can generate realistic mock data:

```bash
python tests/generate_test_logs.py  # Generate test logs
```

## Data Refresh Rate

- Dashboard data refreshes automatically every 30 seconds
- Click the refresh button for immediate updates
- Some visualizations support real-time streaming

## Troubleshooting

If you encounter issues running the dashboard:

1. **Module Not Found Errors**:
   - Ensure all dependencies are installed
   - Check Python environment activation
   
2. **No Data Displayed**:
   - Verify data sources are available
   - Check API endpoints are responding
   - For testing, use mock data generation

3. **Dashboard Not Loading**:
   - Check if port is already in use
   - Verify required libraries are installed
   - Check console for error messages

## Advanced Usage

### Custom Data Sources

The dashboard can be configured to use custom data sources:

1. Edit the config file: `config/dashboard.yaml`
2. Specify custom data sources and API endpoints
3. Restart the dashboard to apply changes

### Development Mode

For development and customization:

1. Enable debug mode: `python run_dashboard.py --debug`
2. Edit templates in the `templates/` directory
3. Modify styles in `static/css/`
4. Add custom JavaScript in `static/js/`

## Technical Details

For more detailed technical information, see the implementation documentation:

- [Dashboard Implementation Guide](docs/DASHBOARD_IMPLEMENTATION.md)

## Next Steps

Future dashboard enhancements include:

1. User-customizable dashboards with saved views
2. Enhanced real-time data streaming
3. Advanced anomaly detection with machine learning
4. Integration with alerting systems
5. Scheduled reports and automated analysis
6. Mobile application companion
7. Collaborative sharing and annotation of insights