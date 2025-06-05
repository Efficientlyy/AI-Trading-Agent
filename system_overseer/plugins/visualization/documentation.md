# Visualization Plugin Documentation

## Overview

The Visualization Plugin is a modular component for the System Overseer that provides real-time chart visualization capabilities for cryptocurrency trading pairs. This document describes the plugin's features, usage, and integration points.

## Features

- **Multiple Chart Types**:
  - Candlestick charts for OHLC visualization
  - Line charts for price tracking
  - Volume charts for trading activity analysis

- **Technical Indicators**:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)

- **Supported Trading Pairs**:
  - BTC/USDC (primary focus)
  - ETH/USDC (primary focus)
  - SOL/USDC (primary focus)
  - BTC/USDT, ETH/USDT, SOL/USDT (secondary support)

- **Supported Time Intervals**:
  - 1m (1 minute)
  - 5m (5 minutes)
  - 15m (15 minutes)
  - 30m (30 minutes)
  - 4h (4 hours)
  - 1d (1 day)
  - 1w (1 week)
  - 1M (1 month)

> **Note**: The '1h' (1 hour) interval is currently not supported by the MEXC API in the format expected by the plugin. Use alternative intervals like 30m or 4h instead.

## Usage

### Telegram Commands

The Visualization Plugin integrates with the Telegram bot to provide chart generation capabilities through commands:

```
/chart <symbol> <type> <interval> [indicators]
```

**Parameters**:
- `symbol`: Trading pair symbol (e.g., BTCUSDT, ETHUSDC)
- `type`: Chart type (candlestick, line, volume)
- `interval`: Time interval (1m, 5m, 15m, 30m, 4h, 1d, 1w, 1M)
- `indicators`: Optional list of indicators (sma, ema)

**Examples**:
```
/chart BTCUSDT candlestick 15m
/chart ETHUSDC line 5m sma ema
/chart SOLUSDT volume 1d
```

### Programmatic Usage

The Visualization Plugin can also be used programmatically through the System Overseer API:

```python
# Get visualization plugin
visualization = system_core.get_plugin("visualization")

# Generate chart
chart_data = visualization.get_chart(
    symbol="BTCUSDT",
    chart_type="candlestick",
    interval="15m",
    indicators=["sma", "ema"]
)

# Save chart to file
with open("btc_chart.png", "wb") as f:
    f.write(chart_data)
```

## Configuration

The Visualization Plugin can be configured through the System Overseer configuration registry:

```json
{
  "visualization": {
    "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
    "default_timeframe": "15m",
    "chart_types": ["candlestick", "line", "volume"],
    "indicators": ["sma", "ema"],
    "auto_refresh": true,
    "refresh_interval": 60,
    "data_provider": "mexc"
  }
}
```

## Data Providers

The Visualization Plugin uses data providers to retrieve market data for chart generation. Currently, the following data providers are supported:

### MEXC Data Provider

The MEXC Data Provider retrieves market data from the MEXC exchange API. It requires the following environment variables to be set:

- `MEXC_API_KEY`: MEXC API key
- `MEXC_API_SECRET`: MEXC API secret

## Error Handling

The Visualization Plugin includes robust error handling to manage various failure scenarios:

- **API Connection Errors**: If the connection to the exchange API fails, an error is logged and the chart generation fails gracefully.
- **Invalid Symbol**: If an invalid trading pair symbol is provided, an error is logged and the chart generation fails gracefully.
- **Invalid Interval**: If an unsupported time interval is provided, an error is logged and the chart generation fails gracefully.
- **Empty Data**: If no data is returned from the API, an error is logged and the chart generation fails gracefully.

## Limitations

- The '1h' (1 hour) interval is currently not supported by the MEXC API in the format expected by the plugin.
- Chart generation may fail if the trading pair has insufficient historical data.
- The plugin currently only supports the MEXC exchange as a data provider.

## Future Enhancements

- Support for additional exchanges as data providers
- More technical indicators (RSI, MACD, Bollinger Bands)
- Interactive charts with zooming and panning
- Historical data caching for improved performance
- Support for custom chart styles and themes

## Troubleshooting

If chart generation fails, check the following:

1. Verify that the trading pair symbol is valid and supported by the exchange
2. Ensure that the time interval is supported (avoid using '1h')
3. Check that the API credentials are correctly configured
4. Verify that the exchange API is accessible and responding

For detailed error information, check the system logs.
