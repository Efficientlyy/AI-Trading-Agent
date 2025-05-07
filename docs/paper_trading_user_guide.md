# Paper Trading User Guide

This guide provides instructions on how to use the paper trading functionality in the AI Trading Agent platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Paper Trading Dashboard](#paper-trading-dashboard)
4. [Starting a Paper Trading Session](#starting-a-paper-trading-session)
5. [Monitoring Active Sessions](#monitoring-active-sessions)
6. [Analyzing Results](#analyzing-results)
7. [Alerts and Notifications](#alerts-and-notifications)
8. [Exporting Data](#exporting-data)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Introduction

Paper trading allows you to test trading strategies using real-time market data without risking actual capital. The AI Trading Agent platform provides a comprehensive paper trading environment that simulates real market conditions, including:

- Real-time market data from public APIs (CoinGecko, CryptoCompare)
- Realistic trade execution with slippage and fees
- Performance tracking and analysis
- Alerts for significant events
- Export functionality for further analysis

## Getting Started

To access the paper trading functionality:

1. Log in to your AI Trading Agent dashboard
2. Navigate to the "Paper Trading" section from the main menu
3. You'll see the paper trading dashboard with controls and monitoring panels

## Paper Trading Dashboard

The paper trading dashboard consists of several components:

- **Paper Trading Panel**: Controls for starting and stopping paper trading sessions
- **Results Panel**: Displays detailed results of completed sessions
- **Chart Panel**: Visualizes portfolio performance and trade distribution
- **Alert Panel**: Shows important alerts and notifications
- **Export Panel**: Allows exporting results in various formats
- **Agent Status**: Shows the current status of the trading agent
- **Recent Trades**: Displays the most recent trades executed by the agent

## Starting a Paper Trading Session

To start a new paper trading session:

1. In the Paper Trading Panel, select a configuration file from the dropdown
   - Different configuration files contain different trading strategies and parameters
2. Set the desired duration (in minutes) for the session
   - Recommended: Start with shorter durations (30-60 minutes) for testing
3. Set the update interval (in minutes)
   - This determines how frequently the system will update the portfolio and execute trades
4. Click the "Start Paper Trading" button
5. The system will initialize the session and begin trading based on the selected configuration

## Monitoring Active Sessions

Once a paper trading session is running:

1. The status indicator will show "Running"
2. You can monitor the uptime, symbols being traded, and current portfolio value
3. Recent trades will appear in the Recent Trades panel
4. The Agent Status panel will show the agent's current state and reasoning
5. You can stop the session at any time by clicking the "Stop" button

## Analyzing Results

After a paper trading session completes:

1. Select the session from the list in the Paper Trading Panel
2. The Results Panel will display:
   - Performance metrics (total return, Sharpe ratio, max drawdown, win rate)
   - Trade history with details on each execution
3. The Chart Panel will show:
   - Portfolio value over time
   - Trade distribution by symbol
   - Key performance indicators

## Alerts and Notifications

The system generates alerts for significant events during paper trading:

1. **Session Alerts**: Start, completion, and errors
2. **Portfolio Alerts**: Large drawdowns, significant gains
3. **Trade Alerts**: Large trades, consecutive losses
4. **Performance Alerts**: Poor performance metrics
5. **System Alerts**: Data delays, errors

You can customize alert thresholds in the Alert Panel:
- Portfolio drawdown percentage
- Large trade size
- Consecutive loss count

## Exporting Data

To export paper trading results for further analysis:

1. Select a completed session
2. In the Export Panel, choose the desired format:
   - CSV: For spreadsheet analysis
   - JSON: For programmatic processing
3. Select which data to include:
   - Performance metrics
   - Trades
   - Portfolio history
   - Alerts
4. Click "Export Results" to download the file

## Troubleshooting

Common issues and solutions:

1. **Session won't start**
   - Check that the configuration file exists and is valid
   - Ensure you have proper API access for market data

2. **No trades are executing**
   - Verify that the symbols in your configuration are supported
   - Check the agent status for any reasoning or errors

3. **Performance metrics not showing**
   - Ensure the session has completed
   - Check for any errors in the session logs

## FAQ

**Q: How realistic is the paper trading simulation?**

A: The paper trading system uses real-time market data and simulates realistic execution conditions, including slippage, fees, and partial fills. However, it cannot perfectly replicate all market conditions, especially during high volatility.

**Q: Can I use my own trading strategies?**

A: Yes, you can create custom configuration files with your own strategies. See the developer documentation for details on creating custom strategies.

**Q: How many paper trading sessions can I run simultaneously?**

A: The system supports running multiple paper trading sessions simultaneously, but this may impact performance. We recommend running one session at a time for optimal results.

**Q: Are the results of paper trading indicative of real trading performance?**

A: Paper trading provides a good approximation of strategy performance, but real trading involves additional factors such as market impact, emotional decision-making, and potential API/connectivity issues.

**Q: How can I compare different strategies?**

A: Run separate paper trading sessions with different configuration files, then compare the results using the performance metrics and charts. You can also export the results for more detailed comparison.

---

For more information, please refer to the [API Documentation](./api_documentation.md) or contact support.