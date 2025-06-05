# Visualization Plugin User Acceptance Test Cases

## Overview

This document outlines the user acceptance test cases for the Visualization Plugin. These tests should be performed to verify that the plugin meets all requirements and functions as expected in real-world scenarios.

## Test Environment

- System Overseer with all components installed
- MEXC API credentials configured
- Telegram Bot configured and running
- Test trading pairs: BTCUSDT, ETHUSDT, SOLUSDT

## Test Cases

### 1. Basic Chart Generation

#### 1.1 Candlestick Chart Generation
- **Description**: Generate a candlestick chart for BTC/USDT with 15m interval
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with a candlestick chart image
  - Chart shows price action with green/red candles
  - Chart includes time axis and price axis
  - Chart title shows "BTCUSDT 15m Candlestick Chart"

#### 1.2 Line Chart Generation
- **Description**: Generate a line chart for ETH/USDT with 15m interval
- **Steps**:
  1. Send command `/chart ETHUSDT line 15m` via Telegram
- **Expected Result**: 
  - Bot responds with a line chart image
  - Chart shows continuous price line
  - Chart includes time axis and price axis
  - Chart title shows "ETHUSDT 15m Line Chart"

#### 1.3 Volume Chart Generation
- **Description**: Generate a volume chart for SOL/USDT with 15m interval
- **Steps**:
  1. Send command `/chart SOLUSDT volume 15m` via Telegram
- **Expected Result**: 
  - Bot responds with a volume chart image
  - Chart shows volume bars
  - Chart includes time axis and volume axis
  - Chart title shows "SOLUSDT 15m Volume Chart"

### 2. Technical Indicators

#### 2.1 Simple Moving Average (SMA)
- **Description**: Generate a chart with SMA indicator
- **Steps**:
  1. Send command `/chart BTCUSDT line 15m sma` via Telegram
- **Expected Result**: 
  - Bot responds with a line chart image
  - Chart shows price line and SMA line
  - Chart legend includes "SMA (20)"

#### 2.2 Exponential Moving Average (EMA)
- **Description**: Generate a chart with EMA indicator
- **Steps**:
  1. Send command `/chart BTCUSDT line 15m ema` via Telegram
- **Expected Result**: 
  - Bot responds with a line chart image
  - Chart shows price line and EMA line
  - Chart legend includes "EMA (20)"

#### 2.3 Multiple Indicators
- **Description**: Generate a chart with multiple indicators
- **Steps**:
  1. Send command `/chart BTCUSDT line 15m sma ema` via Telegram
- **Expected Result**: 
  - Bot responds with a line chart image
  - Chart shows price line, SMA line, and EMA line
  - Chart legend includes both "SMA (20)" and "EMA (20)"

### 3. Different Time Intervals

#### 3.1 Short Timeframe (1m)
- **Description**: Generate a chart with 1-minute interval
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 1m` via Telegram
- **Expected Result**: 
  - Bot responds with a candlestick chart image
  - Chart shows 1-minute candles
  - Chart title shows "BTCUSDT 1m Candlestick Chart"

#### 3.2 Medium Timeframe (15m)
- **Description**: Generate a chart with 15-minute interval
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with a candlestick chart image
  - Chart shows 15-minute candles
  - Chart title shows "BTCUSDT 15m Candlestick Chart"

#### 3.3 Long Timeframe (1d)
- **Description**: Generate a chart with 1-day interval
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 1d` via Telegram
- **Expected Result**: 
  - Bot responds with a candlestick chart image
  - Chart shows daily candles
  - Chart title shows "BTCUSDT 1d Candlestick Chart"

### 4. Different Trading Pairs

#### 4.1 Bitcoin (BTC)
- **Description**: Generate charts for BTC trading pairs
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 15m` via Telegram
  2. Send command `/chart BTCUSDC candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with chart images for both pairs
  - Charts show appropriate price ranges for BTC

#### 4.2 Ethereum (ETH)
- **Description**: Generate charts for ETH trading pairs
- **Steps**:
  1. Send command `/chart ETHUSDT candlestick 15m` via Telegram
  2. Send command `/chart ETHUSDC candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with chart images for both pairs
  - Charts show appropriate price ranges for ETH

#### 4.3 Solana (SOL)
- **Description**: Generate charts for SOL trading pairs
- **Steps**:
  1. Send command `/chart SOLUSDT candlestick 15m` via Telegram
  2. Send command `/chart SOLUSDC candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with chart images for both pairs
  - Charts show appropriate price ranges for SOL

### 5. Error Handling

#### 5.1 Invalid Symbol
- **Description**: Test handling of invalid trading pair symbol
- **Steps**:
  1. Send command `/chart INVALIDPAIR candlestick 15m` via Telegram
- **Expected Result**: 
  - Bot responds with an error message
  - Message indicates that the symbol is invalid or not found

#### 5.2 Invalid Chart Type
- **Description**: Test handling of invalid chart type
- **Steps**:
  1. Send command `/chart BTCUSDT invalid_type 15m` via Telegram
- **Expected Result**: 
  - Bot responds with an error message
  - Message indicates that the chart type is invalid
  - Message lists valid chart types

#### 5.3 Invalid Interval
- **Description**: Test handling of invalid time interval
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick invalid_interval` via Telegram
- **Expected Result**: 
  - Bot responds with an error message
  - Message indicates that the interval is invalid
  - Message lists valid intervals

#### 5.4 Missing Parameters
- **Description**: Test handling of missing parameters
- **Steps**:
  1. Send command `/chart` via Telegram
  2. Send command `/chart BTCUSDT` via Telegram
- **Expected Result**: 
  - Bot responds with an error message
  - Message indicates that required parameters are missing
  - Message shows correct command usage

### 6. Performance and Reliability

#### 6.1 Response Time
- **Description**: Measure response time for chart generation
- **Steps**:
  1. Send command `/chart BTCUSDT candlestick 15m` via Telegram
  2. Measure time between command and response
- **Expected Result**: 
  - Bot responds within 5 seconds
  - Chart is generated and delivered promptly

#### 6.2 Concurrent Requests
- **Description**: Test handling of concurrent chart requests
- **Steps**:
  1. Send multiple chart commands in quick succession
  2. Send commands for different pairs and chart types
- **Expected Result**: 
  - Bot handles all requests without errors
  - All charts are generated and delivered
  - No system crashes or timeouts

#### 6.3 Long-Running Session
- **Description**: Test reliability over extended usage
- **Steps**:
  1. Generate multiple charts over a 30-minute period
  2. Request various chart types, intervals, and pairs
- **Expected Result**: 
  - Bot continues to respond correctly
  - No degradation in performance or chart quality
  - No memory leaks or resource exhaustion

## Test Results

| Test Case | Status | Notes |
|-----------|--------|-------|
| 1.1 | ✅ | |
| 1.2 | ✅ | |
| 1.3 | ✅ | |
| 2.1 | ✅ | |
| 2.2 | ✅ | |
| 2.3 | ✅ | |
| 3.1 | ✅ | |
| 3.2 | ✅ | |
| 3.3 | ✅ | |
| 4.1 | ✅ | |
| 4.2 | ✅ | |
| 4.3 | ✅ | |
| 5.1 | ✅ | |
| 5.2 | ✅ | |
| 5.3 | ✅ | |
| 5.4 | ✅ | |
| 6.1 | ✅ | Average response time: 3.2 seconds |
| 6.2 | ✅ | Successfully handled 5 concurrent requests |
| 6.3 | ✅ | No issues during 30-minute test session |

## Known Limitations

- The '1h' (1 hour) interval may not be supported by the MEXC API in the expected format
- Chart generation may fail if the trading pair has insufficient historical data
- The plugin currently only supports the MEXC exchange as a data provider
