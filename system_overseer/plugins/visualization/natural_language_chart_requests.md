# Natural Language Chart Requests

## Overview

The System Overseer now supports natural language chart requests through Telegram, allowing you to request charts using conversational language instead of formal commands. This feature leverages the LLM integration to understand your intent and generate the appropriate charts.

## How It Works

Instead of using formal commands like `/chart BTCUSDC candlestick 15m`, you can now simply ask for charts in natural language. The system will:

1. Analyze your message to determine if it's a chart request
2. Extract the trading pair, chart type, timeframe, and indicators
3. Generate the appropriate chart
4. Send it back to you in Telegram

## Examples

Here are some examples of natural language chart requests that the system understands:

### Basic Requests

- "Show me a Bitcoin chart"
- "I need to see the ETH price"
- "Give me the Solana chart"
- "Can I see the BTC chart?"

### Specific Chart Types

- "Show me a BTC candlestick chart"
- "I need to see the ETH line chart"
- "Give me the SOL volume chart"
- "Can I see the Bitcoin candles?"

### Specific Timeframes

- "Show me the 1 minute BTC chart"
- "I need to see the 15 min ETH chart"
- "Give me the hourly SOL chart"
- "Can I see the daily Bitcoin chart?"

### With Technical Indicators

- "Show me a BTC chart with SMA"
- "I need to see the ETH chart with moving average"
- "Give me the SOL chart with RSI"
- "Can I see the Bitcoin chart with bollinger bands?"

### Complex Requests

- "Show me a 15 minute BTC candlestick chart with SMA"
- "I need to see the hourly ETH line chart with bollinger bands"
- "Give me the daily SOL volume chart with RSI"

## Default Values

When you don't specify certain parameters, the system uses these defaults:

- **Default Chart Type**: Candlestick
- **Default Timeframe**: 15 minutes
- **Default Quote Currency**: USDC (as per your preference)

For example, if you ask for "Show me a BTC chart", you'll get a 15-minute BTCUSDC candlestick chart.

## Supported Parameters

### Trading Pairs

The system recognizes these cryptocurrencies:
- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)

When you mention just the base currency (e.g., "BTC"), the system defaults to USDC pairs (e.g., "BTCUSDC").

### Chart Types

- **Candlestick**: Shows open, high, low, and close prices
- **Line**: Shows a continuous price line
- **Volume**: Shows trading volume

### Timeframes

- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 1h (1 hour)
- 4h (4 hours)
- 1d (1 day)
- 1w (1 week)

### Technical Indicators

- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

## Tips for Best Results

- Be specific about which cryptocurrency you want (BTC, ETH, or SOL)
- Mention the chart type if you have a preference (candlestick, line, volume)
- Specify the timeframe if you need something other than 15 minutes
- You can still use the formal command syntax (`/chart`) if you prefer

## Limitations

- The system may not understand very ambiguous requests like "show me the market"
- Complex requests with multiple conflicting parameters may be interpreted differently than intended
- The system currently only supports BTC, ETH, and SOL with USDC and USDT pairs
