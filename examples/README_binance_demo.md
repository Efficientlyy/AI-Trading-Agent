# Binance API Demo

This directory contains demos for interacting with the Binance cryptocurrency exchange API.

## Simple Binance Demo

The `simple_binance_demo.py` script demonstrates how to interact with the Binance API directly using `aiohttp`, without relying on the project's exchange connector infrastructure. This is useful for testing and understanding the Binance API.

### Features

- Fetches public market data (no authentication required):
  - Server connectivity test
  - Server time
  - Exchange information
  - Ticker data (price, volume, etc.)
  - Orderbook data

- Makes authenticated requests (requires API keys):
  - Account information
  - Balance information
  - Open orders
  - Placing test orders
  - Cancelling orders

### Usage

1. Install the required dependencies:
   ```
   pip install aiohttp
   ```

2. Run the demo:
   ```
   python examples/simple_binance_demo.py
   ```

3. For authenticated endpoints, set your API keys as environment variables:
   ```
   # Windows PowerShell
   $env:BINANCE_API_KEY="your_api_key"
   $env:BINANCE_API_SECRET="your_api_secret"
   
   # Windows Command Prompt
   set BINANCE_API_KEY=your_api_key
   set BINANCE_API_SECRET=your_api_secret
   
   # Linux/macOS
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_API_SECRET="your_api_secret"
   ```

### Testnet vs Production

By default, the demo uses the Binance Testnet, which is a sandbox environment for testing without real funds. To use the production environment, change the `USE_TESTNET` variable to `False` in the script.

To get Testnet API keys:
1. Visit [Binance Testnet](https://testnet.binance.vision/)
2. Log in with a GitHub account
3. Generate API keys for testing

## Project Exchange Connector

The project includes a more comprehensive `BinanceExchangeConnector` class in `src/execution/exchange/binance.py`, which provides a standardized interface for interacting with Binance as part of the trading system.

However, due to dependencies on the project's configuration system, it may be more complex to use directly in standalone scripts. The simple demo provides a more straightforward approach for testing and learning purposes. 