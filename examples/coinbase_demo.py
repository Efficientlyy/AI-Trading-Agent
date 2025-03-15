"""
Coinbase Exchange Connector Demo Script

This script demonstrates the usage of the CoinbaseExchangeConnector by
connecting to the Coinbase exchange and retrieving public market data.

Note: This demo only uses public API endpoints and does not require API keys.
"""

import asyncio
import os
import sys
import datetime
import logging
from decimal import Decimal
from pprint import pprint

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.exchange.coinbase import CoinbaseExchangeConnector
from src.common.logging import get_logger

logger = get_logger('coinbase_demo')

async def run_demo():
    """Run the Coinbase connector demo."""
    logger.info("Starting Coinbase connector demo")
    
    # Initialize the connector (no API keys needed for public endpoints)
    connector = CoinbaseExchangeConnector(sandbox=False)
    
    try:
        # Initialize the connector
        logger.info("Initializing Coinbase connector...")
        success = await connector.initialize()
        
        if not success:
            logger.error("Failed to initialize Coinbase connector")
            return
        
        logger.info("Coinbase connector initialized successfully")
        
        # Get exchange info
        logger.info("Fetching exchange information...")
        exchange_info = await connector.get_exchange_info()
        logger.info(f"Retrieved information for {len(exchange_info['symbols'])} symbols")
        
        # Display first 5 symbols
        print("\n==== First 5 Available Symbols ====")
        for symbol in exchange_info['symbols'][:5]:
            print(f"Symbol: {symbol['product_id']}")
        
        # Get ticker data for BTC-USD
        symbol = "BTC-USD"
        logger.info(f"Fetching ticker data for {symbol}...")
        ticker = await connector.get_ticker(symbol)
        
        print(f"\n==== {symbol} Ticker Data ====")
        pprint(ticker)
        
        # Get orderbook for BTC-USD
        logger.info(f"Fetching orderbook for {symbol}...")
        orderbook = await connector.get_orderbook(symbol, limit=5)
        
        print(f"\n==== {symbol} Orderbook (Top 5) ====")
        print("Asks:")
        for ask in orderbook.get('asks', [])[:5]:
            print(f"  Price: {ask[0]}, Size: {ask[1]}")
            
        print("Bids:")
        for bid in orderbook.get('bids', [])[:5]:
            print(f"  Price: {bid[0]}, Size: {bid[1]}")
        
        # Get recent trades for BTC-USD
        logger.info(f"Fetching recent trades for {symbol}...")
        trades = await connector.get_trade_history(symbol, limit=5)
        
        print(f"\n==== {symbol} Recent Trades (Last 5) ====")
        for trade in trades[:5]:
            side = "BUY" if trade['side'].name == "BUY" else "SELL"
            print(f"  ID: {trade['id']}, Price: {trade['price']}, Size: {trade['quantity']}, Side: {side}, Time: {trade['time']}")
    
    except Exception as e:
        logger.error(f"Error in Coinbase demo: {e}")
    
    finally:
        # Shutdown the connector
        logger.info("Shutting down Coinbase connector...")
        await connector.shutdown()
        logger.info("Coinbase connector demo completed")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demo
    asyncio.run(run_demo()) 