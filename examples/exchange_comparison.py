"""
Exchange Comparison Demo

This script demonstrates side-by-side comparison between different exchange connectors
(Binance and Coinbase) to show how market data differs across exchanges.
"""

import asyncio
import os
import sys
import logging
import time
from decimal import Decimal
from typing import Dict, Any, List
from tabulate import tabulate

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.exchange.binance import BinanceExchangeConnector
from src.execution.exchange.coinbase import CoinbaseExchangeConnector
from src.common.logging import get_logger

logger = get_logger('exchange_comparison')

# Configure which symbols to compare
SYMBOLS = [
    "BTC/USD",  # Universal format, will be converted to exchange-specific format
    "ETH/USD",
    "SOL/USD"
]

async def format_ticker_data(exchange_name: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
    """Format ticker data for consistent display."""
    if not ticker:
        return {
            "exchange": exchange_name,
            "price": "N/A",
            "volume": "N/A",
            "high": "N/A",
            "low": "N/A"
        }
    
    # Convert values to strings with proper formatting
    price = ticker.get("price", "0")
    price = f"{float(price):.2f}" if price != "0" else "N/A"
    
    volume = ticker.get("volume", "0") 
    volume = f"{float(volume):.2f}" if volume != "0" else "N/A"
    
    high = ticker.get("high", "0")
    high = f"{float(high):.2f}" if high != "0" else "N/A"
    
    low = ticker.get("low", "0")
    low = f"{float(low):.2f}" if low != "0" else "N/A"
    
    return {
        "exchange": exchange_name,
        "price": price,
        "volume": volume,
        "high": high,
        "low": low
    }

async def calculate_price_difference(binance_price: str, coinbase_price: str) -> str:
    """Calculate percentage difference between prices."""
    if binance_price == "N/A" or coinbase_price == "N/A":
        return "N/A"
    
    try:
        binance_val = float(binance_price)
        coinbase_val = float(coinbase_price)
        
        if binance_val == 0 or coinbase_val == 0:
            return "N/A"
        
        diff = (coinbase_val - binance_val) / binance_val * 100
        return f"{diff:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

async def compare_exchanges():
    """Run the exchange comparison demo."""
    logger.info("Starting exchange comparison demo")
    
    # Initialize connectors (no API keys needed for public endpoints)
    binance = BinanceExchangeConnector(testnet=False)
    coinbase = CoinbaseExchangeConnector(sandbox=False)
    
    try:
        # Initialize the connectors
        logger.info("Initializing exchange connectors...")
        binance_init = await binance.initialize()
        coinbase_init = await coinbase.initialize()
        
        if not binance_init:
            logger.error("Failed to initialize Binance connector")
            return
        
        if not coinbase_init:
            logger.error("Failed to initialize Coinbase connector")
            return
        
        logger.info("Exchange connectors initialized successfully")
        
        # Compare ticker data for each symbol
        for symbol in SYMBOLS:
            logger.info(f"Comparing {symbol} across exchanges...")
            
            # Get ticker data from Binance
            binance_ticker = await binance.get_ticker(symbol)
            binance_formatted = await format_ticker_data("Binance", binance_ticker)
            
            # Get ticker data from Coinbase
            coinbase_ticker = await coinbase.get_ticker(symbol)
            coinbase_formatted = await format_ticker_data("Coinbase", coinbase_ticker)
            
            # Calculate price difference
            price_diff = await calculate_price_difference(
                binance_formatted["price"], 
                coinbase_formatted["price"]
            )
            
            # Display comparison
            print(f"\n=== {symbol} Market Data Comparison ===")
            
            table_data = [
                ["Exchange", "Price", "24h Volume", "24h High", "24h Low"],
                [
                    binance_formatted["exchange"],
                    binance_formatted["price"],
                    binance_formatted["volume"],
                    binance_formatted["high"],
                    binance_formatted["low"]
                ],
                [
                    coinbase_formatted["exchange"],
                    coinbase_formatted["price"],
                    coinbase_formatted["volume"],
                    coinbase_formatted["high"],
                    coinbase_formatted["low"]
                ]
            ]
            
            print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
            print(f"Price Difference (Coinbase vs Binance): {price_diff}")
            
            # Compare order book depth
            print(f"\n=== {symbol} Orderbook Comparison (Top 3 Levels) ===")
            
            # Get order books with small depth to keep the output clean
            binance_orderbook = await binance.get_orderbook(symbol, limit=3)
            coinbase_orderbook = await coinbase.get_orderbook(symbol, limit=3)
            
            # Binance orderbook display
            print("BINANCE:")
            if binance_orderbook and "bids" in binance_orderbook and "asks" in binance_orderbook:
                bid_data = []
                for bid in binance_orderbook["bids"][:3]:
                    bid_data.append([f"{float(bid[0]):.2f}", f"{float(bid[1]):.5f}"])
                
                ask_data = []
                for ask in binance_orderbook["asks"][:3]:
                    ask_data.append([f"{float(ask[0]):.2f}", f"{float(ask[1]):.5f}"])
                
                print("  Bids (Price, Quantity):")
                print(tabulate(bid_data, tablefmt="simple"))
                print("  Asks (Price, Quantity):")
                print(tabulate(ask_data, tablefmt="simple"))
            else:
                print("  No orderbook data available")
            
            # Coinbase orderbook display
            print("\nCOINBASE:")
            if coinbase_orderbook and "bids" in coinbase_orderbook and "asks" in coinbase_orderbook:
                bid_data = []
                for bid in coinbase_orderbook["bids"][:3]:
                    bid_data.append([f"{float(bid[0]):.2f}", f"{float(bid[1]):.5f}"])
                
                ask_data = []
                for ask in coinbase_orderbook["asks"][:3]:
                    ask_data.append([f"{float(ask[0]):.2f}", f"{float(ask[1]):.5f}"])
                
                print("  Bids (Price, Quantity):")
                print(tabulate(bid_data, tablefmt="simple"))
                print("  Asks (Price, Quantity):")
                print(tabulate(ask_data, tablefmt="simple"))
            else:
                print("  No orderbook data available")
            
            # Add a separator between symbols
            print("\n" + "-" * 80)
            
            # Small delay to avoid rate limits
            await asyncio.sleep(1)
        
        # Summary
        print("\n=== Exchange Comparison Summary ===")
        print("- Different exchanges often have slightly different prices for the same asset")
        print("- This price difference creates arbitrage opportunities")
        print("- Order book depth varies between exchanges, affecting liquidity")
        print("- Smart Order Routing can optimize execution across multiple exchanges")
        print("- The exchange with better liquidity may offer lower slippage for large orders")
    
    except Exception as e:
        logger.error(f"Error in exchange comparison demo: {e}")
    
    finally:
        # Shutdown the connectors
        logger.info("Shutting down exchange connectors...")
        await binance.shutdown()
        await coinbase.shutdown()
        logger.info("Exchange comparison demo completed")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demo
    asyncio.run(compare_exchanges()) 