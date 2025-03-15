#!/usr/bin/env python3
"""
Binance Exchange Connector Demo

This script demonstrates how to use the BinanceExchangeConnector to interact with 
the Binance cryptocurrency exchange. It shows:

1. Setting up the connector with API credentials
2. Fetching market data like ticker information and order books
3. Placing and managing orders
4. Retrieving account information

Note: You need valid Binance API credentials to run the authenticated parts of this demo.
To test without real trading, you can use the Binance Testnet.
"""

import asyncio
import os
import json
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(parent_dir))

# Import directly from the binance module to avoid config system dependencies
from src.execution.exchange.binance import BinanceExchangeConnector
from src.models.order import Order, OrderType, OrderSide, TimeInForce, OrderStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("binance_demo")

# Create output directory
output_dir = Path("examples/output/binance_demo")
output_dir.mkdir(parents=True, exist_ok=True)


async def demo_public_api(connector: BinanceExchangeConnector):
    """
    Demonstrate accessing public market data (doesn't require authentication).
    
    Args:
        connector: Initialized BinanceExchangeConnector instance
    """
    logger.info("DEMO: Accessing Public Market Data")
    
    # Fetch exchange information
    exchange_info = await connector.get_exchange_info()
    symbol_count = len(exchange_info.get("symbols", []))
    logger.info(f"Exchange info retrieved - {symbol_count} trading pairs available")
    
    # Save exchange info for reference
    with open(output_dir / "exchange_info.json", "w") as f:
        json.dump(exchange_info, f, indent=2, default=str)
    
    # Fetch ticker information for BTC/USDT
    ticker = await connector.get_ticker("BTC/USDT")
    logger.info(f"BTC/USDT Ticker: Last Price: ${ticker['last']}, 24h Change: {ticker['change_percent_24h']}%")
    
    # Get orderbook data
    orderbook = await connector.get_orderbook("BTC/USDT", limit=10)
    best_bid = orderbook["bids"][0] if orderbook["bids"] else None
    best_ask = orderbook["asks"][0] if orderbook["asks"] else None
    
    if best_bid and best_ask:
        logger.info(f"BTC/USDT Orderbook: Best Bid: ${best_bid[0]} ({best_bid[1]} BTC), "
                  f"Best Ask: ${best_ask[0]} ({best_ask[1]} BTC)")
    
    # Get multiple tickers
    symbols = ["ETH/USDT", "SOL/USDT", "LINK/USDT"]
    results = await asyncio.gather(*(connector.get_ticker(symbol) for symbol in symbols))
    
    for ticker in results:
        symbol = ticker["symbol"]
        logger.info(f"{symbol} - Price: ${ticker['last']}, 24h Vol: {ticker['volume']}")


async def demo_authenticated_api(connector: BinanceExchangeConnector, use_testnet: bool = True):
    """
    Demonstrate authenticated API endpoints (requires API keys).
    
    Args:
        connector: Initialized BinanceExchangeConnector instance
        use_testnet: Whether to use test orders or real orders
    """
    if not connector.api_key or not connector.api_secret:
        logger.warning("Skipping authenticated API demo - no API credentials provided")
        return
    
    logger.info("DEMO: Accessing Authenticated Endpoints")
    
    # Check account balances
    balances = await connector.get_account_balance()
    logger.info(f"Account has {len(balances)} assets with non-zero balances")
    
    # Log some top balances
    for asset, amount in list(balances.items())[:5]:
        logger.info(f"Balance: {amount} {asset}")
    
    # Get open orders
    open_orders = await connector.get_open_orders()
    logger.info(f"Found {len(open_orders)} open orders")
    
    # Place a test order if using testnet
    if use_testnet:
        # Create a small limit buy order for BTC/USDT
        # In testnet, this doesn't use real funds
        ticker = await connector.get_ticker("BTC/USDT")
        current_price = float(ticker["last"])
        
        # Set limit price 5% below current price
        limit_price = current_price * 0.95
        
        # Create the order (very small quantity for testing)
        test_order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.001,  # Very small amount
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            client_order_id=f"test_order_{int(datetime.now().timestamp())}"
        )
        
        logger.info(f"Placing test limit buy order for 0.001 BTC at ${limit_price:.2f}")
        success, order_id, error = await connector.create_order(test_order)
        
        if success and order_id:
            logger.info(f"Order placed successfully! Order ID: {order_id}")
            
            # Get order details
            order_details = await connector.get_order(order_id, "BTC/USDT")
            if order_details:
                logger.info(f"Order status: {order_details['status']}")
                
                # Cancel the order
                logger.info(f"Cancelling order {order_id}")
                cancel_success, cancel_error = await connector.cancel_order(order_id, "BTC/USDT")
                
                if cancel_success:
                    logger.info("Order cancelled successfully")
                else:
                    logger.error(f"Failed to cancel order: {cancel_error}")
            else:
                logger.error("Failed to retrieve order details")
        else:
            logger.error(f"Failed to place order: {error}")


async def run_demo():
    """Main function to run the demo."""
    logger.info("Starting Binance Exchange Connector Demo")
    
    # First, try to load API keys from environment variables
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    
    # Decide whether to use testnet based on credentials
    use_testnet = True
    
    # Create the connector directly without using the config system
    connector = BinanceExchangeConnector(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        testnet=use_testnet
    )
    
    try:
        # Initialize the connector
        initialized = await connector.initialize()
        
        if not initialized:
            logger.error("Failed to initialize Binance connector")
            return
        
        logger.info(f"Binance connector initialized successfully (Testnet: {use_testnet})")
        
        # Demonstrate public API endpoints
        await demo_public_api(connector)
        
        # Demonstrate authenticated API endpoints
        if api_key and api_secret:
            await demo_authenticated_api(connector, use_testnet=use_testnet)
        else:
            logger.warning("Skipping authenticated API demo - no API credentials provided")
            logger.info("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables to run this part")
        
    except Exception as e:
        logger.exception(f"Error in demo: {str(e)}")
    finally:
        # Always shutdown the connector properly
        await connector.shutdown()
        logger.info("Binance connector shut down")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 