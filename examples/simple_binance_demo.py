#!/usr/bin/env python3
"""
Simple Binance API Demo

This script demonstrates how to interact with the Binance API directly using aiohttp,
without relying on the project's exchange connector infrastructure.

It shows:
1. Fetching public market data
2. Making authenticated requests (if API keys are provided)
"""

import asyncio
import hmac
import hashlib
import json
import time
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlencode
from pathlib import Path

try:
    import aiohttp
except ImportError:
    raise ImportError("aiohttp is required for this demo. Install it with 'pip install aiohttp'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_binance_demo")

# Create output directory
output_dir = Path("examples/output/binance_demo")
output_dir.mkdir(parents=True, exist_ok=True)

# Binance API configuration
BASE_URL = "https://api.binance.com"
API_V3 = "/api/v3"

# For testing, use the testnet
USE_TESTNET = True
if USE_TESTNET:
    BASE_URL = "https://testnet.binance.vision"

# API credentials (from environment variables)
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")


async def public_request(endpoint: str, params: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict:
    """
    Make a public request to the Binance API.
    
    Args:
        endpoint: API endpoint path
        params: Query parameters
        method: HTTP method (GET, POST, DELETE)
        
    Returns:
        API response as a dictionary
    """
    url = f"{BASE_URL}{API_V3}{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        if method == "GET":
            if params:
                url = f"{url}?{urlencode(params)}"
            async with session.get(url) as response:
                return await response.json()
        elif method == "POST":
            async with session.post(url, data=params) as response:
                return await response.json()
        elif method == "DELETE":
            async with session.delete(url, data=params) as response:
                return await response.json()
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")


async def private_request(endpoint: str, params: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict:
    """
    Make an authenticated request to the Binance API.
    
    Args:
        endpoint: API endpoint path
        params: Query parameters
        method: HTTP method (GET, POST, DELETE)
        
    Returns:
        API response as a dictionary
    """
    if not API_KEY or not API_SECRET:
        raise ValueError("API key and secret are required for authenticated requests")
    
    # Create parameter dictionary if not provided
    if params is None:
        params = {}
    
    # Add timestamp parameter required for authenticated requests
    params['timestamp'] = int(time.time() * 1000)
    
    # Create the query string and signature
    query_string = urlencode(params)
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Add signature to parameters
    params['signature'] = signature
    
    # Create the URL
    url = f"{BASE_URL}{API_V3}{endpoint}"
    
    # Set up headers with API key
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    async with aiohttp.ClientSession() as session:
        if method == "GET":
            async with session.get(url, params=params, headers=headers) as response:
                return await response.json()
        elif method == "POST":
            async with session.post(url, data=params, headers=headers) as response:
                return await response.json()
        elif method == "DELETE":
            async with session.delete(url, data=params, headers=headers) as response:
                return await response.json()
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")


async def demo_public_api():
    """Demonstrate public API endpoints."""
    logger.info("DEMO: Accessing Public Market Data")
    
    # Test server connectivity
    ping_result = await public_request("/ping")
    logger.info(f"Ping result: {ping_result}")
    
    # Get server time
    time_result = await public_request("/time")
    server_time = datetime.fromtimestamp(time_result['serverTime'] / 1000)
    logger.info(f"Server time: {server_time}")
    
    # Get exchange information
    exchange_info = await public_request("/exchangeInfo")
    symbol_count = len(exchange_info.get("symbols", []))
    logger.info(f"Exchange info retrieved - {symbol_count} trading pairs available")
    
    # Save exchange info to file
    exchange_info_file = output_dir / "exchange_info.json"
    with open(exchange_info_file, "w") as f:
        json.dump(exchange_info, f, indent=2, default=str)
    logger.info(f"Exchange info saved to {exchange_info_file}")
    
    # Get ticker for BTC/USDT
    ticker = await public_request("/ticker/24hr", {"symbol": "BTCUSDT"})
    logger.info(f"BTC/USDT 24h stats:")
    logger.info(f"  Last price: ${float(ticker['lastPrice'])}")
    logger.info(f"  24h change: {float(ticker['priceChangePercent'])}%")
    logger.info(f"  24h volume: {float(ticker['volume'])} BTC")
    logger.info(f"  24h trades: {ticker['count']}")
    
    # Save ticker info to file
    ticker_file = output_dir / "btcusdt_ticker.json"
    with open(ticker_file, "w") as f:
        json.dump(ticker, f, indent=2, default=str)
    logger.info(f"Ticker info saved to {ticker_file}")
    
    # Get orderbook for BTC/USDT
    orderbook = await public_request("/depth", {"symbol": "BTCUSDT", "limit": 5})
    logger.info("BTC/USDT Orderbook (top 5 levels):")
    
    logger.info("Bids:")
    for bid in orderbook["bids"][:5]:
        logger.info(f"  ${float(bid[0])}: {float(bid[1])} BTC")
    
    logger.info("Asks:")
    for ask in orderbook["asks"][:5]:
        logger.info(f"  ${float(ask[0])}: {float(ask[1])} BTC")
    
    # Save orderbook to file
    orderbook_file = output_dir / "btcusdt_orderbook.json"
    with open(orderbook_file, "w") as f:
        json.dump(orderbook, f, indent=2, default=str)
    logger.info(f"Orderbook saved to {orderbook_file}")


async def demo_authenticated_api():
    """Demonstrate authenticated API endpoints."""
    if not API_KEY or not API_SECRET:
        logger.warning("Skipping authenticated API demo - no API credentials provided")
        return
    
    logger.info("DEMO: Accessing Authenticated Endpoints")
    
    try:
        # Get account information
        account_info = await private_request("/account")
        
        # Extract balances with non-zero amounts
        balances = {
            asset["asset"]: float(asset["free"]) + float(asset["locked"])
            for asset in account_info["balances"]
            if float(asset["free"]) > 0 or float(asset["locked"]) > 0
        }
        
        logger.info(f"Account has {len(balances)} assets with non-zero balances")
        
        # Log some top balances
        for asset, amount in list(balances.items())[:5]:
            logger.info(f"Balance: {amount} {asset}")
        
        # Get open orders
        open_orders = await private_request("/openOrders")
        logger.info(f"Found {len(open_orders)} open orders")
        
        # Place a test order
        if USE_TESTNET:
            # Get current price for BTC/USDT
            ticker = await public_request("/ticker/price", {"symbol": "BTCUSDT"})
            current_price = float(ticker["price"])
            
            # Set limit price 5% below current price
            limit_price = current_price * 0.95
            
            # Place a test order
            logger.info(f"Placing test limit buy order for 0.001 BTC at ${limit_price:.2f}")
            
            order_params = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": "0.001",
                "price": f"{limit_price:.2f}",
                "newClientOrderId": f"test_order_{int(time.time())}"
            }
            
            # Use the test order endpoint first
            test_result = await private_request("/order/test", order_params, "POST")
            logger.info(f"Test order result: {test_result}")
            
            # Place the actual order
            order_result = await private_request("/order", order_params, "POST")
            
            if "orderId" in order_result:
                order_id = order_result["orderId"]
                logger.info(f"Order placed successfully! Order ID: {order_id}")
                
                # Get order status
                order_status = await private_request("/order", {"symbol": "BTCUSDT", "orderId": order_id})
                logger.info(f"Order status: {order_status['status']}")
                
                # Cancel the order
                logger.info(f"Cancelling order {order_id}")
                cancel_result = await private_request(
                    "/order", 
                    {"symbol": "BTCUSDT", "orderId": order_id}, 
                    "DELETE"
                )
                
                if "orderId" in cancel_result:
                    logger.info("Order cancelled successfully")
                else:
                    logger.error(f"Failed to cancel order: {cancel_result}")
            else:
                logger.error(f"Failed to place order: {order_result}")
    
    except Exception as e:
        logger.exception(f"Error in authenticated API demo: {str(e)}")


async def run_demo():
    """Main function to run the demo."""
    logger.info("Starting Simple Binance API Demo")
    logger.info(f"Using {'testnet' if USE_TESTNET else 'production'} environment")
    
    try:
        # Demonstrate public API endpoints
        await demo_public_api()
        
        # Demonstrate authenticated API endpoints
        await demo_authenticated_api()
        
    except Exception as e:
        logger.exception(f"Error in demo: {str(e)}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 