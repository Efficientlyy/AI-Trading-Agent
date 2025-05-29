"""
MEXC Spot V3 API Integration Tests

This script tests the integration with the MEXC Spot V3 API, verifying
basic functionality for market data retrieval and WebSocket connections.
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_trading_agent.data_acquisition.mexc_spot_v3_client import MexcSpotV3Client
from ai_trading_agent.data_acquisition.mexc_trading_connector import MexcTradingConnector
from ai_trading_agent.config.mexc_config import MEXC_CONFIG, TRADING_PAIRS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mexc_v3_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Test functions
async def test_rest_api():
    """Test basic REST API functionality."""
    client = MexcSpotV3Client()
    
    logger.info("Testing MEXC Spot V3 REST API...")
    
    # Test system status
    try:
        status = await client.get_system_status()
        logger.info(f"System status: {status}")
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
    
    # Test exchange info
    try:
        symbol = TRADING_PAIRS[0]
        exchange_info = await client.get_exchange_info(symbol)
        logger.info(f"Exchange info for {symbol}: {json.dumps(exchange_info, indent=2)[:200]}...")
    except Exception as e:
        logger.error(f"Failed to get exchange info: {e}")
    
    # Test ticker
    try:
        symbol = TRADING_PAIRS[0]
        ticker = await client.get_ticker(symbol)
        logger.info(f"Ticker for {symbol}: {json.dumps(ticker, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to get ticker: {e}")
    
    # Test order book
    try:
        symbol = TRADING_PAIRS[0]
        orderbook = await client.get_orderbook(symbol, 10)
        logger.info(f"Order book for {symbol} (top 10): {json.dumps(orderbook, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to get order book: {e}")
    
    # Test klines
    try:
        symbol = TRADING_PAIRS[0]
        klines = await client.get_klines(symbol, "1m", limit=10)
        logger.info(f"Klines for {symbol} (1m, last 10): {json.dumps(klines, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to get klines: {e}")
    
    # Clean up
    await client.close()

async def test_websocket():
    """Test WebSocket functionality."""
    client = MexcSpotV3Client()
    
    logger.info("Testing MEXC Spot V3 WebSocket...")
    
    # Test ticker WebSocket
    ticker_received = asyncio.Event()
    
    async def ticker_callback(data):
        logger.info(f"Received ticker data: {json.dumps(data, indent=2)}")
        ticker_received.set()
    
    # Subscribe to ticker for first trading pair
    symbol = TRADING_PAIRS[0]
    
    try:
        ticker_task = await client.subscribe_ticker(symbol, ticker_callback)
        
        # Wait for ticker data (with timeout)
        try:
            await asyncio.wait_for(ticker_received.wait(), timeout=30)
            logger.info("Successfully received ticker data")
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for ticker data")
        
        # Cancel the subscription
        ticker_task.cancel()
        
    except Exception as e:
        logger.error(f"Failed to subscribe to ticker: {e}")
    
    # Clean up
    await client.close()

async def test_trading_connector():
    """Test the trading connector."""
    connector = MexcTradingConnector()
    
    logger.info("Testing MEXC Trading Connector...")
    
    # Test getting ticker data
    try:
        symbol = TRADING_PAIRS[0]
        ticker = await connector.get_ticker(symbol)
        logger.info(f"Ticker from connector for {symbol}: {json.dumps(ticker, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to get ticker from connector: {e}")
    
    # Test getting order book
    try:
        symbol = TRADING_PAIRS[0]
        orderbook = await connector.get_orderbook(symbol, 10)
        logger.info(f"Order book from connector for {symbol} (top 10): {json.dumps(orderbook, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to get order book from connector: {e}")
    
    # Test WebSocket callbacks
    data_received = asyncio.Event()
    
    async def test_ticker_callback(symbol, data):
        logger.info(f"Connector ticker callback for {symbol}: {json.dumps(data, indent=2)}")
        data_received.set()
    
    # Register the callback
    connector.register_ticker_callback(test_ticker_callback)
    
    # Subscribe to ticker for first trading pair
    symbol = TRADING_PAIRS[0]
    await connector.subscribe_to_tickers([symbol])
    
    # Wait for data (with timeout)
    try:
        await asyncio.wait_for(data_received.wait(), timeout=30)
        logger.info("Successfully received data through connector")
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for data through connector")
    
    # Clean up
    await connector.close()

async def run_tests():
    """Run all tests."""
    logger.info("Starting MEXC Spot V3 API integration tests...")
    
    # Check API keys
    if not MEXC_CONFIG.get('API_KEY') or not MEXC_CONFIG.get('API_SECRET'):
        logger.warning("API keys not found. Some tests may fail.")
    
    # Run tests
    await test_rest_api()
    await test_websocket()
    await test_trading_connector()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
