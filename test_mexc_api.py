#!/usr/bin/env python
"""
Test script for MEXC API connectivity and data retrieval

This script tests the connection to MEXC API and retrieves market data
to verify that the API credentials are working correctly.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from datetime import datetime

# Import Trading-Agent components
from env_loader import load_environment_variables
from optimized_mexc_client import OptimizedMEXCClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_mexc_api")

def test_api_connection():
    """Test connection to MEXC API"""
    logger.info("Loading environment variables...")
    env = load_environment_variables()
    
    if not env['MEXC_API_KEY'] or not env['MEXC_SECRET_KEY']:
        logger.error("MEXC API credentials not found in environment variables")
        return False
    
    logger.info("Initializing MEXC client...")
    client = OptimizedMEXCClient(
        api_key=env['MEXC_API_KEY'],
        api_secret=env['MEXC_SECRET_KEY']
    )
    
    logger.info("Testing server time...")
    server_time = client.get_server_time()
    logger.info(f"Server time: {server_time}")
    
    logger.info("Testing ticker data...")
    ticker = client.get_ticker("BTCUSDT")
    logger.info(f"BTC/USDC price: {ticker['lastPrice']}")
    
    return True

def test_market_data_retrieval():
    """Test market data retrieval from MEXC API"""
    logger.info("Loading environment variables...")
    env = load_environment_variables()
    
    logger.info("Initializing MEXC client...")
    client = OptimizedMEXCClient(
        api_key=env['MEXC_API_KEY'],
        api_secret=env['MEXC_SECRET_KEY']
    )
    
    symbol = "BTCUSDT"
    timeframe = "5m"
    limit = 100  # Start with a smaller limit to avoid timeouts
    
    logger.info(f"Retrieving {limit} candles for {symbol} {timeframe}...")
    start_time = time.time()
    
    try:
        candles = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(candles)} candles in {elapsed:.2f} seconds")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Print sample data
        logger.info(f"Sample data:\n{df.head()}")
        
        return True
    except Exception as e:
        logger.error(f"Error retrieving market data: {e}")
        return False

def test_pattern_recognition_data():
    """Test data retrieval for pattern recognition"""
    logger.info("Loading environment variables...")
    env = load_environment_variables()
    
    logger.info("Initializing MEXC client...")
    client = OptimizedMEXCClient(
        api_key=env['MEXC_API_KEY'],
        api_secret=env['MEXC_SECRET_KEY']
    )
    
    symbol = "BTCUSDT"
    timeframe = "5m"
    limit = 1000  # Full limit to test potential timeout issues
    
    logger.info(f"Retrieving {limit} candles for {symbol} {timeframe}...")
    start_time = time.time()
    
    try:
        candles = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(candles)} candles in {elapsed:.2f} seconds")
        
        # Save to file for analysis
        with open("test_pattern_data.json", "w") as f:
            json.dump(candles, f)
        
        logger.info(f"Saved {len(candles)} candles to test_pattern_data.json")
        
        return True
    except Exception as e:
        logger.error(f"Error retrieving pattern recognition data: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting MEXC API tests...")
    
    # Test API connection
    if not test_api_connection():
        logger.error("API connection test failed")
        sys.exit(1)
    
    # Test market data retrieval
    if not test_market_data_retrieval():
        logger.error("Market data retrieval test failed")
        sys.exit(1)
    
    # Test pattern recognition data
    if not test_pattern_recognition_data():
        logger.error("Pattern recognition data test failed")
        sys.exit(1)
    
    logger.info("All tests passed successfully")
