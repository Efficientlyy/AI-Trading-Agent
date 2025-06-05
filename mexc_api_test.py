#!/usr/bin/env python
"""
MEXC API Connectivity Test

This script tests connectivity to the MEXC API using the provided credentials.
"""

import os
import sys
import json
import time
import logging
import requests
import hmac
import hashlib
from urllib.parse import urlencode
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mexc_api_test.log")
    ]
)
logger = logging.getLogger("mexc_api_test")

# Load environment variables
load_dotenv('.env-secure/.env')

# MEXC API credentials
API_KEY = os.getenv('MEXC_API_KEY')
API_SECRET = os.getenv('MEXC_API_SECRET')

# MEXC API endpoints
BASE_URL = 'https://api.mexc.com'
SPOT_API_URL = BASE_URL + '/api/v3'
CONTRACT_API_URL = BASE_URL + '/api/v1/contract'

def generate_signature(query_string):
    """Generate HMAC SHA256 signature for MEXC API.
    
    Args:
        query_string: Query string to sign
        
    Returns:
        str: Signature
    """
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def test_spot_api():
    """Test MEXC Spot API connectivity.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing MEXC Spot API connectivity...")
    
    # Test endpoint: Get account information
    endpoint = '/account'
    url = SPOT_API_URL + endpoint
    
    # Generate timestamp
    timestamp = int(time.time() * 1000)
    
    # Generate query string
    query_string = f'timestamp={timestamp}'
    
    # Generate signature
    signature = generate_signature(query_string)
    
    # Add signature to query string
    query_string += f'&signature={signature}'
    
    # Add query string to URL
    url += '?' + query_string
    
    # Set headers
    headers = {
        'X-MEXC-APIKEY': API_KEY
    }
    
    try:
        # Send request
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            logger.info("MEXC Spot API connectivity test successful!")
            logger.info(f"Account data: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            logger.error(f"MEXC Spot API connectivity test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"MEXC Spot API connectivity test failed with exception: {e}")
        return False

def test_spot_market_data():
    """Test MEXC Spot Market Data API.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing MEXC Spot Market Data API...")
    
    # Test endpoint: Get exchange information
    endpoint = '/exchangeInfo'
    url = SPOT_API_URL + endpoint
    
    try:
        # Send request
        response = requests.get(url)
        
        # Check response
        if response.status_code == 200:
            logger.info("MEXC Spot Market Data API test successful!")
            
            # Extract some useful information
            data = response.json()
            symbols = data.get('symbols', [])
            
            # Log some trading pairs
            trading_pairs = [symbol['symbol'] for symbol in symbols[:10]]
            logger.info(f"Available trading pairs (first 10): {trading_pairs}")
            
            # Check for specific pairs
            target_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BTCUSDC', 'ETHUSDC', 'SOLUSDC']
            available_pairs = [symbol['symbol'] for symbol in symbols]
            
            for pair in target_pairs:
                if pair in available_pairs:
                    logger.info(f"Trading pair {pair} is available")
                else:
                    logger.info(f"Trading pair {pair} is NOT available")
            
            return True
        else:
            logger.error(f"MEXC Spot Market Data API test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"MEXC Spot Market Data API test failed with exception: {e}")
        return False

def test_spot_klines():
    """Test MEXC Spot Klines API.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing MEXC Spot Klines API...")
    
    # Test endpoint: Get klines
    endpoint = '/klines'
    url = SPOT_API_URL + endpoint
    
    # Set parameters
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',  # Changed from '1h' to '1m' - MEXC API requires specific interval formats
        'limit': 10
    }
    
    try:
        # Send request
        response = requests.get(url, params=params)
        
        # Check response
        if response.status_code == 200:
            logger.info("MEXC Spot Klines API test successful!")
            
            # Extract klines
            klines = response.json()
            
            # Log klines
            logger.info(f"Received {len(klines)} klines")
            
            # Log first kline
            if klines:
                logger.info(f"First kline: {klines[0]}")
            
            return True
        else:
            logger.error(f"MEXC Spot Klines API test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"MEXC Spot Klines API test failed with exception: {e}")
        return False

def test_spot_order_book():
    """Test MEXC Spot Order Book API.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing MEXC Spot Order Book API...")
    
    # Test endpoint: Get order book
    endpoint = '/depth'
    url = SPOT_API_URL + endpoint
    
    # Set parameters
    params = {
        'symbol': 'BTCUSDT',
        'limit': 10
    }
    
    try:
        # Send request
        response = requests.get(url, params=params)
        
        # Check response
        if response.status_code == 200:
            logger.info("MEXC Spot Order Book API test successful!")
            
            # Extract order book
            order_book = response.json()
            
            # Log order book
            logger.info(f"Order book data: {json.dumps(order_book, indent=2)}")
            
            return True
        else:
            logger.error(f"MEXC Spot Order Book API test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"MEXC Spot Order Book API test failed with exception: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting MEXC API connectivity tests...")
    
    # Check if API credentials are available
    if not API_KEY or not API_SECRET:
        logger.error("MEXC API credentials not found in environment variables")
        return False
    
    # Test MEXC Spot API connectivity
    spot_api_success = test_spot_api()
    
    # Test MEXC Spot Market Data API
    market_data_success = test_spot_market_data()
    
    # Test MEXC Spot Klines API
    klines_success = test_spot_klines()
    
    # Test MEXC Spot Order Book API
    order_book_success = test_spot_order_book()
    
    # Summarize results
    logger.info("MEXC API connectivity test results:")
    logger.info(f"Spot API: {'SUCCESS' if spot_api_success else 'FAILED'}")
    logger.info(f"Market Data API: {'SUCCESS' if market_data_success else 'FAILED'}")
    logger.info(f"Klines API: {'SUCCESS' if klines_success else 'FAILED'}")
    logger.info(f"Order Book API: {'SUCCESS' if order_book_success else 'FAILED'}")
    
    # Overall result
    overall_success = spot_api_success and market_data_success and klines_success and order_book_success
    logger.info(f"Overall result: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    main()
