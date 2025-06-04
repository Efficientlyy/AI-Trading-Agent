#!/usr/bin/env python
"""
Direct Test for MEXC Klines Endpoint

This script directly tests the MEXC klines endpoint for BTCUSDC
without going through the MultiAssetDataService.
"""

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("direct_klines_test")

def test_klines_endpoint():
    """Test klines endpoint directly"""
    # MEXC API base URL
    base_url = "https://api.mexc.com"
    
    # Test parameters
    symbol = "BTCUSDC"
    intervals = ["1m", "5m", "15m", "30m", "60m", "4h", "1d", "1w"]
    limit = 100
    
    results = {}
    
    for interval in intervals:
        url = f"{base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        logger.info(f"Testing URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Received {len(data)} candles for {interval}")
                results[interval] = len(data)
                
                # Print first candle for verification
                if data:
                    logger.info(f"First candle: {data[0]}")
            else:
                logger.error(f"Error response: {response.text}")
                results[interval] = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            results[interval] = f"Exception: {str(e)}"
    
    return results

if __name__ == "__main__":
    logger.info("Starting direct klines test")
    results = test_klines_endpoint()
    
    # Save results
    output_file = "direct_klines_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("=== Test Summary ===")
    for interval, result in results.items():
        logger.info(f"{interval}: {result}")
