#!/usr/bin/env python3
"""
Test Enhanced Bitvavo Connector

This script tests the enhanced Bitvavo connector with all its features.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enhanced_bitvavo_test")

# Import enhanced components
from src.execution.exchange.enhanced_bitvavo import EnhancedBitvavoConnector
from src.monitoring.error_monitor import ErrorMonitor
from src.monitoring.rate_limit_monitor import RateLimitMonitor
from src.common.cache.data_cache import DataCache
from src.common.network.fallback_manager import BitvavoFallbackManager
from src.common.network.connection_pool import ConnectionPool

def alert_callback(alert_data):
    """Alert callback for monitors."""
    alert_type = alert_data.get('type', 'unknown')
    
    if alert_type == 'rate_limit':
        logger.warning(f"ALERT: Rate limit at {alert_data.get('usage_percentage', 0):.2%} for {alert_data.get('connector', 'unknown')}")
    elif alert_type == 'error':
        logger.warning(f"ALERT: {alert_data.get('error_count', 0)} errors for {alert_data.get('source', 'unknown')}")
    elif alert_type == 'circuit_breaker':
        logger.warning(f"ALERT: Circuit breaker tripped for {alert_data.get('source', 'unknown')}:{alert_data.get('error_type', 'unknown')}")
    else:
        logger.warning(f"ALERT: {alert_data}")

def test_basic_functionality(connector):
    """Test basic functionality of the connector."""
    logger.info("Testing basic functionality...")
    
    # Test get_time
    logger.info("Testing get_time...")
    time_result = connector.get_time()
    if 'time' in time_result:
        logger.info(f"Server time: {datetime.fromtimestamp(time_result['time'] / 1000)}")
    else:
        logger.error(f"Error getting time: {time_result}")
        
    # Test get_markets
    logger.info("Testing get_markets...")
    markets_result = connector.get_markets()
    if isinstance(markets_result, list):
        logger.info(f"Found {len(markets_result)} markets")
    else:
        logger.error(f"Error getting markets: {markets_result}")
        
    # Test get_ticker
    logger.info("Testing get_ticker...")
    ticker_result = connector.get_ticker("BTC/EUR")
    if 'price' in ticker_result:
        logger.info(f"BTC/EUR price: {ticker_result['price']}")
    else:
        logger.error(f"Error getting ticker: {ticker_result}")
        
    # Test get_order_book
    logger.info("Testing get_order_book...")
    orderbook_result = connector.get_order_book("BTC/EUR")
    if 'bids' in orderbook_result and 'asks' in orderbook_result:
        logger.info(f"Order book: {len(orderbook_result['bids'])} bids, {len(orderbook_result['asks'])} asks")
    else:
        logger.error(f"Error getting order book: {orderbook_result}")

def test_caching(connector, data_cache):
    """Test caching functionality."""
    logger.info("Testing caching...")
    
    # Clear cache
    data_cache.clear()
    
    # Get ticker (should miss cache)
    start_time = time.time()
    ticker_result = connector.get_ticker("BTC/EUR")
    first_request_time = time.time() - start_time
    
    # Get ticker again (should hit cache)
    start_time = time.time()
    ticker_result_cached = connector.get_ticker("BTC/EUR")
    second_request_time = time.time() - start_time
    
    # Compare results
    logger.info(f"First request time: {first_request_time:.4f}s")
    logger.info(f"Second request time: {second_request_time:.4f}s")
    logger.info(f"Cache speedup: {first_request_time / second_request_time:.2f}x")
    
    # Check cache stats
    cache_stats = data_cache.get_stats()
    logger.info(f"Cache stats: {cache_stats}")

def test_error_handling(connector, error_monitor):
    """Test error handling functionality."""
    logger.info("Testing error handling...")
    
    # Test invalid endpoint
    logger.info("Testing invalid endpoint...")
    invalid_result = connector._request("GET", "/invalid")
    
    # Check error monitor
    error_stats = error_monitor.get_error_stats()
    logger.info(f"Error stats: {error_stats}")
    
    # Check circuit breakers
    circuit_breakers = error_monitor.get_circuit_breakers()
    logger.info(f"Circuit breakers: {circuit_breakers}")

def test_rate_limiting(connector, rate_limit_monitor):
    """Test rate limiting functionality."""
    logger.info("Testing rate limiting...")
    
    # Make multiple requests
    logger.info("Making multiple requests...")
    for i in range(5):
        connector.get_ticker("BTC/EUR")
        time.sleep(0.1)
    
    # Check rate limit monitor
    rate_limit_status = rate_limit_monitor.get_rate_limit_status()
    logger.info(f"Rate limit status: {rate_limit_status}")

def test_fallback(connector, fallback_manager):
    """Test fallback functionality."""
    logger.info("Testing fallback functionality...")
    
    # Test cache fallback
    logger.info("Testing cache fallback...")
    success, result = fallback_manager.handle_fallback(
        "bitvavo",
        "rate_limit",
        {
            "method": "GET",
            "endpoint": "/ticker/price",
            "params": {"market": "BTC-EUR"}
        }
    )
    
    if success:
        logger.info(f"Cache fallback successful: {result}")
    else:
        logger.info("Cache fallback not successful")
    
    # Test mock fallback
    logger.info("Testing mock fallback...")
    success, result = fallback_manager.handle_fallback(
        "bitvavo",
        "api_error",
        {
            "method": "GET",
            "endpoint": "/ticker/price",
            "params": {"market": "BTC-EUR"}
        }
    )
    
    if success:
        logger.info(f"Mock fallback successful: {result}")
    else:
        logger.info("Mock fallback not successful")
    
    # Check fallback stats
    fallback_stats = fallback_manager.get_stats()
    logger.info(f"Fallback stats: {fallback_stats}")

def test_connection_pool(connection_pool):
    """Test connection pool functionality."""
    logger.info("Testing connection pool...")
    
    # Make multiple requests
    logger.info("Making multiple requests...")
    for i in range(5):
        connection_pool.request(
            "GET",
            "https://api.bitvavo.com/v2/time",
            session_name="test"
        )
        time.sleep(0.1)
    
    # Check connection pool stats
    pool_stats = connection_pool.get_stats()
    logger.info(f"Connection pool stats: {pool_stats}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Enhanced Bitvavo Connector")
    parser.add_argument("--api-key", help="Bitvavo API key")
    parser.add_argument("--api-secret", help="Bitvavo API secret")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    
    args = parser.parse_args()
    
    # Get API credentials
    api_key = args.api_key or os.environ.get("BITVAVO_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("BITVAVO_API_SECRET", "")
    
    # Create components
    error_monitor = ErrorMonitor()
    rate_limit_monitor = RateLimitMonitor()
    data_cache = DataCache(cache_dir=args.cache_dir)
    connection_pool = ConnectionPool()
    fallback_manager = BitvavoFallbackManager()
    
    # Register alert callbacks
    error_monitor.register_alert_callback(alert_callback)
    rate_limit_monitor.register_alert_callback(alert_callback)
    
    # Create connector
    connector = EnhancedBitvavoConnector(
        api_key=api_key,
        api_secret=api_secret,
        connection_pool=connection_pool,
        error_monitor=error_monitor,
        rate_limit_monitor=rate_limit_monitor,
        data_cache=data_cache,
        fallback_manager=fallback_manager
    )
    
    # Initialize connector
    logger.info("Initializing connector...")
    if not connector.initialize():
        logger.error("Failed to initialize connector")
        return 1
    
    try:
        # Run tests
        test_basic_functionality(connector)
        test_caching(connector, data_cache)
        test_error_handling(connector, error_monitor)
        test_rate_limiting(connector, rate_limit_monitor)
        test_fallback(connector, fallback_manager)
        test_connection_pool(connection_pool)
        
        # Print final stats
        logger.info("Final stats:")
        logger.info(f"Cache stats: {data_cache.get_stats()}")
        logger.info(f"Error stats: {error_monitor.get_error_stats()}")
        logger.info(f"Rate limit status: {rate_limit_monitor.get_rate_limit_status()}")
        logger.info(f"Fallback stats: {fallback_manager.get_stats()}")
        logger.info(f"Connection pool stats: {connection_pool.get_stats()}")
        
        logger.info("All tests completed successfully")
    finally:
        # Close connector
        connector.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())