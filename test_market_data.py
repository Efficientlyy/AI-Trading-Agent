"""
Test script for the MarketDataProvider implementation.

This script tests the functionality of the MarketDataProvider class
to verify it can correctly retrieve cryptocurrency market data from
multiple sources with proper validation and fallbacks.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)

# Import the MarketDataProvider
from ai_trading_agent.market_data import MarketDataProvider

def test_market_data_provider():
    """Test the core functionality of the MarketDataProvider."""
    print("\n============ MARKET DATA PROVIDER TEST ============")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create the provider
    provider = MarketDataProvider()
    print("✓ Created MarketDataProvider instance")
    
    # Test getting BTC/USD price
    print("\n--- Real-time Price Test ---")
    print("Fetching BTC/USD price...")
    start_time = time.time()
    btc_price = provider.get_current_price("BTC/USD")
    elapsed = time.time() - start_time
    
    if btc_price:
        source = provider.data_source.get("BTC/USD", "Unknown")
        print(f"✓ BTC/USD: ${btc_price:,.2f} from {source} (took {elapsed:.2f}s)")
    else:
        print("✗ Failed to get BTC/USD price")
    
    # Test getting ETH/USD price
    print("\nFetching ETH/USD price...")
    start_time = time.time()
    eth_price = provider.get_current_price("ETH/USD")
    elapsed = time.time() - start_time
    
    if eth_price:
        source = provider.data_source.get("ETH/USD", "Unknown")
        print(f"✓ ETH/USD: ${eth_price:,.2f} from {source} (took {elapsed:.2f}s)")
    else:
        print("✗ Failed to get ETH/USD price")
    
    # Test price validation
    print("\n--- Price Validation Test ---")
    print("Validating BTC/USD price across sources...")
    start_time = time.time()
    price, source, confidence = provider._fetch_real_time_price("BTC/USD", validate=True)
    elapsed = time.time() - start_time
    
    if price:
        print(f"✓ Validated price: ${price:,.2f}")
        print(f"  Source: {source}")
        print(f"  Confidence: {confidence}%")
        print(f"  Time taken: {elapsed:.2f}s")
    else:
        print("✗ Failed to validate price")
    
    # Test provider stats
    print("\n--- Provider Statistics ---")
    for provider_name, stats in provider.api_stats.items():
        if stats['success'] > 0:
            success_rate = stats['success'] / (stats['success'] + stats['failure'] or 1) * 100
            print(f"• {provider_name}: {stats['success']} successes, {stats['failure']} failures ({success_rate:.1f}% success rate)")
    
    # Test historical data
    print("\n--- Historical Data Test ---")
    print("Fetching BTC/USD 7-day historical data...")
    start_time = time.time()
    btc_history = provider.get_historical_data("BTC/USD", days=7, interval='daily')
    elapsed = time.time() - start_time
    
    if btc_history is not None and not btc_history.empty:
        print(f"✓ Got {len(btc_history)} data points (took {elapsed:.2f}s)")
        print(f"  First day: {btc_history.iloc[0]['date'].strftime('%Y-%m-%d')}, Price: ${btc_history.iloc[0]['price']:,.2f}")
        print(f"  Last day: {btc_history.iloc[-1]['date'].strftime('%Y-%m-%d')}, Price: ${btc_history.iloc[-1]['price']:,.2f}")
    else:
        print("✗ Failed to get historical data")
    
    print("\n============ TEST COMPLETED ============")

if __name__ == "__main__":
    test_market_data_provider()
