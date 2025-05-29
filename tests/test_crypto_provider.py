"""
Integration test for the Cryptocurrency Data Provider.

This script tests the functionality of the CryptoDataProvider class
and its API clients to ensure they can properly retrieve cryptocurrency data.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)

# Import the provider
from ai_trading_agent.data_providers.crypto import (
    CryptoDataProvider, 
    BinanceClient, 
    CoinGeckoClient, 
    CryptoCompareClient
)

def test_individual_clients():
    """Test each API client individually."""
    print("\n=== Testing Individual API Clients ===")
    
    # Test Binance client
    print("\nTesting Binance Client:")
    binance = BinanceClient()
    test_client(binance, "Binance")
    
    # Test CoinGecko client
    print("\nTesting CoinGecko Client:")
    coingecko = CoinGeckoClient()
    test_client(coingecko, "CoinGecko")
    
    # Test CryptoCompare client
    print("\nTesting CryptoCompare Client:")
    crypto_compare = CryptoCompareClient()
    test_client(crypto_compare, "CryptoCompare")

def test_client(client, name):
    """Test a specific API client."""
    # Test BTC price
    start = time.time()
    btc_price = client.get_price("BTC/USD")
    elapsed = time.time() - start
    
    if btc_price:
        print(f"  ✓ {name} BTC/USD: ${btc_price:.2f} (took {elapsed:.2f}s)")
    else:
        print(f"  ✗ {name} BTC/USD: Failed to get price")
    
    # Test ETH price
    start = time.time()
    eth_price = client.get_price("ETH/USD")
    elapsed = time.time() - start
    
    if eth_price:
        print(f"  ✓ {name} ETH/USD: ${eth_price:.2f} (took {elapsed:.2f}s)")
    else:
        print(f"  ✗ {name} ETH/USD: Failed to get price")

def test_crypto_data_provider():
    """Test the CryptoDataProvider class."""
    print("\n=== Testing CryptoDataProvider ===")
    provider = CryptoDataProvider()
    
    # Subscribe to symbols
    provider.subscribe("BTC/USD")
    provider.subscribe("ETH/USD")
    
    # Start data provider
    provider.start()
    print("  ✓ Provider started")
    
    # Give it time to fetch initial data
    print("  Waiting for initial data fetch...")
    time.sleep(3)
    
    # Test getting prices
    btc_price = provider.get_current_price("BTC/USD")
    if btc_price:
        source = provider.data_source.get("BTC/USD", "Unknown")
        print(f"  ✓ BTC/USD: ${btc_price:.2f} from {source}")
    else:
        print(f"  ✗ BTC/USD: Failed to get price")
    
    eth_price = provider.get_current_price("ETH/USD")
    if eth_price:
        source = provider.data_source.get("ETH/USD", "Unknown")
        print(f"  ✓ ETH/USD: ${eth_price:.2f} from {source}")
    else:
        print(f"  ✗ ETH/USD: Failed to get price")
    
    # Test price validation
    print("\n  Testing price validation:")
    price, source, confidence = provider._fetch_real_time_price("BTC/USD", validate=True)
    print(f"  ✓ BTC/USD Validated: ${price:.2f} from {source} with {confidence}% confidence")
    
    # Test API stats
    print("\n  API Provider Statistics:")
    for provider_name, stats in provider.api_stats.items():
        if stats['success'] > 0:
            print(f"  - {provider_name}: {stats['success']} successful calls")
    
    # Stop provider
    provider.stop()
    print("  ✓ Provider stopped")

def test_historical_data():
    """Test fetching historical data."""
    print("\n=== Testing Historical Data ===")
    provider = CryptoDataProvider()
    
    # Test getting daily historical data
    print("\nFetching BTC/USD daily data (7 days):")
    btc_daily = provider.get_historical_data("BTC/USD", days=7, interval='daily')
    if btc_daily is not None and not btc_daily.empty:
        print(f"  ✓ Got {len(btc_daily)} daily data points")
        print(f"  First day: {btc_daily.iloc[0]['date'].strftime('%Y-%m-%d')}, Price: ${btc_daily.iloc[0]['price']:.2f}")
        print(f"  Last day: {btc_daily.iloc[-1]['date'].strftime('%Y-%m-%d')}, Price: ${btc_daily.iloc[-1]['price']:.2f}")
    else:
        print("  ✗ Failed to get daily historical data")
    
    # Test getting hourly historical data
    print("\nFetching ETH/USD hourly data (1 day):")
    eth_hourly = provider.get_historical_data("ETH/USD", days=1, interval='hourly')
    if eth_hourly is not None and not eth_hourly.empty:
        print(f"  ✓ Got {len(eth_hourly)} hourly data points")
        print(f"  First hour: {eth_hourly.iloc[0]['date'].strftime('%Y-%m-%d %H:%M')}, Price: ${eth_hourly.iloc[0]['price']:.2f}")
        print(f"  Last hour: {eth_hourly.iloc[-1]['date'].strftime('%Y-%m-%d %H:%M')}, Price: ${eth_hourly.iloc[-1]['price']:.2f}")
    else:
        print("  ✗ Failed to get hourly historical data")

if __name__ == "__main__":
    print("\n==== CRYPTOCURRENCY DATA PROVIDER TEST ====")
    print("Testing at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Run tests
    test_individual_clients()
    test_crypto_data_provider()
    test_historical_data()
    
    print("\n==== TEST COMPLETED ====")
