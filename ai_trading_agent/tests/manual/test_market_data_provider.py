"""
Test script for verifying the MarketDataProvider functionality.

This script tests both Alpha Vantage and Yahoo Finance data sources,
fetching historical and latest data for specified symbols.
"""

import asyncio
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading_agent.data.market_data_provider import MarketDataProvider

async def test_market_data_provider():
    """
    Test the MarketDataProvider with both Alpha Vantage and Yahoo Finance.
    Tests historical data retrieval and latest data for specified symbols.
    """
    print("\n=== Testing MarketDataProvider ===\n")
    
    # Test symbols - mix of stocks and crypto
    test_symbols = ["AAPL", "MSFT", "BTC-USD", "ETH-USD"]
    
    # Get one month of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Test with Alpha Vantage
    print("\n--- Testing Alpha Vantage ---\n")
    try:
        alpha_vantage_provider = MarketDataProvider(
            primary_source="alpha_vantage",
            fallback_source="yahoo_finance"
        )
        
        print(f"Fetching historical data for {test_symbols} from Alpha Vantage...")
        alpha_vantage_data = await alpha_vantage_provider.fetch_historical_data(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        print("Results:")
        for symbol, data in alpha_vantage_data.items():
            print(f"  {symbol}: {len(data)} data points")
            if not data.empty:
                print(f"  Sample data (first 3 rows):")
                print(data.head(3))
                print("\n")
        
        await alpha_vantage_provider.close()
    except Exception as e:
        print(f"Error testing Alpha Vantage: {e}")
    
    # Test with Yahoo Finance
    print("\n--- Testing Yahoo Finance ---\n")
    try:
        yahoo_provider = MarketDataProvider(
            primary_source="yahoo_finance",
            fallback_source="alpha_vantage"
        )
        
        print(f"Fetching historical data for {test_symbols} from Yahoo Finance...")
        yahoo_data = await yahoo_provider.fetch_historical_data(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        print("Results:")
        for symbol, data in yahoo_data.items():
            print(f"  {symbol}: {len(data)} data points")
            if not data.empty:
                print(f"  Sample data (first 3 rows):")
                print(data.head(3))
                print("\n")
        
        # Test fetching latest data
        print("Fetching latest data for all symbols...")
        latest_data = await yahoo_provider.fetch_latest_data(test_symbols)
        
        print("Latest data results:")
        for symbol, data in latest_data.items():
            print(f"  {symbol}:")
            print(f"    Date: {data.get('timestamp')}")
            print(f"    Close: {data.get('close')}")
            print(f"    Volume: {data.get('volume')}")
            print("\n")
        
        await yahoo_provider.close()
    except Exception as e:
        print(f"Error testing Yahoo Finance: {e}")

if __name__ == "__main__":
    asyncio.run(test_market_data_provider())
