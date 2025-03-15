#!/usr/bin/env python
"""Simple script to test the Market Regime Detection API."""

import requests
import json
from datetime import datetime, timedelta
import sys

# API endpoint
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(response.json())
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {API_BASE_URL}")
        return False

def test_methods():
    """Test the methods endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/methods")
        print(f"Methods check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {API_BASE_URL}")
        return False

def test_detect_regimes():
    """Test the detect-regimes endpoint with synthetic data."""
    # Create synthetic data
    dates = []
    prices = []
    volumes = []
    returns = []
    
    base_date = datetime.now() - timedelta(days=365)
    price = 100.0
    
    for i in range(252):  # One year of trading days
        date = base_date + timedelta(days=i)
        dates.append(date.isoformat())
        
        # Simple random walk for price
        price = price * (1 + (0.0002 * (i % 5 - 2)))  # Small oscillation
        prices.append(price)
        
        # Volume
        volume = 1000000 * (1 + 0.1 * (i % 7 - 3))
        volumes.append(volume)
        
        # Return
        if i > 0:
            ret = (prices[i] / prices[i-1]) - 1
        else:
            ret = 0
        returns.append(ret)
    
    # Create API request data
    data_points = []
    for i in range(len(dates)):
        data_points.append({
            "date": dates[i],
            "price": prices[i],
            "volume": volumes[i],
            "return_value": returns[i]
        })
    
    market_data = {
        "symbol": "TEST",
        "data": data_points
    }
    
    payload = {
        "market_data": market_data,
        "methods": ["volatility", "momentum"],
        "lookback_window": 21,
        "include_statistics": True,
        "include_visualization": False
    }
    
    try:
        print("Testing detect-regimes endpoint...")
        response = requests.post(f"{API_BASE_URL}/detect-regimes", json=payload)
        print(f"Detect regimes: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Request ID: {result['request_id']}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            print(f"Number of regimes detected: {len(result['regimes'])}")
            
            # Print first few labels for each method
            for method, labels in result['regimes'].items():
                print(f"{method}: {labels[:10]}...")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Testing Market Regime Detection API...")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Make sure the API is running.")
        return 1
    
    # Test methods endpoint
    if not test_methods():
        print("Methods check failed.")
        return 1
    
    # Test detect-regimes endpoint
    if not test_detect_regimes():
        print("Detect regimes test failed.")
        return 1
    
    print("\nAll tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 