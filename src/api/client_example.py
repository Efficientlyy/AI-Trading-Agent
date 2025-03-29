"""Example client for the Market Regime Detection API."""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import json

# API endpoint
API_BASE_URL = "http://localhost:8000"

def download_market_data(symbol, start_date, end_date):
    """Download market data from Yahoo Finance."""
    print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Calculate returns
    data["return"] = data['Close'].pct_change()
    
    # Convert to API format
    api_data = []
    for date, row in data.iterrows():
        api_data.append({
            "date": date.isoformat(),
            "price": float(row['Close']),
            "volume": float(row['Volume']),
            "return_value": float(row['return']) if not np.isnan(row['return']) else 0,
            "high": float(row['High']),
            "low": float(row['Low'])
        })
    
    return {
        "symbol": symbol,
        "data": api_data
    }

def detect_regimes(market_data, methods=None, lookback_window=63):
    """Call the detect-regimes API endpoint."""
    if methods is None:
        methods = ["volatility", "momentum", "hmm"]
    
    payload = {
        "market_data": market_data,
        "methods": methods,
        "lookback_window": lookback_window,
        "include_statistics": True,
        "include_visualization": True
    }
    
    print("Detecting market regimes...")
    response = requests.post(f"{API_BASE_URL}/detect-regimes", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def run_backtest(market_data, strategy_type="momentum", regime_methods=None):
    """Call the backtest API endpoint."""
    if regime_methods is None:
        regime_methods = ["volatility", "momentum"]
    
    payload = {
        "market_data": market_data,
        "strategy_type": strategy_type,
        "regime_methods": regime_methods,
        "train_test_split": 0.7,
        "walk_forward": True
    }
    
    print(f"Running backtest with {strategy_type} strategy...")
    response = requests.post(f"{API_BASE_URL}/backtest", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def plot_regimes(market_data, regime_results):
    """Plot market data with detected regimes."""
    # Extract data
    dates = [datetime.fromisoformat(dp["date"]) for dp in market_data["data"]]
    prices = [dp["price"] for dp in market_data["data"]]
    
    # Create figure
    fig, axes = plt.subplots(len(regime_results["regimes"]) + 1, 1, figsize=(12, 10), sharex=True)
    
    # Plot price
    axes[0].plot(dates, prices)
    axes[0].set_title(f"{market_data['symbol']} Price")
    axes[0].grid(True)
    
    # Plot regimes
    for i, (method, labels) in enumerate(regime_results["regimes"].items(), 1):
        axes[i].plot(dates, labels)
        axes[i].set_title(f"{method.capitalize()} Regime")
        axes[i].set_yticks([0, 1, 2])
        axes[i].set_yticklabels(["Low", "Normal", "High"])
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate API usage."""
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print(f"API is not running or not healthy. Status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {API_BASE_URL}. Make sure the API server is running.")
        return
    
    # Get available methods
    methods_response = requests.get(f"{API_BASE_URL}/methods")
    available_methods = methods_response.json()
    print("Available methods:", available_methods["methods"])
    print("Available strategies:", available_methods["strategies"])
    
    # Download market data
    symbol = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data
    
    market_data = download_market_data(
        symbol, 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d")
    )
    
    # Detect regimes
    regime_results = detect_regimes(
        market_data,
        methods=["volatility", "momentum", "hmm", "trend"],
        lookback_window=63
    )
    
    if regime_results:
        print("\nRegime Detection Results:")
        print(f"Request ID: {regime_results['request_id']}")
        print(f"Symbol: {regime_results['symbol']}")
        print(f"Execution Time: {regime_results['execution_time']:.2f} seconds")
        
        # Plot regimes
        plot_regimes(market_data, regime_results)
        
        # Run backtest
        backtest_results = run_backtest(
            market_data,
            strategy_type="momentum",
            regime_methods=["volatility", "momentum"]
        )
        
        if backtest_results:
            print("\nBacktest Results:")
            print(f"Request ID: {backtest_results['request_id']}")
            print(f"Symbol: {backtest_results['symbol']}")
            print(f"Strategy: {backtest_results['strategy']}")
            print(f"Execution Time: {backtest_results['execution_time']:.2f} seconds")
            
            print("\nPerformance Metrics:")
            for metric, value in backtest_results["performance"].items():
                if value is not None:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
            
            print("\nRegime Metrics:")
            for metric, value in backtest_results["regime_metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    main() 