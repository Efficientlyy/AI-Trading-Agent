#!/usr/bin/env python
"""Script to download market data for the Market Regime Detection API."""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

def download_data(symbol, start_date, end_date, output_dir=None):
    """Download market data from Yahoo Finance and save it as JSON."""
    print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print(f"No data found for {symbol} in the specified date range.")
        return None
    
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
    
    market_data = {
        "symbol": symbol,
        "data": api_data
    }
    
    # Save to file if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"{symbol}_{start_date}_{end_date}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(market_data, f, indent=2)
        
        print(f"Data saved to {filepath}")
    
    return market_data

def download_multiple(symbols, start_date, end_date, output_dir=None):
    """Download data for multiple symbols."""
    results = {}
    
    for symbol in symbols:
        data = download_data(symbol, start_date, end_date, output_dir)
        if data:
            results[symbol] = data
    
    return results

def main():
    """Main function to parse arguments and download data."""
    parser = argparse.ArgumentParser(description="Download market data for the Market Regime Detection API")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to download (e.g., SPY AAPL)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=365, help="Number of days to download if start date is not specified")
    parser.add_argument("--output", default="./data", help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    # Set dates
    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")
    
    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=args.days)
        start_date = start_dt.strftime("%Y-%m-%d")
    
    # Download data
    download_multiple(args.symbols, start_date, end_date, args.output)

if __name__ == "__main__":
    main() 