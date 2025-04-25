"""
Generate mock data for the dashboard

This script generates mock sentiment data that can be used by the dashboard
without hitting the Alpha Vantage API rate limits.
"""

import os
import sys
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ai_trading_agent.common.logging_config import setup_logging
from ai_trading_agent.api.sentiment_api import SentimentAPI

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_mock_data(symbols=None):
    """Generate mock data for the dashboard."""
    # Create mock data directory
    mock_dir = Path(project_root) / "mock_data" / "sentiment"
    mock_dir.mkdir(parents=True, exist_ok=True)
    
    # Default symbols
    if symbols is None:
        symbols = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOGE"]
    
    logger.info(f"Generating mock data for symbols: {symbols}")
    
    # Create SentimentAPI instance for generating mock data
    api = SentimentAPI(use_mock=False, mock_data_path=str(mock_dir))
    
    try:
        # Try to use the API to generate real data for mocks
        return api.generate_mock_data(symbols)
    except Exception as e:
        logger.error(f"Error using API to generate mock data: {e}")
        logger.info("Generating synthetic mock data instead")
        
        # If the API fails, generate synthetic mock data
        return generate_synthetic_mock_data(symbols, mock_dir)

def generate_synthetic_mock_data(symbols, mock_dir):
    """Generate synthetic mock data when API fails."""
    mock_data = {}
    
    # Generate sentiment summary
    sentiment_data = {}
    for symbol in symbols:
        # Generate random sentiment signal
        signal = random.choice(["buy", "sell", "hold"])
        strength = round(random.uniform(0.1, 0.9), 2)
        score = round(random.uniform(-0.8, 0.8), 2)
        
        sentiment_data[symbol] = {
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "score": score,
            "trend": round(random.uniform(-0.2, 0.2), 2),
            "volatility": round(random.uniform(0.05, 0.3), 2),
            "timestamp": datetime.now().isoformat()
        }
    
    summary = {
        "sentimentData": sentiment_data,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary
    summary_path = mock_dir / "sentiment_summary_symbols_BTC_ETH_XRP_ADA_SOL_DOGE.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    mock_data["summary"] = {
        "file": str(summary_path),
        "symbols": symbols,
        "count": len(sentiment_data)
    }
    
    # Generate historical data for each symbol
    historical = {}
    for symbol in symbols:
        # Generate 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = []
        current_date = start_date
        
        # Create a reasonable sentiment trend with some randomness
        base_score = random.uniform(-0.3, 0.3)
        trend = random.uniform(-0.01, 0.01)
        
        while current_date <= end_date:
            # Add some randomness to the score
            noise = random.uniform(-0.2, 0.2)
            days_passed = (current_date - start_date).days
            score = base_score + (trend * days_passed) + noise
            score = max(-0.9, min(0.9, score))  # Clamp between -0.9 and 0.9
            
            data.append({
                "timestamp": current_date.isoformat(),
                "score": round(score, 2),
                "raw_score": round(score - (noise / 2), 2)  # Slightly different for raw score
            })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Save historical data
        for timeframe in ["1D", "1W", "1M", "3M"]:
            # Adjust data based on timeframe
            if timeframe == "1D":
                tf_data = data[-1:]  # Just the last day
            elif timeframe == "1W":
                tf_data = data[-7:]  # Last week
            elif timeframe == "1M":
                tf_data = data  # All 30 days
            else:  # 3M
                # Generate some extra data for 3M
                extra_data = []
                extra_start = start_date - timedelta(days=60)
                current_date = extra_start
                
                while current_date < start_date:
                    noise = random.uniform(-0.2, 0.2)
                    days_passed = (current_date - extra_start).days
                    score = base_score + (trend * days_passed) + noise
                    score = max(-0.9, min(0.9, score))
                    
                    extra_data.append({
                        "timestamp": current_date.isoformat(),
                        "score": round(score, 2),
                        "raw_score": round(score - (noise / 2), 2)
                    })
                    
                    current_date += timedelta(days=1)
                
                tf_data = extra_data + data
            
            # Save for this timeframe
            hist_path = mock_dir / f"historical_sentiment_symbol_{symbol}_timeframe_{timeframe}.json"
            with open(hist_path, "w") as f:
                json.dump(tf_data, f, indent=2)
            
            historical[f"{symbol}_{timeframe}"] = len(tf_data)
    
    mock_data["historical"] = historical
    
    return mock_data

if __name__ == "__main__":
    print("Generating mock data for the dashboard...")
    result = generate_mock_data()
    print(f"Generated mock data: {result}")
    print("Mock data generation complete!")