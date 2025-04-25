"""
Alpha Vantage Sentiment Integration Example

This script demonstrates how to:
1. Fetch sentiment data from Alpha Vantage
2. Process and transform it for the trading system
3. Visualize the sentiment data
4. Export it for the dashboard
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import components
from ai_trading_agent.sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector
from ai_trading_agent.api.sentiment_api import SentimentAPI
from ai_trading_agent.visualization.sentiment_viz import (
    visualize_sentiment_trends,
    visualize_multi_asset_sentiment,
    visualize_sentiment_vs_price,
    export_sentiment_data_for_dashboard
)
from ai_trading_agent.common.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_market_data(symbols, start_date, end_date):
    """
    Load market price data for testing.
    This is a simple mock implementation - in a real system, you would
    fetch actual market data from your data source.
    """
    market_data = {}
    
    for symbol in symbols:
        # Create a mock DataFrame with price data
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Generate mock prices (you would replace this with real data)
        if symbol == 'BTC':
            base_price = 30000
        elif symbol == 'ETH':
            base_price = 2000
        else:
            base_price = 100
        
        # Simple random walk
        import numpy as np
        np.random.seed(42)  # For reproducibility
        random_walk = np.random.normal(0, 0.02, size=len(date_range)).cumsum()
        prices = base_price * (1 + random_walk)
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'volume': np.random.randint(1000, 10000, size=len(date_range))
        }, index=date_range)
        
        market_data[symbol] = df
    
    return market_data

def main():
    try:
        logger.info("Starting Alpha Vantage sentiment integration example")
        
        # Check for API key
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.error("Alpha Vantage API key not found in environment variables")
            return 1
        
        # Create output directory for visualizations
        output_dir = os.path.join(project_root, "output", "sentiment")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define symbols to analyze
        symbols = ["BTC", "ETH", "SOL"]
        
        # 1. Fetch and process sentiment data
        logger.info(f"Fetching sentiment data for {symbols}")
        connector = AlphaVantageSentimentConnector()
        
        sentiment_data = {}
        for symbol in symbols:
            try:
                # Use a shorter lookback period to respect API limits
                df = connector.get_sentiment_for_symbol(symbol, days_back=7)
                sentiment_data[symbol] = df
                logger.info(f"Retrieved {len(df)} sentiment records for {symbol}")
                
                # Generate signals
                signals = connector.get_sentiment_signals(symbol)
                logger.info(f"Signal for {symbol}: {signals['signal']} (strength: {signals['strength']:.2f})")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # 2. Generate visualizations
        logger.info("Generating visualizations")
        
        # Load price data for comparison
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        price_data = load_market_data(symbols, start_date, end_date)
        
        # Create visualizations for each symbol
        for symbol in symbols:
            if symbol in sentiment_data and not sentiment_data[symbol].empty:
                # Sentiment trends
                output_file = os.path.join(output_dir, f"{symbol}_sentiment_trends.png")
                visualize_sentiment_trends(sentiment_data[symbol], symbol, output_file)
                
                # Sentiment vs price
                output_file = os.path.join(output_dir, f"{symbol}_sentiment_vs_price.png")
                visualize_sentiment_vs_price(sentiment_data[symbol], price_data[symbol], symbol, output_file)
        
        # Multi-asset sentiment comparison
        output_file = os.path.join(output_dir, "multi_asset_sentiment.png")
        visualize_multi_asset_sentiment(symbols, days_back=7, output_file=output_file)
        
        # 3. Export data for dashboard
        logger.info("Exporting data for dashboard")
        output_file = os.path.join(output_dir, "dashboard_sentiment_data.json")
        dashboard_data = export_sentiment_data_for_dashboard(symbols, days_back=7, output_file=output_file)
        
        # 4. Generate mock data for API
        logger.info("Generating mock data for API")
        api = SentimentAPI(use_mock=False, mock_data_path=os.path.join(project_root, "mock_data/sentiment"))
        mock_data_info = api.generate_mock_data(symbols[:2])  # Use fewer symbols to respect API limits
        
        # Print summary
        logger.info("Integration example completed successfully")
        logger.info(f"Visualizations saved to {output_dir}")
        logger.info(f"Dashboard data exported to {output_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in integration example: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())