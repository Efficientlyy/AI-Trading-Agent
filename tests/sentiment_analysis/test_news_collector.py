"""
Tests for the News API collector.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading_agent.sentiment_analysis.collectors.news_collector import NewsAPICollector


class TestNewsAPICollector(unittest.TestCase):
    """Test the News API collector."""

    def setUp(self):
        """Set up test fixtures."""
        # Create cache directory if it doesn't exist
        os.makedirs("tests/data/cache/news", exist_ok=True)
        
        self.collector = NewsAPICollector(
            cache_dir="tests/data/cache/news",
            cache_expiry=3600,
            default_api="newsapi",
            rate_limit_wait=1
        )
        
        # Ensure we're using mock data
        self.collector.use_mock_data = True
        
        # Test parameters
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.start_date = datetime.now() - timedelta(days=7)
        self.end_date = datetime.now()

    def test_fetch_sentiment_data(self):
        """Test fetching sentiment data."""
        # Fetch data
        df = self.collector.fetch_sentiment_data(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check that we got a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check that it's not empty
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = ["timestamp", "content", "symbol", "source", "title", "url"]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check that we have data for all symbols
        symbols_in_data = df["symbol"].unique()
        for symbol in self.symbols:
            self.assertIn(symbol, symbols_in_data)
        
        # Check that timestamps are within range - convert to numpy datetime64 for comparison
        start_date_np = np.datetime64(self.start_date)
        end_date_np = np.datetime64(self.end_date)
        
        # Convert timestamps to numpy datetime64 if they aren't already
        timestamps = pd.to_datetime(df["timestamp"]).to_numpy()
        
        self.assertTrue(np.all(timestamps >= start_date_np))
        self.assertTrue(np.all(timestamps <= end_date_np))
        
        # Check that source is "news_mock"
        self.assertTrue(all(df["source"] == "news_mock"))

    def test_cache_functionality(self):
        """Test that caching works."""
        # Set a fixed seed for random data generation
        np.random.seed(42)
        
        # First call should generate data
        df1 = self.collector.fetch_sentiment_data(
            symbols=["AAPL"],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Set the same seed again
        np.random.seed(42)
        
        # Second call should use cached data
        df2 = self.collector.fetch_sentiment_data(
            symbols=["AAPL"],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Data should be the same
        self.assertEqual(len(df1), len(df2))
        self.assertEqual(set(df1.columns), set(df2.columns))
        
        # Check that the symbols are the same
        self.assertEqual(
            set(df1["symbol"].unique()),
            set(df2["symbol"].unique())
        )

    def test_mock_data_generation(self):
        """Test mock data generation."""
        # Generate mock data
        df = self.collector._generate_mock_data(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check that we got a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check that it's not empty
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = ["timestamp", "content", "symbol", "source", "title", "url"]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check that we have data for all symbols
        symbols_in_data = df["symbol"].unique()
        for symbol in self.symbols:
            self.assertIn(symbol, symbols_in_data)


if __name__ == "__main__":
    unittest.main()
