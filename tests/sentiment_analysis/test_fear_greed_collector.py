"""
Tests for the Fear & Greed Index collector.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading_agent.sentiment_analysis.collectors.fear_greed_collector import FearGreedIndexCollector


class TestFearGreedIndexCollector(unittest.TestCase):
    """Test the Fear & Greed Index collector."""

    def setUp(self):
        """Set up test fixtures."""
        # Create cache directory if it doesn't exist
        os.makedirs("tests/data/cache/fear_greed", exist_ok=True)
        
        self.collector = FearGreedIndexCollector(
            cache_dir="tests/data/cache/fear_greed",
            cache_expiry=3600,
            historical_data_source="alternative"
        )
        
        # Ensure we're using mock data for tests
        self.collector.use_mock_data = True
        
        # Test parameters
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.start_date = datetime.now() - timedelta(days=30)
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
        required_columns = ["timestamp", "content", "symbol", "source", "value", "classification"]
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
        
        # Check that source is "fear_greed_index"
        self.assertTrue(all(df["source"] == "fear_greed_index"))
        
        # Check that values are between 0 and 100
        self.assertTrue(all(df["value"] >= 0))
        self.assertTrue(all(df["value"] <= 100))
        
        # Check that classifications are valid
        valid_classifications = [
            "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
        ]
        for classification in df["classification"].unique():
            self.assertIn(classification, valid_classifications)

    def test_get_classification_from_value(self):
        """Test classification mapping function."""
        test_cases = [
            (0, "Extreme Fear"),
            (10, "Extreme Fear"),
            (25, "Extreme Fear"),
            (26, "Fear"),
            (35, "Fear"),
            (45, "Fear"),
            (46, "Neutral"),
            (50, "Neutral"),
            (55, "Neutral"),
            (56, "Greed"),
            (65, "Greed"),
            (75, "Greed"),
            (76, "Extreme Greed"),
            (90, "Extreme Greed"),
            (100, "Extreme Greed")
        ]
        
        for value, expected_classification in test_cases:
            actual_classification = self.collector._get_classification_from_value(value)
            self.assertEqual(
                actual_classification, 
                expected_classification,
                f"Value {value} should be classified as {expected_classification}, got {actual_classification}"
            )

    def test_normalize_index_value(self):
        """Test normalization function."""
        test_cases = [
            (0, -1.0),
            (25, -0.5),
            (50, 0.0),
            (75, 0.5),
            (100, 1.0)
        ]
        
        for value, expected_normalized in test_cases:
            actual_normalized = self.collector.normalize_index_value(value)
            self.assertAlmostEqual(
                actual_normalized, 
                expected_normalized,
                msg=f"Value {value} should normalize to {expected_normalized}, got {actual_normalized}"
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
        required_columns = ["timestamp", "content", "symbol", "source", "value", "classification"]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check that we have data for all symbols
        symbols_in_data = df["symbol"].unique()
        for symbol in self.symbols:
            self.assertIn(symbol, symbols_in_data)
        
        # Check that we have one data point per day per symbol
        expected_days = (self.end_date - self.start_date).days + 1  # +1 to include end date
        expected_data_points = expected_days * len(self.symbols)
        self.assertEqual(len(df), expected_data_points)


if __name__ == "__main__":
    unittest.main()
