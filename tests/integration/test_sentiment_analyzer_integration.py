"""
Integration test for the sentiment analyzer with Alpha Vantage API.

This module tests the integration between the sentiment analyzer and the Alpha Vantage API,
verifying that we can fetch real sentiment data and process it correctly.
"""

import pytest
import os
import pandas as pd
from decimal import Decimal
from dotenv import load_dotenv

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient


# Skip these tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("ALPHA_VANTAGE_API_KEY"),
    reason="Alpha Vantage API key not available"
)


class TestSentimentAnalyzerIntegration:
    """Integration test class for the sentiment analyzer."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures for the class."""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        cls.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Skip if no API key is available
        if not cls.api_key:
            pytest.skip("Alpha Vantage API key not available")
        
        # Initialize the sentiment analyzer
        cls.analyzer = SentimentAnalyzer({
            "alpha_vantage_api_key": cls.api_key,
            "default_lags": [1, 2, 3],
            "default_windows": [2, 3],
            "sentiment_threshold": Decimal("0.2"),
            "sentiment_window": 2
        })

    def test_alpha_vantage_client_connection(self):
        """Test that we can connect to the Alpha Vantage API."""
        client = AlphaVantageClient(api_key=self.api_key)
        response = client.get_sentiment_by_topic("blockchain", days_back=3)
        
        # Verify that we got a response without errors
        assert "error" in response
        assert response["error"] is None
        assert "data" in response
        assert response["data"] is not None
        
        # Verify that we got some feed data
        assert "feed" in response["data"]
        
        # Print the number of articles for debugging
        print(f"Number of articles: {len(response['data']['feed'])}")

    def test_fetch_sentiment_data_by_topic(self):
        """Test fetching sentiment data by topic."""
        df = self.analyzer.fetch_sentiment_data(topic="blockchain", days_back=3)
        
        # Verify that we got some data
        assert not df.empty
        
        # Verify that we have the expected columns
        assert "overall_sentiment_score" in df.columns
        assert "overall_sentiment_label" in df.columns
        assert "vader_sentiment_score" in df.columns
        
        # Print the first few rows for debugging
        print(f"First few rows:\n{df.head()}")

    def test_full_sentiment_analysis_pipeline(self):
        """Test the full sentiment analysis pipeline with real data."""
        df = self.analyzer.analyze_sentiment(topic="blockchain", days_back=3)
        
        # Verify that we got some data
        assert not df.empty
        
        # Verify that we have the expected columns
        assert "overall_sentiment_score" in df.columns
        assert "weighted_sentiment_score" in df.columns
        assert "signal" in df.columns
        
        # Verify that we have time series features
        assert "sentiment_trend" in df.columns
        assert "sentiment_momentum" in df.columns
        
        # Print the signals for debugging
        signals = df[["time_published", "overall_sentiment_score", "weighted_sentiment_score", "signal"]].tail(5)
        print(f"Last 5 signals:\n{signals}")
        
        # Count the number of each signal type
        signal_counts = df["signal"].value_counts()
        print(f"Signal counts:\n{signal_counts}")
        
        # Verify that we have at least some non-zero signals
        assert (df["signal"] != 0).any()
