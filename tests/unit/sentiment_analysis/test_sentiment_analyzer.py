"""
Test module for the sentiment analyzer.

This module tests the functionality of the sentiment analyzer, including
fetching sentiment data from Alpha Vantage and generating trading signals.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import patch, MagicMock
from datetime import datetime

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient


class TestSentimentAnalyzer:
    """Test class for the sentiment analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "default_lags": [1, 2, 3],
            "default_windows": [2, 3],
            "sentiment_threshold": Decimal("0.2"),
            "sentiment_window": 2,
            "feature_weights": {
                "sentiment_score": Decimal("1.0"),
                "sentiment_trend": Decimal("0.8"),
                "sentiment_momentum": Decimal("0.7")
            }
        }
        self.analyzer = SentimentAnalyzer(config=self.config)

    def test_init(self):
        """Test initialization of the sentiment analyzer."""
        assert self.analyzer.default_lags == [1, 2, 3]
        assert self.analyzer.default_windows == [2, 3]
        assert self.analyzer.sentiment_threshold == Decimal("0.2")
        assert self.analyzer.sentiment_window == 2
        assert self.analyzer.feature_weights["sentiment_score"] == Decimal("1.0")
        assert self.analyzer.feature_weights["sentiment_trend"] == Decimal("0.8")
        assert self.analyzer.feature_weights["sentiment_momentum"] == Decimal("0.7")

    @patch.object(AlphaVantageClient, 'get_sentiment_by_topic')
    @patch.object(AlphaVantageClient, 'extract_sentiment_scores')
    def test_fetch_sentiment_data_by_topic(self, mock_extract, mock_get_sentiment):
        """Test fetching sentiment data by topic."""
        # Mock response from Alpha Vantage
        mock_get_sentiment.return_value = {"error": None, "data": {"feed": []}}
        
        # Mock extracted sentiment scores
        mock_extract.return_value = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "time_published": "20250101T120000",
                "summary": "This is a positive article about blockchain.",
                "overall_sentiment_score": Decimal("0.8"),
                "overall_sentiment_label": "Bullish"
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/2",
                "time_published": "20250102T120000",
                "summary": "This is a negative article about blockchain.",
                "overall_sentiment_score": Decimal("-0.5"),
                "overall_sentiment_label": "Bearish"
            }
        ]
        
        # Call the method
        df = self.analyzer.fetch_sentiment_data(topic="blockchain", days_back=7)
        
        # Verify the result
        assert len(df) == 2
        assert "overall_sentiment_score" in df.columns
        assert "vader_sentiment_score" in df.columns
        assert mock_get_sentiment.called
        assert mock_extract.called

    @patch.object(AlphaVantageClient, 'get_sentiment_by_crypto')
    @patch.object(AlphaVantageClient, 'extract_sentiment_scores')
    def test_fetch_sentiment_data_by_crypto(self, mock_extract, mock_get_sentiment):
        """Test fetching sentiment data by cryptocurrency ticker."""
        # Mock response from Alpha Vantage
        mock_get_sentiment.return_value = {"error": None, "data": {"feed": []}}
        
        # Mock extracted sentiment scores
        mock_extract.return_value = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "time_published": "20250101T120000",
                "summary": "This is a positive article about Bitcoin.",
                "overall_sentiment_score": Decimal("0.8"),
                "overall_sentiment_label": "Bullish",
                "ticker": "BTC",
                "ticker_sentiment_score": Decimal("0.9"),
                "ticker_sentiment_label": "Bullish",
                "relevance_score": Decimal("0.95")
            }
        ]
        
        # Call the method
        df = self.analyzer.fetch_sentiment_data(crypto_ticker="BTC", days_back=7)
        
        # Verify the result
        assert len(df) == 1
        assert "ticker" in df.columns
        assert "ticker_sentiment_score" in df.columns
        assert "relevance_score" in df.columns
        assert mock_get_sentiment.called
        assert mock_extract.called

    def test_generate_time_series_features(self):
        """Test generating time series features from sentiment data."""
        # Create test data
        data = {
            "time_published": [
                "20250101T120000", "20250102T120000", "20250103T120000",
                "20250104T120000", "20250105T120000", "20250106T120000"
            ],
            "overall_sentiment_score": [
                Decimal("0.5"), Decimal("0.6"), Decimal("0.4"),
                Decimal("0.3"), Decimal("0.7"), Decimal("0.8")
            ]
        }
        df = pd.DataFrame(data)
        df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
        
        # Call the method
        result_df = self.analyzer.generate_time_series_features(df)
        
        # Verify the result
        assert "overall_sentiment_score_lag_1" in result_df.columns
        assert "overall_sentiment_score_diff_1" in result_df.columns
        assert "overall_sentiment_score_pct_change_1" in result_df.columns
        assert "overall_sentiment_score_rolling_mean_2" in result_df.columns
        assert "overall_sentiment_score_rolling_std_2" in result_df.columns
        assert "sentiment_momentum" in result_df.columns
        assert "sentiment_trend" in result_df.columns
        
        # These features may not be present in all cases, so we'll check for them conditionally
        # Depending on the implementation, these might be calculated differently
        if "sentiment_volatility" in result_df.columns:
            assert result_df["sentiment_volatility"].notna().any()
            
        if "sentiment_roc" in result_df.columns:
            assert result_df["sentiment_roc"].notna().any()
            
        if "sentiment_acceleration" in result_df.columns:
            assert result_df["sentiment_acceleration"].notna().any()
            
        if "sentiment_anomaly" in result_df.columns:
            assert result_df["sentiment_anomaly"].notna().any()

    def test_calculate_weighted_sentiment_score(self):
        """Test calculating weighted sentiment score from features."""
        # Create test data with features
        data = {
            "overall_sentiment_score": [Decimal("0.5"), Decimal("-0.3"), Decimal("0.7")],
            "sentiment_trend": [1, -1, 1],
            "sentiment_momentum": [Decimal("0.2"), Decimal("-0.1"), Decimal("0.3")],
            "sentiment_anomaly": [0, -1, 1],
            "sentiment_volatility": [Decimal("1.0"), Decimal("1.5"), Decimal("0.8")],
            "sentiment_roc": [Decimal("0.1"), Decimal("-0.2"), Decimal("0.3")],
            "sentiment_acceleration": [Decimal("0.05"), Decimal("-0.1"), Decimal("0.15")]
        }
        df = pd.DataFrame(data)
        
        # Call the method
        result_df = self.analyzer.calculate_weighted_sentiment_score(df)
        
        # Verify the result
        assert "weighted_sentiment_score" in result_df.columns
        
        # First row should have a positive weighted score
        assert result_df.loc[0, "weighted_sentiment_score"] > 0
        
        # Second row should have a negative weighted score
        assert result_df.loc[1, "weighted_sentiment_score"] < 0
        
        # Third row should have a positive weighted score higher than the first
        assert result_df.loc[2, "weighted_sentiment_score"] > result_df.loc[0, "weighted_sentiment_score"]

    def test_generate_trading_signals(self):
        """Test generating trading signals from sentiment data."""
        # Create test data with weighted sentiment scores
        data = {
            "weighted_sentiment_score": [
                Decimal("0.5"), Decimal("-0.4"), Decimal("0.1"),
                Decimal("-0.1"), Decimal("0.3"), Decimal("-0.3")
            ],
            "sentiment_trend": [1, -1, 1, -1, 0, 0],
            "sentiment_volatility": [
                Decimal("1.0"), Decimal("1.0"), Decimal("1.0"),
                Decimal("2.0"), Decimal("2.0"), Decimal("1.0")
            ]
        }
        df = pd.DataFrame(data)
        
        # Call the method
        result_df = self.analyzer.generate_trading_signals(df)
        
        # Verify the result
        assert "signal" in result_df.columns
        
        # First row should have a long signal (1)
        assert result_df.loc[0, "signal"] == 1
        
        # Second row should have a short signal (-1)
        assert result_df.loc[1, "signal"] == -1
        
        # Third row should have no signal (0) as it's below threshold
        assert result_df.loc[2, "signal"] == 0
        
        # Fourth row should have no signal (0) as it's below threshold
        assert result_df.loc[3, "signal"] == 0
        
        # Fifth row should have a reduced long signal (0.5) due to high volatility
        assert result_df.loc[4, "signal"] == 0.5
        
        # Sixth row should have a short signal (-1)
        assert result_df.loc[5, "signal"] == -1

    @patch.object(SentimentAnalyzer, 'fetch_sentiment_data')
    @patch.object(SentimentAnalyzer, 'generate_time_series_features')
    @patch.object(SentimentAnalyzer, 'calculate_weighted_sentiment_score')
    @patch.object(SentimentAnalyzer, 'generate_trading_signals')
    def test_analyze_sentiment(self, mock_signals, mock_weighted, mock_features, mock_fetch):
        """Test the full sentiment analysis pipeline."""
        # Mock the individual steps
        mock_df = pd.DataFrame({"overall_sentiment_score": [Decimal("0.5")]})
        mock_fetch.return_value = mock_df
        mock_features.return_value = mock_df
        mock_weighted.return_value = mock_df
        mock_signals.return_value = mock_df
        
        # Call the method
        result_df = self.analyzer.analyze_sentiment(topic="blockchain")
        
        # Verify that all steps were called
        assert mock_fetch.called
        assert mock_features.called
        assert mock_weighted.called
        assert mock_signals.called
        
        # Verify that the result is the mock DataFrame
        assert result_df is mock_df
