"""Unit tests for the SentimentValidator class.

This module contains tests for the sentiment data validation system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.sentiment_validator import SentimentValidator


class TestSentimentValidator:
    """Tests for the SentimentValidator class."""
    
    def test_validate_sentiment_basic(self):
        """Test basic sentiment validation."""
        # Create validator
        validator = SentimentValidator()
        
        # Validate sentiment with default parameters
        is_valid, reason, adjusted_confidence = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="social_media",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result
        assert is_valid is True
        assert reason == "Validation passed"
        assert adjusted_confidence == 0.8  # Unchanged
    
    def test_validate_sentiment_low_credibility(self):
        """Test validation with low source credibility."""
        # Create validator with high minimum credibility
        validator = SentimentValidator(min_credibility=0.9)
        
        # Set social_media credibility to 0.7
        validator.source_credibility["social_media"] = 0.7
        
        # Validate sentiment
        is_valid, reason, adjusted_confidence = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="social_media",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result
        assert is_valid is False
        assert "credibility below threshold" in reason.lower()
        assert adjusted_confidence == 0.8  # Unchanged
    
    def test_validate_sentiment_anomaly(self):
        """Test validation with anomalous sentiment value."""
        # Create validator
        validator = SentimentValidator(anomaly_threshold=2.0)
        
        # Add historical data
        symbol = "BTC/USDT"
        source = "news"
        validator.historical_data[symbol] = {
            "values": [0.5, 0.52, 0.48, 0.51, 0.49],
            "timestamps": [datetime.utcnow() - timedelta(hours=i) for i in range(5, 0, -1)],
            "sources": [source] * 5
        }
        
        # Validate sentiment with anomalous value
        is_valid, reason, adjusted_confidence = validator.validate_sentiment(
            symbol=symbol,
            source=source,
            sentiment_value=0.9,  # Very different from historical values
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result
        assert adjusted_confidence < 0.8  # Confidence should be reduced
        
        # Validate with extreme anomaly
        is_valid, reason, adjusted_confidence = validator.validate_sentiment(
            symbol=symbol,
            source=source,
            sentiment_value=0.99,  # Extreme anomaly
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result
        assert is_valid is False
        assert "anomaly detected" in reason.lower()
        assert adjusted_confidence < 0.8  # Confidence should be reduced
    
    def test_historical_data_management(self):
        """Test historical data management."""
        # Create validator with small history size
        validator = SentimentValidator(history_size=3)
        
        # Add multiple data points
        symbol = "ETH/USDT"
        source = "social_media"
        
        for i in range(5):
            validator.validate_sentiment(
                symbol=symbol,
                source=source,
                sentiment_value=0.5 + (i * 0.1),
                confidence=0.8,
                timestamp=datetime.utcnow() - timedelta(hours=5-i)
            )
        
        # Verify historical data was limited to history_size
        assert len(validator.historical_data[symbol]["values"]) == 3
        assert len(validator.historical_data[symbol]["timestamps"]) == 3
        assert len(validator.historical_data[symbol]["sources"]) == 3
        
        # Verify we kept the most recent values
        assert validator.historical_data[symbol]["values"] == [0.7, 0.8, 0.9]
    
    def test_source_credibility_management(self):
        """Test source credibility management."""
        # Create validator
        validator = SentimentValidator()
        
        # Get initial credibility
        initial_cred = validator.get_source_credibility("social_media")
        
        # Update credibility
        validator.update_source_credibility("social_media", 0.9)
        
        # Verify update
        assert validator.get_source_credibility("social_media") == 0.9
        
        # Test bounds enforcement
        validator.update_source_credibility("social_media", 1.5)  # Above max
        assert validator.get_source_credibility("social_media") == 1.0
        
        validator.update_source_credibility("social_media", 0.05)  # Below min
        assert validator.get_source_credibility("social_media") == 0.1
    
    def test_unknown_source(self):
        """Test validation with unknown source."""
        # Create validator
        validator = SentimentValidator()
        
        # Validate sentiment with unknown source
        is_valid, reason, adjusted_confidence = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="unknown_source",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result uses default credibility
        assert validator.get_source_credibility("unknown_source") == 0.6  # Default value
        
        # Update unknown source credibility
        validator.update_source_credibility("unknown_source", 0.75)
        
        # Verify update
        assert validator.get_source_credibility("unknown_source") == 0.75
