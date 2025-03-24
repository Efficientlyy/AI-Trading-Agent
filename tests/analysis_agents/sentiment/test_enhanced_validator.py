"""Unit tests for the EnhancedSentimentValidator class.

This module contains tests for the enhanced sentiment validation system.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from src.analysis_agents.sentiment.enhanced_validator import (
    ContentFilter,
    SourceCredibilityTracker,
    EnhancedSentimentValidator
)


class TestContentFilter:
    """Tests for the ContentFilter class."""
    
    def test_filter_content_clean(self):
        """Test filtering clean content."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test clean content
        clean_content = "Bitcoin's fundamentals remain strong despite market volatility."
        is_acceptable, details = content_filter.filter_content(clean_content)
        
        # Verify result
        assert is_acceptable is True
        assert details["spam_score"] < 0.2
        assert details["manipulation_score"] < 0.2
        assert details["extreme_score"] < 0.2
        assert details["total_score"] < 0.5
        
    def test_filter_content_spam(self):
        """Test filtering spam content."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test spam content
        spam_content = "BUY NOW! Bitcoin to the moon! Guaranteed 100% profit! Join my group for more tips!"
        is_acceptable, details = content_filter.filter_content(spam_content)
        
        # Verify result
        assert is_acceptable is False
        assert details["spam_score"] > 0.5
        assert len(details["spam_matches"]) > 0
        assert "Multiple spam patterns" in details["reason"]
        
    def test_filter_content_manipulation(self):
        """Test filtering manipulation content."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test manipulation content
        manipulation_content = "Coordinated pump and dump attack on Bitcoin happening now. Bots are manipulating the price."
        is_acceptable, details = content_filter.filter_content(manipulation_content)
        
        # Verify result
        assert details["manipulation_score"] > 0.5
        assert len(details["manipulation_matches"]) > 0
        
    def test_filter_content_extreme(self):
        """Test filtering extreme sentiment content."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test extreme content
        extreme_content = "Bitcoin is the worst investment ever! It's going to zero! Completely worthless!"
        is_acceptable, details = content_filter.filter_content(extreme_content)
        
        # Verify result
        assert details["extreme_score"] > 0.5
        assert len(details["extreme_matches"]) > 0
        
    def test_banned_phrases(self):
        """Test filtering content with banned phrases."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test content with banned phrase
        banned_content = "This is not financial advice, but you should buy Bitcoin now."
        is_acceptable, details = content_filter.filter_content(banned_content)
        
        # Verify result
        assert is_acceptable is False
        assert "banned phrase" in details["reason"].lower()
        
    def test_confidence_adjustment(self):
        """Test confidence adjustment based on filter results."""
        # Create filter
        content_filter = ContentFilter()
        
        # Test with different content qualities
        clean_content = "Bitcoin's fundamentals remain strong despite market volatility."
        _, clean_details = content_filter.filter_content(clean_content)
        
        borderline_content = "Bitcoin might go to the moon soon! Price is looking very bullish!"
        _, borderline_details = content_filter.filter_content(borderline_content)
        
        problematic_content = "Bitcoin to the MOON! Buy now before it's too late! 1000x gains coming!"
        _, problematic_details = content_filter.filter_content(problematic_content)
        
        # Test confidence adjustments
        original_confidence = 0.8
        
        clean_adjusted = content_filter.adjust_confidence(original_confidence, clean_details)
        borderline_adjusted = content_filter.adjust_confidence(original_confidence, borderline_details)
        problematic_adjusted = content_filter.adjust_confidence(original_confidence, problematic_details)
        
        # Verify adjustments
        assert clean_adjusted == original_confidence  # No reduction for clean content
        assert borderline_adjusted <= original_confidence  # Some reduction for borderline
        assert problematic_adjusted < borderline_adjusted  # More reduction for problematic


class TestSourceCredibilityTracker:
    """Tests for the SourceCredibilityTracker class."""
    
    def test_initial_credibility(self):
        """Test initial credibility scores."""
        # Create tracker
        tracker = SourceCredibilityTracker()
        
        # Verify initial scores
        assert tracker.get_credibility("social_media") == 0.7
        assert tracker.get_credibility("news") == 0.8
        assert tracker.get_credibility("market_sentiment") == 0.9
        assert tracker.get_credibility("onchain") == 0.85
        
    def test_record_performance(self):
        """Test recording performance data."""
        # Create tracker
        tracker = SourceCredibilityTracker()
        
        # Record performance
        now = datetime.utcnow()
        tracker.record_performance(
            source="social_media",
            prediction_value=0.7,
            actual_value=0.8,
            prediction_time=now - timedelta(hours=2),
            actual_time=now
        )
        
        # Verify performance was recorded
        assert len(tracker.performance_history["social_media"]) == 1
        
        # Verify metadata was updated
        assert "last_update" in tracker.source_metadata["social_media"]
        
    def test_update_credibility_scores(self):
        """Test updating credibility scores based on performance."""
        # Create tracker
        tracker = SourceCredibilityTracker()
        
        # Record good performance for news
        now = datetime.utcnow()
        for i in range(10):
            tracker.record_performance(
                source="news",
                prediction_value=0.7,
                actual_value=0.7,  # Perfect prediction
                prediction_time=now - timedelta(hours=i+2),
                actual_time=now - timedelta(hours=i)
            )
            
        # Record poor performance for social_media
        for i in range(10):
            tracker.record_performance(
                source="social_media",
                prediction_value=0.7,
                actual_value=0.3,  # Poor prediction
                prediction_time=now - timedelta(hours=i+2),
                actual_time=now - timedelta(hours=i)
            )
            
        # Update credibility scores
        updated_scores = tracker.update_credibility_scores()
        
        # Verify scores were updated
        assert updated_scores["news"] > tracker.default_scores["news"]
        assert updated_scores["social_media"] < tracker.default_scores["social_media"]
        
    def test_source_metadata(self):
        """Test source metadata management."""
        # Create tracker
        tracker = SourceCredibilityTracker()
        
        # Get initial metadata
        initial_metadata = tracker.get_source_metadata("social_media")
        
        # Record performance to update metadata
        now = datetime.utcnow()
        for i in range(5):
            # Alternating predictions to create volatility
            prediction = 0.7 if i % 2 == 0 else 0.3
            tracker.record_performance(
                source="social_media",
                prediction_value=prediction,
                actual_value=0.5,
                prediction_time=now - timedelta(hours=i+2),
                actual_time=now - timedelta(hours=i)
            )
            
        # Get updated metadata
        updated_metadata = tracker.get_source_metadata("social_media")
        
        # Verify metadata was updated
        assert updated_metadata["last_update"] > initial_metadata["last_update"]
        assert "volatility" in updated_metadata
        assert "noise" in updated_metadata
        assert "lag" in updated_metadata


class TestEnhancedSentimentValidator:
    """Tests for the EnhancedSentimentValidator class."""
    
    def test_validate_sentiment_basic(self):
        """Test basic sentiment validation."""
        # Create validator
        validator = EnhancedSentimentValidator()
        
        # Validate sentiment
        is_valid, reason, adjusted_confidence, details = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="news",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Verify result
        assert is_valid is True
        assert reason == "Validation passed"
        assert "original_confidence" in details
        assert "adjustments" in details
        
    def test_validate_sentiment_with_content(self):
        """Test sentiment validation with content filtering."""
        # Create validator
        validator = EnhancedSentimentValidator()
        
        # Validate sentiment with clean content
        is_valid, reason, adjusted_confidence, details = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="social_media",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow(),
            content="Bitcoin's fundamentals remain strong despite market volatility."
        )
        
        # Verify result
        assert is_valid is True
        assert "content_filter" in details
        
        # Validate sentiment with problematic content
        is_valid, reason, adjusted_confidence, details = validator.validate_sentiment(
            symbol="BTC/USDT",
            source="social_media",
            sentiment_value=0.9,
            confidence=0.8,
            timestamp=datetime.utcnow(),
            content="BUY NOW! Bitcoin to the moon! Guaranteed 100% profit! Join my group for more tips!"
        )
        
        # Verify result
        assert is_valid is False
        assert "content_filter" in details
        assert "Content filtering failed" in reason
        
    def test_anomaly_detection(self):
        """Test anomaly detection in sentiment validation."""
        # Create validator
        validator = EnhancedSentimentValidator()
        
        # Add historical data
        symbol = "BTC/USDT"
        source = "news"
        now = datetime.utcnow()
        
        # Add consistent historical data
        for i in range(10):
            validator.validate_sentiment(
                symbol=symbol,
                source=source,
                sentiment_value=0.5,  # Neutral sentiment
                confidence=0.8,
                timestamp=now - timedelta(hours=i)
            )
            
        # Validate with anomalous value
        is_valid, reason, adjusted_confidence, details = validator.validate_sentiment(
            symbol=symbol,
            source=source,
            sentiment_value=0.9,  # Very bullish (anomaly)
            confidence=0.8,
            timestamp=now
        )
        
        # Verify anomaly detection
        assert "anomaly_detection" in details
        assert details["anomaly_detection"]["z_score"] > validator.anomaly_threshold
        
        # Validate with extreme anomaly
        is_valid, reason, adjusted_confidence, details = validator.validate_sentiment(
            symbol=symbol,
            source=source,
            sentiment_value=0.99,  # Extreme bullish (extreme anomaly)
            confidence=0.8,
            timestamp=now
        )
        
        # Verify extreme anomaly detection
        assert is_valid is False
        assert "Extreme anomaly detected" in reason
        
    def test_validation_stats(self):
        """Test validation statistics tracking."""
        # Create validator
        validator = EnhancedSentimentValidator()
        
        # Perform some validations
        validator.validate_sentiment(
            symbol="BTC/USDT",
            source="news",
            sentiment_value=0.7,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        validator.validate_sentiment(
            symbol="ETH/USDT",
            source="social_media",
            sentiment_value=0.3,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        # Get validation stats
        stats = validator.get_validation_stats()
        
        # Verify stats
        assert stats["total_processed"] = = 2
        assert "rejected_count" in stats
        assert "anomaly_count" in stats
        assert "adjusted_count" in stats
        assert "last_update" in stats
        
    def test_performance_recording(self):
        """Test performance recording for credibility tracking."""
        # Create validator
        validator = EnhancedSentimentValidator()
        
        # Record performance
        now = datetime.utcnow()
        validator.record_performance(
            source="news",
            prediction_value=0.7,
            actual_value=0.8,
            prediction_time=now - timedelta(hours=2),
            actual_time=now
        )
        
        # Update credibility scores
        updated_scores = validator.update_credibility_scores()
        
        # Verify scores were updated
        assert "news" in updated_scores
