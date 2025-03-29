"""Unit tests for the AdaptiveSentimentWeights class.

This module contains tests for the adaptive weights system that learns from past performance.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.adaptive_weights import AdaptiveSentimentWeights


class TestAdaptiveSentimentWeights:
    """Tests for the AdaptiveSentimentWeights class."""
    
    def test_initial_weights(self):
        """Test initial weights configuration."""
        # Create adaptive weights
        adaptive_weights = AdaptiveSentimentWeights()
        
        # Get initial weights
        weights = adaptive_weights.get_weights()
        
        # Verify default weights
        assert weights["social_media"] = = 0.25
        assert weights["news"] = = 0.25
        assert weights["market_sentiment"] = = 0.3
        assert weights["onchain"] = = 0.2
        
        # Verify weights sum to 1
        assert sum(weights.values()) == 1.0
    
    def test_record_performance(self):
        """Test recording performance data."""
        # Create adaptive weights
        adaptive_weights = AdaptiveSentimentWeights()
        
        # Record performance for a source
        now = datetime.utcnow()
        adaptive_weights.record_performance(
            source="social_media",
            prediction=0.7,  # Bullish prediction
            actual_outcome=0.8,  # Strong positive outcome
            timestamp=now
        )
        
        # Verify performance was recorded
        assert len(adaptive_weights.source_performance["social_media"]) == 1
        perf = adaptive_weights.source_performance["social_media"][0]
        assert perf["prediction"] = = 0.7
        assert perf["actual"] = = 0.8
        assert perf["timestamp"] = = now
        assert perf["accuracy"] > 0.9  # High accuracy for close prediction
    
    def test_performance_window_cleanup(self):
        """Test that old performance data is cleaned up."""
        # Create adaptive weights with short window
        adaptive_weights = AdaptiveSentimentWeights(performance_window=5)
        
        # Add performance data with various timestamps
        now = datetime.utcnow()
        
        # Recent data (within window)
        for i in range(3):
            adaptive_weights.record_performance(
                source="news",
                prediction=0.6,
                actual_outcome=0.6,
                timestamp=now - timedelta(days=i)
            )
        
        # Old data (outside window)
        for i in range(3):
            adaptive_weights.record_performance(
                source="news",
                prediction=0.4,
                actual_outcome=0.4,
                timestamp=now - timedelta(days=i+10)
            )
        
        # Verify only recent data is kept
        assert len(adaptive_weights.source_performance["news"]) == 3
        
        # Verify all timestamps are within the window
        for perf in adaptive_weights.source_performance["news"]:
            assert (now - perf["timestamp"]).days < 6
    
    def test_weight_adjustment(self):
        """Test weight adjustment based on performance."""
        # Create adaptive weights with high learning rate for testing
        adaptive_weights = AdaptiveSentimentWeights(
            learning_rate=0.1,
            min_samples=2
        )
        
        # Record good performance for social_media
        now = datetime.utcnow()
        for i in range(5):
            adaptive_weights.record_performance(
                source="social_media",
                prediction=0.7,
                actual_outcome=0.7,  # Perfect prediction
                timestamp=now - timedelta(hours=i)
            )
        
        # Record poor performance for news
        for i in range(5):
            adaptive_weights.record_performance(
                source="news",
                prediction=0.7,
                actual_outcome=0.3,  # Poor prediction
                timestamp=now - timedelta(hours=i)
            )
        
        # Record neutral performance for other sources
        for source in ["market_sentiment", "onchain"]:
            for i in range(5):
                adaptive_weights.record_performance(
                    source=source,
                    prediction=0.5,
                    actual_outcome=0.5,
                    timestamp=now - timedelta(hours=i)
                )
        
        # Update weights
        new_weights = adaptive_weights.update_weights()
        
        # Verify weights were adjusted
        assert new_weights["social_media"] > 0.25  # Should increase
        assert new_weights["news"] < 0.25  # Should decrease
        
        # Verify weights still sum to 1
        assert round(sum(new_weights.values()), 10) == 1.0
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        # Create adaptive weights
        adaptive_weights = AdaptiveSentimentWeights()
        
        # Manually set weights to invalid values
        adaptive_weights.source_weights = {
            "social_media": 0.4,
            "news": 0.4,
            "market_sentiment": 0.4,
            "onchain": 0.4
        }
        
        # Update weights
        new_weights = adaptive_weights.update_weights()
        
        # Verify weights sum to 1
        assert round(sum(new_weights.values()), 10) == 1.0
    
    def test_min_samples_requirement(self):
        """Test that weights aren't adjusted without minimum samples."""
        # Create adaptive weights with high min_samples
        adaptive_weights = AdaptiveSentimentWeights(min_samples=10)
        
        # Record just a few performance points
        now = datetime.utcnow()
        for i in range(5):
            for source in adaptive_weights.source_performance.keys():
                adaptive_weights.record_performance(
                    source=source,
                    prediction=0.7,
                    actual_outcome=0.7 if source == "social_media" else 0.3,
                    timestamp=now - timedelta(hours=i)
                )
        
        # Get initial weights
        initial_weights = adaptive_weights.source_weights.copy()
        
        # Update weights
        new_weights = adaptive_weights.update_weights()
        
        # Verify weights weren't changed (not enough samples)
        assert new_weights == initial_weights
    
    def test_time_decay(self):
        """Test that more recent performance has higher weight."""
        # Create adaptive weights with strong decay
        adaptive_weights = AdaptiveSentimentWeights(
            decay_factor=0.5,
            min_samples=2
        )
        
        # Record mixed performance for social_media with different timestamps
        now = datetime.utcnow()
        
        # Recent good performance
        for i in range(3):
            adaptive_weights.record_performance(
                source="social_media",
                prediction=0.7,
                actual_outcome=0.7,  # Good prediction
                timestamp=now - timedelta(hours=i)
            )
        
        # Older bad performance
        for i in range(3):
            adaptive_weights.record_performance(
                source="social_media",
                prediction=0.7,
                actual_outcome=0.3,  # Bad prediction
                timestamp=now - timedelta(days=i+1)
            )
        
        # Add minimum samples for other sources
        for source in ["news", "market_sentiment", "onchain"]:
            for i in range(3):
                adaptive_weights.record_performance(
                    source=source,
                    prediction=0.5,
                    actual_outcome=0.5,
                    timestamp=now - timedelta(hours=i)
                )
        
        # Update weights
        new_weights = adaptive_weights.update_weights()
        
        # Verify social_media weight increased despite mixed performance
        # (recent good performance should outweigh older bad performance)
        assert new_weights["social_media"] > 0.25
