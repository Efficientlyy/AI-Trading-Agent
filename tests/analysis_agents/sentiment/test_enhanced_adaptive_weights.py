"""Unit tests for the EnhancedAdaptiveWeights class.

This module contains tests for the enhanced adaptive learning system.
"""

import pytest
import os
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.analysis_agents.sentiment.enhanced_adaptive_weights import EnhancedAdaptiveWeights


@pytest.fixture
def adaptive_weights():
    """Create an EnhancedAdaptiveWeights instance for testing."""
    # Use a temporary directory for visualizations
    test_dir = "/tmp/test_sentiment_visualization"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create with test parameters
    weights = EnhancedAdaptiveWeights(
        learning_rate=0.05,  # Higher for testing
        performance_window=10,
        min_samples=3,  # Lower for testing
        visualization_dir=test_dir
    )
    
    return weights


class TestEnhancedAdaptiveWeights:
    """Tests for the EnhancedAdaptiveWeights class."""
    
    def test_initial_weights(self, adaptive_weights):
        """Test initial weights configuration."""
        # Get initial weights
        weights = adaptive_weights.get_weights()
        
        # Verify default weights
        assert weights["social_media"] = = 0.25
        assert weights["news"] = = 0.25
        assert weights["market_sentiment"] = = 0.3
        assert weights["onchain"] = = 0.2
        
        # Verify weights sum to 1
        assert sum(weights.values()) == 1.0
        
        # Verify market condition weights
        for condition in ["bullish", "bearish", "neutral", "volatile"]:
            condition_weights = adaptive_weights.get_weights(market_condition=condition)
            assert sum(condition_weights.values()) == 1.0
    
    def test_record_performance(self, adaptive_weights):
        """Test recording performance data."""
        # Record performance for a source and symbol
        now = datetime.utcnow()
        adaptive_weights.record_performance(
            source="social_media",
            symbol="BTC/USDT",
            prediction=0.7,  # Bullish prediction
            actual_outcome=0.8,  # Strong positive outcome
            timestamp=now,
            market_condition="bullish"
        )
        
        # Verify performance was recorded
        assert "social_media" in adaptive_weights.source_performance
        assert "BTC/USDT" in adaptive_weights.source_performance["social_media"]
        assert len(adaptive_weights.source_performance["social_media"]["BTC/USDT"]) == 1
        
        # Verify performance metrics were updated
        metrics = adaptive_weights.get_performance_metrics()
        assert metrics["global_accuracy"] > 0
        assert "social_media" in metrics["by_source"]
        assert "BTC/USDT" in metrics["by_symbol"]
    
    def test_update_global_weights(self, adaptive_weights):
        """Test updating global weights based on performance."""
        # Record good performance for social_media
        now = datetime.utcnow()
        for i in range(5):
            adaptive_weights.record_performance(
                source="social_media",
                symbol="BTC/USDT",
                prediction=0.7,
                actual_outcome=0.7,  # Perfect prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bullish"
            )
        
        # Record poor performance for news
        for i in range(5):
            adaptive_weights.record_performance(
                source="news",
                symbol="BTC/USDT",
                prediction=0.7,
                actual_outcome=0.3,  # Poor prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bullish"
            )
        
        # Record neutral performance for other sources
        for source in ["market_sentiment", "onchain"]:
            for i in range(5):
                adaptive_weights.record_performance(
                    source=source,
                    symbol="BTC/USDT",
                    prediction=0.5,
                    actual_outcome=0.5,
                    timestamp=now - timedelta(hours=i),
                    market_condition="neutral"
                )
        
        # Get initial weights
        initial_weights = adaptive_weights.source_weights.copy()
        
        # Update weights
        new_weights = adaptive_weights.update_weights()
        
        # Verify weights were adjusted
        assert new_weights["social_media"] > initial_weights["social_media"]
        assert new_weights["news"] < initial_weights["news"]
        
        # Verify weights still sum to 1
        assert round(sum(new_weights.values()), 10) == 1.0
        
        # Verify weight history was recorded
        assert len(adaptive_weights.weight_history["global"]) > 0
    
    def test_symbol_specific_weights(self, adaptive_weights):
        """Test symbol-specific weight adjustments."""
        # Record different performance for different symbols
        now = datetime.utcnow()
        
        # Good performance for social_media on BTC
        for i in range(5):
            adaptive_weights.record_performance(
                source="social_media",
                symbol="BTC/USDT",
                prediction=0.7,
                actual_outcome=0.7,  # Good prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bullish"
            )
        
        # Good performance for news on ETH
        for i in range(5):
            adaptive_weights.record_performance(
                source="news",
                symbol="ETH/USDT",
                prediction=0.7,
                actual_outcome=0.7,  # Good prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bullish"
            )
        
        # Add minimum data for other sources
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            for source in adaptive_weights.source_weights.keys():
                if (source == "social_media" and symbol == "BTC/USDT") or (source == "news" and symbol == "ETH/USDT"):
                    continue  # Already added
                
                for i in range(5):
                    adaptive_weights.record_performance(
                        source=source,
                        symbol=symbol,
                        prediction=0.5,
                        actual_outcome=0.5,
                        timestamp=now - timedelta(hours=i),
                        market_condition="neutral"
                    )
        
        # Update weights for BTC
        btc_weights = adaptive_weights.update_weights(symbol="BTC/USDT")
        
        # Update weights for ETH
        eth_weights = adaptive_weights.update_weights(symbol="ETH/USDT")
        
        # Verify symbol-specific weights
        assert btc_weights["social_media"] > adaptive_weights.source_weights["social_media"]
        assert eth_weights["news"] > adaptive_weights.source_weights["news"]
        
        # Verify weights still sum to 1
        assert round(sum(btc_weights.values()), 10) == 1.0
        assert round(sum(eth_weights.values()), 10) == 1.0
        
        # Verify weight history was recorded
        assert "BTC/USDT" in adaptive_weights.weight_history["by_symbol"]
        assert "ETH/USDT" in adaptive_weights.weight_history["by_symbol"]
    
    def test_market_condition_weights(self, adaptive_weights):
        """Test market condition-specific weight adjustments."""
        # Record different performance for different market conditions
        now = datetime.utcnow()
        
        # Good performance for social_media in bullish markets
        for i in range(5):
            adaptive_weights.record_performance(
                source="social_media",
                symbol="BTC/USDT",
                prediction=0.7,
                actual_outcome=0.7,  # Good prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bullish"
            )
        
        # Good performance for news in bearish markets
        for i in range(5):
            adaptive_weights.record_performance(
                source="news",
                symbol="BTC/USDT",
                prediction=0.3,
                actual_outcome=0.3,  # Good prediction
                timestamp=now - timedelta(hours=i),
                market_condition="bearish"
            )
        
        # Add minimum data for other sources and conditions
        for condition in ["bullish", "bearish"]:
            for source in adaptive_weights.source_weights.keys():
                if (source == "social_media" and condition == "bullish") or (source == "news" and condition == "bearish"):
                    continue  # Already added
                
                for i in range(5):
                    adaptive_weights.record_performance(
                        source=source,
                        symbol="BTC/USDT",
                        prediction=0.5,
                        actual_outcome=0.5,
                        timestamp=now - timedelta(hours=i),
                        market_condition=condition
                    )
        
        # Update weights for bullish condition
        bullish_weights = adaptive_weights.update_weights(market_condition="bullish")
        
        # Update weights for bearish condition
        bearish_weights = adaptive_weights.update_weights(market_condition="bearish")
        
        # Verify condition-specific weights
        assert bullish_weights["social_media"] > adaptive_weights.source_weights["social_media"]
        assert bearish_weights["news"] > adaptive_weights.source_weights["news"]
        
        # Verify weights still sum to 1
        assert round(sum(bullish_weights.values()), 10) == 1.0
        assert round(sum(bearish_weights.values()), 10) == 1.0
        
        # Verify weight history was recorded
        assert "bullish" in adaptive_weights.weight_history["by_market_condition"]
        assert "bearish" in adaptive_weights.weight_history["by_market_condition"]
    
    def test_performance_metrics(self, adaptive_weights):
        """Test performance metrics calculation."""
        # Record mixed performance
        now = datetime.utcnow()
        
        # Different performance for different sources
        performances = {
            "social_media": {"accuracy": 0.9, "count": 5},
            "news": {"accuracy": 0.7, "count": 5},
            "market_sentiment": {"accuracy": 0.5, "count": 5},
            "onchain": {"accuracy": 0.8, "count": 5}
        }
        
        for source, perf in performances.items():
            for i in range(perf["count"]):
                adaptive_weights.record_performance(
                    source=source,
                    symbol="BTC/USDT",
                    prediction=0.5,
                    actual_outcome=0.5 + (perf["accuracy"] - 0.5) * 0.5,  # Scale to create desired accuracy
                    timestamp=now - timedelta(hours=i),
                    market_condition="neutral"
                )
        
        # Get performance metrics
        metrics = adaptive_weights.get_performance_metrics()
        
        # Verify global accuracy
        assert 0.7 <= metrics["global_accuracy"] <= 0.8  # Should be average of all sources
        
        # Verify source-specific accuracy
        for source, perf in performances.items():
            assert abs(metrics["by_source"][source]["accuracy"] - perf["accuracy"]) < 0.1
            assert metrics["by_source"][source]["sample_count"] == perf["count"]
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_weights(self, mock_savefig, adaptive_weights):
        """Test weight visualization."""
        # Record some performance to create weight history
        now = datetime.utcnow()
        
        # Add performance data
        for i in range(5):
            for source in adaptive_weights.source_weights.keys():
                adaptive_weights.record_performance(
                    source=source,
                    symbol="BTC/USDT",
                    prediction=0.5,
                    actual_outcome=0.5 + (0.1 * (source == "social_media")),  # Better for social_media
                    timestamp=now - timedelta(days=i),
                    market_condition="neutral"
                )
        
        # Update weights multiple times to create history
        for i in range(3):
            adaptive_weights.update_weights()
        
        # Test global visualization
        output_file = adaptive_weights.visualize_weights()
        
        # Verify visualization was created
        mock_savefig.assert_called()
        assert output_file.endswith(".png")
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_performance(self, mock_savefig, adaptive_weights):
        """Test performance visualization."""
        # Record some performance
        now = datetime.utcnow()
        
        # Add performance data
        for i in range(5):
            for source in adaptive_weights.source_weights.keys():
                adaptive_weights.record_performance(
                    source=source,
                    symbol="BTC/USDT",
                    prediction=0.5,
                    actual_outcome=0.5 + (0.1 * (source == "social_media")),  # Better for social_media
                    timestamp=now - timedelta(days=i),
                    market_condition="neutral"
                )
        
        # Test performance visualization
        output_file = adaptive_weights.visualize_performance(by_source=True, by_market_condition=True)
        
        # Verify visualization was created
        mock_savefig.assert_called()
        assert output_file.endswith(".png")
    
    def test_export_import_weights(self, adaptive_weights):
        """Test exporting and importing weights."""
        # Record some performance and update weights
        now = datetime.utcnow()
        
        # Add performance data
        for i in range(5):
            for source in adaptive_weights.source_weights.keys():
                adaptive_weights.record_performance(
                    source=source,
                    symbol="BTC/USDT",
                    prediction=0.5,
                    actual_outcome=0.5 + (0.1 * (source == "social_media")),  # Better for social_media
                    timestamp=now - timedelta(days=i),
                    market_condition="neutral"
                )
        
        # Update weights
        adaptive_weights.update_weights()
        
        # Export weights
        export_file = adaptive_weights.export_weights()
        
        # Create a new instance
        new_weights = EnhancedAdaptiveWeights(visualization_dir="/tmp/test_sentiment_visualization")
        
        # Import weights
        success = new_weights.import_weights(export_file)
        
        # Verify import
        assert success is True
        assert new_weights.source_weights == adaptive_weights.source_weights
        
        # Verify symbol weights were imported
        for symbol in adaptive_weights.symbol_weights:
            assert symbol in new_weights.symbol_weights
            assert new_weights.symbol_weights[symbol] == adaptive_weights.symbol_weights[symbol]
        
        # Verify market condition weights were imported
        for condition in adaptive_weights.market_condition_weights:
            assert condition in new_weights.market_condition_weights
            assert new_weights.market_condition_weights[condition] == adaptive_weights.market_condition_weights[condition]
