"""
Tests for the Multi-Model Consensus System

This module contains unit tests for the consensus system that
combines sentiment signals from multiple sources and models.
"""

import asyncio
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.analysis_agents.sentiment.consensus_system import ConsensusSystem, MultiModelConsensusAgent
from src.common.logging import setup_logging


class TestConsensusSystem(unittest.TestCase):
    """Test the ConsensusSystem class."""
    
    def setUp(self):
        """Set up the test."""
        setup_logging()
        # Create a consensus system with test configuration
        self.consensus = ConsensusSystem("test")
        self.consensus.min_sources = 2
        self.consensus.confidence_threshold = 0.6
        self.consensus.max_age_hours = 24
        
        # Create test sentiment data
        self.sentiment_data = [
            {
                "value": 0.7,
                "direction": "bullish",
                "confidence": 0.8,
                "source_type": "llm",
                "model": "gpt-4",
                "timestamp": datetime.utcnow() - timedelta(minutes=30)
            },
            {
                "value": 0.65,
                "direction": "bullish",
                "confidence": 0.75,
                "source_type": "social_media",
                "model": "finbert",
                "timestamp": datetime.utcnow() - timedelta(hours=1)
            },
            {
                "value": 0.3,
                "direction": "bearish",
                "confidence": 0.7,
                "source_type": "news",
                "model": "distilbert",
                "timestamp": datetime.utcnow() - timedelta(hours=2)
            }
        ]
    
    def test_compute_consensus_basic(self):
        """Test basic consensus computation."""
        # Compute consensus
        result = self.consensus.compute_consensus(self.sentiment_data)
        
        # Check that the result contains expected fields
        self.assertIn("value", result)
        self.assertIn("direction", result)
        self.assertIn("confidence", result)
        self.assertIn("source_count", result)
        self.assertIn("unique_source_types", result)
        self.assertIn("disagreement_level", result)
        
        # Check values are in expected range
        self.assertTrue(0 <= result["value"] <= 1)
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertTrue(0 <= result["disagreement_level"] <= 1)
        
        # Check source count
        self.assertEqual(result["source_count"], 3)
        self.assertEqual(result["unique_source_types"], 3)
    
    def test_compute_consensus_with_insufficient_data(self):
        """Test consensus with insufficient data."""
        # Test with empty data
        result = self.consensus.compute_consensus([])
        self.assertEqual(result["value"], 0.5)  # Should return default neutral
        self.assertEqual(result["direction"], "neutral")
        self.assertEqual(result["confidence"], 0.0)
        
        # Test with single data point (below min_sources)
        result = self.consensus.compute_consensus([self.sentiment_data[0]])
        self.assertEqual(result["value"], 0.5)  # Should return default neutral
        self.assertEqual(result["confidence"], 0.0)
    
    def test_compute_consensus_with_old_data(self):
        """Test consensus with old data that should be filtered out."""
        # Create data with old timestamps
        old_data = self.sentiment_data.copy()
        old_data.append({
            "value": 0.8,
            "direction": "bullish",
            "confidence": 0.9,
            "source_type": "market",
            "model": "fear_greed",
            "timestamp": datetime.utcnow() - timedelta(hours=48)  # Older than max_age_hours
        })
        
        # Compute consensus
        result = self.consensus.compute_consensus(old_data)
        
        # The old data should be filtered out
        self.assertEqual(result["source_count"], 3)
    
    def test_compute_consensus_with_low_confidence(self):
        """Test consensus with low confidence data that should be filtered out."""
        # Create data with low confidence
        low_confidence_data = self.sentiment_data.copy()
        low_confidence_data.append({
            "value": 0.8,
            "direction": "bullish",
            "confidence": 0.4,  # Below confidence threshold
            "source_type": "market",
            "model": "fear_greed",
            "timestamp": datetime.utcnow()
        })
        
        # Compute consensus
        result = self.consensus.compute_consensus(low_confidence_data)
        
        # The low confidence data should be filtered out
        self.assertEqual(result["source_count"], 3)
    
    def test_calculate_disagreement(self):
        """Test the disagreement calculation."""
        # Test with uniform values (no disagreement)
        uniform_values = np.array([0.7, 0.7, 0.7])
        disagreement = self.consensus._calculate_disagreement(uniform_values)
        self.assertEqual(disagreement, 0.0)
        
        # Test with very different values (high disagreement)
        diverse_values = np.array([0.1, 0.5, 0.9])
        disagreement = self.consensus._calculate_disagreement(diverse_values)
        self.assertGreater(disagreement, 0.5)
    
    def test_bayesian_aggregation(self):
        """Test the Bayesian aggregation method."""
        # Create test data
        values = np.array([0.7, 0.3, 0.8])
        confidences = np.array([0.8, 0.6, 0.9])
        weights = np.array([1.0, 0.8, 0.9])
        
        # Perform Bayesian aggregation
        value, confidence = self.consensus._bayesian_aggregation(values, confidences, weights)
        
        # Check results
        self.assertTrue(0 <= value <= 1)
        self.assertTrue(0 <= confidence <= 1)
    
    def test_record_performance(self):
        """Test recording and retrieving performance metrics."""
        # Record some performance data
        self.consensus.record_performance(
            source_type="llm",
            model="gpt-4",
            prediction=0.7,
            actual_outcome=0.75,
            timestamp=datetime.utcnow()
        )
        
        self.consensus.record_performance(
            source_type="social_media",
            model="finbert",
            prediction=0.6,
            actual_outcome=0.4,
            timestamp=datetime.utcnow()
        )
        
        # Get performance metrics
        performance = self.consensus.get_model_performance()
        
        # Check that the metrics exist
        self.assertIn("llm:gpt-4", performance)
        self.assertIn("social_media:finbert", performance)
        
        # Check that values are in the expected range
        self.assertTrue(0 <= performance["llm:gpt-4"] <= 1)
        self.assertTrue(0 <= performance["social_media:finbert"] <= 1)
        
        # Check filtering
        llm_performance = self.consensus.get_model_performance(source_type="llm")
        self.assertIn("llm:gpt-4", llm_performance)
        self.assertNotIn("social_media:finbert", llm_performance)


@pytest.mark.asyncio
class TestMultiModelConsensusAgent:
    """Test the MultiModelConsensusAgent class."""
    
    @pytest.fixture
    async def agent(self):
        """Create and initialize a test agent."""
        agent = MultiModelConsensusAgent("test")
        # Mock components to avoid actual initialization
        agent.consensus = MagicMock()
        agent._connect_consensus_agent = MagicMock()
        agent.publish_sentiment_event = MagicMock()
        
        # Quick patch to make the agent think it's initialized & started
        agent.initialized = True
        agent.running = True
        
        yield agent
        
        # Clean up
        if hasattr(agent, "update_task") and agent.update_task:
            agent.update_task.cancel()
            try:
                await agent.update_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_submit_sentiment(self, agent):
        """Test submitting sentiment data."""
        # Submit some test sentiment
        await agent.submit_sentiment(
            symbol="BTC/USDT",
            value=0.7,
            direction="bullish",
            confidence=0.8,
            source_type="llm",
            model="gpt-4",
            metadata={"explanation": "Test explanation"}
        )
        
        # Check that sentiment points were stored
        assert "BTC/USDT" in agent.sentiment_points
        assert len(agent.sentiment_points["BTC/USDT"]) == 1
        
        # Check data values
        data = agent.sentiment_points["BTC/USDT"][0]
        assert data["value"] = = 0.7
        assert data["direction"] = = "bullish"
        assert data["confidence"] = = 0.8
        assert data["source_type"] = = "llm"
        assert data["model"] = = "gpt-4"
        assert "explanation" in data
    
    @pytest.mark.asyncio
    async def test_process_consensus(self, agent):
        """Test processing consensus data."""
        # Setup
        agent.sentiment_points["BTC/USDT"] = [
            {
                "value": 0.7,
                "direction": "bullish",
                "confidence": 0.8,
                "source_type": "llm",
                "model": "gpt-4",
                "timestamp": datetime.utcnow()
            },
            {
                "value": 0.65,
                "direction": "bullish",
                "confidence": 0.75,
                "source_type": "social_media",
                "model": "finbert",
                "timestamp": datetime.utcnow()
            }
        ]
        
        # Mock consensus result
        mock_result = {
            "value": 0.68,
            "direction": "bullish",
            "confidence": 0.85,
            "source_count": 2,
            "unique_source_types": 2,
            "unique_models": 2,
            "disagreement_level": 0.1,
            "source_types": ["llm", "social_media"],
            "models": ["gpt-4", "finbert"],
            "direction_counts": {"bullish": 2, "bearish": 0, "neutral": 0},
            "last_update": datetime.utcnow()
        }
        agent.consensus.compute_consensus.return_value = mock_result
        
        # Process consensus
        await agent._process_consensus("BTC/USDT")
        
        # Check that consensus was computed
        agent.consensus.compute_consensus.assert_called_once()
        
        # Check that consensus cache was updated
        assert "BTC/USDT" in agent.consensus_cache
        assert agent.consensus_cache["BTC/USDT"] == mock_result
        
        # Check that event was published for high confidence
        agent.publish_sentiment_event.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_get_consensus(self, agent):
        """Test getting consensus results."""
        # Setup
        agent.consensus_cache["BTC/USDT"] = {
            "value": 0.68,
            "direction": "bullish",
            "confidence": 0.85,
            "source_count": 2,
            "disagreement_level": 0.1
        }
        
        # Get consensus
        result = agent.get_consensus("BTC/USDT")
        
        # Check result
        assert result is not None
        assert result["value"] = = 0.68
        assert result["direction"] = = "bullish"
        
        # Test for non-existent symbol
        result = agent.get_consensus("ETH/USDT")
        assert result is None


if __name__ == '__main__':
    unittest.main()