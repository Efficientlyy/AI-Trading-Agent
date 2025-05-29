"""
Integration tests for LLM Oversight system.

This module contains tests to verify the integration between the LLM Oversight service
and the Adaptive Health Orchestrator.
"""

import os
import sys
import unittest
import json
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ai_trading_agent.oversight.client import OversightClient, OversightAction
from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
from ai_trading_agent.oversight.llm_oversight import LLMProvider, OversightLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockResponse:
    """Mock response object for requests."""
    
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


class TestLLMOversightClient(unittest.TestCase):
    """Test the LLM Oversight client functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.client = OversightClient(base_url="http://test-oversight-service", timeout=1)
    
    @patch('requests.Session.get')
    def test_check_health(self, mock_get):
        """Test the health check endpoint."""
        # Mock response for health check
        mock_get.return_value = MockResponse({"status": "healthy", "timestamp": 1621234567.89})
        
        # Call health check method
        result = self.client.check_health()
        
        # Verify health check request and response
        self.assertTrue(result)
        mock_get.assert_called_once_with(
            "http://test-oversight-service/health", 
            params=None, 
            timeout=1
        )
    
    @patch('requests.Session.post')
    def test_validate_trading_decision(self, mock_post):
        """Test the trading decision validation."""
        # Sample decision and context
        decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "price": 150.00,
            "confidence": 0.85
        }
        
        context = {
            "market_regime": {"global": {"regime_type": "BULL", "volatility_type": "LOW"}},
            "portfolio": {"value": 100000, "drawdown": 0.02}
        }
        
        # Mock response for validation
        mock_response_data = {
            "success": True,
            "result": {
                "action": "approve",
                "confidence": 0.92,
                "reason": "Decision aligns with current market conditions and risk parameters",
                "recommendation": "Consider increasing position size given strong bullish signals"
            },
            "processing_time": 0.5,
            "oversight_level": "advise",
            "timestamp": "2025-05-18 23:05:00"
        }
        mock_post.return_value = MockResponse(mock_response_data)
        
        # Call validate decision method
        result = self.client.validate_trading_decision(decision, context)
        
        # Verify validation request and response
        self.assertEqual(result, mock_response_data["result"])
        mock_post.assert_called_once()
        
    @patch('requests.Session.post')
    def test_get_decision_action(self, mock_post):
        """Test getting the decision action."""
        # Sample decision and context
        decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "price": 150.00
        }
        
        context = {
            "market_regime": {"global": {"regime_type": "BULL"}}
        }
        
        # Mock response with different actions
        mock_responses = [
            # Approve case
            {
                "success": True,
                "result": {"action": "approve", "confidence": 0.9},
                "oversight_level": "advise",
                "timestamp": "2025-05-18 23:05:00"
            },
            # Reject case
            {
                "success": True,
                "result": {"action": "reject", "confidence": 0.8},
                "oversight_level": "advise",
                "timestamp": "2025-05-18 23:05:00"
            },
            # Modify case
            {
                "success": True,
                "result": {"action": "modify", "confidence": 0.85},
                "oversight_level": "advise",
                "timestamp": "2025-05-18 23:05:00"
            }
        ]
        
        expected_actions = [
            OversightAction.APPROVE,
            OversightAction.REJECT,
            OversightAction.MODIFY
        ]
        
        # Test each action
        for i, mock_response in enumerate(mock_responses):
            mock_post.return_value = MockResponse(mock_response)
            action = self.client.get_decision_action(decision, context)
            self.assertEqual(action, expected_actions[i])
    
    @patch('requests.Session.post')
    def test_analyze_market_conditions(self, mock_post):
        """Test market condition analysis."""
        # Sample market data
        market_data = {
            "AAPL": {
                "close": [150.0, 152.0, 153.0, 151.0, 155.0],
                "volume": [10000000, 12000000, 9000000, 11000000, 15000000],
                "date": ["2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18"]
            },
            "MSFT": {
                "close": [290.0, 295.0, 298.0, 296.0, 300.0],
                "volume": [8000000, 9000000, 7500000, 8500000, 10000000],
                "date": ["2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18"]
            }
        }
        
        # Mock response for market analysis
        mock_response_data = {
            "success": True,
            "result": {
                "regime": "bull",
                "confidence": 0.85,
                "key_insights": [
                    "AAPL showing strong momentum with 3.3% gain over 5 days",
                    "MSFT breaking out above key resistance level at $298",
                    "Volume increasing across tech sector indicating institutional buying"
                ],
                "outlook": "Bullish trend likely to continue with 75% probability",
                "risk_factors": [
                    "Potential profit-taking after multi-day rally",
                    "Sector rotation risk if tech valuations are seen as stretched"
                ]
            },
            "processing_time": 0.8,
            "oversight_level": "advise",
            "timestamp": "2025-05-18 23:10:00"
        }
        mock_post.return_value = MockResponse(mock_response_data)
        
        # Call analyze market conditions method
        result = self.client.analyze_market_conditions(market_data)
        
        # Verify analysis request and response
        self.assertEqual(result, mock_response_data["result"])
        mock_post.assert_called_once()


class TestAdaptiveOrchestratorWithOversight(unittest.TestCase):
    """Test the integration of LLM Oversight with Adaptive Health Orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock oversight client
        self.mock_client = MagicMock()
        self.mock_client.check_health.return_value = True
        self.mock_client.get_config.return_value = {
            "oversight_level": "advise",
            "llm_provider": "openai"
        }
        
        # Create patch for OversightClient
        self.client_patch = patch('ai_trading_agent.oversight.client.OversightClient')
        self.mock_client_class = self.client_patch.start()
        self.mock_client_class.return_value = self.mock_client
        
        # Create orchestrator with LLM oversight enabled
        self.orchestrator = AdaptiveHealthOrchestrator(
            enable_llm_oversight=True,
            llm_oversight_service_url="http://test-oversight-service",
            llm_oversight_level="advise"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.client_patch.stop()
    
    def test_orchestrator_initialization_with_oversight(self):
        """Test that orchestrator initializes with LLM oversight."""
        self.assertTrue(self.orchestrator.enable_llm_oversight)
        self.assertEqual(self.orchestrator.llm_oversight_level, "advise")
        self.assertIsNotNone(self.orchestrator.oversight_client)
    
    def test_validate_trading_decision(self):
        """Test trading decision validation within orchestrator."""
        # Set up mock response for validation
        self.mock_client.validate_trading_decision.return_value = {
            "action": "approve",
            "confidence": 0.9,
            "reason": "Decision aligns with market conditions"
        }
        self.mock_client.get_decision_action.return_value = OversightAction.APPROVE
        
        # Sample decision and context
        decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "price": 155.00
        }
        
        # Call validate decision method
        is_approved, result = self.orchestrator.validate_trading_decision(decision)
        
        # Verify validation call and response
        self.assertTrue(is_approved)
        self.assertEqual(result["action"], "approve")
        self.mock_client.validate_trading_decision.assert_called_once()
    
    def test_run_cycle_with_oversight(self):
        """Test run cycle with LLM oversight."""
        # Mock market data
        self.orchestrator.market_data = {
            "AAPL": {
                "close": [150.0, 152.0, 155.0],
                "volume": [10000000, 12000000, 15000000]
            }
        }
        
        # Mock portfolio state
        self.orchestrator.portfolio_value = 100000
        self.orchestrator.portfolio_drawdown = 0.01
        
        # Set up mock response for market analysis
        self.mock_client.analyze_market_conditions.return_value = {
            "regime": "bull",
            "confidence": 0.85,
            "outlook": "Bullish trend likely to continue"
        }
        
        # Set up mock response for decision validation
        self.mock_client.validate_trading_decision.return_value = {
            "action": "approve",
            "confidence": 0.9
        }
        self.mock_client.get_decision_action.return_value = OversightAction.APPROVE
        
        # Mock the super().run_cycle call
        with patch.object(AdaptiveHealthOrchestrator, 'run_cycle', return_value={"status": "success"}):
            # Call run cycle
            result = self.orchestrator.run_cycle()
        
        # Verify LLM oversight info in result
        self.assertIn("llm_oversight", result)
        self.assertEqual(result["llm_oversight"]["level"], "advise")
    
    def test_cycle_with_rejected_decisions(self):
        """Test cycle with LLM rejecting decisions."""
        # Mock external inputs with trading decisions
        external_inputs = {
            "trend_strategy": [
                {
                    "symbol": "AAPL", 
                    "action": "buy", 
                    "quantity": 10, 
                    "price": 155.00
                }
            ]
        }
        
        # Set up mock to reject the decision
        self.mock_client.validate_trading_decision.return_value = {
            "action": "reject",
            "confidence": 0.9,
            "reason": "Decision contradicts current market regime"
        }
        self.mock_client.get_decision_action.return_value = OversightAction.REJECT
        
        # Mock the super().run_cycle call
        with patch.object(AdaptiveHealthOrchestrator, 'run_cycle', return_value={"status": "success"}) as mock_super_run:
            # Call run cycle with external inputs
            result = self.orchestrator.run_cycle(external_inputs)
            
            # Verify that run_cycle was called with empty list for that agent
            # since the decision was rejected
            super_call_args = mock_super_run.call_args[0][0]
            self.assertEqual(super_call_args["trend_strategy"], [])


# Run the tests if script is executed directly
if __name__ == '__main__':
    unittest.main()
