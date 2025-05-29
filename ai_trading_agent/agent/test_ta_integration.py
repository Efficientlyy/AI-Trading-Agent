"""
Unit tests for the enhanced Technical Analysis Agent integration.

This module provides tests to ensure that the enhanced TA agent properly
integrates with the existing agent ecosystem.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add the project root to the path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.agent.ta_agent_integration import EnhancedTechnicalAnalysisAgent
from ai_trading_agent.agent.agent_definitions import AgentStatus


class TestEnhancedTAIntegration(unittest.TestCase):
    """Tests for the enhanced Technical Analysis Agent integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Basic configuration for testing
        self.config = {
            "agent_id_suffix": "enhanced_ta",
            "name": "Enhanced Technical Analysis",
            "agent_type": "EnhancedTA",
            "symbols": ["BTC/USD", "ETH/USD"],
            "enabled": True,
            "advanced": {
                "enable": True,
                "strategies": [
                    {
                        "type": "MA_Cross",
                        "name": "MA_Cross_Standard",
                        "fast_period": 9,
                        "slow_period": 21,
                        "confirmation_periods": 2
                    },
                    {
                        "type": "RSI_OB_OS",
                        "name": "RSI_Standard",
                        "rsi_period": 14,
                        "overbought": 70,
                        "oversold": 30
                    }
                ],
                "indicators": [
                    {"name": "sma", "params": {"window": 9}},
                    {"name": "sma", "params": {"window": 21}},
                    {"name": "rsi", "params": {"window": 14}}
                ],
                "timeframes": ["1d"],  # Just use daily for testing
                "ml_validator": {
                    "min_confidence": 0.5  # Lower threshold for testing
                },
                "indicator_config": {}
            }
        }
        
        # Initialize agent
        self.agent = EnhancedTechnicalAnalysisAgent(self.config)
        
        # Generate test data
        self.market_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate test market data."""
        # Create sample date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate data for BTC/USD with an uptrend
        btc_data = pd.DataFrame(index=date_range)
        btc_trend = np.linspace(0, 0.5, len(date_range))  # Uptrend
        btc_noise = np.random.normal(0, 0.02, len(date_range))  # Random noise
        btc_returns = btc_trend + btc_noise
        btc_price = 50000 * (1 + btc_returns).cumprod()
        
        btc_data['open'] = btc_price
        btc_data['high'] = btc_price * 1.01
        btc_data['low'] = btc_price * 0.99
        btc_data['close'] = btc_price
        btc_data['volume'] = np.random.uniform(1000000, 2000000, len(date_range))
        
        # Generate data for ETH/USD with a sideways pattern
        eth_data = pd.DataFrame(index=date_range)
        eth_pattern = np.sin(np.linspace(0, 4*np.pi, len(date_range))) * 0.1  # Oscillating pattern
        eth_noise = np.random.normal(0, 0.025, len(date_range))  # Random noise
        eth_returns = eth_pattern + eth_noise
        eth_price = 3000 * (1 + eth_returns).cumprod()
        
        eth_data['open'] = eth_price
        eth_data['high'] = eth_price * 1.015
        eth_data['low'] = eth_price * 0.985
        eth_data['close'] = eth_price
        eth_data['volume'] = np.random.uniform(500000, 1500000, len(date_range))
        
        return {
            "BTC/USD": btc_data,
            "ETH/USD": eth_data
        }
    
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.status, AgentStatus.IDLE)
        self.assertTrue(self.agent.enable_advanced)
        self.assertIsNotNone(self.agent.advanced_agent)
        
        # Test capabilities
        capabilities = self.agent.get_capabilities()
        self.assertEqual(capabilities["name"], "Enhanced Technical Analysis Agent")
        self.assertTrue(capabilities["advanced_enabled"])
        self.assertIn("Multi-timeframe analysis", capabilities["features"])
        self.assertIn("Machine learning signal validation", capabilities["features"])
    
    def test_processing(self):
        """Test that the agent can process market data."""
        # Prepare test message
        message = {
            "market_data": self.market_data,
            "symbols": ["BTC/USD", "ETH/USD"]
        }
        
        # Process message
        signals = self.agent.process(message)
        
        # Verify that signals were generated
        self.assertTrue(len(signals) > 0, "No signals were generated")
        
        # Check signal structure
        for signal in signals:
            self.assertEqual(signal["type"], "technical_signal")
            self.assertIn("payload", signal)
            self.assertIn("symbol", signal["payload"])
            self.assertIn("signal", signal["payload"])
            self.assertIn("strategy", signal["payload"])
            self.assertIn("price_at_signal", signal["payload"])
            
            # Check for advanced features
            self.assertIn("confidence", signal["payload"])
            self.assertIn("validation", signal["payload"])
            
        # Verify agent status
        self.assertEqual(self.agent.status, AgentStatus.IDLE)
        
        # Verify metrics were updated
        self.assertGreater(self.agent.metrics["signals_generated"], 0)
        self.assertIn("signals_validated", self.agent.metrics)
        self.assertIn("current_market_regime", self.agent.metrics)
    
    def test_state_management(self):
        """Test saving and loading agent state."""
        # First process some data to have state to save
        message = {
            "market_data": self.market_data,
            "symbols": ["BTC/USD", "ETH/USD"]
        }
        
        self.agent.process(message)
        
        # Save state
        temp_dir = os.path.join(os.path.dirname(__file__), "test_state")
        success = self.agent.save_state(temp_dir)
        self.assertTrue(success)
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(temp_dir, "base_agent_state.json")))
        self.assertTrue(os.path.exists(os.path.join(temp_dir, "advanced_agent")))
        
        # Create a new agent
        new_agent = EnhancedTechnicalAnalysisAgent(self.config)
        
        # Load state
        success = new_agent.load_state(temp_dir)
        self.assertTrue(success)
        
        # Verify metrics were loaded
        self.assertEqual(
            self.agent.metrics["signals_generated"],
            new_agent.metrics["signals_generated"]
        )
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_fallback_mode(self):
        """Test that the agent falls back to base implementation when advanced features are disabled."""
        # Create agent with advanced features disabled
        config = self.config.copy()
        config["advanced"]["enable"] = False
        
        fallback_agent = EnhancedTechnicalAnalysisAgent(config)
        
        # Verify configuration
        self.assertFalse(fallback_agent.enable_advanced)
        self.assertIsNone(fallback_agent.advanced_agent)
        
        # Process data
        message = {
            "market_data": self.market_data,
            "symbols": ["BTC/USD", "ETH/USD"]
        }
        
        signals = fallback_agent.process(message)
        
        # Verify signals were still generated
        self.assertTrue(len(signals) > 0, "No signals were generated in fallback mode")
        
        # Capabilities should reflect disabled advanced features
        capabilities = fallback_agent.get_capabilities()
        self.assertFalse(capabilities["advanced_enabled"])


if __name__ == "__main__":
    unittest.main()
