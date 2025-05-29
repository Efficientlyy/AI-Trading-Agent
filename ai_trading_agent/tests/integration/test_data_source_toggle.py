"""
Integration tests for the mock/real data source toggle functionality.

These tests validate that the data source toggle works correctly with the 
Technical Analysis Agent and other components in an integrated environment.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.config.data_source_config import get_data_source_config
from ai_trading_agent.data.data_source_factory import get_data_source_factory
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent


class TestDataSourceToggleIntegration(unittest.TestCase):
    """Test the integration of the data source toggle with other components."""
    
    def setUp(self):
        """Set up test environment."""
        # Get global instances
        self.data_source_config = get_data_source_config()
        self.data_source_factory = get_data_source_factory()
        
        # Store original state to restore after tests
        self.original_state = self.data_source_config.use_mock_data
        
        # Create agent configuration for testing
        self.agent_config = {
            "strategies": [
                {
                    "name": "Test Strategy",
                    "indicators": ["rsi", "macd", "bollinger_bands"],
                    "timeframes": ["1h", "4h"],
                    "parameters": {
                        "rsi_period": 14,
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "macd_signal": 9,
                        "bb_period": 20,
                        "bb_stddev": 2
                    }
                }
            ],
            "timeframes": ["1h", "4h", "1d"],
            "ml_validator": {
                "enabled": True,
                "min_confidence": 0.65
            },
            "data_source": {
                "use_mock_data": True,  # Start with mock data
                "mock_data_settings": {
                    "volatility": 0.02,
                    "trend_strength": 0.4,
                    "seed": 42
                }
            }
        }
        
        # Initialize the technical analysis agent
        self.agent = AdvancedTechnicalAnalysisAgent(self.agent_config)
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original data source setting
        self.data_source_config.use_mock_data = self.original_state
    
    def test_data_source_config_singleton(self):
        """Test that data source config is correctly shared as a singleton."""
        # Get a new instance and verify it's the same object
        another_config = get_data_source_config()
        self.assertIs(self.data_source_config, another_config, 
                      "DataSourceConfig should be a singleton")
        
        # Test that changes propagate
        test_value = not self.data_source_config.use_mock_data
        self.data_source_config.use_mock_data = test_value
        self.assertEqual(another_config.use_mock_data, test_value,
                         "Changes to one instance should affect all instances")
    
    def test_data_source_factory_singleton(self):
        """Test that data source factory is correctly shared as a singleton."""
        # Get a new instance and verify it's the same object
        another_factory = get_data_source_factory()
        self.assertIs(self.data_source_factory, another_factory,
                      "DataSourceFactory should be a singleton")
    
    def test_factory_provides_correct_source(self):
        """Test that the factory provides the correct data source based on configuration."""
        # Set to mock data
        self.data_source_config.use_mock_data = True
        provider = self.data_source_factory.get_data_provider()
        self.assertIn("MockDataGenerator", provider.__class__.__name__, 
                      "Factory should provide MockDataGenerator when use_mock_data is True")
        
        # Set to real data
        self.data_source_config.use_mock_data = False
        provider = self.data_source_factory.get_data_provider()
        self.assertIn("MarketDataProvider", provider.__class__.__name__,
                      "Factory should provide MarketDataProvider when use_mock_data is False")
    
    def test_agent_data_source_integration(self):
        """Test that the technical analysis agent correctly integrates with the data source toggle."""
        # Check initial state (should be mock from config)
        self.assertEqual(self.agent.get_data_source_type(), "mock",
                         "Agent should start with mock data as specified in config")
        
        # Toggle to real data
        new_source = self.agent.toggle_data_source()
        self.assertEqual(new_source, "real", "Toggle should return 'real' after changing from mock")
        self.assertEqual(self.agent.get_data_source_type(), "real",
                         "Agent should report 'real' after toggle")
        
        # Toggle back to mock
        new_source = self.agent.toggle_data_source()
        self.assertEqual(new_source, "mock", "Toggle should return 'mock' after changing from real")
        self.assertEqual(self.agent.get_data_source_type(), "mock",
                         "Agent should report 'mock' after toggle")
    
    def test_agent_metrics_include_data_source(self):
        """Test that agent metrics include the current data source type."""
        # Get metrics with mock data
        self.data_source_config.use_mock_data = True
        metrics = self.agent.get_metrics()
        self.assertIn("data_source", metrics, "Metrics should include data_source field")
        self.assertEqual(metrics["data_source"], "mock", 
                         "Metrics should indicate mock data source")
        
        # Get metrics with real data
        self.data_source_config.use_mock_data = False
        metrics = self.agent.get_metrics()
        self.assertEqual(metrics["data_source"], "real",
                         "Metrics should indicate real data source")
    
    def test_config_listener_integration(self):
        """Test that the agent correctly receives data source config change notifications."""
        # Create a tracking variable to count notifications
        self.notification_count = 0
        
        # Define a test listener
        def test_listener(config):
            self.notification_count += 1
        
        # Register the test listener
        self.data_source_config.register_listener(test_listener)
        
        # Make changes and verify notifications
        initial_count = self.notification_count
        self.data_source_config.use_mock_data = not self.data_source_config.use_mock_data
        self.assertEqual(self.notification_count, initial_count + 1,
                         "Listener should be notified of configuration changes")
        
        # Unregister the listener
        self.data_source_config.unregister_listener(test_listener)
        
        # Verify no more notifications after unregistering
        self.data_source_config.use_mock_data = not self.data_source_config.use_mock_data
        self.assertEqual(self.notification_count, initial_count + 1,
                         "Unregistered listener should not receive notifications")
    
    def test_analyze_includes_data_source(self):
        """Test that analyze method correctly handles the data source type."""
        # Create simple test data
        mock_data = {
            "BTC-USD": {
                "1h": pd.DataFrame({
                    "open": np.random.normal(10000, 100, 100),
                    "high": np.random.normal(10100, 100, 100),
                    "low": np.random.normal(9900, 100, 100),
                    "close": np.random.normal(10050, 100, 100),
                    "volume": np.random.normal(1000, 100, 100)
                }, index=pd.date_range(end=datetime.now(), periods=100, freq='H'))
            }
        }
        
        # Set to mock data and analyze
        self.data_source_config.use_mock_data = True
        signals = self.agent.analyze(mock_data, ["BTC-USD"])
        self.assertEqual(self.agent.metrics["data_source"], "mock",
                         "Metrics should show mock data after analysis with mock data")
        
        # Set to real data and analyze
        self.data_source_config.use_mock_data = False
        signals = self.agent.analyze(mock_data, ["BTC-USD"])
        self.assertEqual(self.agent.metrics["data_source"], "real",
                         "Metrics should show real data after analysis with real data")


if __name__ == "__main__":
    unittest.main()
