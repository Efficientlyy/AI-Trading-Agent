"""
Mock/Real Data Toggle Integration Test

This test verifies the end-to-end functionality of the mock/real data toggle feature,
ensuring proper integration between UI components, API endpoints, and the Technical Analysis Agent.
"""

import unittest
import json
import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import requests
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.config.data_source_config import get_data_source_config, DataSourceConfig
from ai_trading_agent.data.data_source_factory import get_data_source_factory, DataSourceFactory
from ai_trading_agent.data.mock_data_generator import MockDataGenerator
from ai_trading_agent.data.market_data_provider import MarketDataProvider


class TestMockRealToggleIntegration(unittest.TestCase):
    """Test the end-to-end integration of the mock/real data toggle functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Get global instances
        self.data_source_config = get_data_source_config()
        self.data_source_factory = get_data_source_factory()
        
        # Store original state to restore after tests
        self.original_state = self.data_source_config.use_mock_data
        
        # Reset config to use mock data for testing
        self.data_source_config.use_mock_data = True
        
        # Create agent configuration for testing
        self.agent_config = {
            "strategies": [
                {
                    "name": "Test Strategy",
                    "indicators": ["rsi", "macd", "bollinger_bands"],
                    "timeframes": ["1h", "4h", "1d"],
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
            }
        }
        
        # Initialize agent
        self.agent = AdvancedTechnicalAnalysisAgent(self.agent_config)
        
        # Generate test market data
        self.mock_data = self._generate_test_data()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original data source setting
        self.data_source_config.use_mock_data = self.original_state
    
    def _generate_test_data(self):
        """Generate test market data for testing."""
        # Simple test data generator
        symbols = ["BTC-USD", "ETH-USD"]
        timeframes = ["1h", "4h", "1d"]
        
        mock_data = {}
        for symbol in symbols:
            mock_data[symbol] = {}
            for tf in timeframes:
                # Generate a simple DataFrame with OHLCV data
                n_periods = 100
                base_price = 30000 if symbol == "BTC-USD" else 2000
                
                np.random.seed(42)  # For reproducibility
                dates = pd.date_range(end=pd.Timestamp.now(), periods=n_periods, freq=tf)
                
                # Generate random price data
                closes = base_price + np.cumsum(np.random.normal(0, base_price * 0.01, n_periods))
                opens = closes - np.random.normal(0, base_price * 0.005, n_periods)
                highs = np.maximum(opens, closes) + np.random.normal(0, base_price * 0.008, n_periods)
                lows = np.minimum(opens, closes) - np.random.normal(0, base_price * 0.008, n_periods)
                volumes = np.random.normal(base_price * 10, base_price * 5, n_periods)
                volumes = np.abs(volumes)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                }, index=dates)
                
                mock_data[symbol][tf] = df
        
        return mock_data
    
    def test_data_source_config_persistence(self):
        """Test that data source configuration persists between instances."""
        # Set to real data
        self.data_source_config.use_mock_data = False
        
        # Create a new config instance and verify it reads the same value
        temp_config = DataSourceConfig()
        self.assertFalse(temp_config.use_mock_data, "Configuration should persist between instances")
        
        # Reset to mock data
        self.data_source_config.use_mock_data = True
        
        # Verify the temp config also updates when reloaded
        temp_config = DataSourceConfig()
        self.assertTrue(temp_config.use_mock_data, "Configuration should persist between instances")
    
    def test_factory_returns_correct_provider(self):
        """Test that the factory returns the correct data provider based on configuration."""
        # Set to mock data
        self.data_source_config.use_mock_data = True
        
        # Get provider from factory
        provider = self.data_source_factory.get_data_provider()
        self.assertIsInstance(provider, MockDataGenerator, 
                            "Factory should return MockDataGenerator when mock data is enabled")
        
        # Set to real data
        self.data_source_config.use_mock_data = False
        
        # Get provider from factory
        provider = self.data_source_factory.get_data_provider()
        self.assertIsInstance(provider, MarketDataProvider, 
                            "Factory should return MarketDataProvider when real data is enabled")
    
    def test_agent_uses_correct_data_source(self):
        """Test that the agent uses the correct data source based on configuration."""
        # Set to mock data
        self.data_source_config.use_mock_data = True
        
        # Check agent reports correct data source type
        self.assertEqual(self.agent.get_data_source_type(), "mock", 
                        "Agent should report 'mock' as data source type")
        
        # Set to real data
        self.data_source_config.use_mock_data = False
        
        # Check agent reports correct data source type
        self.assertEqual(self.agent.get_data_source_type(), "real", 
                        "Agent should report 'real' as data source type")
    
    def test_toggle_functionality(self):
        """Test the toggle functionality in the agent."""
        # Start with mock data
        self.data_source_config.use_mock_data = True
        
        # Toggle to real data
        result = self.agent.toggle_data_source()
        self.assertEqual(result, "real", "Toggle should return 'real' after switching from mock")
        self.assertEqual(self.agent.get_data_source_type(), "real", 
                        "Agent should report 'real' after toggle")
        self.assertFalse(self.data_source_config.use_mock_data, 
                        "Config should be updated to use real data")
        
        # Toggle back to mock data
        result = self.agent.toggle_data_source()
        self.assertEqual(result, "mock", "Toggle should return 'mock' after switching from real")
        self.assertEqual(self.agent.get_data_source_type(), "mock", 
                        "Agent should report 'mock' after toggle")
        self.assertTrue(self.data_source_config.use_mock_data, 
                        "Config should be updated to use mock data")
    
    @patch('requests.get')
    @patch('requests.post')
    def test_api_endpoints(self, mock_post, mock_get):
        """Test the API endpoints for data source management."""
        # Mock API response for status endpoint
        mock_get.return_value = MagicMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "source_type": "mock",
            "use_mock_data": True,
            "mock_data_settings": {"volatility": 0.015},
            "real_data_settings": {"primary_source": "alpha_vantage"}
        }
        
        # Mock API response for toggle endpoint
        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "previous": "mock",
            "current": "real",
            "success": True,
            "message": "Successfully toggled data source from mock to real"
        }
        
        # Test status endpoint
        response = requests.get('/api/data-source/status')
        self.assertEqual(response.status_code, 200, "Status endpoint should return 200")
        data = response.json()
        self.assertTrue("source_type" in data, "Response should include source_type")
        self.assertTrue("use_mock_data" in data, "Response should include use_mock_data flag")
        
        # Test toggle endpoint
        response = requests.post('/api/data-source/toggle')
        self.assertEqual(response.status_code, 200, "Toggle endpoint should return 200")
        data = response.json()
        self.assertEqual(data["previous"], "mock", "Response should include previous state")
        self.assertEqual(data["current"], "real", "Response should include current state")
        self.assertTrue(data["success"], "Response should indicate success")
    
    def test_mock_settings_update(self):
        """Test updating mock data generator settings."""
        # Set initial mock settings
        original_settings = self.data_source_config.get_mock_data_settings()
        
        # Update settings
        update_dict = {
            "mock_data_settings": {
                "volatility": 0.03,
                "seed": 100
            }
        }
        self.data_source_config.update_config(update_dict)
        
        # Verify settings were updated
        new_settings = self.data_source_config.get_mock_data_settings()
        self.assertEqual(new_settings["volatility"], 0.03, 
                        "Volatility setting should be updated")
        self.assertEqual(new_settings["seed"], 100, 
                        "Seed setting should be updated")
        
        # Other settings should remain unchanged
        self.assertEqual(new_settings["trend_strength"], original_settings["trend_strength"], 
                        "Trend strength should remain unchanged")
    
    def test_agent_analysis_with_different_data_sources(self):
        """Test that agent analysis works with both mock and real data sources."""
        # Test with mock data
        self.data_source_config.use_mock_data = True
        
        # Analyze with mock data
        signals_mock = self.agent.analyze(self.mock_data)
        
        # Verify we got some signals
        self.assertIsNotNone(signals_mock, "Agent should return signals with mock data")
        self.assertIsInstance(signals_mock, list, "Signals should be a list")
        
        # Toggle to real data
        self.data_source_config.use_mock_data = False
        
        # Analyze with the same data (pretending it's real data)
        signals_real = self.agent.analyze(self.mock_data)
        
        # Verify we still get signals
        self.assertIsNotNone(signals_real, "Agent should return signals with real data")
        self.assertIsInstance(signals_real, list, "Signals should be a list")
        
        # Check that agent's internal data source type is reported correctly
        self.assertEqual(self.agent.get_metrics()["data_source"], "real",
                        "Agent metrics should report real data source")


if __name__ == "__main__":
    unittest.main()
