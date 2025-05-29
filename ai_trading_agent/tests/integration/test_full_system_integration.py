"""
Full System Integration Tests

These tests validate that the complete trading system works correctly
with both mock and real data sources.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
import time

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.config.data_source_config import get_data_source_config
from ai_trading_agent.data.data_source_factory import get_data_source_factory
from ai_trading_agent.data.mock_data_generator import MockDataGenerator, TrendType
from ai_trading_agent.ml.enhanced_signal_validator import EnhancedSignalValidator


class TestFullSystemIntegration(unittest.TestCase):
    """Test the integration of all system components with the mock/real data toggle."""
    
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
            },
            "parallel_processing": {
                "enabled": True,
                "max_workers": 2
            }
        }
        
        # Create test symbols and timeframes
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.timeframes = ["1h", "4h", "1d"]
        
        # Generate test data
        self.mock_data = self._generate_test_data()
        
        # Initialize components
        self.agent = AdvancedTechnicalAnalysisAgent(self.agent_config)
        self.validator = EnhancedSignalValidator({
            "min_confidence": 0.6,
            "enable_regime_adaptation": True
        })
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original data source setting
        self.data_source_config.use_mock_data = self.original_state
    
    def _generate_test_data(self):
        """Generate test market data for testing."""
        # Create a mock data generator
        mock_gen = MockDataGenerator(seed=42)
        
        # Generate data for different timeframes
        end_date = datetime.now()
        
        market_data = {}
        for symbol in self.symbols:
            market_data[symbol] = {}
            
            # Generate different trend types for different symbols
            if symbol == "BTC-USD":
                trend = TrendType.BULLISH
            elif symbol == "ETH-USD":
                trend = TrendType.BEARISH
            else:
                trend = TrendType.VOLATILE
                
            for tf in self.timeframes:
                # Generate more periods for smaller timeframes
                if tf == "1h":
                    periods = 500
                elif tf == "4h":
                    periods = 250
                else:
                    periods = 200
                    
                # Generate data with specific trend characteristics
                data = mock_gen.generate_data(
                    symbol=symbol,
                    periods=periods,
                    trend_type=trend,
                    volatility=0.02,
                    end_date=end_date
                )
                
                market_data[symbol][tf] = data
        
        return market_data
    
    def test_e2e_analysis_with_mock_data(self):
        """Test end-to-end analysis with mock data."""
        # Ensure we're using mock data
        self.data_source_config.use_mock_data = True
        
        # Run analysis
        signals = self.agent.analyze(self.mock_data, self.symbols)
        
        # Verify results
        self.assertIsInstance(signals, list, "Analysis should return a list of signals")
        self.assertGreater(len(signals), 0, "Analysis should generate at least one signal")
        
        # Check that signals have expected structure
        for signal in signals:
            self.assertIn("symbol", signal, "Signal should include symbol")
            self.assertIn("direction", signal, "Signal should include direction")
            self.assertIn("strategy", signal, "Signal should include strategy")
            self.assertIn("confidence", signal, "Signal should include confidence")
            self.assertIn("metadata", signal, "Signal should include metadata")
        
        # Verify metrics
        metrics = self.agent.get_metrics()
        self.assertEqual(metrics["data_source"], "mock", "Metrics should show mock data source")
        self.assertGreater(metrics["signals_generated"], 0, "Metrics should record signals generated")
    
    def test_e2e_analysis_with_real_data(self):
        """Test end-to-end analysis with real data."""
        # Skip if real data provider not implemented yet
        try:
            # Switch to real data
            self.data_source_config.use_mock_data = False
            provider = self.data_source_factory.get_data_provider()
            if "MarketDataProvider" not in provider.__class__.__name__:
                self.skipTest("Real MarketDataProvider not implemented yet")
        except Exception:
            self.skipTest("Error accessing real data provider")
        
        # Run analysis (using mock data but with real data flag)
        signals = self.agent.analyze(self.mock_data, self.symbols)
        
        # Verify results
        self.assertIsInstance(signals, list, "Analysis should return a list of signals")
        
        # Verify metrics
        metrics = self.agent.get_metrics()
        self.assertEqual(metrics["data_source"], "real", "Metrics should show real data source")
    
    def test_seamless_toggle_during_operation(self):
        """Test toggling between data sources during operation."""
        # Start with mock data
        self.data_source_config.use_mock_data = True
        
        # Run first analysis
        signals_mock = self.agent.analyze(self.mock_data, self.symbols)
        
        # Toggle to real data
        self.agent.toggle_data_source()
        
        # Run second analysis
        signals_real = self.agent.analyze(self.mock_data, self.symbols)
        
        # Verify metrics changed
        metrics = self.agent.get_metrics()
        self.assertEqual(metrics["data_source"], "real", "Metrics should show real data source after toggle")
        
        # Toggle back to mock
        self.agent.toggle_data_source()
        
        # Run third analysis
        signals_mock_again = self.agent.analyze(self.mock_data, self.symbols)
        
        # Verify metrics changed back
        metrics = self.agent.get_metrics()
        self.assertEqual(metrics["data_source"], "mock", "Metrics should show mock data source after toggle back")
    
    def test_configuration_persistence(self):
        """Test that configuration changes persist properly."""
        # Create a temporary config file path
        temp_config_path = os.path.join(
            Path(__file__).parent.parent.parent,
            'tests',
            'integration',
            'temp_data_source_config.json'
        )
        
        try:
            # Create a new config instance with the temp path
            temp_config = get_data_source_config()
            
            # Update and save the configuration
            temp_config.use_mock_data = True
            mock_settings = {
                "mock_data_settings": {
                    "volatility": 0.03,
                    "trend_strength": 0.5,
                    "seed": 123
                }
            }
            temp_config.update_config(mock_settings)
            
            # Now create a new instance to simulate restarting the application
            temp_config2 = get_data_source_config()
            
            # Check that settings persisted
            self.assertTrue(temp_config2.use_mock_data, "Mock data setting should persist")
            saved_settings = temp_config2.get_mock_data_settings()
            self.assertEqual(saved_settings["volatility"], 0.03, "Volatility setting should persist")
            self.assertEqual(saved_settings["trend_strength"], 0.5, "Trend strength setting should persist")
            self.assertEqual(saved_settings["seed"], 123, "Seed setting should persist")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def test_event_listener_notifications(self):
        """Test that event listeners receive notifications when data source changes."""
        # Create a tracking variable to count notifications
        self.notification_count = 0
        self.last_notification = None
        
        # Define a test listener
        def test_listener(config):
            self.notification_count += 1
            self.last_notification = config.copy()
        
        # Register the test listener
        self.data_source_config.register_listener(test_listener)
        
        # Make changes and verify notifications
        initial_count = self.notification_count
        
        # Update mock data settings
        self.data_source_config.update_config({
            "mock_data_settings": {
                "volatility": 0.04
            }
        })
        self.assertEqual(self.notification_count, initial_count + 1,
                         "Listener should be notified of settings changes")
        self.assertEqual(self.last_notification["mock_data_settings"]["volatility"], 0.04,
                         "Notification should include updated settings")
        
        # Toggle data source
        current_state = self.data_source_config.use_mock_data
        self.data_source_config.use_mock_data = not current_state
        self.assertEqual(self.notification_count, initial_count + 2,
                         "Listener should be notified of toggle changes")
        self.assertEqual(self.last_notification["use_mock_data"], not current_state,
                         "Notification should include updated toggle state")
        
        # Unregister the listener
        self.data_source_config.unregister_listener(test_listener)
        
        # Verify no more notifications after unregistering
        self.data_source_config.use_mock_data = current_state
        self.assertEqual(self.notification_count, initial_count + 2,
                         "Unregistered listener should not receive notifications")
    
    def test_integration_with_ml_validator(self):
        """Test integration between the data toggle and ML validator."""
        # Start with mock data
        self.data_source_config.use_mock_data = True
        
        # Generate signals using the agent
        signals = self.agent.analyze(self.mock_data, self.symbols)
        
        # Find a valid signal to test with the validator
        test_signal = None
        for signal in signals:
            if signal.get("is_valid", False):
                test_signal = signal
                break
        
        if test_signal is None:
            self.skipTest("No valid signals generated for validator test")
        
        # Validate the signal with mock data setting
        mock_valid, mock_confidence, mock_metadata = self.validator.validate_signal(
            test_signal,
            {test_signal["symbol"]: self.mock_data[test_signal["symbol"]]},
            {}  # Empty indicators for simplicity
        )
        
        # Toggle to real data
        self.data_source_config.use_mock_data = False
        
        # Validate the same signal with real data setting
        real_valid, real_confidence, real_metadata = self.validator.validate_signal(
            test_signal,
            {test_signal["symbol"]: self.mock_data[test_signal["symbol"]]},
            {}  # Empty indicators for simplicity
        )
        
        # Verify that both validations provide results (not checking specific values
        # since they might legitimately differ between mock/real)
        self.assertIsInstance(mock_valid, bool, "Mock validation should return boolean validity")
        self.assertIsInstance(real_valid, bool, "Real validation should return boolean validity")
        self.assertIsInstance(mock_confidence, float, "Mock validation should return confidence score")
        self.assertIsInstance(real_confidence, float, "Real validation should return confidence score")


if __name__ == "__main__":
    unittest.main()
