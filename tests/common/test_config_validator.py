"""
Tests for the configuration validator.
"""
import os
import sys
import unittest
import copy

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from ai_trading_agent.common.config_validator import (
    validate_agent_config,
    validate_component_config,
    check_config_compatibility
)

class TestConfigValidator(unittest.TestCase):
    """
    Test the configuration validator.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.symbols = ["AAPL", "GOOG", "MSFT"]
        self.start_date = "2020-01-01"
        self.end_date = "2020-12-31"
        self.initial_capital = 100000.0
        
        # Create a valid configuration for testing
        self.valid_config = {
            "data_manager": {
                "type": "SimpleDataManager",
                "config": {
                    "data_dir": os.path.join(project_root, 'temp_data'),
                    "symbols": self.symbols,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "timeframe": "1d",
                    "data_types": ["ohlcv", "sentiment"]
                }
            },
            "strategy": {
                "type": "SentimentStrategy",
                "config": {
                    "name": "SentimentStrategy",
                    "symbols": self.symbols,
                    "sentiment_threshold": 0.3,
                    "position_size_pct": 0.1
                }
            },
            "risk_manager": {
                "type": "SimpleRiskManager",
                "config": {
                    "max_position_size": None,
                    "max_portfolio_risk_pct": 0.05,
                    "stop_loss_pct": 0.05
                }
            },
            "portfolio_manager": {
                "type": "PortfolioManager",
                "config": {
                    "initial_capital": self.initial_capital,
                    "risk_per_trade": 0.02,
                    "max_position_size": 0.2
                }
            },
            "execution_handler": {
                "type": "SimulatedExecutionHandler",
                "config": {
                    "commission_rate": 0.001,
                    "slippage_pct": 0.001
                }
            },
            "backtest": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "symbols": self.symbols,
                "initial_capital": self.initial_capital
            }
        }
    
    def test_valid_config(self):
        """
        Test validation of a valid configuration.
        """
        is_valid, error_message = validate_agent_config(self.valid_config)
        self.assertTrue(is_valid)
        self.assertIsNone(error_message)
    
    def test_missing_required_component(self):
        """
        Test validation of a configuration with a missing required component.
        """
        invalid_config = copy.deepcopy(self.valid_config)
        del invalid_config["data_manager"]
        
        is_valid, error_message = validate_agent_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_message)
    
    def test_missing_required_field(self):
        """
        Test validation of a configuration with a missing required field.
        """
        invalid_config = copy.deepcopy(self.valid_config)
        del invalid_config["data_manager"]["config"]["data_dir"]
        
        is_valid, error_message = validate_agent_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_message)
    
    def test_component_config_validation(self):
        """
        Test validation of a specific component's configuration.
        """
        is_valid, error_message = validate_component_config(
            "data_manager",
            self.valid_config["data_manager"]
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_message)
    
    def test_invalid_component_type(self):
        """
        Test validation of an invalid component type.
        """
        is_valid, error_message = validate_component_config(
            "nonexistent_component",
            {"type": "SomeType", "config": {}}
        )
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_message)
    
    def test_config_compatibility(self):
        """
        Test checking for compatibility issues between components.
        """
        # Create a configuration with compatibility issues
        incompatible_config = copy.deepcopy(self.valid_config)
        incompatible_config["strategy"]["config"]["symbols"] = ["AAPL", "GOOG"]  # Different from backtest symbols
        incompatible_config["portfolio_manager"]["config"]["initial_capital"] = 200000.0  # Different from backtest initial capital
        
        warnings = check_config_compatibility(incompatible_config)
        self.assertEqual(len(warnings), 2)  # Should have two warnings
        
        # Check a compatible configuration
        warnings = check_config_compatibility(self.valid_config)
        self.assertEqual(len(warnings), 0)  # Should have no warnings

if __name__ == "__main__":
    unittest.main()
