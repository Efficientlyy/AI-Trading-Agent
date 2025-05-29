"""
Unit tests for lag features calculation in IndicatorEngine.
These tests specifically test the Python implementation to ensure
it works correctly even when Rust is not available.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Import from project
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

class TestLagFeaturesPython(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Basic config for testing
        config = {
            "logging": {"log_level": "DEBUG"}
        }
        self.engine = IndicatorEngine(config)
        
    def test_calculate_lag_features_python(self):
        """Test lag features calculation using Python implementation."""
        # Create test data
        data = pd.DataFrame({
            'close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'timestamp': pd.to_datetime([f'2023-01-01T00:0{i}:00Z' for i in range(6)])
        }).set_index('timestamp')
        
        # The lags we want to test
        lags_to_test = [1, 3, 5]
        
        # Call the _calculate_lag_features method directly
        lag_features_dict = self.engine._calculate_lag_features(
            df=data.copy(),
            lags=lags_to_test,
            source_column='close'
        )
        
        # Verify the structure of the result
        self.assertIsInstance(lag_features_dict, dict)
        self.assertEqual(len(lag_features_dict), len(lags_to_test))
        
        for lag in lags_to_test:
            feature_key = f'lag_{lag}'
            self.assertIn(feature_key, lag_features_dict)
            lag_series = lag_features_dict[feature_key]
            
            # Verify it's a pandas Series
            self.assertIsInstance(lag_series, pd.Series)
            
            # Verify length matches input data
            self.assertEqual(len(lag_series), len(data))
            
            # Verify that first N values are NaN (where N is the lag)
            self.assertTrue(lag_series.iloc[:lag].isna().all())
            
            # Verify non-NaN values match expected lagged values
            expected_values = data['close'].shift(lag)
            pd.testing.assert_series_equal(
                lag_series,
                expected_values,
                check_dtype=False,
                check_names=False
            )
        
        print("Python lag features calculation test passed.")
    
    def test_lag_features_rs_config(self):
        """Test that LAG_FEATURES_RS_ indicators can be configured."""
        # The lags we want to test
        lags_to_test = [1, 3, 5]
        
        # Create config with LAG_FEATURES_RS_ indicator
        config = {
            "features": {
                "LAG_FEATURES_RS_test": {
                    "enabled": True,
                    "lags_to_calculate": lags_to_test,
                    "source_column": "close"
                }
            },
            "logging": {"log_level": "DEBUG"}
        }
        
        # Create the engine
        engine = IndicatorEngine(config)
        
        # Verify the indicator was registered
        self.assertIn("LAG_FEATURES_RS_test", engine.indicators)
        
        # Verify the settings
        settings = engine.indicators["LAG_FEATURES_RS_test"]
        self.assertEqual(settings["lags_to_calculate"], lags_to_test)
        self.assertEqual(settings["source_column"], "close")
        self.assertTrue(settings["enabled"])
        
        print("LAG_FEATURES_RS_ configuration test passed.")

if __name__ == '__main__':
    unittest.main()
