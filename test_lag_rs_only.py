"""
A standalone test script to verify the LAG_FEATURES_RS_ functionality.
"""

import pandas as pd
import numpy as np
import sys
import os
import unittest

# Add project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from the project
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

class TestLagFeaturesRS(unittest.TestCase):
    """Test case specifically for LAG_FEATURES_RS_ functionality."""
    
    def test_lag_features_rs_direct(self):
        """Test LAG_FEATURES_RS_ through direct calculation."""
        # Create test data
        data = pd.DataFrame({
            'close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'timestamp': pd.to_datetime([f'2023-01-01T00:0{i}:00Z' for i in range(6)])
        }).set_index('timestamp')
        
        # The lags we want to test
        lags_to_test = [1, 3, 5]
        
        # Create a minimal configuration for the engine
        config = {
            "features": {
                "LAG_FEATURES_RS_test": {
                    "enabled": True,
                    "lags_to_calculate": lags_to_test,
                    "source_column": "close"
                }
            }
        }
        
        # Create the engine
        print("Creating IndicatorEngine...")
        engine = IndicatorEngine(config)
        
        # Verify the indicator was registered
        self.assertIn("LAG_FEATURES_RS_test", engine.indicators)
        print(f"LAG_FEATURES_RS_test registered: {engine.indicators['LAG_FEATURES_RS_test']}")
        
        # Get the calculator function
        calculator = engine.indicators["LAG_FEATURES_RS_test"]["calculator"]
        self.assertIsNotNone(calculator)
        
        # Calculate the lag features directly
        print("Calculating lag features directly...")
        result_dict = calculator(data, lags_to_test, 'close')
        
        # Verify the structure of the result
        self.assertIsInstance(result_dict, dict)
        print(f"Result dictionary keys: {list(result_dict.keys())}")
        
        for lag in lags_to_test:
            feature_key = f'lag_{lag}'
            self.assertIn(feature_key, result_dict)
            lag_series = result_dict[feature_key]
            
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
        
        print("Direct calculation test passed.")
        
        # Convert the result to a DataFrame as expected by the test
        result_df = pd.DataFrame(result_dict)
        print(f"Result DataFrame shape: {result_df.shape}")
        print(f"Result DataFrame columns: {result_df.columns.tolist()}")
        
        # Verify DataFrame structure
        for lag in lags_to_test:
            column_name = f"lag_{lag}"
            self.assertIn(column_name, result_df.columns)
        
        print("DataFrame conversion test passed.")
        print("LAG_FEATURES_RS_ test completed successfully.")

if __name__ == "__main__":
    unittest.main()
