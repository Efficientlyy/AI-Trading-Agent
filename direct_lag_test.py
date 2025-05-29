"""
Direct test for lag features calculation using Python implementation.
This standalone script doesn't rely on the test framework or the indicator_engine.py file.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_lag_features_python(df, lags, source_column='close'):
    """Calculate lag features using Python implementation."""
    result = {}
    for lag in lags:
        key = f'lag_{lag}'
        result[key] = df[source_column].shift(lag)
    return result

def test_lag_feature_calculation():
    """Test lag features calculation using direct Python implementation."""
    # Create test data
    data = pd.DataFrame({
        'close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        'timestamp': pd.to_datetime([f'2023-01-01T00:0{i}:00Z' for i in range(6)])
    }).set_index('timestamp')
    
    # The lags we want to test
    lags_to_test = [1, 3, 5]
    
    # Calculate lag features using our function
    lag_features = calculate_lag_features_python(data, lags_to_test)
    
    # Verify the structure of the result
    assert len(lag_features) == len(lags_to_test), f"Expected {len(lags_to_test)} lag features, got {len(lag_features)}"
    
    for lag in lags_to_test:
        feature_key = f'lag_{lag}'
        assert feature_key in lag_features, f"Feature key {feature_key} missing from result"
        lag_series = lag_features[feature_key]
        
        # Verify it's a pandas Series
        assert isinstance(lag_series, pd.Series), f"Expected pandas Series, got {type(lag_series)}"
        
        # Verify length matches input data
        assert len(lag_series) == len(data), f"Expected length {len(data)}, got {len(lag_series)}"
        
        # Verify that first N values are NaN (where N is the lag)
        assert lag_series.iloc[:lag].isna().all(), f"First {lag} values should be NaN"
        
        # Verify non-NaN values match expected lagged values
        for i in range(lag, len(data)):
            expected_value = data['close'].iloc[i - lag]
            assert lag_series.iloc[i] == expected_value, f"Value at index {i} expected {expected_value}, got {lag_series.iloc[i]}"
    
    print("Python lag features calculation test passed!")

if __name__ == '__main__':
    try:
        test_lag_feature_calculation()
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
