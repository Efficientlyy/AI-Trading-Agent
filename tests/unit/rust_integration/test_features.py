"""
Unit tests for Rust-accelerated feature engineering functions.
"""
import pytest
import pandas as pd
import numpy as np

# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Requires missing or unbuilt Rust integration modules (ai_trading_agent_rs)")

from ai_trading_agent.rust_integration.features import create_lag_features, create_lag_features_df

class TestLagFeatures:
    """Test cases for lag features functions."""
    
    def test_create_lag_features_basic(self):
        """Test basic functionality of create_lag_features."""
        # Create a simple time series
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lags = [1, 2]
        
        # Expected result
        expected = np.array([
            [np.nan, np.nan],
            [1.0, np.nan],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0]
        ])
        
        # Get result
        result = create_lag_features(series, lags)
        
        # Check shape
        assert result.shape == (5, 2)
        
        # Check values (need to handle NaN values specially)
        for i in range(5):
            for j in range(2):
                if np.isnan(expected[i, j]):
                    assert np.isnan(result[i, j])
                else:
                    assert result[i, j] == expected[i, j]
    
    def test_create_lag_features_with_pandas_series(self):
        """Test create_lag_features with pandas Series input."""
        # Create a pandas Series
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        lags = [1, 2]
        
        # Expected result
        expected = np.array([
            [np.nan, np.nan],
            [1.0, np.nan],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0]
        ])
        
        # Get result
        result = create_lag_features(series, lags)
        
        # Check shape
        assert result.shape == (5, 2)
        
        # Check values (need to handle NaN values specially)
        for i in range(5):
            for j in range(2):
                if np.isnan(expected[i, j]):
                    assert np.isnan(result[i, j])
                else:
                    assert result[i, j] == expected[i, j]
    
    def test_create_lag_features_with_list(self):
        """Test create_lag_features with list input."""
        # Create a list
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        lags = [1, 2]
        
        # Expected result
        expected = np.array([
            [np.nan, np.nan],
            [1.0, np.nan],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0]
        ])
        
        # Get result
        result = create_lag_features(series, lags)
        
        # Check shape
        assert result.shape == (5, 2)
        
        # Check values (need to handle NaN values specially)
        for i in range(5):
            for j in range(2):
                if np.isnan(expected[i, j]):
                    assert np.isnan(result[i, j])
                else:
                    assert result[i, j] == expected[i, j]
    
    def test_create_lag_features_df(self):
        """Test create_lag_features_df function."""
        # Create a DataFrame
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Create lag features
        result_df = create_lag_features_df(df, columns=['A', 'B'], lags=[1, 2])
        
        # Check that original columns are preserved
        assert 'A' in result_df.columns
        assert 'B' in result_df.columns
        
        # Check that lag columns are created
        assert 'A_lag_1' in result_df.columns
        assert 'A_lag_2' in result_df.columns
        assert 'B_lag_1' in result_df.columns
        assert 'B_lag_2' in result_df.columns
        
        # Check values for A_lag_1
        expected_A_lag_1 = [np.nan, 1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(expected_A_lag_1):
            if np.isnan(val):
                assert np.isnan(result_df['A_lag_1'].iloc[i])
            else:
                assert result_df['A_lag_1'].iloc[i] == val
        
        # Check values for B_lag_2
        expected_B_lag_2 = [np.nan, np.nan, 10.0, 20.0, 30.0]
        for i, val in enumerate(expected_B_lag_2):
            if np.isnan(val):
                assert np.isnan(result_df['B_lag_2'].iloc[i])
            else:
                assert result_df['B_lag_2'].iloc[i] == val
    
    def test_error_handling(self):
        """Test error handling in create_lag_features."""
        # Empty series
        with pytest.raises(ValueError):
            create_lag_features([], [1, 2])
        
        # Empty lags
        with pytest.raises(ValueError):
            create_lag_features([1.0, 2.0, 3.0], [])
        
        # Invalid series type
        with pytest.raises(TypeError):
            create_lag_features("not a series", [1, 2])
        
        # Empty DataFrame
        with pytest.raises(ValueError):
            create_lag_features_df(pd.DataFrame(), columns=['A'], lags=[1])
        
        # Column not in DataFrame
        with pytest.raises(KeyError):
            create_lag_features_df(pd.DataFrame({'A': [1, 2, 3]}), columns=['B'], lags=[1])
