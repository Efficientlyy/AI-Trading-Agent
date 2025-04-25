"""
Test module for feature engineering functions with Decimal values.

This module tests that the feature engineering functions in the rust_integration module
correctly handle Decimal values.
"""
import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from ai_trading_agent.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix
)


class TestDecimalFeatures:
    """Test class for feature engineering functions with Decimal values."""

    def test_create_lag_features_with_decimal(self):
        """Test that create_lag_features correctly handles Decimal values."""
        # Create a series with Decimal values
        series = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0'), Decimal('4.0'), Decimal('5.0')]
        lags = [1, 2]
        
        # Call the function
        result = create_lag_features(series, lags)
        
        # Expected result: first lag values should be [NaN, 1.0, 2.0, 3.0, 4.0]
        # Second lag values should be [NaN, NaN, 1.0, 2.0, 3.0]
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        assert result[1, 0] == 1.0
        assert result[2, 0] == 2.0
        assert result[2, 1] == 1.0
        assert result[3, 0] == 3.0
        assert result[3, 1] == 2.0
        assert result[4, 0] == 4.0
        assert result[4, 1] == 3.0

    def test_create_diff_features_with_decimal(self):
        """Test that create_diff_features correctly handles Decimal values."""
        # Create a series with Decimal values
        series = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0'), Decimal('4.0'), Decimal('5.0')]
        periods = [1, 2]
        
        # Call the function
        result = create_diff_features(series, periods)
        
        # Expected result: first diff values should be [NaN, 1.0, 1.0, 1.0, 1.0]
        # Second diff values should be [NaN, NaN, 2.0, 2.0, 2.0]
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        assert result[1, 0] == 1.0
        assert result[2, 0] == 1.0
        assert result[2, 1] == 2.0
        assert result[3, 0] == 1.0
        assert result[3, 1] == 2.0
        assert result[4, 0] == 1.0
        assert result[4, 1] == 2.0

    def test_create_pct_change_features_with_decimal(self):
        """Test that create_pct_change_features correctly handles Decimal values."""
        # Create a series with Decimal values
        series = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0'), Decimal('4.0'), Decimal('5.0')]
        periods = [1, 2]
        
        # Call the function
        result = create_pct_change_features(series, periods)
        
        # Expected result: first pct_change values should be [NaN, 1.0, 0.5, 0.333..., 0.25]
        # Second pct_change values should be [NaN, NaN, 2.0, 1.0, 0.666...]
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        assert result[1, 0] == 1.0
        assert result[2, 0] == 0.5
        assert result[2, 1] == 2.0
        assert abs(result[3, 0] - 0.333333) < 0.000001
        assert result[3, 1] == 1.0
        assert result[4, 0] == 0.25
        assert abs(result[4, 1] - 0.666667) < 0.000001

    def test_create_rolling_window_features_with_decimal(self):
        """Test that create_rolling_window_features correctly handles Decimal values."""
        # Create a series with Decimal values
        series = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0'), Decimal('4.0'), Decimal('5.0')]
        window_sizes = [2, 3]
        
        # Call the function with mean
        result = create_rolling_window_features(series, window_sizes, 'mean')
        
        # Expected result for window size 2: [NaN, 1.5, 2.5, 3.5, 4.5]
        # Expected result for window size 3: [NaN, NaN, 2.0, 3.0, 4.0]
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        assert result[1, 0] == 1.5
        assert result[2, 0] == 2.5
        assert result[2, 1] == 2.0
        assert result[3, 0] == 3.5
        assert result[3, 1] == 3.0
        assert result[4, 0] == 4.5
        assert result[4, 1] == 4.0

    def test_create_feature_matrix_with_decimal(self):
        """Test that create_feature_matrix correctly handles Decimal values."""
        # Create a series with Decimal values
        series = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0'), Decimal('4.0'), Decimal('5.0')]
        lag_periods = [1]
        diff_periods = [1]
        
        # Call the function
        result = create_feature_matrix(
            series,
            lag_periods=lag_periods,
            diff_periods=diff_periods
        )
        
        # Expected result: first column is lag 1, second column is diff 1
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert result[1, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[2, 0] == 2.0
        assert result[2, 1] == 1.0
        assert result[3, 0] == 3.0
        assert result[3, 1] == 1.0
        assert result[4, 0] == 4.0
        assert result[4, 1] == 1.0

    def test_mixed_decimal_and_float(self):
        """Test that the functions handle a mix of Decimal and float values."""
        # Create a series with mixed Decimal and float values
        series = [Decimal('1.0'), 2.0, Decimal('3.0'), 4.0, Decimal('5.0')]
        lags = [1]
        
        # Call the function
        result = create_lag_features(series, lags)
        
        # Expected result should be the same as with all Decimal values
        assert result.shape == (5, 1)
        assert np.isnan(result[0, 0])
        assert result[1, 0] == 1.0
        assert result[2, 0] == 2.0
        assert result[3, 0] == 3.0
        assert result[4, 0] == 4.0
