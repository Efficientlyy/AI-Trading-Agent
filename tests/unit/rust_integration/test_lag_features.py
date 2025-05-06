"""
Unit tests for the Rust-based lag feature implementations.
"""
import pytest
import numpy as np
import pandas as pd
from ai_trading_agent.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix
)

# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Requires missing or unbuilt Rust integration modules (ai_trading_agent_rs)")


class TestLagFeatures:
    """Tests for the lag features implementation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.series_pd = pd.Series(self.series)
        self.series_list = self.series.tolist()
        
    def test_create_lag_features_basic(self):
        """Test basic lag feature creation."""
        lags = [1, 2, 3]
        result = create_lag_features(self.series, lags)
        
        # Check shape
        assert result.shape == (len(self.series), len(lags))
        
        # Check values
        # First row should have NaN for all lags
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
        
        # Second row should have first lag value, but NaN for others
        assert result[1, 0] == 1.0
        assert np.isnan(result[1, 1])
        assert np.isnan(result[1, 2])
        
        # Third row should have first and second lag values, but NaN for third
        assert result[2, 0] == 2.0
        assert result[2, 1] == 1.0
        assert np.isnan(result[2, 2])
        
        # Fourth row should have all lag values
        assert result[3, 0] == 3.0
        assert result[3, 1] == 2.0
        assert result[3, 2] == 1.0
        
    def test_create_lag_features_with_pandas(self):
        """Test lag feature creation with pandas Series input."""
        lags = [1, 2, 3]
        result = create_lag_features(self.series_pd, lags)
        
        # Check shape
        assert result.shape == (len(self.series_pd), len(lags))
        
        # Check values (same as basic test)
        assert np.isnan(result[0, 0])
        assert result[3, 0] == 3.0
        assert result[3, 1] == 2.0
        assert result[3, 2] == 1.0
        
    def test_create_lag_features_with_list(self):
        """Test lag feature creation with list input."""
        lags = [1, 2, 3]
        result = create_lag_features(self.series_list, lags)
        
        # Check shape
        assert result.shape == (len(self.series_list), len(lags))
        
        # Check values (same as basic test)
        assert np.isnan(result[0, 0])
        assert result[3, 0] == 3.0
        assert result[3, 1] == 2.0
        assert result[3, 2] == 1.0
        
    def test_create_lag_features_empty_input(self):
        """Test lag feature creation with empty input."""
        lags = [1, 2, 3]
        
        with pytest.raises(ValueError):
            create_lag_features([], lags)
            
    def test_create_lag_features_empty_lags(self):
        """Test lag feature creation with empty lags."""
        with pytest.raises(ValueError):
            create_lag_features(self.series, [])


class TestDiffFeatures:
    """Tests for the difference features implementation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
    def test_create_diff_features_basic(self):
        """Test basic difference feature creation."""
        periods = [1, 2, 3]
        result = create_diff_features(self.series, periods)
        
        # Check shape
        assert result.shape == (len(self.series), len(periods))
        
        # Check values
        # First row should have NaN for all periods
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
        
        # Second row should have first period difference, but NaN for others
        assert result[1, 0] == 1.0  # 2 - 1
        assert np.isnan(result[1, 1])
        assert np.isnan(result[1, 2])
        
        # Third row should have first and second period differences, but NaN for third
        assert result[2, 0] == 1.0  # 3 - 2
        assert result[2, 1] == 2.0  # 3 - 1
        assert np.isnan(result[2, 2])
        
        # Fourth row should have all period differences
        assert result[3, 0] == 1.0  # 4 - 3
        assert result[3, 1] == 2.0  # 4 - 2
        assert result[3, 2] == 3.0  # 4 - 1


class TestPctChangeFeatures:
    """Tests for the percentage change features implementation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
    def test_create_pct_change_features_basic(self):
        """Test basic percentage change feature creation."""
        periods = [1, 2, 3]
        result = create_pct_change_features(self.series, periods)
        
        # Check shape
        assert result.shape == (len(self.series), len(periods))
        
        # Check values
        # First row should have NaN for all periods
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
        
        # Second row should have first period percentage change, but NaN for others
        assert result[1, 0] == 1.0  # (2 - 1) / 1
        assert np.isnan(result[1, 1])
        assert np.isnan(result[1, 2])
        
        # Third row should have first and second period percentage changes, but NaN for third
        assert result[2, 0] == 0.5  # (3 - 2) / 2
        assert result[2, 1] == 2.0  # (3 - 1) / 1
        assert np.isnan(result[2, 2])
        
        # Fourth row should have all period percentage changes
        assert result[3, 0] == 1/3  # (4 - 3) / 3
        assert result[3, 1] == 1.0  # (4 - 2) / 2
        assert result[3, 2] == 3.0  # (4 - 1) / 1


class TestRollingWindowFeatures:
    """Tests for the rolling window features implementation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
    def test_create_rolling_mean_features(self):
        """Test rolling mean feature creation."""
        window_sizes = [2, 3, 4]
        result = create_rolling_window_features(self.series, window_sizes, 'mean')
        
        # Check shape
        assert result.shape == (len(self.series), len(window_sizes))
        
        # Check values
        # First row should have NaN for all window sizes
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
        
        # Second row should have mean for window size 2, but NaN for others
        assert result[1, 0] == 1.5  # mean([1, 2])
        assert np.isnan(result[1, 1])
        assert np.isnan(result[1, 2])
        
        # Third row should have mean for window sizes 2 and 3, but NaN for 4
        assert result[2, 0] == 2.5  # mean([2, 3])
        assert result[2, 1] == 2.0  # mean([1, 2, 3])
        assert np.isnan(result[2, 2])
        
        # Fourth row should have mean for all window sizes
        assert result[3, 0] == 3.5  # mean([3, 4])
        assert result[3, 1] == 3.0  # mean([2, 3, 4])
        assert result[3, 2] == 2.5  # mean([1, 2, 3, 4])
        
    def test_create_rolling_min_features(self):
        """Test rolling min feature creation."""
        window_sizes = [2, 3]
        result = create_rolling_window_features(self.series, window_sizes, 'min')
        
        # Check values
        assert result[1, 0] == 1.0  # min([1, 2])
        assert result[2, 1] == 1.0  # min([1, 2, 3])
        
    def test_create_rolling_max_features(self):
        """Test rolling max feature creation."""
        window_sizes = [2, 3]
        result = create_rolling_window_features(self.series, window_sizes, 'max')
        
        # Check values
        assert result[1, 0] == 2.0  # max([1, 2])
        assert result[2, 1] == 3.0  # max([1, 2, 3])
        
    def test_create_rolling_std_features(self):
        """Test rolling std feature creation."""
        window_sizes = [2, 3]
        result = create_rolling_window_features(self.series, window_sizes, 'std')
        
        # Check values (with some tolerance for floating point)
        assert abs(result[1, 0] - 0.7071) < 0.5  # std([1, 2])
        assert abs(result[2, 1] - 1.0) < 0.5     # std([1, 2, 3])
        
    def test_invalid_feature_type(self):
        """Test with invalid feature type."""
        window_sizes = [2, 3]
        
        with pytest.raises(ValueError):
            create_rolling_window_features(self.series, window_sizes, 'invalid')


class TestFeatureMatrix:
    """Tests for the feature matrix creation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample time series data
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
    def test_create_feature_matrix_all_types(self):
        """Test creating a feature matrix with all feature types."""
        lag_periods = [1, 2]
        diff_periods = [1]
        pct_change_periods = [1]
        rolling_windows = {'mean': [2], 'std': [2]}
        
        result = create_feature_matrix(
            self.series,
            lag_periods=lag_periods,
            diff_periods=diff_periods,
            pct_change_periods=pct_change_periods,
            rolling_windows=rolling_windows
        )
        
        # Check shape
        expected_cols = (
            len(lag_periods) +
            len(diff_periods) +
            len(pct_change_periods) +
            len(rolling_windows['mean']) +
            len(rolling_windows['std'])
        )
        assert result.shape == (len(self.series), expected_cols)
        
    def test_create_feature_matrix_no_features(self):
        """Test creating a feature matrix with no feature types."""
        with pytest.raises(ValueError):
            create_feature_matrix(self.series)
            
    def test_create_feature_matrix_single_type(self):
        """Test creating a feature matrix with a single feature type."""
        lag_periods = [1, 2, 3]
        
        result = create_feature_matrix(
            self.series,
            lag_periods=lag_periods
        )
        
        # Check shape
        assert result.shape == (len(self.series), len(lag_periods))
        
        # Check that it's the same as calling create_lag_features directly
        expected = create_lag_features(self.series, lag_periods)
        assert np.array_equal(result, expected, equal_nan=True)
