"""
Unit tests for lag features module.
"""

import unittest
import numpy as np
from ai_trading_agent.features.lag_features import (
    py_create_lag_feature,
    py_create_diff_feature,
    py_create_pct_change_feature,
    lag_feature,
    diff_feature,
    pct_change_feature,
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    RUST_AVAILABLE
)


class TestLagFeatures(unittest.TestCase):
    """Test cases for lag features."""

    def setUp(self):
        """Set up test data."""
        self.series = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.lag = 2
        self.period = 2
        self.lags = [1, 2]
        self.periods = [1, 2]

    def test_py_create_lag_feature(self):
        """Test Python implementation of lag feature creation."""
        result = py_create_lag_feature(self.series, self.lag)
        expected = [float('nan'), float('nan'), 1.0, 2.0, 3.0]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.lag, len(result)):
            self.assertEqual(result[i], expected[i])
        
        # Check NaN values
        for i in range(self.lag):
            self.assertTrue(np.isnan(result[i]))

    def test_py_create_diff_feature(self):
        """Test Python implementation of difference feature creation."""
        result = py_create_diff_feature(self.series, self.period)
        expected = [float('nan'), float('nan'), 2.0, 2.0, 2.0]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.period, len(result)):
            self.assertEqual(result[i], expected[i])
        
        # Check NaN values
        for i in range(self.period):
            self.assertTrue(np.isnan(result[i]))

    def test_py_create_pct_change_feature(self):
        """Test Python implementation of percentage change feature creation."""
        result = py_create_pct_change_feature(self.series, self.period)
        expected = [float('nan'), float('nan'), 2.0, 1.0, 0.6666666666666666]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.period, len(result)):
            self.assertAlmostEqual(result[i], expected[i], places=6)
        
        # Check NaN values
        for i in range(self.period):
            self.assertTrue(np.isnan(result[i]))

    def test_lag_feature(self):
        """Test lag feature function (either Rust or Python implementation)."""
        result = lag_feature(self.series, self.lag)
        expected = [float('nan'), float('nan'), 1.0, 2.0, 3.0]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.lag, len(result)):
            self.assertEqual(result[i], expected[i])
        
        # Check NaN values
        for i in range(self.lag):
            self.assertTrue(np.isnan(result[i]))

    def test_diff_feature(self):
        """Test difference feature function (either Rust or Python implementation)."""
        result = diff_feature(self.series, self.period)
        expected = [float('nan'), float('nan'), 2.0, 2.0, 2.0]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.period, len(result)):
            self.assertEqual(result[i], expected[i])
        
        # Check NaN values
        for i in range(self.period):
            self.assertTrue(np.isnan(result[i]))

    def test_pct_change_feature(self):
        """Test percentage change feature function (either Rust or Python implementation)."""
        result = pct_change_feature(self.series, self.period)
        expected = [float('nan'), float('nan'), 2.0, 1.0, 0.6666666666666666]
        
        # Check length
        self.assertEqual(len(result), len(expected))
        
        # Check values (skipping NaN values)
        for i in range(self.period, len(result)):
            self.assertAlmostEqual(result[i], expected[i], places=6)
        
        # Check NaN values
        for i in range(self.period):
            self.assertTrue(np.isnan(result[i]))

    def test_create_lag_features(self):
        """Test creating multiple lag features."""
        result = create_lag_features(self.series, self.lags)
        
        # Check number of features
        self.assertEqual(len(result), len(self.lags))
        
        # Check each feature
        for i, lag in enumerate(self.lags):
            expected = [float('nan')] * lag + self.series[:-lag]
            
            # Check length
            self.assertEqual(len(result[i]), len(expected))
            
            # Check values (skipping NaN values)
            for j in range(lag, len(result[i])):
                self.assertEqual(result[i][j], expected[j])
            
            # Check NaN values
            for j in range(lag):
                self.assertTrue(np.isnan(result[i][j]))

    def test_create_diff_features(self):
        """Test creating multiple difference features."""
        result = create_diff_features(self.series, self.periods)
        
        # Check number of features
        self.assertEqual(len(result), len(self.periods))
        
        # Check each feature
        for i, period in enumerate(self.periods):
            # Calculate expected values
            expected = [float('nan')] * period
            for j in range(period, len(self.series)):
                expected.append(self.series[j] - self.series[j - period])
            
            # Check length
            self.assertEqual(len(result[i]), len(expected))
            
            # Check values (skipping NaN values)
            for j in range(period, len(result[i])):
                self.assertEqual(result[i][j], expected[j])
            
            # Check NaN values
            for j in range(period):
                self.assertTrue(np.isnan(result[i][j]))

    def test_create_pct_change_features(self):
        """Test creating multiple percentage change features."""
        result = create_pct_change_features(self.series, self.periods)
        
        # Check number of features
        self.assertEqual(len(result), len(self.periods))
        
        # Check each feature
        for i, period in enumerate(self.periods):
            # Calculate expected values
            expected = [float('nan')] * period
            for j in range(period, len(self.series)):
                if self.series[j - period] != 0:
                    expected.append((self.series[j] - self.series[j - period]) / self.series[j - period])
                else:
                    expected.append(float('nan'))
            
            # Check length
            self.assertEqual(len(result[i]), len(expected))
            
            # Check values (skipping NaN values)
            for j in range(period, len(result[i])):
                if not np.isnan(expected[j]):
                    self.assertAlmostEqual(result[i][j], expected[j], places=6)
                else:
                    self.assertTrue(np.isnan(result[i][j]))
            
            # Check NaN values
            for j in range(period):
                self.assertTrue(np.isnan(result[i][j]))

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        np_series = np.array(self.series)
        
        # Test lag features
        result = create_lag_features(np_series, self.lags)
        self.assertEqual(len(result), len(self.lags))
        
        # Test diff features
        result = create_diff_features(np_series, self.periods)
        self.assertEqual(len(result), len(self.periods))
        
        # Test pct change features
        result = create_pct_change_features(np_series, self.periods)
        self.assertEqual(len(result), len(self.periods))


if __name__ == '__main__':
    unittest.main()
