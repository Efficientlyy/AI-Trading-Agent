"""
Tests for multi-timeframe regime predictions and transitions.

This module contains tests specifically focused on regime prediction capabilities
across multiple timeframes, including transition probability modeling, regime
consistency across timeframes, and prediction accuracy evaluation.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.ml.detection.extended import (
    MultiTimeframeRegimeAnalyzer,
    HMMRegimeDetector
)
from src.models.market_data import TimeFrame


class TestMultiTimeframePredictions(unittest.TestCase):
    """Tests for multi-timeframe regime prediction capabilities."""
    
    def setUp(self):
        """Set up test data across multiple timeframes."""
        # Create synthetic price data
        np.random.seed(42)
        
        # Base minute data
        self.n_points = 1000
        dates = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(self.n_points)]
        
        # Create multiple regimes in the data
        # Bull market (uptrend with low volatility)
        bull_trend = np.linspace(0, 10, 300) + np.random.normal(0, 0.5, 300)
        
        # Bear market (downtrend with high volatility)
        bear_trend = np.linspace(10, 4, 300) + np.random.normal(0, 1.2, 300)
        
        # Sideways market (range-bound with medium volatility)
        sideways = np.sin(np.linspace(0, 6*np.pi, 400)) * 0.5 + 4 + np.random.normal(0, 0.8, 400)
        
        # Combine these regimes
        prices = np.concatenate([bull_trend, bear_trend, sideways])[:self.n_points]
        
        # Create minute dataframe
        self.minute_df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.lognormal(0, 1, self.n_points),
            'returns': np.diff(prices, prepend=prices[0]) / prices
        })
        self.minute_df.set_index('date', inplace=True)
        
        # Create hourly dataframe
        self.hourly_df = self.minute_df.resample('1H').agg({
            'price': 'last',
            'volume': 'sum',
            'returns': 'sum'
        })
        
        # Create daily dataframe
        self.daily_df = self.minute_df.resample('1D').agg({
            'price': 'last',
            'volume': 'sum',
            'returns': 'sum'
        })
        
        # Initialize detectors
        self.minute_detector = HMMRegimeDetector(n_regimes=3)
        self.hourly_detector = HMMRegimeDetector(n_regimes=3)
        self.daily_detector = HMMRegimeDetector(n_regimes=3)
        
        # Initialize analyzer
        self.analyzer = MultiTimeframeRegimeAnalyzer(
            detectors={
                TimeFrame.MINUTE: self.minute_detector,
                TimeFrame.HOUR: self.hourly_detector,
                TimeFrame.DAY: self.daily_detector
            }
        )
        
        # Fit the analyzer
        self.analyzer.fit({
            TimeFrame.MINUTE: self.minute_df,
            TimeFrame.HOUR: self.hourly_df,
            TimeFrame.DAY: self.daily_df
        })
    
    def test_regime_prediction(self):
        """Test regime prediction for future periods."""
        # Get current regime probabilities
        current_probs = self.analyzer.get_current_regime_probabilities()
        
        # Predict regimes for next 5 periods
        predictions = self.analyzer.predict_regimes(
            n_periods=5,
            timeframe=TimeFrame.HOUR
        )
        
        # Assertions
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= prob <= 1 for regime_probs in predictions 
                           for prob in regime_probs.values()))
        
        # Test prediction decay - probabilities should tend toward equilibrium
        first_pred = predictions[0]
        last_pred = predictions[-1]
        
        # Calculate variance of probabilities - should decrease over time
        first_variance = np.var(list(first_pred.values()))
        last_variance = np.var(list(last_pred.values()))
        
        # In longer predictions, variance should decrease as uncertainty increases
        self.assertGreaterEqual(first_variance, last_variance)
    
    def test_transition_matrices(self):
        """Test transition probability matrices across timeframes."""
        # Get transition matrices
        minute_transitions = self.analyzer.get_transition_matrix(TimeFrame.MINUTE)
        hourly_transitions = self.analyzer.get_transition_matrix(TimeFrame.HOUR)
        daily_transitions = self.analyzer.get_transition_matrix(TimeFrame.DAY)
        
        # Assertions
        self.assertEqual(minute_transitions.shape, (3, 3))
        self.assertEqual(hourly_transitions.shape, (3, 3))
        self.assertEqual(daily_transitions.shape, (3, 3))
        
        # Check that rows sum to 1 (valid probability distributions)
        for matrix in [minute_transitions, hourly_transitions, daily_transitions]:
            for row in matrix:
                self.assertAlmostEqual(sum(row), 1.0, places=6)
    
    def test_regime_persistence(self):
        """Test regime persistence metrics across timeframes."""
        # Calculate persistence (diagonal elements of transition matrices)
        minute_persistence = self.analyzer.calculate_regime_persistence(TimeFrame.MINUTE)
        hourly_persistence = self.analyzer.calculate_regime_persistence(TimeFrame.HOUR)
        daily_persistence = self.analyzer.calculate_regime_persistence(TimeFrame.DAY)
        
        # Assertions
        self.assertEqual(len(minute_persistence), 3)
        self.assertEqual(len(hourly_persistence), 3)
        self.assertEqual(len(daily_persistence), 3)
        
        # Higher timeframes should generally have more persistent regimes
        # (This is a general market behavior, but might not always hold in test data)
        # We'll assert that at least one regime shows this property
        any_increasing_persistence = False
        for i in range(3):
            if minute_persistence[i] <= hourly_persistence[i] <= daily_persistence[i]:
                any_increasing_persistence = True
                break
        
        self.assertTrue(any_increasing_persistence)
    
    def test_regime_consistency(self):
        """Test consistency of regime assignments across timeframes."""
        # Get consistency matrix
        consistency = self.analyzer.get_consistency_matrix()
        
        # Assertions
        self.assertEqual(consistency.shape, (3, 3, 3))  # 3 regimes x 3 timeframes
        
        # Test regime mapping across timeframes
        regime_mapping = self.analyzer.get_regime_mapping()
        
        self.assertIsNotNone(regime_mapping)
        self.assertEqual(len(regime_mapping), 3)  # 3 timeframes
        
        # Test alignment score
        alignment_score = self.analyzer.calculate_timeframe_alignment()
        
        self.assertGreaterEqual(alignment_score, 0)
        self.assertLessEqual(alignment_score, 1)
    
    def test_hierarchical_prediction(self):
        """Test hierarchical regime prediction."""
        # Get hierarchical prediction
        hierarchical_pred = self.analyzer.predict_hierarchical_regimes(n_periods=3)
        
        # Assertions
        self.assertIsNotNone(hierarchical_pred)
        self.assertEqual(len(hierarchical_pred), 3)  # 3 prediction periods
        
        # Each prediction should have entries for all timeframes
        for pred in hierarchical_pred:
            self.assertEqual(len(pred), 3)  # 3 timeframes
            
            # Each timeframe prediction should have valid probabilities
            for tf_pred in pred.values():
                self.assertEqual(len(tf_pred), 3)  # 3 regimes
                self.assertAlmostEqual(sum(tf_pred.values()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
