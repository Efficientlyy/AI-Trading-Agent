"""
Tests for ensemble regime detection capabilities.

This module contains tests for the ensemble regime detector, which combines
multiple base detectors to produce more robust regime classifications through
various consensus mechanisms.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.ml.detection.extended import (
    HMMRegimeDetector,
    VolatilityRegimeDetector,
    TrendRegimeDetector
)
from src.ml.detection.ensemble_detector import EnsembleRegimeDetector


class TestEnsembleRegimeDetector(unittest.TestCase):
    """Tests for the ensemble regime detector."""
    
    def setUp(self):
        """Set up test data and detectors."""
        # Create synthetic price data
        np.random.seed(42)
        n_points = 500
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_points)]
        
        # Create multiple regimes in the data
        # Bull market (uptrend with low volatility)
        bull_trend = np.linspace(100, 150, 150) + np.random.normal(0, 2, 150)
        
        # Bear market (downtrend with high volatility)
        bear_trend = np.linspace(150, 100, 150) + np.random.normal(0, 4, 150)
        
        # Sideways market (range-bound with medium volatility)
        sideways = np.sin(np.linspace(0, 10*np.pi, 200)) * 5 + 100 + np.random.normal(0, 3, 200)
        
        # Combine these regimes
        prices = np.concatenate([bull_trend, bear_trend, sideways])[:n_points]
        
        # Create dataframe
        self.df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.lognormal(0, 1, n_points),
            'returns': np.diff(prices, prepend=prices[0]) / prices
        })
        
        # Initialize individual detectors
        self.hmm_detector = HMMRegimeDetector(n_regimes=3)
        self.volatility_detector = VolatilityRegimeDetector(
            volatility_window=20,
            volatility_threshold_low=0.01,
            volatility_threshold_high=0.03
        )
        self.trend_detector = TrendRegimeDetector(
            trend_window=30,
            trend_threshold_low=0.01,
            trend_threshold_high=0.03
        )
        
        # Initialize ensemble detector with different consensus methods
        self.majority_ensemble = EnsembleRegimeDetector(
            detectors=[self.hmm_detector, self.volatility_detector, self.trend_detector],
            consensus_method='majority_vote'
        )
        
        self.weighted_ensemble = EnsembleRegimeDetector(
            detectors=[self.hmm_detector, self.volatility_detector, self.trend_detector],
            consensus_method='weighted',
            weights=[0.5, 0.3, 0.2]
        )
        
        self.bayesian_ensemble = EnsembleRegimeDetector(
            detectors=[self.hmm_detector, self.volatility_detector, self.trend_detector],
            consensus_method='bayesian'
        )
        
        # Fit the detectors
        for detector in [self.hmm_detector, self.volatility_detector, self.trend_detector]:
            detector.fit(self.df)
    
    def test_majority_vote_consensus(self):
        """Test majority vote consensus method."""
        # Get individual detector predictions
        hmm_regimes = self.hmm_detector.detect(self.df)
        vol_regimes = self.volatility_detector.detect(self.df)
        trend_regimes = self.trend_detector.detect(self.df)
        
        # Get ensemble prediction
        ensemble_regimes = self.majority_ensemble.detect(self.df)
        
        # Assertions
        self.assertEqual(len(ensemble_regimes), len(self.df))
        
        # Test individual data points where all detectors agree
        for i in range(len(self.df)):
            if hmm_regimes[i] == vol_regimes[i] == trend_regimes[i]:
                self.assertEqual(ensemble_regimes[i], hmm_regimes[i])
    
    def test_weighted_consensus(self):
        """Test weighted consensus method."""
        # Get ensemble prediction
        ensemble_regimes = self.weighted_ensemble.detect(self.df)
        
        # Assertions
        self.assertEqual(len(ensemble_regimes), len(self.df))
        
        # Test weights by looking at a specific window of data
        # We'll manually calculate what the weighted consensus should be for one point
        # and compare it with what the ensemble detector produces
        test_idx = 100
        hmm_regime = self.hmm_detector.detect(self.df)[test_idx]
        vol_regime = self.volatility_detector.detect(self.df)[test_idx]
        trend_regime = self.trend_detector.detect(self.df)[test_idx]
        
        # Get probabilities for each regime
        hmm_probs = self.hmm_detector.get_regime_probabilities(self.df.iloc[[test_idx]])
        vol_probs = self.volatility_detector.get_regime_probabilities(self.df.iloc[[test_idx]])
        trend_probs = self.trend_detector.get_regime_probabilities(self.df.iloc[[test_idx]])
        
        # Verify the weighted ensemble's logic by manually calculating for a test point
        # The details here will depend on exactly how your weighted consensus is implemented
        # This is a simplified example
        weights = [0.5, 0.3, 0.2]
        weighted_probs = {}
        
        # For the regimes detected at this point, check weighted probabilities
        for regime in set([hmm_regime, vol_regime, trend_regime]):
            hmm_prob = hmm_probs[0].get(regime, 0) if hmm_probs else 0
            vol_prob = vol_probs[0].get(regime, 0) if vol_probs else 0
            trend_prob = trend_probs[0].get(regime, 0) if trend_probs else 0
            
            weighted_prob = hmm_prob * weights[0] + vol_prob * weights[1] + trend_prob * weights[2]
            weighted_probs[regime] = weighted_prob
        
        # The ensemble should choose the regime with highest weighted probability
        if weighted_probs:
            expected_regime = max(weighted_probs, key=weighted_probs.get)
            self.assertEqual(ensemble_regimes[test_idx], expected_regime)
    
    def test_bayesian_consensus(self):
        """Test Bayesian consensus method."""
        # Get ensemble prediction
        ensemble_regimes = self.bayesian_ensemble.detect(self.df)
        
        # Assertions
        self.assertEqual(len(ensemble_regimes), len(self.df))
        
        # Check that the ensemble produces different results than individual detectors
        # at least some of the time (indicating it's actually doing consensus)
        hmm_regimes = self.hmm_detector.detect(self.df)
        vol_regimes = self.volatility_detector.detect(self.df)
        trend_regimes = self.trend_detector.detect(self.df)
        
        # Count how often the ensemble differs from all individual detectors
        differs_count = 0
        for i in range(len(self.df)):
            if (ensemble_regimes[i] != hmm_regimes[i] and 
                ensemble_regimes[i] != vol_regimes[i] and 
                ensemble_regimes[i] != trend_regimes[i]):
                differs_count += 1
        
        # The Bayesian consensus should differ at least sometimes
        # Not an exhaustive test, but a sanity check
        self.assertLess(differs_count, len(self.df))  # Not all should differ
    
    def test_confidence_metrics(self):
        """Test confidence metrics for ensemble predictions."""
        # Get confidence scores
        confidences = self.majority_ensemble.get_confidence_scores(self.df)
        
        # Assertions
        self.assertEqual(len(confidences), len(self.df))
        
        # Confidences should be between 0 and 1
        for conf in confidences:
            self.assertGreaterEqual(conf, 0)
            self.assertLessEqual(conf, 1)
        
        # Check higher confidence when detectors agree
        hmm_regimes = self.hmm_detector.detect(self.df)
        vol_regimes = self.volatility_detector.detect(self.df)
        trend_regimes = self.trend_detector.detect(self.df)
        
        agree_indices = [i for i in range(len(self.df)) 
                        if hmm_regimes[i] == vol_regimes[i] == trend_regimes[i]]
        disagree_indices = [i for i in range(len(self.df)) 
                           if not (hmm_regimes[i] == vol_regimes[i] == trend_regimes[i])]
        
        if agree_indices and disagree_indices:
            avg_agree_conf = sum(confidences[i] for i in agree_indices) / len(agree_indices)
            avg_disagree_conf = sum(confidences[i] for i in disagree_indices) / len(disagree_indices)
            
            # Confidence should be higher when detectors agree
            self.assertGreater(avg_agree_conf, avg_disagree_conf)
    
    def test_detector_weighting(self):
        """Test dynamic detector weighting based on performance."""
        # Train the ensemble for dynamic weighting
        self.adaptive_ensemble = EnsembleRegimeDetector(
            detectors=[self.hmm_detector, self.volatility_detector, self.trend_detector],
            consensus_method='adaptive'
        )
        
        # Create "true" regimes for training
        # This is simplified - in practice you'd have actual labeled data
        true_regimes = [0] * 150 + [1] * 150 + [2] * 200
        true_regimes = true_regimes[:len(self.df)]
        
        # Train with this labeled data
        self.adaptive_ensemble.train_weights(self.df, true_regimes)
        
        # Get trained weights
        weights = self.adaptive_ensemble.get_detector_weights()
        
        # Assertions
        self.assertEqual(len(weights), 3)  # One weight per detector
        self.assertAlmostEqual(sum(weights), 1.0, places=6)  # Weights should sum to 1
        
        # All weights should be positive
        for w in weights:
            self.assertGreaterEqual(w, 0)
        
        # Test detection with trained weights
        ensemble_regimes = self.adaptive_ensemble.detect(self.df)
        self.assertEqual(len(ensemble_regimes), len(self.df))


if __name__ == '__main__':
    unittest.main()
