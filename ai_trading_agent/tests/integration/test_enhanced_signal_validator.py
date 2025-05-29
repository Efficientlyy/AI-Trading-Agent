"""
Integration tests for the enhanced ML-based signal validator.

These tests validate that the enhanced signal validator works correctly with
various strategies and market regimes in an integrated environment.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.ml.enhanced_signal_validator import EnhancedSignalValidator
from ai_trading_agent.ml.feature_engineering import FeatureEngineer
from ai_trading_agent.ml.signal_clustering import SignalClusterAnalyzer
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier


class TestEnhancedSignalValidatorIntegration(unittest.TestCase):
    """Test the integration of the enhanced signal validator with other components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create configuration for testing
        self.validator_config = {
            "min_confidence": 0.6,
            "enable_ensemble": True,
            "enable_regime_adaptation": True,
            "performance_tracking": True,
            "uncertainty_quantification": True,
            "model_params": {
                "rf_n_estimators": 50,  # Lower for testing
                "gb_n_estimators": 50,  # Lower for testing
                "use_xgboost": True,
                "ensemble_weights": {"rf": 0.4, "gb": 0.4, "xgb": 0.2}
            }
        }
        
        self.feature_config = {
            "enable_auto_feature_selection": True,
            "pattern_recognition": True,
            "normalize_volatility": True,
            "include_regime_features": True,
            "feature_selection_method": "mutual_info"
        }
        
        self.cluster_config = {
            "n_clusters": 5,
            "similarity_threshold": 0.7,
            "max_history_signals": 100,
            "use_dimension_reduction": True
        }
        
        self.regime_config = {
            "lookback_window": 50,
            "enable_ml_enhancement": True,
            "smoothing_window": 3
        }
        
        # Create test data
        self.market_data = self._generate_test_data()
        self.test_signals = self._generate_test_signals()
        
        # Initialize components
        self.signal_validator = EnhancedSignalValidator(self.validator_config)
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.cluster_analyzer = SignalClusterAnalyzer(self.cluster_config)
        self.regime_classifier = MarketRegimeClassifier(self.regime_config)
    
    def _generate_test_data(self):
        """Generate test market data with different characteristics."""
        # Create date range
        dates = pd.date_range(start="2025-01-01", periods=200, freq="D")
        
        test_data = {}
        
        # Generate trending market data
        trending_data = pd.DataFrame(index=dates)
        trending_data['open'] = np.linspace(100, 200, len(dates)) + np.random.normal(0, 3, len(dates))
        trending_data['high'] = trending_data['open'] + np.random.uniform(1, 5, len(dates))
        trending_data['low'] = trending_data['open'] - np.random.uniform(1, 5, len(dates))
        trending_data['close'] = np.linspace(100, 200, len(dates)) + np.random.normal(0, 3, len(dates))
        trending_data['volume'] = np.random.uniform(1000, 5000, len(dates))
        test_data["TRENDING"] = trending_data
        
        # Generate ranging market data
        ranging_data = pd.DataFrame(index=dates)
        ranging_data['open'] = 150 + np.random.normal(0, 10, len(dates))
        ranging_data['high'] = ranging_data['open'] + np.random.uniform(1, 5, len(dates))
        ranging_data['low'] = ranging_data['open'] - np.random.uniform(1, 5, len(dates))
        ranging_data['close'] = 150 + np.random.normal(0, 10, len(dates))
        ranging_data['volume'] = np.random.uniform(1000, 5000, len(dates))
        test_data["RANGING"] = ranging_data
        
        # Generate volatile market data
        volatile_data = pd.DataFrame(index=dates)
        base = 150 + np.cumsum(np.random.normal(0, 3, len(dates)))
        volatile_data['open'] = base
        volatile_data['high'] = base + np.random.uniform(5, 15, len(dates))
        volatile_data['low'] = base - np.random.uniform(5, 15, len(dates))
        volatile_data['close'] = base + np.random.normal(0, 8, len(dates))
        volatile_data['volume'] = np.random.uniform(2000, 10000, len(dates))
        test_data["VOLATILE"] = volatile_data
        
        return test_data
    
    def _generate_test_signals(self):
        """Generate test trading signals with different characteristics."""
        # Create some test signals
        test_signals = {
            # Strong trend signal with high confidence
            "strong_trend": {
                "signal": 0.85,
                "direction": "buy",
                "signal_type": "TestTrendStrategy",
                "metadata": {
                    "trend_strength": 0.9,
                    "momentum": 0.8,
                    "volume_confirmation": 0.7
                }
            },
            # Weak trend signal with low confidence
            "weak_trend": {
                "signal": 0.35,
                "direction": "buy",
                "signal_type": "TestTrendStrategy",
                "metadata": {
                    "trend_strength": 0.4,
                    "momentum": 0.3,
                    "volume_confirmation": 0.3
                }
            },
            # Strong reversal signal
            "strong_reversal": {
                "signal": 0.75,
                "direction": "sell",
                "signal_type": "TestReversalStrategy",
                "metadata": {
                    "overbought": 0.9,
                    "momentum_divergence": 0.8,
                    "pattern_match": 0.7
                }
            },
            # Contradictory signal with mixed indicators
            "contradictory": {
                "signal": 0.6,
                "direction": "buy",
                "signal_type": "TestMixedStrategy",
                "metadata": {
                    "trend_indicator": 0.7,
                    "oscillator_indicator": -0.5,  # Contradictory
                    "volume_indicator": 0.3
                }
            }
        }
        
        return test_signals
    
    def test_validator_feature_engineer_integration(self):
        """Test integration between signal validator and feature engineer."""
        # Generate features for a test dataset
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        signal = self.test_signals["strong_trend"]
        
        # Extract features using the feature engineer
        features = self.feature_engineer.extract_features(market_data[symbol])
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 5)  # Should have multiple features
        
        # Validate a signal using the extracted features
        indicators = {symbol: features}
        is_valid, confidence, metadata = self.signal_validator.validate_signal(
            signal, market_data, indicators
        )
        
        # Check validation results
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(metadata, dict)
        
        # For a strong trend signal with good confirmation, it should be valid
        self.assertTrue(is_valid)
        self.assertGreater(confidence, 0.6)
    
    def test_validator_clustering_integration(self):
        """Test integration between signal validator and cluster analyzer."""
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        signal = self.test_signals["strong_trend"]
        
        # Generate some historical signals for the cluster analyzer
        historical_signals = []
        for i in range(10):
            # Create variations of the strong trend signal
            sig = self.test_signals["strong_trend"].copy()
            sig["signal"] *= (0.8 + 0.4 * np.random.random())  # Vary signal strength
            sig["metadata"] = sig["metadata"].copy()
            sig["metadata"]["trend_strength"] *= (0.8 + 0.4 * np.random.random())
            sig["metadata"]["timestamp"] = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            sig["metadata"]["outcome"] = 1 if np.random.random() > 0.3 else 0  # 70% success rate
            historical_signals.append(sig)
        
        # Add historical signals to the cluster analyzer
        for hist_signal in historical_signals:
            self.cluster_analyzer.add_historical_signal(hist_signal)
        
        # Find similar signals
        similar_signals = self.cluster_analyzer.find_similar_signals(signal)
        self.assertIsInstance(similar_signals, list)
        
        # Since we added variations of the strong trend signal, we should find similar ones
        self.assertGreater(len(similar_signals), 0)
        
        # Validate signal with cluster information
        # First, prepare the signal with cluster info
        signal_with_cluster = signal.copy()
        signal_with_cluster["metadata"] = signal["metadata"].copy()
        signal_with_cluster["metadata"]["similar_signals"] = similar_signals
        
        # Now validate
        is_valid, confidence, metadata = self.signal_validator.validate_signal(
            signal_with_cluster, market_data, {}
        )
        
        # Check validation results
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(confidence, float)
        self.assertIn("similar_signals_success_rate", metadata)
    
    def test_validator_regime_adaptation(self):
        """Test that the validator adapts thresholds based on market regime."""
        # Test across different market regimes
        regimes = ["TRENDING", "RANGING", "VOLATILE"]
        
        regime_confidences = {}
        
        for regime_symbol in regimes:
            regime_data = {regime_symbol: self.market_data[regime_symbol]}
            signal = self.test_signals["strong_trend"]
            
            # Detect the regime
            detected_regime = self.regime_classifier.classify_regime(regime_data[regime_symbol])
            
            # Validate the signal
            is_valid, confidence, metadata = self.signal_validator.validate_signal(
                signal, regime_data, {}
            )
            
            # Store the confidence for this regime
            regime_confidences[detected_regime] = confidence
            
            # Check that regime is correctly identified in metadata
            self.assertIn("regime", metadata)
            self.assertEqual(metadata["regime"], detected_regime)
            
            # Check that thresholds are adapted based on regime
            self.assertIn("threshold", metadata)
            self.assertIn("regime_adjustment", metadata)
        
        # Thresholds should be different across regimes
        unique_confidences = set(regime_confidences.values())
        self.assertGreater(len(unique_confidences), 1, 
                           "Validator should produce different confidences across regimes")
    
    def test_ensemble_validation(self):
        """Test that the ensemble model provides more robust validation."""
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        
        # Test with a contradictory signal
        signal = self.test_signals["contradictory"]
        
        # First validate with ensemble enabled
        self.signal_validator.config["enable_ensemble"] = True
        is_valid_ensemble, confidence_ensemble, metadata_ensemble = self.signal_validator.validate_signal(
            signal, market_data, {}
        )
        
        # Then validate with ensemble disabled
        self.signal_validator.config["enable_ensemble"] = False
        is_valid_single, confidence_single, metadata_single = self.signal_validator.validate_signal(
            signal, market_data, {}
        )
        
        # Re-enable ensemble for other tests
        self.signal_validator.config["enable_ensemble"] = True
        
        # Ensemble validation should provide model-specific confidences
        self.assertIn("model_confidences", metadata_ensemble)
        
        # Single model validation should not have model confidences
        self.assertNotIn("model_confidences", metadata_single)
        
        # Ensemble validation should be more conservative for contradictory signals
        self.assertLessEqual(confidence_ensemble, confidence_single + 0.1)
    
    def test_uncertainty_quantification(self):
        """Test that uncertainty quantification works correctly."""
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        
        # Test with a strong trend signal (low uncertainty expected)
        strong_signal = self.test_signals["strong_trend"]
        
        # Test with a weak trend signal (higher uncertainty expected)
        weak_signal = self.test_signals["weak_trend"]
        
        # Validate both signals
        _, _, strong_metadata = self.signal_validator.validate_signal(
            strong_signal, market_data, {}
        )
        
        _, _, weak_metadata = self.signal_validator.validate_signal(
            weak_signal, market_data, {}
        )
        
        # Both results should include uncertainty metrics
        self.assertIn("uncertainty", strong_metadata)
        self.assertIn("uncertainty", weak_metadata)
        
        # Weak signal should have higher uncertainty
        self.assertGreater(weak_metadata["uncertainty"], strong_metadata["uncertainty"])
    
    def test_performance_tracking(self):
        """Test that performance tracking records signal outcomes correctly."""
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        signal = self.test_signals["strong_trend"]
        
        # First validate a signal
        is_valid, confidence, _ = self.signal_validator.validate_signal(
            signal, market_data, {}
        )
        
        # Record a successful outcome
        self.signal_validator.record_signal_outcome(
            signal_type=signal["signal_type"],
            regime="trending",
            outcome=True,
            profit_pct=2.5
        )
        
        # Record a failed outcome
        self.signal_validator.record_signal_outcome(
            signal_type=signal["signal_type"],
            regime="trending",
            outcome=False,
            profit_pct=-1.5
        )
        
        # Get performance metrics
        performance = self.signal_validator.get_performance_metrics()
        
        # Check that metrics were recorded
        self.assertIn(signal["signal_type"], performance)
        self.assertIn("trending", performance[signal["signal_type"]])
        
        # Check specific metrics
        strategy_perf = performance[signal["signal_type"]]["trending"]
        self.assertEqual(strategy_perf["total_signals"], 2)
        self.assertEqual(strategy_perf["successful_signals"], 1)
        self.assertEqual(strategy_perf["success_rate"], 0.5)
        self.assertAlmostEqual(strategy_perf["avg_profit"], 0.5)  # (2.5 - 1.5) / 2
    
    def test_full_integration_workflow(self):
        """Test the complete signal validation workflow with all components."""
        symbol = "TRENDING"
        market_data = {symbol: self.market_data[symbol]}
        signal = self.test_signals["strong_trend"]
        
        # Step 1: Classify the market regime
        regime = self.regime_classifier.classify_regime(market_data[symbol])
        
        # Step 2: Extract features from market data
        features = self.feature_engineer.extract_features(market_data[symbol])
        indicators = {symbol: features}
        
        # Step 3: Add historical signal data to the cluster analyzer
        # (we already have test data in the analyzer from previous tests)
        
        # Step 4: Find similar historical signals
        similar_signals = self.cluster_analyzer.find_similar_signals(signal)
        signal_with_cluster = signal.copy()
        signal_with_cluster["metadata"] = signal["metadata"].copy()
        signal_with_cluster["metadata"]["similar_signals"] = similar_signals
        
        # Step 5: Validate the signal with all the enhanced context
        is_valid, confidence, metadata = self.signal_validator.validate_signal(
            signal_with_cluster, market_data, indicators
        )
        
        # Check that everything was integrated correctly
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(confidence, float)
        self.assertEqual(metadata["regime"], regime)
        self.assertIn("model_confidences", metadata)
        self.assertIn("uncertainty", metadata)
        self.assertIn("similar_signals_success_rate", metadata)
        
        # Record an outcome to complete the cycle
        self.signal_validator.record_signal_outcome(
            signal_type=signal["signal_type"],
            regime=regime,
            outcome=True,
            profit_pct=3.0
        )
        
        # Verify performance metrics were updated
        performance = self.signal_validator.get_performance_metrics()
        self.assertIn(signal["signal_type"], performance)
        self.assertIn(regime, performance[signal["signal_type"]])


if __name__ == "__main__":
    unittest.main()
