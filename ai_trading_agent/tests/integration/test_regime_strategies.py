"""
Integration tests for the regime-specific trading strategies.

These tests validate that the regime strategies work correctly with the ML validator
and market regime classifier in an integrated environment.
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

from ai_trading_agent.strategies.regime_strategies import BaseRegimeStrategy
from ai_trading_agent.strategies.range_bound_strategy import RangeBoundStrategy
from ai_trading_agent.strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy
from ai_trading_agent.strategies.mean_reversion_strategy import MeanReversionStrategy
from ai_trading_agent.strategies.regime_transition_strategy import RegimeTransitionStrategy
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier
from ai_trading_agent.ml.enhanced_signal_validator import EnhancedSignalValidator

class TestRegimeStrategyIntegration(unittest.TestCase):
    """Test the integration of regime strategies with other components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create configuration for testing
        self.regime_config = {
            "lookback_window": 50,
            "enable_ml_enhancement": True,
            "smoothing_window": 3
        }
        
        self.strategy_config = {
            "lookback_window": 50,
            "sensitivity": 1.0,
            "confirmation_threshold": 1,  # Lower for testing
            "enable_filters": True,
            "regime_config": self.regime_config
        }
        
        self.validator_config = {
            "min_confidence": 0.6,
            "enable_regime_adaptation": True,
            "performance_tracking": True
        }
        
        # Create test data
        self.market_data = self._generate_test_data()
        
        # Initialize components
        self.regime_classifier = MarketRegimeClassifier(self.regime_config)
        self.signal_validator = EnhancedSignalValidator(self.validator_config)
        
        # Initialize strategies
        self.range_strategy = RangeBoundStrategy(self.strategy_config)
        self.volatility_strategy = VolatilityBreakoutStrategy(self.strategy_config)
        self.mean_reversion_strategy = MeanReversionStrategy(self.strategy_config)
        self.transition_strategy = RegimeTransitionStrategy(self.strategy_config)
    
    def _generate_test_data(self):
        """Generate test market data with different regime characteristics."""
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
        
        # Generate mean-reverting market data
        mean_rev_data = pd.DataFrame(index=dates)
        center = 150
        deviation = np.random.normal(0, 20, len(dates))
        mean_rev_force = 0.3  # Mean reversion strength
        for i in range(1, len(dates)):
            deviation[i] = deviation[i-1] * (1 - mean_rev_force) + np.random.normal(0, 5)
        
        mean_rev_data['open'] = center + deviation
        mean_rev_data['high'] = mean_rev_data['open'] + np.random.uniform(1, 5, len(dates))
        mean_rev_data['low'] = mean_rev_data['open'] - np.random.uniform(1, 5, len(dates))
        mean_rev_data['close'] = center + deviation + np.random.normal(0, 2, len(dates))
        mean_rev_data['volume'] = np.random.uniform(1000, 5000, len(dates))
        test_data["MEAN_REVERTING"] = mean_rev_data
        
        # Generate transition market data (starts ranging, then trends, then volatile)
        transition_data = pd.DataFrame(index=dates)
        
        # First part: ranging
        first_third = len(dates) // 3
        transition_data.loc[dates[:first_third], 'close'] = 150 + np.random.normal(0, 5, first_third)
        
        # Second part: trending
        second_third = 2 * len(dates) // 3
        transition_data.loc[dates[first_third:second_third], 'close'] = np.linspace(
            transition_data.loc[dates[first_third-1], 'close'], 
            transition_data.loc[dates[first_third-1], 'close'] + 50, 
            second_third - first_third
        ) + np.random.normal(0, 3, second_third - first_third)
        
        # Third part: volatile
        transition_data.loc[dates[second_third:], 'close'] = transition_data.loc[dates[second_third-1], 'close'] + np.cumsum(
            np.random.normal(0, 5, len(dates) - second_third)
        )
        
        # Fill in other columns
        transition_data['open'] = transition_data['close'].shift(1).fillna(transition_data['close'][0])
        transition_data['high'] = transition_data[['open', 'close']].max(axis=1) + np.random.uniform(1, 8, len(dates))
        transition_data['low'] = transition_data[['open', 'close']].min(axis=1) - np.random.uniform(1, 8, len(dates))
        transition_data['volume'] = np.random.uniform(1000, 5000, len(dates))
        test_data["TRANSITION"] = transition_data
        
        return test_data
    
    def test_range_bound_strategy(self):
        """Test that RangeBoundStrategy generates appropriate signals for ranging markets."""
        signals = self.range_strategy.generate_signals({"RANGING": self.market_data["RANGING"]})
        
        # Check if signals were generated
        self.assertIn("RANGING", signals)
        signal_data = signals["RANGING"]
        
        # Validate signal properties
        self.assertIn("signal", signal_data)
        self.assertIn("direction", signal_data)
        self.assertIn("signal_type", signal_data)
        self.assertIn("metadata", signal_data)
        
        # Verify it's the correct strategy type
        self.assertEqual(signal_data["signal_type"], "RangeBoundStrategy")
        
        # Verify signal metadata contains expected fields
        metadata = signal_data["metadata"]
        self.assertIn("oscillator_component", metadata)
        self.assertIn("support_resistance_component", metadata)
        self.assertIn("mean_reversion_component", metadata)
        
        # The range strategy should perform better on ranging data than on trending data
        trending_signals = self.range_strategy.generate_signals({"TRENDING": self.market_data["TRENDING"]})
        # Skip if no signals generated for trending data due to filtering
        if "TRENDING" in trending_signals:
            self.assertGreaterEqual(
                abs(signal_data["signal"]),
                abs(trending_signals["TRENDING"]["signal"]),
                "Range strategy should generate stronger signals for ranging markets"
            )
    
    def test_volatility_breakout_strategy(self):
        """Test that VolatilityBreakoutStrategy generates appropriate signals for volatile markets."""
        signals = self.volatility_strategy.generate_signals({"VOLATILE": self.market_data["VOLATILE"]})
        
        # Check if signals were generated
        self.assertIn("VOLATILE", signals)
        signal_data = signals["VOLATILE"]
        
        # Validate signal properties
        self.assertIn("signal", signal_data)
        self.assertIn("direction", signal_data)
        self.assertIn("signal_type", signal_data)
        self.assertIn("metadata", signal_data)
        
        # Verify it's the correct strategy type
        self.assertEqual(signal_data["signal_type"], "VolatilityBreakoutStrategy")
        
        # Verify signal metadata contains expected fields
        metadata = signal_data["metadata"]
        self.assertIn("volatility_expansion_component", metadata)
        self.assertIn("bb_breakout_component", metadata)
        self.assertIn("volume_price_component", metadata)
        
        # The volatility strategy should perform better on volatile data than on ranging data
        ranging_signals = self.volatility_strategy.generate_signals({"RANGING": self.market_data["RANGING"]})
        # Skip if no signals generated for ranging data due to filtering
        if "RANGING" in ranging_signals:
            self.assertGreaterEqual(
                abs(signal_data["signal"]),
                abs(ranging_signals["RANGING"]["signal"]),
                "Volatility strategy should generate stronger signals for volatile markets"
            )
    
    def test_mean_reversion_strategy(self):
        """Test that MeanReversionStrategy generates appropriate signals for mean-reverting markets."""
        signals = self.mean_reversion_strategy.generate_signals({"MEAN_REVERTING": self.market_data["MEAN_REVERTING"]})
        
        # Check if signals were generated
        self.assertIn("MEAN_REVERTING", signals)
        signal_data = signals["MEAN_REVERTING"]
        
        # Validate signal properties
        self.assertIn("signal", signal_data)
        self.assertIn("direction", signal_data)
        self.assertIn("signal_type", signal_data)
        self.assertIn("metadata", signal_data)
        
        # Verify it's the correct strategy type
        self.assertEqual(signal_data["signal_type"], "MeanReversionStrategy")
        
        # Verify signal metadata contains expected fields
        metadata = signal_data["metadata"]
        self.assertIn("zscore_component", metadata)
        self.assertIn("ma_deviation_component", metadata)
        self.assertIn("rsi_extreme_component", metadata)
        
        # The mean reversion strategy should perform better on mean-reverting data than on trending data
        trending_signals = self.mean_reversion_strategy.generate_signals({"TRENDING": self.market_data["TRENDING"]})
        # Skip if no signals generated for trending data due to filtering
        if "TRENDING" in trending_signals:
            self.assertGreaterEqual(
                abs(signal_data["signal"]),
                abs(trending_signals["TRENDING"]["signal"]),
                "Mean reversion strategy should generate stronger signals for mean-reverting markets"
            )
    
    def test_regime_transition_strategy(self):
        """Test that RegimeTransitionStrategy generates appropriate signals for transitioning markets."""
        signals = self.transition_strategy.generate_signals({"TRANSITION": self.market_data["TRANSITION"]})
        
        # Check if signals were generated
        self.assertIn("TRANSITION", signals)
        signal_data = signals["TRANSITION"]
        
        # Validate signal properties
        self.assertIn("signal", signal_data)
        self.assertIn("direction", signal_data)
        self.assertIn("signal_type", signal_data)
        self.assertIn("metadata", signal_data)
        
        # Verify it's the correct strategy type
        self.assertEqual(signal_data["signal_type"], "RegimeTransitionStrategy")
        
        # Verify signal metadata contains expected fields
        metadata = signal_data["metadata"]
        self.assertIn("volatility_change_component", metadata)
        self.assertIn("trend_reversal_component", metadata)
        self.assertIn("correlation_breakdown_component", metadata)
    
    def test_ml_validator_integration(self):
        """Test integration of the ML validator with regime strategies."""
        # Generate signals from the range strategy
        signals = self.range_strategy.generate_signals({"RANGING": self.market_data["RANGING"]})
        
        # Make sure signals were generated
        self.assertIn("RANGING", signals)
        signal = signals["RANGING"]
        
        # Validate signal using the ML validator
        is_valid, confidence, metadata = self.signal_validator.validate_signal(
            signal, 
            {"RANGING": self.market_data["RANGING"]},
            {}  # Empty indicators for simplicity
        )
        
        # Check validation results
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(metadata, dict)
        
        # Verify metadata contains expected fields
        self.assertIn("regime", metadata)
        self.assertIn("confidence", metadata)
        self.assertIn("threshold", metadata)
    
    def test_regime_classifier_integration(self):
        """Test integration of the market regime classifier with strategies."""
        # Test with range-bound data
        regime = self.regime_classifier.classify_regime(self.market_data["RANGING"])
        self.assertEqual(regime, "ranging", "Range-bound data should be classified as 'ranging'")
        
        # Test with trending data
        regime = self.regime_classifier.classify_regime(self.market_data["TRENDING"])
        self.assertEqual(regime, "trending", "Trending data should be classified as 'trending'")
        
        # Test with volatile data
        regime = self.regime_classifier.classify_regime(self.market_data["VOLATILE"])
        self.assertEqual(regime, "volatile", "Volatile data should be classified as 'volatile'")
    
    def test_strategy_selection_based_on_regime(self):
        """Test that the appropriate strategy is selected based on the detected regime."""
        # Simulate strategy selection based on regime
        strategies = {
            "ranging": self.range_strategy,
            "trending": None,  # We don't have a trending strategy yet
            "volatile": self.volatility_strategy,
            "unknown": None
        }
        
        # Test with range-bound data
        regime = self.regime_classifier.classify_regime(self.market_data["RANGING"])
        selected_strategy = strategies.get(regime)
        self.assertEqual(selected_strategy, self.range_strategy)
        
        # Test with volatile data
        regime = self.regime_classifier.classify_regime(self.market_data["VOLATILE"])
        selected_strategy = strategies.get(regime)
        self.assertEqual(selected_strategy, self.volatility_strategy)
    
    def test_end_to_end_signal_generation_and_validation(self):
        """Test the full workflow from regime classification to signal validation."""
        symbol = "RANGING"
        market_data = {symbol: self.market_data[symbol]}
        
        # Step 1: Classify the market regime
        regime = self.regime_classifier.classify_regime(market_data[symbol])
        
        # Step 2: Select the appropriate strategy based on regime
        strategies = {
            "ranging": self.range_strategy,
            "trending": None,  # We don't have a trending strategy yet
            "volatile": self.volatility_strategy,
            "unknown": None
        }
        selected_strategy = strategies.get(regime)
        
        # Skip test if no strategy for detected regime
        if selected_strategy is None:
            self.skipTest(f"No strategy available for {regime} regime")
        
        # Step 3: Generate signals using the selected strategy
        signals = selected_strategy.generate_signals(market_data)
        self.assertIn(symbol, signals)
        
        # Step 4: Validate the signals using the ML validator
        signal = signals[symbol]
        is_valid, confidence, metadata = self.signal_validator.validate_signal(
            signal, market_data, {}
        )
        
        # Check that the end-to-end process works
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(confidence, float)
        self.assertIn("regime", metadata)
        
        # The detected regime in the validator should match the one from the classifier
        self.assertEqual(metadata["regime"], regime)


if __name__ == "__main__":
    unittest.main()
