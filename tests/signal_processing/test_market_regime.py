"""
Tests for the MarketRegimeDetector class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading_agent.signal_processing.regime import (
    MarketRegimeDetector,
    MarketRegime,
    volatility_regime,
    rolling_kmeans_regime
)


class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for the MarketRegimeDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector(
            volatility_window=10,
            trend_window=20,
            volatility_threshold=0.015,
            trend_threshold=0.6,
            range_threshold=0.3
        )
        
        # Create sample price data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()  # Earliest date first
        
        # Create trending price data
        trending_prices = [100]
        for i in range(1, 100):
            # Add trend with some noise
            trending_prices.append(trending_prices[-1] + 0.5 + np.random.normal(0, 0.2))
        
        self.trending_data = pd.DataFrame({
            'open': trending_prices,
            'high': [p + 0.5 for p in trending_prices],
            'low': [p - 0.5 for p in trending_prices],
            'close': trending_prices,
            'volume': [1000 for _ in range(100)]
        }, index=dates)
        
        # Create ranging price data
        ranging_prices = []
        center = 100
        for i in range(100):
            # Oscillate around center with some noise
            ranging_prices.append(center + 3 * np.sin(i / 10) + np.random.normal(0, 0.1))
        
        self.ranging_data = pd.DataFrame({
            'open': ranging_prices,
            'high': [p + 0.3 for p in ranging_prices],
            'low': [p - 0.3 for p in ranging_prices],
            'close': ranging_prices,
            'volume': [1000 for _ in range(100)]
        }, index=dates)
        
        # Create volatile price data
        volatile_prices = [100]
        for i in range(1, 100):
            # Add high volatility
            volatile_prices.append(volatile_prices[-1] + np.random.normal(0, 2.0))
        
        self.volatile_data = pd.DataFrame({
            'open': volatile_prices,
            'high': [p + 1.5 for p in volatile_prices],
            'low': [p - 1.5 for p in volatile_prices],
            'close': volatile_prices,
            'volume': [1000 for _ in range(100)]
        }, index=dates)

    def test_detect_trending_regime(self):
        """Test detection of trending market regime."""
        regime = self.detector.detect_regime(self.trending_data)
        self.assertEqual(regime, MarketRegime.TRENDING)

    def test_detect_ranging_regime(self):
        """Test detection of ranging market regime."""
        regime = self.detector.detect_regime(self.ranging_data)
        self.assertEqual(regime, MarketRegime.RANGING)

    def test_detect_volatile_regime(self):
        """Test detection of volatile market regime."""
        regime = self.detector.detect_regime(self.volatile_data)
        self.assertEqual(regime, MarketRegime.VOLATILE)

    def test_detect_regime_with_series(self):
        """Test regime detection with a Series instead of DataFrame."""
        regime = self.detector.detect_regime(self.trending_data['close'])
        self.assertEqual(regime, MarketRegime.TRENDING)

    def test_get_regime_history(self):
        """Test getting historical regime data."""
        regimes = self.detector.get_regime_history(self.trending_data, window=30)
        
        # Check that regimes were calculated
        self.assertEqual(len(regimes), len(self.trending_data))
        
        # Check that early data points are UNKNOWN
        self.assertEqual(regimes.iloc[0], MarketRegime.UNKNOWN)
        
        # Check that later data points have valid regimes
        self.assertNotEqual(regimes.iloc[-1], MarketRegime.UNKNOWN)

    def test_get_regime_parameters(self):
        """Test getting recommended parameters for different regimes."""
        trending_params = self.detector.get_regime_parameters(MarketRegime.TRENDING)
        ranging_params = self.detector.get_regime_parameters(MarketRegime.RANGING)
        volatile_params = self.detector.get_regime_parameters(MarketRegime.VOLATILE)
        
        # Check that parameters exist for each regime
        self.assertIsInstance(trending_params, dict)
        self.assertIsInstance(ranging_params, dict)
        self.assertIsInstance(volatile_params, dict)
        
        # Check specific parameter values
        self.assertGreater(trending_params['sentiment_weight'], ranging_params['sentiment_weight'])
        self.assertGreater(ranging_params['technical_weight'], volatile_params['technical_weight'])
        self.assertGreater(volatile_params['stop_loss_pct'], ranging_params['stop_loss_pct'])

    def test_volatility_regime_function(self):
        """Test the volatility_regime function."""
        regimes = volatility_regime(self.volatile_data['close'], window=10, threshold=0.01)
        
        # Check that regimes were calculated
        self.assertEqual(len(regimes), len(self.volatile_data))
        
        # Check that most data points are high_vol for volatile data
        high_vol_count = (regimes == 'high_vol').sum()
        self.assertGreater(high_vol_count, len(regimes) // 2)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create small dataset
        small_data = self.trending_data.iloc[:10]
        
        # Should return UNKNOWN for insufficient data
        regime = self.detector.detect_regime(small_data)
        self.assertEqual(regime, MarketRegime.UNKNOWN)


if __name__ == '__main__':
    unittest.main()
