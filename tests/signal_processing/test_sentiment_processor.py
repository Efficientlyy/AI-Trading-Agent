"""
Tests for the SentimentSignalProcessor class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading_agent.signal_processing.sentiment_processor import (
    SentimentSignalProcessor,
    SentimentSignal,
    TradingMode
)


class TestSentimentSignalProcessor(unittest.TestCase):
    """Test cases for the SentimentSignalProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SentimentSignalProcessor(
            threshold=0.2,
            window_size=3,
            sentiment_weight=0.4,
            min_confidence=0.6
        )
        
        # Create sample sentiment data
        dates = [datetime.now() - timedelta(days=i) for i in range(10)]
        dates.reverse()  # Earliest date first
        
        # Create sentiment scores with a clear trend
        sentiment_values = [0.1, 0.15, 0.25, 0.3, 0.35, 0.2, 0.1, -0.1, -0.3, -0.4]
        self.sentiment_data = pd.Series(sentiment_values, index=dates)
        
        # Create sample price data
        self.price_data = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 for _ in range(10)]
        }, index=dates)

    def test_trading_mode_selection(self):
        """Test that trading modes are correctly selected based on timeframe."""
        self.assertEqual(self.processor.select_trading_mode('1m'), TradingMode.SCALPING)
        self.assertEqual(self.processor.select_trading_mode('5m'), TradingMode.SCALPING)
        self.assertEqual(self.processor.select_trading_mode('15m'), TradingMode.INTRADAY)
        self.assertEqual(self.processor.select_trading_mode('1h'), TradingMode.INTRADAY)
        self.assertEqual(self.processor.select_trading_mode('4h'), TradingMode.SWING)
        self.assertEqual(self.processor.select_trading_mode('1d'), TradingMode.SWING)
        self.assertEqual(self.processor.select_trading_mode('1w'), TradingMode.POSITION)

    def test_sentiment_weight_by_trading_mode(self):
        """Test that sentiment weights are correctly assigned based on trading mode."""
        self.assertAlmostEqual(self.processor.get_sentiment_weight(TradingMode.SCALPING), 0.0)
        self.assertAlmostEqual(self.processor.get_sentiment_weight(TradingMode.INTRADAY), 0.3)
        self.assertAlmostEqual(self.processor.get_sentiment_weight(TradingMode.SWING), 0.6)
        self.assertAlmostEqual(self.processor.get_sentiment_weight(TradingMode.POSITION), 0.8)

    def test_no_signals_for_scalping(self):
        """Test that no signals are generated for scalping timeframes."""
        signals = self.processor.process_sentiment_data(
            symbol='BTC',
            historical_sentiment=self.sentiment_data,
            timeframe='1m',
            price_data=self.price_data
        )
        self.assertEqual(len(signals), 0)

    def test_process_sentiment_data(self):
        """Test processing sentiment data into trading signals."""
        signals = self.processor.process_sentiment_data(
            symbol='BTC',
            historical_sentiment=self.sentiment_data,
            timeframe='1d',
            price_data=self.price_data
        )
        
        # Check that signals were generated
        self.assertGreater(len(signals), 0)
        
        # Check signal properties
        for signal in signals:
            self.assertEqual(signal.symbol, 'BTC')
            self.assertIn(signal.signal_type, ['buy', 'sell'])
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)
            self.assertGreaterEqual(signal.confidence, self.processor.min_confidence)
            self.assertLessEqual(signal.confidence, 1.0)
            self.assertEqual(signal.timeframe, '1d')
            self.assertEqual(signal.source, 'sentiment_analysis')
            self.assertIsInstance(signal.metadata, dict)

    def test_signal_strength_calculation(self):
        """Test calculation of signal strength."""
        # Generate raw signals
        raw_signals = self.processor.signal_generator.generate_signals_from_scores(self.sentiment_data)
        
        # Calculate signal strength
        strengths = self.processor._calculate_signal_strength(self.sentiment_data, raw_signals)
        
        # Check that strengths were calculated for all non-hold signals
        for timestamp, signal in raw_signals.items():
            if signal != 0:
                self.assertIn(timestamp, strengths)
                self.assertGreaterEqual(strengths[timestamp], 0.0)
                self.assertLessEqual(strengths[timestamp], 1.0)

    def test_signal_confidence_calculation(self):
        """Test calculation of signal confidence."""
        # Generate raw signals
        raw_signals = self.processor.signal_generator.generate_signals_from_scores(self.sentiment_data)
        
        # Calculate signal confidence
        confidences = self.processor._calculate_signal_confidence(self.sentiment_data, raw_signals)
        
        # Check that confidences were calculated for all non-hold signals
        for timestamp, signal in raw_signals.items():
            if signal != 0:
                self.assertIn(timestamp, confidences)
                self.assertGreaterEqual(confidences[timestamp], 0.0)
                self.assertLessEqual(confidences[timestamp], 1.0)

    def test_signal_to_dict_conversion(self):
        """Test conversion of SentimentSignal to dictionary and back."""
        # Create a sample signal
        signal = SentimentSignal(
            symbol='BTC',
            timestamp=pd.Timestamp('2023-01-01'),
            signal_type='buy',
            strength=0.8,
            confidence=0.7,
            timeframe='1d',
            metadata={'raw_score': 0.5}
        )
        
        # Convert to dictionary
        signal_dict = signal.to_dict()
        
        # Check dictionary values
        self.assertEqual(signal_dict['symbol'], 'BTC')
        self.assertEqual(signal_dict['signal_type'], 'buy')
        self.assertEqual(signal_dict['strength'], 0.8)
        self.assertEqual(signal_dict['confidence'], 0.7)
        self.assertEqual(signal_dict['timeframe'], '1d')
        self.assertEqual(signal_dict['metadata']['raw_score'], 0.5)
        
        # Convert back to signal
        recreated_signal = SentimentSignal.from_dict(signal_dict)
        
        # Check recreated signal
        self.assertEqual(recreated_signal.symbol, 'BTC')
        self.assertEqual(recreated_signal.signal_type, 'buy')
        self.assertEqual(recreated_signal.strength, 0.8)
        self.assertEqual(recreated_signal.confidence, 0.7)
        self.assertEqual(recreated_signal.timeframe, '1d')
        self.assertEqual(recreated_signal.metadata['raw_score'], 0.5)


if __name__ == '__main__':
    unittest.main()
