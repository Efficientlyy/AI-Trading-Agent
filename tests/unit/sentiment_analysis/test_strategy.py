"""
Unit tests for the sentiment-based trading strategy module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.sentiment_analysis.strategy import SentimentStrategy
from src.trading_engine.models import Order, OrderType, OrderSide


class TestSentimentStrategy:
    """Tests for the SentimentStrategy class."""
    
    def test_initialization(self):
        """Test initialization of the strategy."""
        config = {
            'sentiment_threshold_long': 0.4,
            'sentiment_threshold_short': -0.4,
            'sentiment_window': 5,
            'risk_per_trade': 0.03,
            'max_position_size': 0.25
        }
        strategy = SentimentStrategy(config)
        
        assert strategy.config == config
        assert strategy.name == "SentimentStrategy"
        assert strategy.sentiment_threshold_long == 0.4
        assert strategy.sentiment_threshold_short == -0.4
        assert strategy.sentiment_window == 5
        assert strategy.risk_per_trade == 0.03
        assert strategy.max_position_size == 0.25
        
    def test_preprocess_data(self):
        """Test preprocessing of market and sentiment data."""
        config = {}
        strategy = SentimentStrategy(config)
        
        # Create sample market data
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'symbol': ['BTC/USD'] * 5,
            'open': [40000, 41000, 42000, 43000, 44000],
            'high': [41000, 42000, 43000, 44000, 45000],
            'low': [39000, 40000, 41000, 42000, 43000],
            'close': [41000, 42000, 43000, 44000, 45000],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }).set_index('timestamp')
        
        # Create sample sentiment data
        sentiment_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'symbol': ['BTC/USD'] * 5,
            'source': ['twitter'] * 5,
            'sentiment_score_mean': [0.2, 0.3, 0.4, 0.5, 0.6],
            'sentiment_score_count': [100, 110, 120, 130, 140],
            'sentiment_confidence': [0.7, 0.7, 0.8, 0.8, 0.9]
        })
        
        # Call the preprocess_data method
        result = strategy.preprocess_data(market_data, sentiment_data)
        
        # Check that the result has the expected columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert 'symbol' in result.columns
        assert 'twitter_weighted_score' in result.columns
        assert 'combined_sentiment_score' in result.columns
        
        # Check that ATR is calculated
        assert 'atr' in result.columns
        
    def test_generate_signals(self):
        """Test generation of trading signals."""
        config = {
            'sentiment_threshold_long': 0.3,
            'sentiment_threshold_short': -0.3
        }
        strategy = SentimentStrategy(config)
        
        # Create sample data with combined market and sentiment data
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5),
            'symbol': ['BTC/USD'] * 5,
            'open': [40000, 41000, 42000, 43000, 44000],
            'high': [41000, 42000, 43000, 44000, 45000],
            'low': [39000, 40000, 41000, 42000, 43000],
            'close': [41000, 42000, 43000, 44000, 45000],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'combined_sentiment_score': [0.1, 0.2, 0.4, -0.4, -0.2],
            'atr': [1000, 1000, 1000, 1000, 1000]
        }).set_index('timestamp')
        
        # Call the generate_signals method
        result = strategy.generate_signals(data)
        
        # Check that the result has the expected columns
        assert 'signal' in result.columns
        assert 'signal_strength' in result.columns
        assert 'position_size' in result.columns
        assert 'stop_loss' in result.columns
        assert 'take_profit' in result.columns
        
        # Check that signals are generated correctly
        assert result['signal'].iloc[0] == 0  # Below threshold
        assert result['signal'].iloc[1] == 0  # Below threshold
        assert result['signal'].iloc[2] == 1  # Above long threshold
        assert result['signal'].iloc[3] == -1  # Below short threshold
        assert result['signal'].iloc[4] == 0  # Between thresholds
        
        # Check that position sizes are calculated
        assert result['position_size'].iloc[2] > 0
        assert result['position_size'].iloc[3] > 0
        
        # Check that stop loss and take profit are calculated
        assert result['stop_loss'].iloc[2] < result['close'].iloc[2]  # Long stop loss below price
        assert result['take_profit'].iloc[2] > result['close'].iloc[2]  # Long take profit above price
        assert result['stop_loss'].iloc[3] > result['close'].iloc[3]  # Short stop loss above price
        assert result['take_profit'].iloc[3] < result['close'].iloc[3]  # Short take profit below price
        
    def test_volatility_position_sizing(self):
        """Test volatility-based position sizing."""
        config = {
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'stop_loss_atr_multiplier': 2.0
        }
        strategy = SentimentStrategy(config)
        
        # Create a sample row with market data
        row = pd.Series({
            'close': 40000,
            'atr': 2000
        })
        
        # Call the _volatility_position_sizing method
        position_size, stop_loss, take_profit = strategy._volatility_position_sizing(
            row, signal=1, signal_strength=0.8
        )
        
        # Check that position size is calculated correctly
        # risk_per_trade / stop_distance_percentage * signal_strength
        # 0.02 / (2000*2/40000) * 0.8 = 0.02 / 0.1 * 0.8 = 0.16
        assert position_size == pytest.approx(0.16, abs=0.01)
        
        # Check that stop loss is calculated correctly
        # close - atr * stop_loss_atr_multiplier
        assert stop_loss == pytest.approx(40000 - 2000 * 2.0, abs=0.01)
        
        # Check that take profit is calculated correctly
        # For long: close + atr * stop_loss_atr_multiplier * (take_profit_atr_multiplier / stop_loss_atr_multiplier)
        assert take_profit > row['close']
        
        # Test with short signal
        position_size, stop_loss, take_profit = strategy._volatility_position_sizing(
            row, signal=-1, signal_strength=0.8
        )
        
        # Check that stop loss is above price for short positions
        assert stop_loss > row['close']
        
        # Check that take profit is below price for short positions
        assert take_profit < row['close']
        
    def test_kelly_position_sizing(self):
        """Test Kelly criterion position sizing."""
        config = {
            'kelly_fraction': 0.5,
            'win_rate_window': 10,
            'max_position_size': 0.2
        }
        strategy = SentimentStrategy(config)
        
        # Create a sample row with market data
        row = pd.Series({
            'close': 40000,
            'atr': 2000
        })
        
        # Add some trade history
        strategy.trade_history = [
            {'profit': 100} for _ in range(7)  # 7 winning trades
        ] + [
            {'profit': -50} for _ in range(3)  # 3 losing trades
        ]
        
        # Call the _kelly_position_sizing method
        position_size, stop_loss, take_profit = strategy._kelly_position_sizing(
            row, signal=1, signal_strength=0.8
        )
        
        # Check that position size is calculated correctly and within limits
        assert 0 <= position_size <= strategy.max_position_size
        
        # Check that stop loss and take profit are calculated
        assert stop_loss < row['close']  # Long stop loss below price
        assert take_profit > row['close']  # Long take profit above price
        
    def test_fixed_position_sizing(self):
        """Test fixed position sizing."""
        config = {
            'max_position_size': 0.2
        }
        strategy = SentimentStrategy(config)
        
        # Create a sample row with market data
        row = pd.Series({
            'close': 40000,
            'atr': 2000
        })
        
        # Call the _fixed_position_sizing method
        position_size, stop_loss, take_profit = strategy._fixed_position_sizing(
            row, signal=1, signal_strength=0.8
        )
        
        # Check that position size is calculated correctly
        # max_position_size * signal_strength
        assert position_size == pytest.approx(0.2 * 0.8, abs=0.01)
        
        # Check that stop loss and take profit are calculated
        assert stop_loss < row['close']  # Long stop loss below price
        assert take_profit > row['close']  # Long take profit above price
        
    def test_generate_orders(self):
        """Test generation of orders from signals."""
        config = {}
        strategy = SentimentStrategy(config)
        
        # Create sample signals
        signals = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=3),
            'symbol': ['BTC/USD', 'ETH/USD', 'XRP/USD'],
            'signal': [1, -1, 0],  # Long, short, none
            'position_size': [0.1, 0.05, 0],
            'stop_loss': [38000, 3200, None],
            'take_profit': [45000, 2800, None]
        }).set_index('timestamp')
        
        # Current positions (empty)
        current_positions = {}
        
        # Call the generate_orders method
        timestamp = pd.Timestamp('2023-01-01')
        orders = strategy.generate_orders(signals, timestamp, current_positions)
        
        # Check that the correct number of orders are generated
        # 2 symbols with signals * 3 orders each (entry, stop loss, take profit)
        assert len(orders) == 6
        
        # Check that the orders have the correct properties
        btc_orders = [o for o in orders if o.symbol == 'BTC/USD']
        eth_orders = [o for o in orders if o.symbol == 'ETH/USD']
        
        assert len(btc_orders) == 3
        assert len(eth_orders) == 3
        
        # Check BTC orders (long)
        btc_entry = next(o for o in btc_orders if o.order_type == OrderType.MARKET)
        btc_sl = next(o for o in btc_orders if o.order_type == OrderType.STOP)
        btc_tp = next(o for o in btc_orders if o.order_type == OrderType.LIMIT)
        
        assert btc_entry.side == OrderSide.BUY
        assert btc_sl.side == OrderSide.SELL
        assert btc_tp.side == OrderSide.SELL
        assert btc_sl.price == 38000
        assert btc_tp.price == 45000
        
        # Check ETH orders (short)
        eth_entry = next(o for o in eth_orders if o.order_type == OrderType.MARKET)
        eth_sl = next(o for o in eth_orders if o.order_type == OrderType.STOP)
        eth_tp = next(o for o in eth_orders if o.order_type == OrderType.LIMIT)
        
        assert eth_entry.side == OrderSide.SELL
        assert eth_sl.side == OrderSide.BUY
        assert eth_tp.side == OrderSide.BUY
        assert eth_sl.price == 3200
        assert eth_tp.price == 2800
        
        # Test with existing positions
        current_positions = {
            'BTC/USD': {'side': 'long', 'quantity': 0.1},
            'ETH/USD': {'side': 'long', 'quantity': 0.2}
        }
        
        orders = strategy.generate_orders(signals, timestamp, current_positions)
        
        # Check that the correct number of orders are generated
        # BTC: Already long, no new orders
        # ETH: Close long position (1 order) + Open short position (3 orders)
        assert len(orders) == 4
        
        # Check that the ETH close order is generated
        eth_close = next(o for o in orders if o.symbol == 'ETH/USD' and 'close' in o.order_id)
        assert eth_close.side == OrderSide.SELL
        assert eth_close.quantity == 0.2
        
    def test_update_trade_history(self):
        """Test updating trade history."""
        config = {
            'win_rate_window': 5
        }
        strategy = SentimentStrategy(config)
        
        # Add some initial trade history
        strategy.trade_history = [
            {'profit': 100} for _ in range(3)
        ]
        
        # Update with a new trade
        strategy.update_trade_history({'profit': -50})
        
        # Check that the trade history is updated
        assert len(strategy.trade_history) == 4
        assert strategy.trade_history[-1]['profit'] == -50
        
        # Add many trades to test limiting
        for i in range(100):
            strategy.update_trade_history({'profit': i})
            
        # Check that the trade history is limited
        assert len(strategy.trade_history) == 100  # Default max_history is 100
