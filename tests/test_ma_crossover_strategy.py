"""Tests for the moving average crossover strategy."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.models.market_data import TimeFrame
from src.models.signals import SignalType
from src.strategy.ma_crossover import MovingAverageCrossoverStrategy


@pytest.mark.asyncio
async def test_strategy_initialization():
    """Test strategy initialization."""
    # Create a strategy instance
    strategy = MovingAverageCrossoverStrategy()
    
    # Check default configuration
    assert strategy.fast_ma_type == "EMA"
    assert strategy.fast_ma_period == 12
    assert strategy.slow_ma_type == "EMA"
    assert strategy.slow_ma_period == 26
    assert strategy.min_confidence == 0.6
    assert strategy.use_stop_loss is True
    assert strategy.use_take_profit is True


@pytest.mark.asyncio
async def test_process_indicator():
    """Test processing of indicator events."""
    # Create a strategy instance with patched publish_signal method
    strategy = MovingAverageCrossoverStrategy()
    strategy.publish_signal = MagicMock()
    
    # Mock the timeframe
    timeframe = TimeFrame("1h")
    symbol = "BTC/USDT"
    
    # Create some test indicator data - no crossover yet
    now = datetime.utcnow()
    t1 = now - timedelta(hours=2)
    t2 = now - timedelta(hours=1)
    
    # Test EMA indicator values - no crossover
    ema_values = {
        t1: {"EMA12": 10000, "EMA26": 10100},
        t2: {"EMA12": 10050, "EMA26": 10120}
    }
    
    # Process the indicator
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that no signal was published
    strategy.publish_signal.assert_not_called()
    
    # Now test a bullish crossover
    t3 = now
    ema_values = {
        t3: {"EMA12": 10150, "EMA26": 10130}
    }
    
    # Process the indicator with the crossover
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that a long signal was published
    strategy.publish_signal.assert_called_once()
    args, kwargs = strategy.publish_signal.call_args
    assert kwargs["symbol"] = = symbol
    assert kwargs["signal_type"] = = SignalType.ENTRY
    assert kwargs["direction"] = = "long"
    assert kwargs["timeframe"] = = timeframe
    
    # Reset the mock and test a bearish crossover
    strategy.publish_signal.reset_mock()
    
    # First simulate that we have an active long signal
    strategy.active_signals[symbol] = MagicMock()
    strategy.active_signals[symbol].direction = "long"
    
    t4 = now + timedelta(hours=1)
    t5 = now + timedelta(hours=2)
    
    # Prices move up then start falling
    ema_values = {
        t4: {"EMA12": 10200, "EMA26": 10150},
        t5: {"EMA12": 10120, "EMA26": 10140}
    }
    
    # Process the indicator with the bearish crossover
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that an exit signal was published
    strategy.publish_signal.assert_called()
    args, kwargs = strategy.publish_signal.call_args_list[0]
    assert kwargs["symbol"] = = symbol
    assert kwargs["signal_type"] = = SignalType.EXIT
    assert kwargs["direction"] = = "long"
    assert kwargs["timeframe"] = = timeframe


@pytest.mark.asyncio
async def test_signal_confidence():
    """Test signal confidence calculation."""
    # Create a strategy instance with patched publish methods
    strategy = MovingAverageCrossoverStrategy()
    strategy.publish_signal = MagicMock()
    strategy.publish_status = MagicMock()
    strategy.publish_error = MagicMock()
    
    # Mock the timeframe and symbol
    timeframe = TimeFrame("1h")
    symbol = "BTC/USDT"
    
    # Set a higher minimum confidence to test filtering
    strategy.min_confidence = 0.8
    
    # Create test data with a small crossover (low confidence)
    now = datetime.utcnow()
    t1 = now - timedelta(hours=1)
    t2 = now
    
    # Small crossover (low confidence)
    ema_values = {
        t1: {"EMA12": 10000, "EMA26": 10010},
        t2: {"EMA12": 10011, "EMA26": 10010}
    }
    
    # Process the indicator
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that no signal was published due to low confidence
    strategy.publish_signal.assert_not_called()
    
    # Now test a stronger crossover (high confidence)
    t3 = now + timedelta(hours=1)
    ema_values = {
        t3: {"EMA12": 10100, "EMA26": 10000}
    }
    
    # Process the indicator
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that a signal was published this time
    strategy.publish_signal.assert_called_once()


@pytest.mark.asyncio
async def test_stop_loss_take_profit():
    """Test stop loss and take profit calculation."""
    # Create a strategy instance with patched publish methods
    strategy = MovingAverageCrossoverStrategy()
    strategy.publish_signal = MagicMock()
    
    # Configure stop loss and take profit
    strategy.use_stop_loss = True
    strategy.stop_loss_pct = 0.05  # 5%
    strategy.use_take_profit = True
    strategy.take_profit_pct = 0.1  # 10%
    
    # Mock the timeframe and symbol
    timeframe = TimeFrame("1h")
    symbol = "BTC/USDT"
    
    # Create test data with a crossover
    now = datetime.utcnow()
    t1 = now - timedelta(hours=1)
    t2 = now
    
    # Bullish crossover
    ema_values = {
        t1: {"EMA12": 10000, "EMA26": 10100},
        t2: {"EMA12": 10200, "EMA26": 10150}
    }
    
    # Process the indicator
    await strategy.process_indicator(symbol, timeframe, "EMA", ema_values)
    
    # Check that a signal was published with correct stop loss and take profit
    strategy.publish_signal.assert_called_once()
    args, kwargs = strategy.publish_signal.call_args
    assert kwargs["symbol"] = = symbol
    assert kwargs["direction"] = = "long"
    assert kwargs["price"] = = 10200  # The current fast MA value
    assert kwargs["stop_loss"] = = 10200 * 0.95  # 5% below entry price
    assert kwargs["take_profit"] = = 10200 * 1.10  # 10% above entry price 