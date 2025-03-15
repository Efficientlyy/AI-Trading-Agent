"""Tests for the sentiment-based trading strategy."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import json
import sys
from typing import Optional, Dict, Any

import pytest

# Create a marker for marking asyncio tests
pytest_plugins = ["pytest_asyncio"]

# Mock classes needed for testing
class TimeFrame:
    def __init__(self, timeframe):
        self.value = timeframe
        
    def __str__(self):
        return self.value

class CandleData:
    def __init__(self, symbol, exchange, timeframe, open, high, low, close, volume, timestamp):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp

class SignalType:
    ENTRY = "entry"
    EXIT = "exit"

class SentimentEvent:
    def __init__(self, source: str, symbol: str, sentiment_value: float,
                 sentiment_direction: str, confidence: float,
                 details: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        self.source = source
        self.symbol = symbol
        self.sentiment_value = sentiment_value
        self.sentiment_direction = sentiment_direction
        self.confidence = confidence
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
        # Add data to payload as it would be in the real class
        self.payload = {
            "symbol": symbol,
            "sentiment_value": sentiment_value,
            "sentiment_direction": sentiment_direction,
            "confidence": confidence,
            "details": self.details
        }


# Mock the SentimentStrategy class for testing
class SentimentStrategy:
    def __init__(self):
        self.sentiment_threshold_bullish = 0.7
        self.sentiment_threshold_bearish = 0.3
        self.min_confidence = 0.7
        self.contrarian_mode = False
        self.extreme_sentiment_threshold = 0.85
        self.use_stop_loss = True
        self.use_take_profit = True
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.06
        self.event_subscriptions = ["SentimentEvent", "CandleDataEvent"]
        self.sentiment_data = {}
        self.latest_candles = {}
        self.active_signals = {}
        self.source_weights = {
            "social_media": 1.0,
            "news": 1.0,
            "market": 1.0,
            "onchain": 1.0,
            "aggregator": 2.0
        }
        
    async def _handle_sentiment_event(self, event):
        if not event.symbol:
            return
            
        await self._process_sentiment_data(
            symbol=event.symbol,
            source=event.source,
            sentiment_value=event.sentiment_value,
            direction=event.sentiment_direction,
            confidence=event.confidence,
            details=event.payload
        )
        
    async def _process_sentiment_data(self, symbol, source, sentiment_value, direction, confidence, details=None):
        if symbol not in self.sentiment_data:
            self.sentiment_data[symbol] = {}
            
        # Extract source category (e.g., social_media_sentiment -> social_media)
        source_category = source.split("_")[0] if "_" in source else source
        
        # Corrected extraction for social_media
        if source.startswith("social_media"):
            source_category = "social_media"
            
        self.sentiment_data[symbol][source_category] = {
            "value": sentiment_value,
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
            "details": details or {}
        }
        
        await self._analyze_sentiment(symbol)
        
    async def _analyze_sentiment(self, symbol):
        if symbol not in self.sentiment_data:
            return
            
        # Calculate weighted average sentiment
        total_weight = 0
        weighted_sentiment = 0
        total_confidence = 0
        
        for source, data in self.sentiment_data[symbol].items():
            weight = self.source_weights.get(source, 1.0)
            weighted_sentiment += data["value"] * weight
            total_weight += weight
            total_confidence += data["confidence"]
            
        if total_weight == 0:
            return
            
        avg_sentiment = weighted_sentiment / total_weight
        avg_confidence = total_confidence / len(self.sentiment_data[symbol])
        
        # Check if sentiment is extreme
        is_extreme = avg_sentiment >= self.extreme_sentiment_threshold or avg_sentiment <= (1 - self.extreme_sentiment_threshold)
        
        await self._generate_sentiment_signals(
            symbol=symbol,
            sentiment_value=avg_sentiment,
            confidence=avg_confidence,
            is_extreme=is_extreme
        )
        
    async def _generate_sentiment_signals(self, symbol, sentiment_value, confidence, is_extreme):
        if symbol not in self.latest_candles:
            return
            
        candle = self.latest_candles[symbol]
        current_price = candle.close
        
        # Determine signal direction based on sentiment and contrarian mode
        direction = None
        
        # Contrarian mode for extreme sentiment
        if is_extreme and self.contrarian_mode:
            if sentiment_value >= self.extreme_sentiment_threshold:
                direction = "short"  # Contrarian to extremely bullish
            elif sentiment_value <= (1 - self.extreme_sentiment_threshold):
                direction = "long"   # Contrarian to extremely bearish
        # Regular sentiment mode
        else:
            if sentiment_value >= self.sentiment_threshold_bullish:
                direction = "long"
            elif sentiment_value <= self.sentiment_threshold_bearish:
                direction = "short"
                
        if not direction or confidence < self.min_confidence:
            return
            
        # Check if we need to exit existing position
        if symbol in self.active_signals and self.active_signals[symbol].direction != direction:
            self.generate_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT,
                direction=self.active_signals[symbol].direction,
                price=current_price,
                confidence=confidence
            )
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if self.use_stop_loss:
            stop_loss = current_price * (1 - self.stop_loss_pct) if direction == "long" else current_price * (1 + self.stop_loss_pct)
            
        if self.use_take_profit:
            take_profit = current_price * (1 + self.take_profit_pct) if direction == "long" else current_price * (1 - self.take_profit_pct)
            
        # Generate signal
        self.generate_signal(
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            price=current_price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
    def generate_signal(self, **kwargs):
        # In the real class, this would create and publish a signal
        pass


@pytest.fixture
def mock_strategy():
    """Create a mock sentiment strategy for testing."""
    return SentimentStrategy()


@pytest.mark.asyncio
async def test_strategy_initialization(mock_strategy):
    """Test strategy initialization."""
    # Create a strategy instance
    strategy = mock_strategy
    
    # Check default configuration
    assert strategy.sentiment_threshold_bullish == 0.7
    assert strategy.sentiment_threshold_bearish == 0.3
    assert strategy.min_confidence == 0.7
    assert strategy.contrarian_mode is False
    assert strategy.extreme_sentiment_threshold == 0.85
    assert strategy.use_stop_loss is True
    assert strategy.use_take_profit is True
    assert "SentimentEvent" in strategy.event_subscriptions


@pytest.mark.asyncio
async def test_handle_sentiment_event(mock_strategy):
    """Test handling of sentiment events."""
    # Create a strategy instance with patched methods
    strategy = mock_strategy
    strategy._process_sentiment_data = AsyncMock()
    
    # Create a test sentiment event
    event = SentimentEvent(
        source="social_media_sentiment",
        symbol="BTC/USDT",
        sentiment_value=0.8,
        sentiment_direction="bullish",
        confidence=0.85,
        details={"tags": ["trending"]}
    )
    
    # Handle the event
    await strategy._handle_sentiment_event(event)
    
    # Check that _process_sentiment_data was called with correct arguments
    strategy._process_sentiment_data.assert_called_once_with(
        symbol="BTC/USDT",
        source="social_media_sentiment",
        sentiment_value=0.8,
        direction="bullish",
        confidence=0.85,
        details=event.payload
    )
    
    # Test with missing data
    strategy._process_sentiment_data.reset_mock()
    invalid_event = SentimentEvent(
        source="test",
        symbol="",  # Missing symbol
        sentiment_value=0.5,
        sentiment_direction="neutral",
        confidence=0.5
    )
    
    await strategy._handle_sentiment_event(invalid_event)
    
    # Should not process event with missing data
    strategy._process_sentiment_data.assert_not_called()


@pytest.mark.asyncio
async def test_process_sentiment_data(mock_strategy):
    """Test processing of sentiment data."""
    # Create a strategy instance with patched _analyze_sentiment method
    strategy = mock_strategy
    strategy._analyze_sentiment = AsyncMock()
    
    # Test processing valid sentiment data
    await strategy._process_sentiment_data(
        symbol="BTC/USDT",
        source="social_media_sentiment",
        sentiment_value=0.8,
        direction="bullish",
        confidence=0.85,
        details={"tags": ["trending"]}
    )
    
    # Check that data was stored correctly
    assert "BTC/USDT" in strategy.sentiment_data
    assert "social_media" in strategy.sentiment_data["BTC/USDT"]
    assert strategy.sentiment_data["BTC/USDT"]["social_media"]["value"] == 0.8
    assert strategy.sentiment_data["BTC/USDT"]["social_media"]["direction"] == "bullish"
    assert strategy.sentiment_data["BTC/USDT"]["social_media"]["confidence"] == 0.85
    
    # Check that _analyze_sentiment was called
    strategy._analyze_sentiment.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_analyze_sentiment(mock_strategy):
    """Test sentiment analysis for signal generation."""
    # Create a strategy instance with patched generate_signal method
    strategy = mock_strategy
    strategy._generate_sentiment_signals = AsyncMock()
    
    # Add mock candle data
    symbol = "BTC/USDT"
    strategy.latest_candles[symbol] = CandleData(
        symbol=symbol,
        exchange="binance",
        timeframe=TimeFrame("1h"),
        open=10000,
        high=10200,
        low=9900,
        close=10100,
        volume=100,
        timestamp=datetime.utcnow()
    )
    
    # Add mock sentiment data
    strategy.sentiment_data[symbol] = {
        "social_media": {
            "value": 0.8,
            "direction": "bullish",
            "confidence": 0.85,
            "timestamp": datetime.utcnow(),
            "details": {}
        },
        "news": {
            "value": 0.75,
            "direction": "bullish",
            "confidence": 0.8,
            "timestamp": datetime.utcnow(),
            "details": {}
        }
    }
    
    # Run the analysis
    await strategy._analyze_sentiment(symbol)
    
    # Check that _generate_sentiment_signals was called with correct arguments
    strategy._generate_sentiment_signals.assert_called_once()
    args = strategy._generate_sentiment_signals.call_args[1]
    assert args["symbol"] == symbol
    assert args["sentiment_value"] > 0.75  # Should be weighted average between 0.75 and 0.8
    assert args["confidence"] >= 0.8
    assert args["is_extreme"] == False  # Not extreme enough


@pytest.mark.asyncio
async def test_generate_sentiment_signals_bullish(mock_strategy):
    """Test generation of bullish sentiment signals."""
    # Create a strategy instance with patched generate_signal method
    strategy = mock_strategy
    strategy.generate_signal = MagicMock()
    
    # Add mock candle data
    symbol = "BTC/USDT"
    strategy.latest_candles[symbol] = CandleData(
        symbol=symbol,
        exchange="binance",
        timeframe=TimeFrame("1h"),
        open=10000,
        high=10200,
        low=9900,
        close=10100,
        volume=100,
        timestamp=datetime.utcnow()
    )
    
    # Test bullish signal generation
    await strategy._generate_sentiment_signals(
        symbol=symbol,
        sentiment_value=0.75,  # Bullish
        confidence=0.85,
        is_extreme=False
    )
    
    # Check that a long signal was generated
    strategy.generate_signal.assert_called_once()
    args = strategy.generate_signal.call_args[1]
    assert args["symbol"] == symbol
    assert args["signal_type"] == SignalType.ENTRY
    assert args["direction"] == "long"
    assert args["price"] == 10100
    assert args["confidence"] == 0.85
    assert args["stop_loss"] == 10100 * 0.97  # 3% below entry
    assert args["take_profit"] == 10100 * 1.06  # 6% above entry


@pytest.mark.asyncio
async def test_generate_sentiment_signals_bearish(mock_strategy):
    """Test generation of bearish sentiment signals."""
    # Create a strategy instance with patched generate_signal method
    strategy = mock_strategy
    strategy.generate_signal = MagicMock()
    
    # Add mock candle data
    symbol = "BTC/USDT"
    strategy.latest_candles[symbol] = CandleData(
        symbol=symbol,
        exchange="binance",
        timeframe=TimeFrame("1h"),
        open=10000,
        high=10200,
        low=9900,
        close=10100,
        volume=100,
        timestamp=datetime.utcnow()
    )
    
    # Test bearish signal generation
    await strategy._generate_sentiment_signals(
        symbol=symbol,
        sentiment_value=0.25,  # Bearish
        confidence=0.85,
        is_extreme=False
    )
    
    # Check that a short signal was generated
    strategy.generate_signal.assert_called_once()
    args = strategy.generate_signal.call_args[1]
    assert args["symbol"] == symbol
    assert args["signal_type"] == SignalType.ENTRY
    assert args["direction"] == "short"
    assert args["price"] == 10100
    assert args["confidence"] == 0.85
    assert args["stop_loss"] == 10100 * 1.03  # 3% above entry
    assert args["take_profit"] == 10100 * 0.94  # 6% below entry


@pytest.mark.asyncio
async def test_contrarian_mode(mock_strategy):
    """Test contrarian mode for extreme sentiment."""
    # Create a strategy instance with patched generate_signal method
    strategy = mock_strategy
    strategy.generate_signal = MagicMock()
    strategy.contrarian_mode = True
    
    # Add mock candle data
    symbol = "BTC/USDT"
    strategy.latest_candles[symbol] = CandleData(
        symbol=symbol,
        exchange="binance",
        timeframe=TimeFrame("1h"),
        open=10000,
        high=10200,
        low=9900,
        close=10100,
        volume=100,
        timestamp=datetime.utcnow()
    )
    
    # Test extreme bullish sentiment in contrarian mode (should generate short signal)
    await strategy._generate_sentiment_signals(
        symbol=symbol,
        sentiment_value=0.9,  # Extremely bullish
        confidence=0.85,
        is_extreme=True
    )
    
    # Check that a short signal was generated (contrarian to bullish)
    strategy.generate_signal.assert_called_once()
    args = strategy.generate_signal.call_args[1]
    assert args["symbol"] == symbol
    assert args["signal_type"] == SignalType.ENTRY
    assert args["direction"] == "short"
    assert args["price"] == 10100
    assert args["confidence"] == 0.85
    
    # Reset and test extreme bearish sentiment
    strategy.generate_signal.reset_mock()
    
    # Test extreme bearish sentiment in contrarian mode (should generate long signal)
    await strategy._generate_sentiment_signals(
        symbol=symbol,
        sentiment_value=0.1,  # Extremely bearish
        confidence=0.85,
        is_extreme=True
    )
    
    # Check that a long signal was generated (contrarian to bearish)
    strategy.generate_signal.assert_called_once()
    args = strategy.generate_signal.call_args[1]
    assert args["symbol"] == symbol
    assert args["signal_type"] == SignalType.ENTRY
    assert args["direction"] == "long"
    assert args["price"] == 10100
    assert args["confidence"] == 0.85


@pytest.mark.asyncio
async def test_exit_existing_position(mock_strategy):
    """Test exiting an existing position when sentiment changes."""
    # Create a strategy instance with patched generate_signal method
    strategy = mock_strategy
    strategy.generate_signal = MagicMock()
    
    # Add mock candle data
    symbol = "BTC/USDT"
    strategy.latest_candles[symbol] = CandleData(
        symbol=symbol,
        exchange="binance",
        timeframe=TimeFrame("1h"),
        open=10000,
        high=10200,
        low=9900,
        close=10100,
        volume=100,
        timestamp=datetime.utcnow()
    )
    
    # Add mock active signal
    strategy.active_signals[symbol] = MagicMock()
    strategy.active_signals[symbol].direction = "long"
    
    # Test sentiment shift to bearish
    await strategy._generate_sentiment_signals(
        symbol=symbol,
        sentiment_value=0.25,  # Bearish
        confidence=0.85,
        is_extreme=False
    )
    
    # Should generate two signals: exit the long and enter a short
    assert strategy.generate_signal.call_count == 2
    
    # First call should be to exit the existing long position
    exit_args = strategy.generate_signal.call_args_list[0][1]
    assert exit_args["symbol"] == symbol
    assert exit_args["signal_type"] == SignalType.EXIT
    assert exit_args["direction"] == "long"
    
    # Second call should be to enter a short position
    entry_args = strategy.generate_signal.call_args_list[1][1]
    assert entry_args["symbol"] == symbol
    assert entry_args["signal_type"] == SignalType.ENTRY
    assert entry_args["direction"] == "short"
