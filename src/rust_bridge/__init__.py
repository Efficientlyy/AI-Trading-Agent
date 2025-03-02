"""
Python bridge to Rust components.

This module provides access to high-performance Rust components through Python bindings.
"""

import importlib.util
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Setup logging
logger = logging.getLogger(__name__)

# Try to import the Rust library
_rust_available = False
_crypto_trading_engine = None

try:
    # Check if the compiled library exists
    from crypto_trading_engine import market_data, technical, backtesting
    _rust_available = True
    logger.info("Rust components successfully loaded")
except ImportError:
    logger.warning("Could not import Rust components. Falling back to pure Python implementations.")
    _rust_available = False

def is_rust_available() -> bool:
    """Check if Rust components are available."""
    return _rust_available

def initialize_rust() -> bool:
    """Initialize the Rust engine."""
    if not _rust_available:
        return False
    
    from crypto_trading_engine import initialize
    return initialize()

# Market data functions
def create_candle(symbol: str, timestamp: int, open_price: float, high: float, 
                  low: float, close: float, volume: float, timeframe: str) -> Any:
    """Create a candle data object."""
    if _rust_available:
        return market_data.create_candle(
            symbol, timestamp, open_price, high, low, close, volume, timeframe
        )
    else:
        # Python fallback implementation
        from .market_data_py import create_candle as create_candle_py
        return create_candle_py(symbol, timestamp, open_price, high, low, close, volume, timeframe)

def create_order_book(symbol: str, timestamp: int, bids: List[Tuple[float, float]], 
                      asks: List[Tuple[float, float]]) -> Any:
    """Create an order book data object."""
    if _rust_available:
        return market_data.create_order_book(symbol, timestamp, bids, asks)
    else:
        # Fallback to Python implementation
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "bids": bids,
            "asks": asks,
        }

def calculate_mid_price(order_book: Any) -> float:
    """Calculate mid price from an order book."""
    if _rust_available and hasattr(order_book, "calculate_mid_price"):
        return order_book.calculate_mid_price()
    else:
        # Fallback to Python implementation
        if not order_book["bids"] or not order_book["asks"]:
            return 0.0
        bid = order_book["bids"][0][0] if order_book["bids"] else 0
        ask = order_book["asks"][0][0] if order_book["asks"] else 0
        return (bid + ask) / 2 if bid and ask else 0

# Technical indicator functions
class SMA:
    """Simple Moving Average implementation."""
    
    def __init__(self, period: int):
        """Initialize the SMA with a period."""
        self.period = period
        if _rust_available:
            self._impl = technical.SMA(period)
        else:
            self._values = []
    
    def update(self, value: float) -> float:
        """Update the SMA with a new value."""
        if _rust_available:
            return self._impl.update(value)
        else:
            self._values.append(value)
            if len(self._values) > self.period:
                self._values.pop(0)
            return self.current_value()
    
    def current_value(self) -> float:
        """Get the current SMA value."""
        if _rust_available:
            return self._impl.current_value()
        else:
            if not self._values:
                return 0.0
            return sum(self._values) / len(self._values)
    
    def reset(self) -> None:
        """Reset the SMA."""
        if _rust_available:
            self._impl.reset()
        else:
            self._values = []

class EMA:
    """Exponential Moving Average implementation."""
    
    def __init__(self, period: int):
        """Initialize the EMA with a period."""
        self.period = period
        if _rust_available:
            self._impl = technical.EMA(period)
        else:
            self._values = []
            self._last_ema = 0.0
            self._is_initialized = False
            self._alpha = 2.0 / (period + 1)
    
    def update(self, value: float) -> float:
        """Update the EMA with a new value."""
        if _rust_available:
            return self._impl.update(value)
        else:
            if not self._is_initialized:
                if len(self._values) < self.period:
                    self._values.append(value)
                    if len(self._values) == self.period:
                        self._last_ema = sum(self._values) / self.period
                        self._is_initialized = True
                    return self._last_ema
            
            self._last_ema = value * self._alpha + self._last_ema * (1 - self._alpha)
            return self._last_ema
    
    def current_value(self) -> float:
        """Get the current EMA value."""
        if _rust_available:
            return self._impl.current_value()
        else:
            return self._last_ema
    
    def reset(self) -> None:
        """Reset the EMA."""
        if _rust_available:
            self._impl.reset()
        else:
            self._values = []
            self._last_ema = 0.0
            self._is_initialized = False

class WMA:
    """Weighted Moving Average implementation."""
    
    def __init__(self, period: int):
        """Initialize the WMA with a period."""
        self.period = period
        if _rust_available:
            self._impl = technical.WMA(period)
        else:
            self._values = []
    
    def update(self, value: float) -> float:
        """Update the WMA with a new value."""
        if _rust_available:
            return self._impl.update(value)
        else:
            self._values.append(value)
            if len(self._values) > self.period:
                self._values.pop(0)
            return self.current_value()
    
    def current_value(self) -> float:
        """Get the current WMA value."""
        if _rust_available:
            return self._impl.current_value()
        else:
            if not self._values:
                return 0.0
            
            weights = list(range(1, len(self._values) + 1))
            total_weight = sum(weights)
            
            return sum(val * weight for val, weight in zip(self._values, weights)) / total_weight
    
    def reset(self) -> None:
        """Reset the WMA."""
        if _rust_available:
            self._impl.reset()
        else:
            self._values = []

class MACrossover:
    """Moving Average Crossover detector."""
    
    def __init__(self, fast_period: int, slow_period: int, fast_type: str = "SMA", slow_type: str = "SMA"):
        """Initialize the MA Crossover detector."""
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        if _rust_available:
            self._impl = technical.MACrossover(fast_period, slow_period, fast_type, slow_type)
        else:
            # Create appropriate MA types
            if fast_type == "SMA":
                self._fast_ma = SMA(fast_period)
            elif fast_type == "EMA":
                self._fast_ma = EMA(fast_period)
            elif fast_type == "WMA":
                self._fast_ma = WMA(fast_period)
            else:
                raise ValueError(f"Unsupported MA type: {fast_type}")
            
            if slow_type == "SMA":
                self._slow_ma = SMA(slow_period)
            elif slow_type == "EMA":
                self._slow_ma = EMA(slow_period)
            elif slow_type == "WMA":
                self._slow_ma = WMA(slow_period)
            else:
                raise ValueError(f"Unsupported MA type: {slow_type}")
            
            self._last_fast = 0.0
            self._last_slow = 0.0
            self._last_position = 0  # 0: no position, 1: above, -1: below
    
    def update(self, value: float) -> str:
        """Update the MAs with a new value and check for crossover."""
        if _rust_available:
            return self._impl.update(value)
        else:
            fast_val = self._fast_ma.update(value)
            slow_val = self._slow_ma.update(value)
            
            if fast_val == 0 or slow_val == 0:
                return "none"
            
            current_position = 1 if fast_val > slow_val else -1 if fast_val < slow_val else 0
            
            if self._last_position == 0:
                self._last_position = current_position
                return "none"
            
            signal = "none"
            if self._last_position == -1 and current_position == 1:
                signal = "bullish"
            elif self._last_position == 1 and current_position == -1:
                signal = "bearish"
            
            self._last_position = current_position
            return signal
    
    def reset(self) -> None:
        """Reset the crossover detector."""
        if _rust_available:
            self._impl.reset()
        else:
            self._fast_ma.reset()
            self._slow_ma.reset()
            self._last_position = 0

def calculate_sma(values: List[float], period: int) -> List[float]:
    """Calculate SMA for a list of values."""
    if _rust_available:
        return technical.calc_sma(values, period)
    else:
        if len(values) < period:
            return [0.0] * len(values)
        
        result = []
        for i in range(len(values)):
            if i < period - 1:
                result.append(0.0)
                continue
            
            window = values[i - period + 1:i + 1]
            result.append(sum(window) / period)
        
        return result

def calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate EMA for a list of values."""
    if _rust_available:
        return technical.calc_ema(values, period)
    else:
        if len(values) < period:
            return [0.0] * len(values)
        
        ema = EMA(period)
        result = [0.0] * (period - 1)
        
        # Initialize with SMA
        sma_val = sum(values[:period]) / period
        result.append(sma_val)
        ema._last_ema = sma_val
        ema._is_initialized = True
        
        # Calculate EMA for remaining values
        for i in range(period, len(values)):
            result.append(ema.update(values[i]))
        
        return result

def detect_crossover(fast_ma: List[float], slow_ma: List[float]) -> List[str]:
    """Detect crossovers between two moving averages."""
    if _rust_available:
        return technical.detect_crossover(fast_ma, slow_ma)
    else:
        if len(fast_ma) != len(slow_ma):
            raise ValueError("Moving averages must have the same length")
        
        signals = ["none"] * len(fast_ma)
        last_position = 0  # 0: no position, 1: above, -1: below
        
        for i in range(len(fast_ma)):
            if fast_ma[i] == 0 or slow_ma[i] == 0:
                continue
                
            current_position = 1 if fast_ma[i] > slow_ma[i] else -1 if fast_ma[i] < slow_ma[i] else 0
            
            if last_position == 0:
                last_position = current_position
                continue
            
            if last_position == -1 and current_position == 1:
                signals[i] = "bullish"
            elif last_position == 1 and current_position == -1:
                signals[i] = "bearish"
            
            last_position = current_position
        
        return signals

# Backtesting functions
def get_backtest_engine(
    initial_balance: float,
    symbols: List[str],
    start_time: int,
    end_time: int,
    mode: str = "candles",
    commission_rate: float = 0.001,
    slippage: float = 0.0005,
    enable_fractional_sizing: bool = True
) -> Any:
    """Create a new backtest engine."""
    if _rust_available:
        return backtesting.create_backtest_engine(
            initial_balance=initial_balance,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            mode=mode,
            commission_rate=commission_rate,
            slippage=slippage,
            enable_fractional_sizing=enable_fractional_sizing
        )
    else:
        # This should never be called, as the backtesting module will handle the fallback
        raise NotImplementedError("Rust implementation not available")

# Class to expose Rust BacktestStats in Python
class BacktestStats:
    """Wrapper for the Rust BacktestStats class."""
    
    def __init__(self, rust_stats: Any):
        """Initialize with a Rust stats object."""
        self._stats = rust_stats
    
    @property
    def start_time(self) -> int:
        """Get start time."""
        return self._stats.start_time
    
    @property
    def end_time(self) -> int:
        """Get end time."""
        return self._stats.end_time
    
    @property
    def initial_balance(self) -> float:
        """Get initial balance."""
        return self._stats.initial_balance
    
    @property
    def final_balance(self) -> float:
        """Get final balance."""
        return self._stats.final_balance
    
    @property
    def total_trades(self) -> int:
        """Get total trades."""
        return self._stats.total_trades
    
    @property
    def winning_trades(self) -> int:
        """Get winning trades."""
        return self._stats.winning_trades
    
    @property
    def losing_trades(self) -> int:
        """Get losing trades."""
        return self._stats.losing_trades
    
    @property
    def total_profit(self) -> float:
        """Get total profit."""
        return self._stats.total_profit
    
    @property
    def total_loss(self) -> float:
        """Get total loss."""
        return self._stats.total_loss
    
    @property
    def max_drawdown(self) -> float:
        """Get max drawdown."""
        return self._stats.max_drawdown
    
    @property
    def max_drawdown_pct(self) -> float:
        """Get max drawdown as percentage."""
        return self._stats.max_drawdown_pct
    
    @property
    def sharpe_ratio(self) -> Optional[float]:
        """Get Sharpe ratio."""
        return self._stats.sharpe_ratio
    
    @property
    def profit_factor(self) -> Optional[float]:
        """Get profit factor."""
        return self._stats.profit_factor
    
    @property
    def win_rate(self) -> float:
        """Get win rate."""
        return self._stats.win_rate
    
    @property
    def avg_win(self) -> Optional[float]:
        """Get average win."""
        return self._stats.avg_win
    
    @property
    def avg_loss(self) -> Optional[float]:
        """Get average loss."""
        return self._stats.avg_loss
    
    @property
    def largest_win(self) -> Optional[float]:
        """Get largest win."""
        return self._stats.largest_win
    
    @property
    def largest_loss(self) -> Optional[float]:
        """Get largest loss."""
        return self._stats.largest_loss
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if hasattr(self._stats, "to_dict"):
            return self._stats.to_dict()
        else:
            return {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "initial_balance": self.initial_balance,
                "final_balance": self.final_balance,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "total_profit": self.total_profit,
                "total_loss": self.total_loss,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
                "sharpe_ratio": self.sharpe_ratio,
                "profit_factor": self.profit_factor,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
            }
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._stats)
    
    def __repr__(self) -> str:
        """Representation."""
        return repr(self._stats)

# Add new function for order book processor
def create_order_book_processor(symbol: str, exchange: str, max_depth: int = 100) -> 'OrderBookProcessor':
    """
    Create a high-performance order book processor.
    
    Args:
        symbol: The trading pair symbol (e.g., "BTC/USD")
        exchange: The exchange name (e.g., "binance")
        max_depth: Maximum depth to maintain per side
        
    Returns:
        An OrderBookProcessor instance
    """
    if _rust_available:
        try:
            return OrderBookProcessor(symbol, exchange, max_depth)
        except Exception as e:
            logger.warning(f"Error creating Rust order book processor: {e}. Falling back to Python implementation.")
            from .market_data_py import OrderBookProcessor as PyOrderBookProcessor
            return PyOrderBookProcessor(symbol, exchange, max_depth)
    else:
        from .market_data_py import OrderBookProcessor as PyOrderBookProcessor
        return PyOrderBookProcessor(symbol, exchange, max_depth)

# Add OrderBookProcessor class
class OrderBookProcessor:
    """
    High-performance order book processor powered by Rust.
    
    This class maintains an order book and provides efficient methods for:
    - Processing real-time order book updates
    - Calculating market impact for potential orders
    - Computing order book analytics (spreads, liquidity, imbalance)
    
    For environments without Rust, a Python fallback implementation is used.
    """
    
    def __init__(self, symbol: str, exchange: str, max_depth: int = 100):
        """
        Initialize the order book processor.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USD")
            exchange: The exchange name (e.g., "binance")
            max_depth: Maximum depth to maintain per side
        """
        if not _rust_available:
            from .market_data_py import OrderBookProcessor as PyOrderBookProcessor
            self._processor = PyOrderBookProcessor(symbol, exchange, max_depth)
        else:
            self._processor = market_data.create_order_book_processor(symbol, exchange, max_depth)
        
        self.symbol = symbol
        self.exchange = exchange
        self.max_depth = max_depth
    
    def process_updates(self, updates: List[Dict[str, Any]]) -> float:
        """
        Process a batch of order book updates.
        
        Args:
            updates: List of update dictionaries. Each update should have:
                     - price: float
                     - side: str ("buy"/"sell" or "bid"/"ask")
                     - quantity: float
                     - timestamp: float (optional, defaults to current time)
                     - sequence: int (optional, for ordering)
        
        Returns:
            Processing time in milliseconds
        """
        return self._processor.process_updates(updates)
    
    def calculate_market_impact(self, side: str, size: float) -> Dict[str, Any]:
        """
        Calculate the market impact for a given order size.
        
        Args:
            side: Trade side ("buy"/"sell" or "bid"/"ask")
            size: Order size
            
        Returns:
            Dict with market impact metrics:
            - avg_price: Average execution price
            - slippage_pct: Price slippage as percentage
            - total_value: Total value of the order
            - fillable_quantity: Total quantity that can be filled
            - levels_consumed: Number of price levels needed to fill
        """
        return self._processor.calculate_market_impact(side, size)
    
    def best_bid_price(self) -> float:
        """Get the best bid price."""
        return self._processor.best_bid_price()
    
    def best_ask_price(self) -> float:
        """Get the best ask price."""
        return self._processor.best_ask_price()
    
    def mid_price(self) -> float:
        """Get the mid price."""
        return self._processor.mid_price()
    
    def spread(self) -> float:
        """Get the current bid-ask spread."""
        return self._processor.spread()
    
    def spread_pct(self) -> float:
        """Get the current bid-ask spread as a percentage of the mid price."""
        return self._processor.spread_pct()
    
    def vwap(self, side: str, depth: int) -> float:
        """
        Calculate the volume-weighted average price (VWAP) for a given side and depth.
        
        Args:
            side: "buy"/"sell" or "bid"/"ask"
            depth: Number of levels to include
            
        Returns:
            VWAP price
        """
        return self._processor.vwap(side, depth)
    
    def liquidity_up_to(self, side: str, price_depth: float) -> float:
        """
        Calculate the total liquidity available up to a given price depth.
        
        Args:
            side: "buy"/"sell" or "bid"/"ask"
            price_depth: Price depth away from best bid/ask
            
        Returns:
            Total available quantity
        """
        return self._processor.liquidity_up_to(side, price_depth)
    
    def book_imbalance(self, depth: int) -> float:
        """
        Detect order book imbalance (ratio of buy to sell liquidity).
        
        Args:
            depth: Number of levels to include
            
        Returns:
            Imbalance ratio (> 1 means more bids than asks)
        """
        return self._processor.book_imbalance(depth)
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current order book.
        
        Returns:
            Dict with symbol, exchange, timestamp, bids, and asks
        """
        return self._processor.snapshot()
    
    def processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict with processing statistics
        """
        return self._processor.processing_stats()
    
    def reset(self) -> None:
        """Reset the order book processor."""
        self._processor.reset() 