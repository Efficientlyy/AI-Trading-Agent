"""
Technical indicators implemented using Rust for high performance.

This module provides technical indicators that are accelerated by Rust
when available, with automatic fallback to pure Python implementations.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple

# Import Rust implementations
from src.rust_bridge import Technical, is_rust_available

logger = logging.getLogger(__name__)
RUST_AVAILABLE = is_rust_available()

if RUST_AVAILABLE:
    logger.info("Using Rust-accelerated technical indicators")
else:
    logger.warning("Rust acceleration not available, using pure Python implementations")

# Simple Moving Average
def sma(values: List[float], period: int) -> List[float]:
    """
    Calculate Simple Moving Average.
    
    This function uses the Rust implementation when available,
    with automatic fallback to Python.
    
    Args:
        values: List of price values
        period: Period of the moving average
        
    Returns:
        List of SMA values
    """
    return Technical.sma(values, period)

# Exponential Moving Average
def ema(values: List[float], period: int) -> List[float]:
    """
    Calculate Exponential Moving Average.
    
    This function uses the Rust implementation when available,
    with automatic fallback to Python.
    
    Args:
        values: List of price values
        period: Period of the moving average
        
    Returns:
        List of EMA values
    """
    return Technical.ema(values, period)

# Detect Moving Average Crossover
def detect_crossover(fast_ma: List[float], slow_ma: List[float]) -> List[str]:
    """
    Detect crossovers between fast and slow moving averages.
    
    Args:
        fast_ma: List of fast moving average values
        slow_ma: List of slow moving average values
        
    Returns:
        List of crossover signals ("bullish_crossover", "bearish_crossover", or "no_signal")
    """
    return Technical.detect_crossover(fast_ma, slow_ma)

# Streaming SMA calculator for real-time updates
class SMA:
    """
    Simple Moving Average calculator for streaming data.
    
    This class allows calculating SMA in real-time with each new price update.
    It uses Rust implementation when available.
    """
    
    def __init__(self, period: int):
        """
        Initialize the SMA calculator.
        
        Args:
            period: Period of the moving average
        """
        self._impl = Technical.SMA(period)
        self.period = period
    
    def update(self, value: float) -> Optional[float]:
        """
        Update the SMA with a new value.
        
        Args:
            value: New price value
            
        Returns:
            Updated SMA value, or None if not enough data yet
        """
        return self._impl.update(value)
    
    def current(self) -> Optional[float]:
        """
        Get the current SMA value.
        
        Returns:
            Current SMA value, or None if not enough data yet
        """
        return self._impl.current()
    
    def reset(self) -> None:
        """Reset the calculator."""
        self._impl.reset()

# Streaming EMA calculator for real-time updates
class EMA:
    """
    Exponential Moving Average calculator for streaming data.
    
    This class allows calculating EMA in real-time with each new price update.
    It uses Rust implementation when available.
    """
    
    def __init__(self, period: int):
        """
        Initialize the EMA calculator.
        
        Args:
            period: Period of the moving average
        """
        self._impl = Technical.EMA(period)
        self.period = period
    
    def update(self, value: float) -> Optional[float]:
        """
        Update the EMA with a new value.
        
        Args:
            value: New price value
            
        Returns:
            Updated EMA value, or None if not enough data yet
        """
        return self._impl.update(value)
    
    def current(self) -> Optional[float]:
        """
        Get the current EMA value.
        
        Returns:
            Current EMA value, or None if not enough data yet
        """
        return self._impl.current()
    
    def reset(self) -> None:
        """Reset the calculator."""
        self._impl.reset()

# Moving Average Crossover detector for real-time signals
class MACrossover:
    """
    Moving Average Crossover detector for streaming data.
    
    This class detects crossovers between fast and slow moving averages
    in real-time with each new price update.
    """
    
    def __init__(
        self,
        fast_period: int,
        slow_period: int,
        fast_type: str = "simple",
        slow_type: str = "simple"
    ):
        """
        Initialize the MA Crossover detector.
        
        Args:
            fast_period: Period of the fast moving average
            slow_period: Period of the slow moving average
            fast_type: Type of the fast MA ("simple" or "exponential")
            slow_type: Type of the slow MA ("simple" or "exponential")
        """
        self._impl = Technical.MACrossover(
            fast_period, slow_period, fast_type, slow_type
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_type = fast_type
        self.slow_type = slow_type
    
    def update(self, price: float) -> str:
        """
        Update with a new price and check for crossover.
        
        Args:
            price: New price value
            
        Returns:
            Crossover signal ("bullish_crossover", "bearish_crossover", or "no_signal")
        """
        return self._impl.update(price)
    
    def last_signal(self) -> Optional[str]:
        """
        Get the last crossover signal.
        
        Returns:
            Last crossover signal, or None if no signal yet
        """
        return self._impl.last_signal()
    
    def reset(self) -> None:
        """Reset the detector."""
        self._impl.reset()

# Example of implementing the MA Crossover strategy with Rust acceleration
class MACrossoverStrategy:
    """
    Moving Average Crossover trading strategy.
    
    This strategy generates signals based on crossovers between
    fast and slow moving averages.
    """
    
    def __init__(
        self,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        fast_type: str = "exponential",
        slow_type: str = "exponential",
        signal_threshold: float = 0.0
    ):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            fast_period: Period of the fast moving average
            slow_period: Period of the slow moving average
            fast_type: Type of the fast MA ("simple" or "exponential")
            slow_type: Type of the slow MA ("simple" or "exponential")
            signal_threshold: Minimum distance between MAs to generate signal
        """
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_threshold = signal_threshold
        
        # Create MA Crossover detector with Rust acceleration
        self.crossover = MACrossover(
            fast_period, slow_period, fast_type, slow_type
        )
        
        # Additionally, keep the individual MAs for analysis
        if fast_type == "simple":
            self.fast_ma = SMA(fast_period)
        else:
            self.fast_ma = EMA(fast_period)
            
        if slow_type == "simple":
            self.slow_ma = SMA(slow_period)
        else:
            self.slow_ma = EMA(slow_period)
        
        # Store last values for analysis
        self.last_price = None
        self.last_fast_ma = None
        self.last_slow_ma = None
    
    def update(self, price: float) -> Dict[str, Any]:
        """
        Update the strategy with a new price.
        
        Args:
            price: New price value
            
        Returns:
            Dictionary with signal details
        """
        # Update individual MAs for tracking
        fast_value = self.fast_ma.update(price)
        slow_value = self.slow_ma.update(price)
        
        # Update crossover detector and get signal
        signal = self.crossover.update(price)
        
        # Store last values
        self.last_price = price
        self.last_fast_ma = fast_value
        self.last_slow_ma = slow_value
        
        # Calculate signal confidence based on distance between MAs
        confidence = 0.0
        if fast_value is not None and slow_value is not None:
            # Calculate percentage difference between MAs
            ma_diff_pct = abs(fast_value - slow_value) / slow_value * 100
            
            # Scale to confidence (0.5-1.0 range)
            if ma_diff_pct > self.signal_threshold:
                confidence = min(0.5 + ma_diff_pct / 10, 1.0)
        
        # Build result
        result = {
            "symbol": self.symbol,
            "price": price,
            "signal": signal,
            "confidence": confidence,
            "fast_ma": fast_value,
            "slow_ma": slow_value
        }
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the strategy.
        
        Returns:
            Dictionary with current state
        """
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "last_fast_ma": self.last_fast_ma,
            "last_slow_ma": self.last_slow_ma,
            "last_signal": self.crossover.last_signal()
        }
    
    def reset(self) -> None:
        """Reset the strategy."""
        self.crossover.reset()
        self.fast_ma.reset()
        self.slow_ma.reset()
        self.last_price = None
        self.last_fast_ma = None
        self.last_slow_ma = None 