"""Advanced technical indicators for market analysis."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class TechnicalIndicatorParams:
    """Parameters for technical indicators."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    fibonacci_levels: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)

class TechnicalIndicators:
    """Advanced technical indicators for market analysis."""
    
    def __init__(self, params: Optional[TechnicalIndicatorParams] = None):
        """Initialize with parameters."""
        self.params = params or TechnicalIndicatorParams()
    
    def calculate_macd(
        self,
        prices: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Array of prices
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, self.params.macd_fast)
        ema_slow = self._calculate_ema(prices, self.params.macd_slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self._calculate_ema(macd_line, self.params.macd_signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self,
        prices: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Array of prices
        
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        # Calculate middle band (SMA)
        middle_band = self._calculate_sma(prices, self.params.bb_period)
        
        # Calculate standard deviation
        rolling_std = np.array([
            np.std(prices[max(0, i - self.params.bb_period + 1):i + 1])
            for i in range(len(prices))
        ])
        
        # Calculate bands
        upper_band = middle_band + (rolling_std * self.params.bb_std)
        lower_band = middle_band - (rolling_std * self.params.bb_std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_atr(self, high: NDArray[np.float64], low: NDArray[np.float64],
                     close: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate Average True Range.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            Array of ATR values
        """
        # Calculate True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR
        atr = self._calculate_ema(tr, self.params.atr_period)
        return atr
    
    def calculate_adx(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Average Directional Index.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Calculate True Range
        tr = np.zeros_like(close)
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Calculate smoothed values
        period = self.params.adx_period
        tr_smooth = self._calculate_ema(tr, period)
        plus_dm_smooth = self._calculate_ema(plus_dm, period)
        minus_dm_smooth = self._calculate_ema(minus_dm, period)
        
        # Calculate DI lines
        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100
        
        # Calculate DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = self._calculate_ema(dx, period)
        
        return adx, plus_di, minus_di
    
    def calculate_stochastic(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Stochastic Oscillator.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            Tuple of (%K, %D)
        """
        # Calculate %K
        k_period = self.params.stoch_k_period
        stoch_k = np.zeros_like(close)
        
        for i in range(k_period - 1, len(close)):
            window_high = np.max(high[i-k_period+1:i+1])
            window_low = np.min(low[i-k_period+1:i+1])
            stoch_k[i] = ((close[i] - window_low) /
                         (window_high - window_low) * 100)
        
        # Calculate %D (SMA of %K)
        stoch_d = self._calculate_sma(
            stoch_k,
            self.params.stoch_d_period
        )
        
        return stoch_k, stoch_d
    
    def calculate_fibonacci_levels(
        self,
        high: float,
        low: float
    ) -> NDArray[np.float64]:
        """Calculate Fibonacci retracement levels.
        
        Args:
            high: Highest price in range
            low: Lowest price in range
        
        Returns:
            Array of Fibonacci levels
        """
        price_range = high - low
        levels = np.array([
            high - (price_range * level)
            for level in self.params.fibonacci_levels
        ])
        return levels
    
    def calculate_ichimoku(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Ichimoku Cloud components.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            Tuple of (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B,
                     Chikou Span)
        """
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = np.array([
            (np.max(high[max(0, i-9):i+1]) +
             np.min(low[max(0, i-9):i+1])) / 2
            for i in range(len(close))
        ])
        
        # Calculate Kijun-sen (Base Line)
        kijun_sen = np.array([
            (np.max(high[max(0, i-26):i+1]) +
             np.min(low[max(0, i-26):i+1])) / 2
            for i in range(len(close))
        ])
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = np.array([
            (np.max(high[max(0, i-52):i+1]) +
             np.min(low[max(0, i-52):i+1])) / 2
            for i in range(len(close))
        ])
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = np.roll(close, -26)
        
        return (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
                chikou_span)
    
    def _calculate_sma(
        self,
        data: NDArray[np.float64],
        period: int
    ) -> NDArray[np.float64]:
        """Calculate Simple Moving Average.
        
        Args:
            data: Input data array
            period: Moving average period
        
        Returns:
            Array of SMA values
        """
        sma = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    def _calculate_ema(
        self,
        data: NDArray[np.float64],
        period: int
    ) -> NDArray[np.float64]:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Input data array
            period: Moving average period
        
        Returns:
            Array of EMA values
        """
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        
        # Initialize EMA with SMA
        ema[period-1] = np.mean(data[:period])
        
        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = ((data[i] - ema[i-1]) * multiplier) + ema[i-1]
        
        return ema 