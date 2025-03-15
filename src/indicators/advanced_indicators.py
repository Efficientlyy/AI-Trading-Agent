"""Advanced technical indicators for market analysis.

This module provides implementations of sophisticated technical indicators
that go beyond basic price and volume analysis.
"""

import numpy as np
from typing import List, Dict, Union, cast
from scipy import stats
from .types import (
    HeikinAshiCandles,
    KeltnerChannels,
    MarketRegime,
    PriceData,
    VolumeData,
    IndicatorOutput
)


def calculate_heikin_ashi(
    opens: PriceData,
    highs: PriceData,
    lows: PriceData,
    closes: PriceData
) -> HeikinAshiCandles:
    """Calculate Heikin Ashi candles.
    
    Args:
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices
        
    Returns:
        Heikin Ashi candle data
    """
    # Convert inputs to numpy arrays
    o = np.array(opens, dtype=np.float64)
    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)
    
    # Calculate Heikin Ashi values
    ha_close = (o + h + l + c) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, len(closes)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    
    return 


def calculate_keltner_channels(
    closes: PriceData,
    highs: PriceData,
    lows: PriceData,
    period: int = 20,
    atr_mult: float = 2.0
) -> KeltnerChannels:
    """Calculate Keltner Channels.
    
    Args:
        closes: Close prices
        highs: High prices
        lows: Low prices
        period: Calculation period
        atr_mult: ATR multiplier
        
    Returns:
        Keltner channel data
    """
    # Convert inputs to numpy arrays
    c = np.array(closes, dtype=np.float64)
    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    
    # Calculate middle line (EMA)
    middle = np.zeros_like(c)
    alpha = 2 / (period + 1)
    middle[0] = c[0]
    for i in range(1, len(c)):
        middle[i] = alpha * c[i] + (1 - alpha) * middle[i-1]
    
    # Calculate ATR
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - np.roll(c, 1)),
        np.abs(l - np.roll(c, 1))
    ])
    tr[0] = h[0] - l[0]
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    # Calculate bands
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
    return {
        "upper": upper.tolist(),
        "middle": middle.tolist(),
        "lower": lower.tolist()
    }


def calculate_atr(
    highs: PriceData,
    lows: PriceData,
    closes: PriceData,
    period: int = 14
) -> IndicatorOutput:
    """Calculate Average True Range.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Calculation period
        
    Returns:
        ATR values
    """
    # Convert inputs to numpy arrays
    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)
    
    # Calculate true range
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - np.roll(c, 1)),
        np.abs(l - np.roll(c, 1))
    ])
    tr[0] = h[0] - l[0]
    
    # Calculate ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    return atr.tolist()


def calculate_ema(
    values: List[float],
    period: int
) -> List[float]:
    """Calculate Exponential Moving Average (EMA).
    
    Args:
        values: List of values
        period: The period for EMA calculation
        
    Returns:
        List of EMA values
    """
    ema_list = []
    multiplier = 2 / (period + 1)
    
    for i in range(len(values)):
        if i < period:
            ema = sum(values[:i+1]) / (i + 1)
        else:
            prev_ema = ema_list[-1]
            ema = (values[i] - prev_ema) * multiplier + prev_ema
        ema_list.append(ema)
    
    return ema_list


def detect_market_regime(
    closes: PriceData,
    period: int = 20,
    std_threshold: float = 0.02
) -> MarketRegime:
    """Detect market regime (trending/ranging).
    
    Args:
        closes: Close prices
        period: Analysis period
        std_threshold: Standard deviation threshold for regime detection
        
    Returns:
        Market regime indicators
    """
    # Convert input to numpy array
    c = np.array(closes, dtype=np.float64)
    
    # Calculate returns
    returns = np.diff(np.log(c))
    
    # Calculate rolling standard deviation
    rolling_std = np.zeros_like(c)
    for i in range(period, len(c)):
        rolling_std[i] = np.std(returns[i-period:i])
    
    # Detect regimes
    trending = rolling_std > std_threshold
    ranging = ~trending
    
    return {
        "trending": trending,
        "ranging": ranging
    }


def calculate_roc(
    values: List[float],
    period: int = 14
) -> List[float]:
    """Calculate Rate of Change (ROC).
    
    Args:
        values: List of values
        period: The period for ROC calculation
        
    Returns:
        List of ROC values
    """
    roc_list = []
    
    for i in range(len(values)):
        if i < period:
            roc = 0
        else:
            roc = ((values[i] - values[i-period]) / values[i-period]) * 100
        roc_list.append(roc)
    
    return roc_list


def calculate_cmf(
    highs: PriceData,
    lows: PriceData,
    closes: PriceData,
    volumes: VolumeData,
    period: int = 20
) -> IndicatorOutput:
    """Calculate Chaikin Money Flow.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        period: Calculation period
        
    Returns:
        CMF values
    """
    # Convert inputs to numpy arrays
    h = np.array(highs, dtype=np.float64)
    l = np.array(lows, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)
    v = np.array(volumes, dtype=np.float64)
    
    # Calculate Money Flow Multiplier
    mf_mult = ((c - l) - (h - c)) / (h - l)
    mf_mult = np.where(h == l, 0, mf_mult)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mf_vol = mf_mult * v
    
    # Calculate CMF
    cmf = np.zeros_like(c)
    for i in range(period, len(c)):
        cmf[i] = np.sum(mf_vol[i-period:i]) / np.sum(v[i-period:i])
    
    return cmf.tolist()


def calculate_stoch_rsi(
    closes: PriceData,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> IndicatorOutput:
    """Calculate Stochastic RSI.
    
    Args:
        closes: Close prices
        period: RSI period
        smooth_k: %K smoothing period
        smooth_d: %D smoothing period
        
    Returns:
        Stochastic RSI values
    """
    # Convert input to numpy array
    c = np.array(closes, dtype=np.float64)
    
    # Calculate price changes
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Calculate RSI
    avg_gain = np.zeros_like(c)
    avg_loss = np.zeros_like(c)
    
    # First value
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Calculate smoothed averages
    alpha = 1 / period
    for i in range(period + 1, len(c)):
        avg_gain[i] = alpha * gains[i-1] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * losses[i-1] + (1 - alpha) * avg_loss[i-1]
    
    rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    stoch_rsi = np.zeros_like(c)
    for i in range(period, len(c)):
        window = rsi[i-period:i]
        if len(window) > 0:
            min_rsi = np.min(window)
            max_rsi = np.max(window)
            if max_rsi - min_rsi != 0:
                stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
    
    return stoch_rsi.tolist()


def calculate_sma(
    values: List[float],
    period: int
) -> List[float]:
    """Calculate Simple Moving Average (SMA).
    
    Args:
        values: List of values
        period: The period for SMA calculation
        
    Returns:
        List of SMA values
    """
    sma_list = []
    
    for i in range(len(values)):
        if i < period:
            sma = sum(values[:i+1]) / (i + 1)
        else:
            sma = sum(values[i-period:i]) / period
        sma_list.append(sma)
    
    return sma_list 