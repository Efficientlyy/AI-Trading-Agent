"""
Market data utility functions.

This module provides utility functions for generating mock market data
when real data is not available.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from .market_models import HistoricalBar, TimeFrame, MarketRegime, VolatilityRegime

def generate_mock_historical_data(
    symbol: str,
    timeframe: TimeFrame,
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[HistoricalBar]:
    """
    Generate mock historical price data with realistic patterns.
    
    This function creates synthetic price data that follows realistic market patterns
    including trends, volatility clusters, and appropriate time intervals.
    
    Args:
        symbol: Trading symbol
        timeframe: Time interval between data points
        limit: Number of data points to generate
        start_date: Optional start date for historical data
        end_date: Optional end date for historical data
        
    Returns:
        List of HistoricalBar objects
    """
    # Set end date to now if not specified
    if end_date is None:
        end_date = datetime.now()
        
    # Calculate start date based on limit and timeframe if not specified
    if start_date is None:
        # Map timeframe to timedelta
        timeframe_map = {
            TimeFrame.MINUTE_1: timedelta(minutes=1),
            TimeFrame.MINUTE_5: timedelta(minutes=5),
            TimeFrame.MINUTE_15: timedelta(minutes=15),
            TimeFrame.MINUTE_30: timedelta(minutes=30),
            TimeFrame.HOUR_1: timedelta(hours=1),
            TimeFrame.HOUR_4: timedelta(hours=4),
            TimeFrame.DAY_1: timedelta(days=1),
            TimeFrame.WEEK_1: timedelta(weeks=1),
            TimeFrame.MONTH_1: timedelta(days=30),  # Approximate
        }
        
        # Calculate start date
        delta = timeframe_map.get(timeframe, timedelta(days=1))
        start_date = end_date - (delta * limit)
    
    # Generate timestamps
    delta = (end_date - start_date) / max(1, limit - 1)
    timestamps = [start_date + delta * i for i in range(limit)]
    
    # Set seed based on symbol for consistent results
    seed = sum(ord(c) for c in symbol)
    np.random.seed(seed)
    
    # Base price depends on the symbol
    if "BTC" in symbol:
        base_price = 105000.0  # Starting price for BTC
        volatility = 0.02  # 2% daily volatility
    elif "ETH" in symbol:
        base_price = 2500.0  # Starting price for ETH
        volatility = 0.025  # 2.5% daily volatility
    elif "SOL" in symbol:
        base_price = 140.0  # Starting price for SOL
        volatility = 0.035  # 3.5% daily volatility
    elif "XRP" in symbol:
        base_price = 0.65  # Starting price for XRP
        volatility = 0.03  # 3% daily volatility
    elif "ADA" in symbol:
        base_price = 0.45  # Starting price for ADA
        volatility = 0.03  # 3% daily volatility
    elif "DOGE" in symbol:
        base_price = 0.12  # Starting price for DOGE
        volatility = 0.04  # 4% daily volatility
    else:
        base_price = 100.0  # Generic starting price
        volatility = 0.02  # 2% daily volatility
    
    # Adjust volatility based on timeframe
    timeframe_volatility_factor = {
        TimeFrame.MINUTE_1: 0.2,
        TimeFrame.MINUTE_5: 0.3,
        TimeFrame.MINUTE_15: 0.4,
        TimeFrame.MINUTE_30: 0.5,
        TimeFrame.HOUR_1: 0.6,
        TimeFrame.HOUR_4: 0.8,
        TimeFrame.DAY_1: 1.0,
        TimeFrame.WEEK_1: 1.5,
        TimeFrame.MONTH_1: 2.0,
    }
    
    volatility = volatility * timeframe_volatility_factor.get(timeframe, 1.0)
    
    # Generate price data using Geometric Brownian Motion
    returns = np.random.normal(0.0002, volatility, size=limit)  # Small positive drift
    
    # Add a trend component (bullish or bearish based on seed)
    trend = 0.0003 if seed % 2 == 0 else -0.0001
    returns = returns + trend
    
    # Add some mean reversion
    mean_reversion = 0.1
    for i in range(1, len(returns)):
        returns[i] = returns[i] - (returns[i-1] * mean_reversion)
    
    # Calculate price path
    prices = [base_price]
    for ret in returns[:-1]:  # Exclude the last return as we already have the first price
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    bars = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        if i >= len(prices):
            continue

        # Calculate realistic OHLC values
        high_factor = 1 + abs(np.random.normal(0, volatility * 0.5))
        low_factor = 1 - abs(np.random.normal(0, volatility * 0.5))
        
        open_price = price * (1 + np.random.normal(0, volatility * 0.3))
        high_price = max(price, open_price) * high_factor
        low_price = min(price, open_price) * low_factor
        close_price = price
        
        # Ensure high is highest and low is lowest
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate realistic volume
        if "BTC" in symbol:
            base_volume = 1000000000  # $1B for BTC
        elif "ETH" in symbol:
            base_volume = 500000000  # $500M for ETH
        else:
            base_volume = 100000000  # $100M for others
        
        # Volume is higher when price moves more
        price_change_factor = 1 + abs((close_price - open_price) / open_price) * 10
        volume = base_volume * price_change_factor * np.random.lognormal(0, 0.5)
        
        bar = HistoricalBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        bars.append(bar)
    
    return bars

def detect_market_regime(
    historical_data: List[HistoricalBar],
    window: int = 20
) -> Tuple[MarketRegime, VolatilityRegime]:
    """
    Detect the current market regime based on historical data.
    
    This function analyzes recent price action to determine the current
    market regime (bull, bear, sideways, volatile) and volatility regime.
    
    Args:
        historical_data: List of historical price bars
        window: Number of bars to analyze
        
    Returns:
        Tuple of (MarketRegime, VolatilityRegime)
    """
    if not historical_data or len(historical_data) < window:
        return MarketRegime.UNKNOWN, VolatilityRegime.UNKNOWN
    
    # Get the most recent data window
    recent_data = historical_data[-window:]
    
    # Extract close prices
    closes = [bar.close for bar in recent_data]
    
    # Calculate returns
    returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
    
    # Calculate mean return and volatility
    mean_return = sum(returns) / len(returns)
    volatility = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
    
    # Calculate trend strength using linear regression
    x = list(range(len(closes)))
    mean_x = sum(x) / len(x)
    mean_y = sum(closes) / len(closes)
    numerator = sum((x[i] - mean_x) * (closes[i] - mean_y) for i in range(len(x)))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    slope = numerator / denominator if denominator != 0 else 0
    
    # Normalize slope
    norm_slope = slope / (closes[0] / len(closes)) if closes[0] != 0 else 0
    
    # Determine market regime
    if volatility > 0.03:  # High volatility threshold
        if norm_slope > 0.01:
            market_regime = MarketRegime.VOLATILE_BULL
        elif norm_slope < -0.01:
            market_regime = MarketRegime.VOLATILE_BEAR
        else:
            market_regime = MarketRegime.VOLATILE
    else:
        if norm_slope > 0.01:
            market_regime = MarketRegime.BULL
        elif norm_slope < -0.01:
            market_regime = MarketRegime.BEAR
        else:
            market_regime = MarketRegime.SIDEWAYS
    
    # Determine volatility regime
    if volatility < 0.005:
        volatility_regime = VolatilityRegime.VERY_LOW
    elif volatility < 0.01:
        volatility_regime = VolatilityRegime.LOW
    elif volatility < 0.02:
        volatility_regime = VolatilityRegime.MODERATE
    elif volatility < 0.03:
        volatility_regime = VolatilityRegime.HIGH
    elif volatility < 0.05:
        volatility_regime = VolatilityRegime.VERY_HIGH
    else:
        volatility_regime = VolatilityRegime.EXTREME
    
    return market_regime, volatility_regime

def get_technical_indicators(historical_data: List[HistoricalBar]) -> Dict[str, Any]:
    """
    Calculate technical indicators based on historical price data.
    
    This function calculates common technical indicators used for 
    technical analysis including moving averages, RSI, MACD, etc.
    
    Args:
        historical_data: List of historical price bars
        
    Returns:
        Dictionary of technical indicators
    """
    if not historical_data or len(historical_data) < 50:
        return {}
    
    # Extract close prices
    closes = pd.Series([bar.close for bar in historical_data])
    
    # Calculate indicators
    indicators = {}
    
    # Moving Averages
    indicators['sma_20'] = closes.rolling(window=20).mean().iloc[-1]
    indicators['sma_50'] = closes.rolling(window=50).mean().iloc[-1]
    indicators['sma_200'] = closes.rolling(window=50).mean().iloc[-1] if len(closes) >= 200 else None
    
    # Exponential Moving Averages
    indicators['ema_12'] = closes.ewm(span=12).mean().iloc[-1]
    indicators['ema_26'] = closes.ewm(span=26).mean().iloc[-1]
    
    # MACD
    ema_12 = closes.ewm(span=12).mean()
    ema_26 = closes.ewm(span=26).mean()
    indicators['macd'] = (ema_12 - ema_26).iloc[-1]
    indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
    indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
    
    # RSI (14-period)
    delta = closes.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    indicators['rsi_14'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    # Bollinger Bands (20-period, 2 standard deviations)
    sma_20 = closes.rolling(window=20).mean()
    std_20 = closes.rolling(window=20).std()
    indicators['bollinger_upper'] = (sma_20 + 2 * std_20).iloc[-1]
    indicators['bollinger_middle'] = sma_20.iloc[-1]
    indicators['bollinger_lower'] = (sma_20 - 2 * std_20).iloc[-1]
    
    # Trading signals based on indicators
    indicators['signal_macd'] = 'buy' if indicators['macd'] > indicators['macd_signal'] else 'sell'
    indicators['signal_rsi'] = 'oversold' if indicators['rsi_14'] < 30 else ('overbought' if indicators['rsi_14'] > 70 else 'neutral')
    indicators['signal_ma'] = 'bullish' if closes.iloc[-1] > indicators['sma_50'] else 'bearish'
    
    return indicators
