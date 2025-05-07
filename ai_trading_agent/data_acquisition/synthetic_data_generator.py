"""
Synthetic data generator for testing the AI Trading Agent.

This module provides functions to generate synthetic price and sentiment data
for backtesting and testing the trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple


def generate_price_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1d',
    base_price: Optional[Dict[str, float]] = None,
    volatility: Optional[Dict[str, float]] = None,
    trend: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic price data for backtesting.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date for the data
        end_date: End date for the data
        timeframe: Timeframe for the data ('1m', '5m', '15m', '1h', '4h', '1d')
        base_price: Dictionary mapping symbols to their base price
        volatility: Dictionary mapping symbols to their volatility
        trend: Dictionary mapping symbols to their trend (daily percentage change)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping symbols to their price DataFrames
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate date range
    if timeframe == '1m':
        freq = 'T'
    elif timeframe == '5m':
        freq = '5T'
    elif timeframe == '15m':
        freq = '15T'
    elif timeframe == '1h':
        freq = 'H'
    elif timeframe == '4h':
        freq = '4H'
    else:  # Default to daily
        freq = 'D'
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate price data for each symbol
    price_data = {}
    
    for symbol in symbols:
        # Set default values if not provided
        symbol_base_price = base_price.get(symbol, 100.0) if base_price else 100.0
        symbol_volatility = volatility.get(symbol, 0.02) if volatility else 0.02  # 2% daily volatility
        symbol_trend = trend.get(symbol, 0.0005) if trend else 0.0005  # 0.05% daily trend
        
        # Create a DataFrame with the date range
        df = pd.DataFrame(index=date_range)
        
        # Generate random returns
        returns = np.random.normal(symbol_trend, symbol_volatility, size=len(df))
        
        # Add some autocorrelation to the returns
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Calculate price series
        prices = symbol_base_price * np.cumprod(1 + returns)
        
        # Add columns to DataFrame
        df['open'] = prices
        df['close'] = prices * (1 + returns)  # Close prices slightly different from open
        df['high'] = df['open'] * (1 + np.random.uniform(0, 0.01, size=len(df)))
        df['low'] = df['open'] * (1 - np.random.uniform(0, 0.01, size=len(df)))
        df['volume'] = np.random.lognormal(10, 1, size=len(df))
        
        # Ensure high is the highest price and low is the lowest price
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Store in dictionary
        price_data[symbol] = df
    
    return price_data


def generate_sentiment_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1d',
    base_sentiment: Optional[Dict[str, float]] = None,
    volatility: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic sentiment data for backtesting.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date for the data
        end_date: End date for the data
        timeframe: Timeframe for the data ('1m', '5m', '15m', '1h', '4h', '1d')
        base_sentiment: Dictionary mapping symbols to their base sentiment
        volatility: Dictionary mapping symbols to their sentiment volatility
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping symbols to their sentiment DataFrames
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate date range
    if timeframe == '1m':
        freq = 'T'
    elif timeframe == '5m':
        freq = '5T'
    elif timeframe == '15m':
        freq = '15T'
    elif timeframe == '1h':
        freq = 'H'
    elif timeframe == '4h':
        freq = '4H'
    else:  # Default to daily
        freq = 'D'
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate sentiment data for each symbol
    sentiment_data = {}
    
    for symbol in symbols:
        # Set default values if not provided
        symbol_base_sentiment = base_sentiment.get(symbol, 0.0) if base_sentiment else 0.0
        symbol_volatility = volatility.get(symbol, 0.2) if volatility else 0.2
        
        # Create DataFrame with the date range
        df = pd.DataFrame(index=date_range)
        
        # Generate random sentiment scores
        sentiment_scores = np.random.normal(symbol_base_sentiment, symbol_volatility, size=len(df))
        
        # Add some autocorrelation to the sentiment scores
        for i in range(1, len(sentiment_scores)):
            sentiment_scores[i] = 0.7 * sentiment_scores[i-1] + 0.3 * sentiment_scores[i]
        
        # Clip sentiment scores to [-1, 1]
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        # Add columns to DataFrame
        df['sentiment_score'] = sentiment_scores
        df['confidence_score'] = np.random.uniform(0.5, 1.0, size=len(df))
        df['source_count'] = np.random.randint(1, 100, size=len(df))
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Store in dictionary
        sentiment_data[symbol] = df
    
    return sentiment_data


def generate_market_regimes(
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1d',
    regime_probabilities: Optional[Dict[str, float]] = None,
    regime_duration_range: Optional[Tuple[int, int]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic market regime data for backtesting.
    
    Args:
        start_date: Start date for the data
        end_date: End date for the data
        timeframe: Timeframe for the data ('1m', '5m', '15m', '1h', '4h', '1d')
        regime_probabilities: Dictionary mapping regimes to their probabilities
        regime_duration_range: Tuple of (min_duration, max_duration) in days
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with market regime data
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Set default values if not provided
    if regime_probabilities is None:
        regime_probabilities = {
            'normal': 0.6,
            'trending': 0.2,
            'volatile': 0.15,
            'crisis': 0.05
        }
    
    if regime_duration_range is None:
        regime_duration_range = (5, 30)  # 5 to 30 days
    
    # Generate date range
    if timeframe == '1m':
        freq = 'T'
    elif timeframe == '5m':
        freq = '5T'
    elif timeframe == '15m':
        freq = '15T'
    elif timeframe == '1h':
        freq = 'H'
    elif timeframe == '4h':
        freq = '4H'
    else:  # Default to daily
        freq = 'D'
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create DataFrame with the date range
    df = pd.DataFrame(index=date_range)
    
    # Calculate number of periods per day
    periods_per_day = 1
    if timeframe == '1m':
        periods_per_day = 24 * 60
    elif timeframe == '5m':
        periods_per_day = 24 * 12
    elif timeframe == '15m':
        periods_per_day = 24 * 4
    elif timeframe == '1h':
        periods_per_day = 24
    elif timeframe == '4h':
        periods_per_day = 6
    
    # Generate regimes
    regimes = []
    volatility_levels = []
    trend_strengths = []
    
    # Generate regime changes
    regime_changes = []
    current_date = start_date
    
    while current_date <= end_date:
        # Select a random regime based on probabilities
        regime = np.random.choice(
            list(regime_probabilities.keys()),
            p=list(regime_probabilities.values())
        )
        
        # Determine regime duration
        min_duration, max_duration = regime_duration_range
        duration_days = np.random.randint(min_duration, max_duration + 1)
        duration = timedelta(days=duration_days)
        
        # Set volatility and trend strength based on regime
        if regime == 'normal':
            volatility = np.random.uniform(0.1, 0.2)
            trend_strength = np.random.uniform(0.0002, 0.001)
        elif regime == 'trending':
            volatility = np.random.uniform(0.15, 0.25)
            trend_strength = np.random.uniform(0.001, 0.003)
        elif regime == 'volatile':
            volatility = np.random.uniform(0.25, 0.4)
            trend_strength = np.random.uniform(0.0001, 0.0005)
        else:  # crisis
            volatility = np.random.uniform(0.4, 0.6)
            trend_strength = np.random.uniform(0.002, 0.005) * (-1 if np.random.random() < 0.8 else 1)
        
        # Add regime change
        regime_changes.append({
            'start_date': current_date,
            'end_date': min(current_date + duration, end_date),
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength
        })
        
        # Move to next period
        current_date = min(current_date + duration, end_date) + timedelta(days=1)
    
    # Fill in regimes for each date
    for date in date_range:
        # Find the regime for this date
        for change in regime_changes:
            if change['start_date'] <= date <= change['end_date']:
                regimes.append(change['regime'])
                volatility_levels.append(change['volatility'])
                trend_strengths.append(change['trend_strength'])
                break
        else:
            # If no regime found, use the last regime
            if regimes:
                regimes.append(regimes[-1])
                volatility_levels.append(volatility_levels[-1])
                trend_strengths.append(trend_strengths[-1])
            else:
                # Default to normal regime if no previous regime
                regimes.append('normal')
                volatility_levels.append(0.15)
                trend_strengths.append(0.0005)
    
    # Ensure we have the right number of values
    if len(regimes) > len(date_range):
        regimes = regimes[:len(date_range)]
        volatility_levels = volatility_levels[:len(date_range)]
        trend_strengths = trend_strengths[:len(date_range)]
    elif len(regimes) < len(date_range):
        # Fill with the last values
        while len(regimes) < len(date_range):
            regimes.append(regimes[-1] if regimes else 'normal')
            volatility_levels.append(volatility_levels[-1] if volatility_levels else 0.15)
            trend_strengths.append(trend_strengths[-1] if trend_strengths else 0.0005)
    
    # Add columns to DataFrame
    df['regime'] = regimes
    df['volatility'] = volatility_levels
    df['trend_strength'] = trend_strengths
    
    return df


def generate_synthetic_data_for_backtest(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1d',
    include_sentiment: bool = True,
    include_regimes: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a complete set of synthetic data for backtesting.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date for the data
        end_date: End date for the data
        timeframe: Timeframe for the data ('1m', '5m', '15m', '1h', '4h', '1d')
        include_sentiment: Whether to include sentiment data
        include_regimes: Whether to include market regime data
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with price_data, sentiment_data, and market_regimes
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Generate market regimes first if included
    market_regimes = None
    if include_regimes:
        market_regimes = generate_market_regimes(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            seed=seed
        )
    
    # Generate price data with regime-aware parameters if available
    if market_regimes is not None:
        # Extract regime data for price generation
        volatility = {}
        trend = {}
        
        for symbol in symbols:
            # Use average volatility and trend from regimes
            volatility[symbol] = market_regimes['volatility'].mean()
            trend[symbol] = market_regimes['trend_strength'].mean()
        
        price_data = generate_price_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            volatility=volatility,
            trend=trend,
            seed=seed
        )
    else:
        # Generate price data with default parameters
        price_data = generate_price_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            seed=seed
        )
    
    # Generate sentiment data if included
    sentiment_data = None
    if include_sentiment:
        sentiment_data = generate_sentiment_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            seed=seed
        )
    
    # Return all generated data
    result = {
        'price_data': price_data
    }
    
    if sentiment_data is not None:
        result['sentiment_data'] = sentiment_data
    
    if market_regimes is not None:
        result['market_regimes'] = market_regimes
    
    return result
