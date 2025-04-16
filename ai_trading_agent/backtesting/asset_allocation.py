"""
Asset Allocation Strategies for Multi-Asset Backtesting.

This module provides various asset allocation strategies to be used with the
multi-asset backtesting framework, including equal weight, risk parity,
minimum variance, and sentiment-weighted allocation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import scipy.optimize as sco


def equal_weight_allocation(data: Dict[str, pd.DataFrame], bar_idx: int) -> Dict[str, float]:
    """
    Simple equal weight allocation strategy.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    n_assets = len(symbols)
    
    if n_assets == 0:
        return {}
        
    weight = 1.0 / n_assets
    return {symbol: weight for symbol in symbols}


def market_cap_weight_allocation(
    data: Dict[str, pd.DataFrame], 
    bar_idx: int,
    market_caps: Dict[str, float]
) -> Dict[str, float]:
    """
    Market capitalization weighted allocation.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        market_caps: Dictionary mapping symbols to market capitalizations
        
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    
    # Filter to symbols that have market cap data
    valid_symbols = [s for s in symbols if s in market_caps and market_caps[s] > 0]
    
    if not valid_symbols:
        return equal_weight_allocation(data, bar_idx)
    
    # Calculate total market cap
    total_market_cap = sum(market_caps[s] for s in valid_symbols)
    
    # Calculate weights
    weights = {s: market_caps[s] / total_market_cap for s in valid_symbols}
    
    # Add zero weights for symbols without market cap data
    for s in symbols:
        if s not in weights:
            weights[s] = 0.0
            
    return weights


def minimum_variance_allocation(
    data: Dict[str, pd.DataFrame], 
    bar_idx: int,
    lookback_period: int = 60,
    max_weight: float = 0.4
) -> Dict[str, float]:
    """
    Minimum variance portfolio allocation.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        lookback_period: Number of bars to use for calculating covariance
        max_weight: Maximum weight for any single asset
        
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    n_assets = len(symbols)
    
    if n_assets == 0:
        return {}
        
    if bar_idx < lookback_period:
        return equal_weight_allocation(data, bar_idx)
    
    # Extract returns for lookback period
    returns_data = {}
    for symbol, df in data.items():
        if bar_idx < len(df):
            # Calculate daily returns
            prices = df.iloc[bar_idx - lookback_period:bar_idx]['close'].values
            returns = np.diff(prices) / prices[:-1]
            returns_data[symbol] = returns
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Handle missing data
    returns_df = returns_df.fillna(0)
    
    # If we don't have enough data, fall back to equal weight
    if returns_df.empty or returns_df.shape[0] < 10:
        return equal_weight_allocation(data, bar_idx)
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov().values
    
    # Handle case where covariance matrix is not invertible
    if np.linalg.det(cov_matrix) == 0:
        # Add small diagonal values to make it invertible
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6
    
    # Define optimization problem
    def portfolio_variance(weights):
        weights = np.array(weights)
        return weights.dot(cov_matrix).dot(weights.T)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
    
    # Bounds (0 <= weight <= max_weight)
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    
    # Initial guess (equal weight)
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Solve optimization problem
    result = sco.minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # If optimization fails, fall back to equal weight
    if not result.success:
        return equal_weight_allocation(data, bar_idx)
    
    # Convert result to dictionary
    weights = {symbol: weight for symbol, weight in zip(symbols, result.x)}
    
    return weights


def risk_parity_allocation(
    data: Dict[str, pd.DataFrame], 
    bar_idx: int,
    lookback_period: int = 60,
    risk_target: float = 0.1
) -> Dict[str, float]:
    """
    Risk parity portfolio allocation where each asset contributes equally to portfolio risk.
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        lookback_period: Number of bars to use for volatility estimation
        risk_target: Target portfolio risk (not always used)
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    n_assets = len(symbols)
    if n_assets == 0:
        return {}
    if bar_idx < lookback_period:
        return equal_weight_allocation(data, bar_idx)
    # Compute returns for each asset
    returns_data = {}
    for symbol, df in data.items():
        if bar_idx < len(df):
            prices = df.iloc[bar_idx - lookback_period:bar_idx]['close'].values
            returns = np.diff(prices) / prices[:-1]
            returns_data[symbol] = returns
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.fillna(0)
    if returns_df.empty or returns_df.shape[0] < 10:
        return equal_weight_allocation(data, bar_idx)
    # Calculate asset volatilities
    volatilities = returns_df.std()
    inv_vol = 1.0 / volatilities
    inv_vol = inv_vol.replace([np.inf, -np.inf], 0)
    total_inv_vol = inv_vol.sum()
    if total_inv_vol == 0:
        return equal_weight_allocation(data, bar_idx)
    weights = {symbol: inv_vol[symbol] / total_inv_vol for symbol in symbols}
    return weights


def sentiment_weighted_allocation(
    data: Dict[str, pd.DataFrame], 
    bar_idx: int,
    sentiment_scores: Dict[str, float],
    base_allocation_fn=equal_weight_allocation,
    sentiment_weight: float = 0.5
) -> Dict[str, float]:
    """
    Sentiment-weighted allocation that adjusts a base allocation strategy
    based on sentiment scores.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        sentiment_scores: Dictionary mapping symbols to sentiment scores (-1 to 1)
        base_allocation_fn: Base allocation function to adjust
        sentiment_weight: Weight given to sentiment adjustment (0 to 1)
        
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    
    # Get base allocation
    base_weights = base_allocation_fn(data, bar_idx)
    
    # If no sentiment data, return base allocation
    if not sentiment_scores:
        return base_weights
    
    # Normalize sentiment scores to range [0, 1]
    normalized_scores = {}
    for symbol in symbols:
        score = sentiment_scores.get(symbol, 0)
        normalized_scores[symbol] = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    # Calculate total sentiment score
    total_score = sum(normalized_scores.values())
    
    # If total score is 0, return base allocation
    if total_score == 0:
        return base_weights
    
    # Calculate sentiment-based weights
    sentiment_weights = {symbol: score / total_score for symbol, score in normalized_scores.items()}
    
    # Combine base and sentiment weights
    combined_weights = {}
    for symbol in symbols:
        base = base_weights.get(symbol, 0)
        sentiment = sentiment_weights.get(symbol, 0)
        combined_weights[symbol] = (1 - sentiment_weight) * base + sentiment_weight * sentiment
    
    # Normalize weights to sum to 1
    total_weight = sum(combined_weights.values())
    if total_weight > 0:
        combined_weights = {symbol: weight / total_weight for symbol, weight in combined_weights.items()}
    
    return combined_weights


def momentum_allocation(
    data: Dict[str, pd.DataFrame], 
    bar_idx: int,
    lookback_period: int = 60,
    top_n: Optional[int] = None
) -> Dict[str, float]:
    """
    Momentum-based allocation that weights assets based on their recent performance.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        lookback_period: Number of bars to use for calculating momentum
        top_n: Number of top performers to include (None = include all)
        
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    n_assets = len(symbols)
    
    if n_assets == 0:
        return {}
        
    if bar_idx < lookback_period:
        return equal_weight_allocation(data, bar_idx)
    
    # Calculate momentum for each asset
    momentum_scores = {}
    for symbol, df in data.items():
        if bar_idx < len(df) and bar_idx - lookback_period >= 0:
            start_price = df.iloc[bar_idx - lookback_period]['close']
            end_price = df.iloc[bar_idx - 1]['close']
            
            if start_price > 0:
                momentum = (end_price / start_price) - 1
                momentum_scores[symbol] = momentum
    
    # If no momentum data, return equal weight
    if not momentum_scores:
        return equal_weight_allocation(data, bar_idx)
    
    # Filter to top N performers if specified
    if top_n is not None and top_n < len(momentum_scores):
        top_symbols = sorted(momentum_scores.keys(), key=lambda s: momentum_scores[s], reverse=True)[:top_n]
        momentum_scores = {s: momentum_scores[s] for s in top_symbols}
    
    # Handle negative momentum scores
    min_score = min(momentum_scores.values())
    if min_score < 0:
        # Shift all scores to be positive
        momentum_scores = {s: score - min_score + 0.01 for s, score in momentum_scores.items()}
    
    # Calculate total momentum
    total_momentum = sum(momentum_scores.values())
    
    # Calculate weights
    weights = {symbol: score / total_momentum for symbol, score in momentum_scores.items()}
    
    # Add zero weights for symbols without momentum data
    for s in symbols:
        if s not in weights:
            weights[s] = 0.0
    
    return weights


def momentum_weight_allocation(
    data: Dict[str, pd.DataFrame],
    bar_idx: int,
    lookback_period: int = 30,
    top_n: int = None
) -> Dict[str, float]:
    """
    Momentum-based allocation: weights proportional to recent returns.
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        bar_idx: Current bar index
        lookback_period: Number of bars to use for calculating momentum
        top_n: Number of top performers to include (None = include all)
    Returns:
        Dictionary mapping symbols to target weights
    """
    symbols = list(data.keys())
    n_assets = len(symbols)
    if n_assets == 0:
        return {}
    if bar_idx < lookback_period:
        return equal_weight_allocation(data, bar_idx)
    momentum_scores = {}
    for symbol, df in data.items():
        if bar_idx < len(df) and bar_idx - lookback_period >= 0:
            start_price = df.iloc[bar_idx - lookback_period]["close"]
            end_price = df.iloc[bar_idx - 1]["close"]
            if start_price > 0:
                momentum = (end_price / start_price) - 1
                momentum_scores[symbol] = momentum
    if not momentum_scores:
        return equal_weight_allocation(data, bar_idx)
    # Filter to top N performers if specified
    if top_n is not None and top_n < len(momentum_scores):
        top_symbols = sorted(momentum_scores.keys(), key=lambda s: momentum_scores[s], reverse=True)[:top_n]
        momentum_scores = {s: momentum_scores[s] for s in top_symbols}
    # Handle negative momentum scores
    min_score = min(momentum_scores.values())
    if min_score < 0:
        momentum_scores = {s: score - min_score + 0.01 for s, score in momentum_scores.items()}
    total_momentum = sum(momentum_scores.values())
    weights = {symbol: score / total_momentum for symbol, score in momentum_scores.items()}
    for s in symbols:
        if s not in weights:
            weights[s] = 0.0
    return weights
