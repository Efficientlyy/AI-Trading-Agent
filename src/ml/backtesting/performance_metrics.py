"""Performance metrics calculation for backtesting.

This module provides functions for calculating various performance metrics
for trading strategies, including risk-adjusted returns, drawdowns, and
trading statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, cast
from datetime import datetime


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate simple returns from prices.
    
    Args:
        prices: Array of prices
        
    Returns:
        Array of returns (percentage change)
    """
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] / prices[:-1]) - 1
    return returns


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate logarithmic returns from prices.
    
    Args:
        prices: Array of prices
        
    Returns:
        Array of log returns
    """
    returns = np.zeros_like(prices)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    return returns


def calculate_cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative returns from simple returns.
    
    Args:
        returns: Array of simple returns
        
    Returns:
        Array of cumulative returns
    """
    return np.cumprod(1 + returns) - 1


def calculate_drawdowns(returns: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdowns from returns.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tuple of (drawdowns, max_drawdown, max_drawdown_duration)
    """
    cum_returns = calculate_cumulative_returns(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns) / (1 + running_max) - 1
    
    # Calculate maximum drawdown and its duration
    max_drawdown = float(np.min(drawdowns))
    
    # Calculate drawdown duration
    is_in_drawdown = drawdowns < 0
    drawdown_start = np.where(np.diff(is_in_drawdown.astype(int)) == 1)[0] + 1
    drawdown_end = np.where(np.diff(is_in_drawdown.astype(int)) == -1)[0] + 1
    
    if len(drawdown_start) == 0 or len(drawdown_end) == 0:
        max_drawdown_duration = 0
    else:
        # Ensure we have matching start and end points
        if drawdown_start[0] > drawdown_end[0]:
            drawdown_end = drawdown_end[1:]
        if len(drawdown_start) > len(drawdown_end):
            drawdown_start = drawdown_start[:-1]
            
        # Calculate durations and find the maximum
        durations = drawdown_end - drawdown_start
        max_drawdown_duration = int(np.max(durations)) if len(durations) > 0 else 0
    
    return drawdowns, max_drawdown, max_drawdown_duration


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate downside deviation (std dev of negative returns only)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(negative_returns)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annual_return = np.mean(returns) * periods_per_year
    _, max_drawdown, _ = calculate_drawdowns(returns)
    
    if max_drawdown == 0:
        return float('inf') if annual_return > 0 else 0.0
    
    return float(-annual_return / max_drawdown)


def calculate_win_rate(pnl_values: np.ndarray) -> float:
    """
    Calculate win rate from trade P&L values.
    
    Args:
        pnl_values: Array of P&L values for each trade
        
    Returns:
        Win rate (proportion of winning trades)
    """
    if len(pnl_values) == 0:
        return 0.0
    
    winning_trades = np.sum(pnl_values > 0)
    return winning_trades / len(pnl_values)


def calculate_profit_factor(pnl_values: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        pnl_values: Array of P&L values for each trade
        
    Returns:
        Profit factor
    """
    if len(pnl_values) == 0:
        return 0.0
    
    gross_profit = np.sum(pnl_values[pnl_values > 0])
    gross_loss = np.abs(np.sum(pnl_values[pnl_values < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(pnl_values: np.ndarray) -> float:
    """
    Calculate the expectancy (average trade P&L).
    
    Args:
        pnl_values: Array of P&L values for each trade
        
    Returns:
        Expectancy (average trade P&L)
    """
    if len(pnl_values) == 0:
        return 0.0
    
    return float(np.mean(pnl_values))


def calculate_regime_metrics(
    returns: np.ndarray,
    regimes: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Calculate performance metrics for each market regime.
    
    Args:
        returns: Array of returns
        regimes: Array of regime labels
        regime_names: Optional dictionary mapping regime IDs to names
        
    Returns:
        Dictionary of performance metrics for each regime
    """
    if len(returns) != len(regimes):
        raise ValueError("Length of returns and regimes arrays must match")
    
    unique_regimes = np.unique(regimes)
    metrics = {}
    
    for regime in unique_regimes:
        regime_returns = returns[regimes == regime]
        
        if len(regime_returns) == 0:
            continue
        
        regime_metrics = {
            'count': len(regime_returns),
            'total_return': np.prod(1 + regime_returns) - 1,
            'mean_return': np.mean(regime_returns),
            'std_return': np.std(regime_returns),
            'sharpe': calculate_sharpe_ratio(regime_returns),
            'max_drawdown': calculate_drawdowns(regime_returns)[1],
            'duration_pct': len(regime_returns) / len(returns)
        }
        
        metrics[int(regime)] = regime_metrics
    
    return metrics


def calculate_comprehensive_metrics(
    returns: np.ndarray, 
    equity_curve: np.ndarray,
    trades: List[Dict[str, Union[float, str, datetime]]],
    regimes: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Union[float, Dict[Any, Any]]]:
    """
    Calculate comprehensive performance metrics for a backtest.
    
    Args:
        returns: Array of returns
        equity_curve: Array of equity values
        trades: List of trade dictionaries
        regimes: Optional array of regime labels
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return {}
    
    # Extract trade P&L values - ensure we're only using numeric values
    pnl_values = np.array([float(trade.get('pnl', 0)) if isinstance(trade.get('pnl', 0), (int, float)) else 0.0 
                          for trade in trades])
    
    # Calculate drawdowns
    drawdowns, max_drawdown, max_drawdown_duration = calculate_drawdowns(returns)
    
    # Total return
    total_return = equity_curve[-1] / equity_curve[0] - 1
    
    # Annual return
    annual_return = np.mean(returns) * periods_per_year
    
    # Annual volatility
    annual_volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    # Consecutive wins/losses
    trade_results = np.array([1 if isinstance(trade.get('pnl', 0), (int, float)) and float(trade.get('pnl', 0)) > 0 
                             else 0 for trade in trades])
    
    win_streaks = []
    loss_streaks = []
    current_streak = 0
    current_type = None
    
    for result in trade_results:
        if result == current_type or current_type is None:
            current_streak += 1
            current_type = result
        else:
            if current_type == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
            current_type = result
    
    # Don't forget the last streak
    if current_type == 1:
        win_streaks.append(current_streak)
    elif current_type == 0 and current_streak > 0:
        loss_streaks.append(current_streak)
    
    max_consecutive_wins = max(win_streaks) if win_streaks else 0
    max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'win_rate': calculate_win_rate(pnl_values),
        'profit_factor': calculate_profit_factor(pnl_values),
        'expectancy': calculate_expectancy(pnl_values),
        'num_trades': len(trades),
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
    }
    
    # Add regime-specific metrics if provided
    if regimes is not None:
        metrics["regime_metrics"] = calculate_regime_metrics(returns, regimes)
    
    return metrics 