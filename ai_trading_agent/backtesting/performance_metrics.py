"""
Performance metrics module for AI Trading Agent.

This module provides functions for calculating and analyzing trading performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    daily_returns: pd.Series
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    avg_profit_loss_ratio: float
    
    # Exposure metrics
    avg_exposure: float
    time_in_market: float
    
    # Additional metrics
    calmar_ratio: float
    omega_ratio: float

    # Advanced risk metrics
    value_at_risk: float
    conditional_value_at_risk: float
    
    # Raw data
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_summary: pd.DataFrame


def calculate_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) at the specified alpha level.
    VaR is the maximum expected loss at a given confidence level.
    """
    if returns.empty:
        return 0.0
    return np.percentile(returns, 100 * alpha)

def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR, Expected Shortfall) at the specified alpha level.
    CVaR is the expected loss given that the loss is beyond the VaR threshold.
    """
    if returns.empty:
        return 0.0
    var = calculate_var(returns, alpha)
    return returns[returns <= var].mean()

def get_drawdown_data(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    Generate comprehensive drawdown data for visualization and analysis.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Dict with drawdown data including timestamps, equity values, drawdown values, and underwater periods
    """
    # Calculate drawdown curve
    drawdown_curve = calculate_drawdown(equity_curve)
    
    # Get underwater periods
    underwater_periods = get_drawdown_periods(drawdown_curve)
    
    # Prepare data for frontend
    return {
        'timestamps': [str(ts) for ts in equity_curve.index.tolist()],
        'equity': equity_curve.tolist(),
        'drawdown': drawdown_curve.tolist(),
        'underwater_periods': underwater_periods
    }


def calculate_trade_statistics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate detailed trade statistics from trade history.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dict with trade statistics
    """
    if trades_df.empty:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'average_duration': 0.0
        }
    
    # Filter to closed trades
    closed_trades = trades_df[trades_df['status'] == 'closed']
    
    if closed_trades.empty:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'average_duration': 0.0
        }
    
    # Calculate win rate
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0.0
    
    # Calculate profit factor
    gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    # Calculate average win/loss
    average_win = winning_trades['pnl_percent'].mean() if not winning_trades.empty else 0.0
    average_loss = losing_trades['pnl_percent'].mean() if not losing_trades.empty else 0.0
    
    # Calculate risk/reward ratio
    risk_reward_ratio = abs(average_win / average_loss) if average_loss != 0 else float('inf') if average_win > 0 else 0.0
    
    # Calculate max consecutive wins/losses
    if 'streak' in closed_trades.columns:
        max_consecutive_wins = closed_trades['streak'].max() if not closed_trades.empty else 0
        max_consecutive_losses = abs(closed_trades['streak'].min()) if not closed_trades.empty else 0
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0
    
    # Calculate average duration
    average_duration = closed_trades['duration'].mean() if 'duration' in closed_trades.columns else 0.0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'average_win': average_win,
        'average_loss': average_loss,
        'risk_reward_ratio': risk_reward_ratio,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'average_duration': average_duration
    }


def calculate_metrics(
    portfolio_history: List[Dict[str, Any]],
    trade_history: List,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        portfolio_history: List of portfolio snapshots
        trade_history: List of trade objects
        initial_capital: Initial portfolio capital
        risk_free_rate: Annual risk-free rate (decimal)
        target_return: Target return for downside deviation calculation
        
    Returns:
        PerformanceMetrics: Object containing all performance metrics
    """
    # Convert portfolio history to DataFrame
    portfolio_df = pd.DataFrame([
        {'timestamp': p['timestamp'], 'total_value': p['total_value']}
        for p in portfolio_history
    ]).set_index('timestamp')
    
    # Calculate equity curve
    equity_curve = portfolio_df['total_value']
    
    # Calculate returns
    returns = equity_curve.pct_change().fillna(0)
    
    # Calculate drawdowns
    drawdown_curve = calculate_drawdown(equity_curve)
    
    # Calculate basic return metrics
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    
    # Annualize returns based on trading frequency
    trading_days = len(equity_curve)
    calendar_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if calendar_days == 0:
        calendar_days = 1  # Avoid division by zero
    
    # Estimate trading days per year
    days_per_year = 252  # Standard trading days in a year
    estimated_annual_periods = days_per_year * (trading_days / calendar_days)
    
    annualized_return = (1 + total_return) ** (days_per_year / calendar_days) - 1
    
    # Calculate risk metrics
    volatility = returns.std() * np.sqrt(estimated_annual_periods)

    # Advanced risk metrics: VaR and CVaR (using daily returns)
    value_at_risk = calculate_var(returns, alpha=0.05)
    conditional_value_at_risk = calculate_cvar(returns, alpha=0.05)
    
    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0
    
    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < target_return]
    downside_deviation = downside_returns.std() * np.sqrt(estimated_annual_periods)
    sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
    
    # Maximum drawdown
    max_drawdown = drawdown_curve.min()
    
    # Maximum drawdown duration
    drawdown_periods = get_drawdown_periods(drawdown_curve)
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    
    # Process trade history
    trade_df = process_trade_history(trade_history)
    
    # Calculate trade metrics
    total_trades = len(trade_df) if not trade_df.empty else 0
    
    if total_trades > 0:
        win_rate = len(trade_df[trade_df['pnl'] > 0]) / total_trades
        
        total_profit = trade_df[trade_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trade_df[trade_df['pnl'] < 0]['pnl'].sum())
        
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        winning_trades = trade_df[trade_df['pnl'] > 0]
        losing_trades = trade_df[trade_df['pnl'] < 0]
        
        avg_profit_per_trade = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss_per_trade = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        avg_profit_loss_ratio = abs(avg_profit_per_trade / avg_loss_per_trade) if avg_loss_per_trade != 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_profit_per_trade = 0.0
        avg_loss_per_trade = 0.0
        avg_profit_loss_ratio = 0.0
    
    # Calculate exposure metrics
    positions_df = create_positions_df(portfolio_history)
    
    if not positions_df.empty:
        # Average exposure (as percentage of portfolio)
        avg_exposure = positions_df['exposure'].mean()
        
        # Time in market (percentage of time with open positions)
        time_in_market = len(positions_df[positions_df['exposure'] > 0]) / len(positions_df)
    else:
        avg_exposure = 0.0
        time_in_market = 0.0
    
    # Calculate additional metrics
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Omega ratio (probability-weighted ratio of gains versus losses)
    threshold = 0  # Can be set to risk-free rate or target return
    omega_ratio = calculate_omega_ratio(returns, threshold)
    
    metrics = PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        daily_returns=returns,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_profit_per_trade=avg_profit_per_trade,
        avg_loss_per_trade=avg_loss_per_trade,
        avg_profit_loss_ratio=avg_profit_loss_ratio,
        avg_exposure=avg_exposure,
        time_in_market=time_in_market,
        calmar_ratio=calmar_ratio,
        omega_ratio=omega_ratio,
        value_at_risk=value_at_risk,
        conditional_value_at_risk=conditional_value_at_risk,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trade_summary=trade_df
    )
    return metrics


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        pd.Series: Drawdown series (negative values)
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown as percentage of running maximum
    drawdown = (equity_curve / running_max) - 1
    
    return drawdown


def get_drawdown_periods(drawdown_curve: pd.Series) -> List[Dict[str, Any]]:
    """
    Calculate drawdown periods (consecutive periods below zero).
    
    Args:
        drawdown_curve: Series of drawdown values
        
    Returns:
        List[Dict]: List of drawdown periods with start, end, depth, and duration
    """
    # Initialize variables
    in_drawdown = False
    current_start = 0
    periods = []
    
    # Iterate through drawdown curve
    for i, (timestamp, dd) in enumerate(drawdown_curve.items()):
        if dd < 0:
            # We're in a drawdown
            if not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                current_start = i
        else:
            # We're not in a drawdown
            if in_drawdown:
                # End of a drawdown period
                current_end = i - 1
                # Get the deepest drawdown in this period
                period_slice = drawdown_curve.iloc[current_start:current_end+1]
                max_depth = period_slice.min()
                max_depth_idx = period_slice.idxmin()
                
                periods.append({
                    'start': current_start,
                    'end': current_end,
                    'depth': max_depth,
                    'duration': current_end - current_start + 1,
                    'start_date': drawdown_curve.index[current_start],
                    'end_date': drawdown_curve.index[current_end],
                    'max_depth_date': max_depth_idx
                })
                
                in_drawdown = False
    
    # Check if we're still in a drawdown at the end
    if in_drawdown:
        current_end = len(drawdown_curve) - 1
        period_slice = drawdown_curve.iloc[current_start:current_end+1]
        max_depth = period_slice.min()
        max_depth_idx = period_slice.idxmin()
        
        periods.append({
            'start': current_start,
            'end': current_end,
            'depth': max_depth,
            'duration': current_end - current_start + 1,
            'start_date': drawdown_curve.index[current_start],
            'end_date': drawdown_curve.index[current_end],
            'max_depth_date': max_depth_idx
        })
    
    return periods


def process_trade_history(trade_history: List) -> pd.DataFrame:
    """
    Process trade history into a DataFrame with calculated metrics.
    
    Args:
        trade_history: List of trade objects
        
    Returns:
        pd.DataFrame: DataFrame with trade metrics
    """
    if not trade_history:
        return pd.DataFrame()
    
    # Extract trade data
    trades = []
    for trade in trade_history:
        trade_data = {
            'id': getattr(trade, 'id', f"{trade.symbol}_{trade.entry_time}"),
            'symbol': trade.symbol,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'side': trade.side,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'status': trade.status
        }
        trades.append(trade_data)
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate additional metrics
    if not trades_df.empty and 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        # Convert timestamps to datetime if they're not already
        for col in ['entry_time', 'exit_time']:
            if trades_df[col].dtype == 'object':
                trades_df[col] = pd.to_datetime(trades_df[col])
        
        # Calculate trade duration
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60  # in minutes
        
        # Calculate consecutive wins/losses
        if 'pnl' in trades_df.columns and not trades_df.empty:
            # Sort by exit time
            trades_df = trades_df.sort_values('exit_time')
            
            # Initialize streak counters
            current_streak = 0
            streak_type = None
            trades_df['streak'] = 0
            
            # Calculate streaks
            for i, row in trades_df.iterrows():
                if row['status'] == 'closed':
                    is_win = row['pnl'] > 0
                    
                    if streak_type is None:
                        # First trade
                        streak_type = 'win' if is_win else 'loss'
                        current_streak = 1
                    elif (is_win and streak_type == 'win') or (not is_win and streak_type == 'loss'):
                        # Continuing the streak
                        current_streak += 1
                    else:
                        # Breaking the streak
                        streak_type = 'win' if is_win else 'loss'
                        current_streak = 1
                    
                    trades_df.at[i, 'streak'] = current_streak if streak_type == 'win' else -current_streak
    
    return trades_df


def create_positions_df(portfolio_history: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame with position and exposure information.
    
    Args:
        portfolio_history: List of portfolio snapshots
        
    Returns:
        pd.DataFrame: DataFrame with position and exposure information
    """
    if not portfolio_history:
        return pd.DataFrame()
    
    # Extract position information
    positions = []
    for snapshot in portfolio_history:
        timestamp = snapshot['timestamp']
        total_value = snapshot['total_value']
        
        # Calculate total position value
        position_value = 0
        for symbol, pos in snapshot.get('positions', {}).items():
            position_value += pos.get('value', 0)
        
        # Calculate exposure
        exposure = position_value / total_value if total_value > 0 else 0
        
        positions.append({
            'timestamp': timestamp,
            'total_value': total_value,
            'position_value': position_value,
            'exposure': exposure
        })
    
    return pd.DataFrame(positions).set_index('timestamp')


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    """
    Calculate the Omega ratio.
    
    The Omega ratio is the probability-weighted ratio of gains versus losses
    for some threshold return target.
    
    Args:
        returns: Series of returns
        threshold: Return threshold
        
    Returns:
        float: Omega ratio
    """
    # Separate returns into gains and losses relative to threshold
    gains = returns[returns >= threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    # Calculate expected gains and losses
    expected_gain = gains.sum() / len(returns) if not gains.empty else 0
    expected_loss = losses.sum() / len(returns) if not losses.empty else 0
    
    # Calculate Omega ratio
    omega = expected_gain / expected_loss if expected_loss != 0 else float('inf')
    return omega
    return omega
    
    return omega


def calculate_asset_metrics(portfolio_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate asset-level returns and drawdowns.

    Args:
        portfolio_history: List of portfolio snapshots with 'positions' key.

    Returns:
        Dict mapping symbol to metrics dict.
    """
    import pandas as pd

    # Build time series per asset
    asset_values = {}
    for snapshot in portfolio_history:
        ts = snapshot['timestamp']
        for symbol, pos in snapshot.get('positions', {}).items():
            if symbol not in asset_values:
                asset_values[symbol] = []
            value = pos.get('quantity', 0) * pos.get('entry_price', 0)
            asset_values[symbol].append((ts, value))

    metrics = {}
    for symbol, series in asset_values.items():
        df = pd.DataFrame(series, columns=["timestamp", "value"]).set_index("timestamp")
        returns = df["value"].pct_change().fillna(0)
        total_return = (df["value"].iloc[-1] / df["value"].iloc[0]) - 1 if df["value"].iloc[0] != 0 else 0
        max_drawdown = (df["value"] / df["value"].cummax() - 1).min()
        metrics[symbol] = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "volatility": returns.std() * (252 ** 0.5)
        }
    return metrics

def calculate_portfolio_diversification(correlation_matrix: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate a simple diversification score based on average pairwise correlations.

    Args:
        correlation_matrix: Nested dict of correlations.

    Returns:
        Diversification score (lower average correlation = higher diversification).
    """
    corrs = []
    symbols = list(correlation_matrix.keys())
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            corrs.append(abs(correlation_matrix[sym1].get(sym2, 0)))
    if not corrs:
        return 1.0  # No diversification info
    avg_corr = sum(corrs) / len(corrs)
    diversification_score = 1 - avg_corr
    return diversification_score
