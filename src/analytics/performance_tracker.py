"""Performance analytics module for trading strategies.

This module provides functionality for tracking and analyzing trading performance,
including trade statistics, equity curves, and performance metrics.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TradeStats:
    """Statistics for a single trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    return_pct: float
    holding_period: timedelta
    max_drawdown: float
    max_runup: float
    risk_reward_ratio: float
    strategy_name: str

@dataclass
class PerformanceMetrics:
    """Overall strategy performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winner: float
    largest_loser: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_holding_period: timedelta
    total_pnl: float
    return_pct: float

class PerformanceTracker:
    """Track and analyze trading performance."""
    
    def __init__(self, initial_capital: float):
        """Initialize performance tracker.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.trades: List[TradeStats] = []
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.drawdown_curve: List[Tuple[datetime, float]] = [(datetime.now(), 0.0)]
        
        # Performance metrics
        self.metrics = PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_return=0.0,
            avg_winning_trade=0.0,
            avg_losing_trade=0.0,
            largest_winner=0.0,
            largest_loser=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            avg_holding_period=timedelta(),
            total_pnl=0.0,
            return_pct=0.0
        )
        
        # Strategy-specific metrics
        self.strategy_metrics: Dict[str, PerformanceMetrics] = {}
    
    def add_trade(self, trade: TradeStats) -> None:
        """Add a completed trade to the performance history.
        
        Args:
            trade: Trade statistics
        """
        self.trades.append(trade)
        self.current_capital += trade.pnl
        
        # Update equity curve
        self.equity_curve.append((trade.exit_time, self.current_capital))
        
        # Update drawdown curve
        peak_capital = max(capital for _, capital in self.equity_curve)
        current_drawdown = (peak_capital - self.current_capital) / peak_capital
        self.drawdown_curve.append((trade.exit_time, current_drawdown))
        
        # Update performance metrics
        self._update_metrics()
        
        # Update strategy-specific metrics
        if trade.strategy_name not in self.strategy_metrics:
            self.strategy_metrics[trade.strategy_name] = PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_return=0.0,
                avg_winning_trade=0.0,
                avg_losing_trade=0.0,
                largest_winner=0.0,
                largest_loser=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                avg_holding_period=timedelta(),
                total_pnl=0.0,
                return_pct=0.0
            )
        self._update_strategy_metrics(trade.strategy_name)
    
    def _update_metrics(self) -> None:
        """Update overall performance metrics."""
        if not self.trades:
            return
        
        # Basic trade statistics
        self.metrics.total_trades = len(self.trades)
        self.metrics.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        self.metrics.losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        
        # Win rate and returns
        self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        self.metrics.total_pnl = sum(t.pnl for t in self.trades)
        self.metrics.return_pct = self.metrics.total_pnl / self.initial_capital * 100
        
        # Average trade metrics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        self.metrics.avg_winning_trade = (
            sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades else 0.0
        )
        self.metrics.avg_losing_trade = (
            sum(t.pnl for t in losing_trades) / len(losing_trades)
            if losing_trades else 0.0
        )
        
        # Largest trades
        self.metrics.largest_winner = max((t.pnl for t in self.trades), default=0.0)
        self.metrics.largest_loser = min((t.pnl for t in self.trades), default=0.0)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        self.metrics.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )
        
        # Average holding period
        total_holding_time = sum(
            (t.exit_time - t.entry_time for t in self.trades),
            timedelta()
        )
        self.metrics.avg_holding_period = (
            total_holding_time / self.metrics.total_trades
        )
        
        # Calculate returns for ratio metrics
        returns = [t.return_pct for t in self.trades]
        
        # Sharpe ratio (assuming daily returns)
        excess_returns = np.mean(returns) - 0.02/252  # Assuming 2% risk-free rate
        self.metrics.sharpe_ratio = (
            float(excess_returns / (np.std(returns) + 1e-10) * np.sqrt(252))
        )
        
        # Sortino ratio (penalizes only downside volatility)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0
        self.metrics.sortino_ratio = (
            float(excess_returns / (downside_std + 1e-10) * np.sqrt(252))
        )
        
        # Calmar ratio (return / max drawdown)
        self.metrics.max_drawdown = max((dd for _, dd in self.drawdown_curve), default=0.0)
        self.metrics.calmar_ratio = (
            self.metrics.return_pct / (self.metrics.max_drawdown * 100)
            if self.metrics.max_drawdown > 0 else 0.0
        )
    
    def _update_strategy_metrics(self, strategy_name: str) -> None:
        """Update performance metrics for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
        """
        strategy_trades = [t for t in self.trades if t.strategy_name == strategy_name]
        if not strategy_trades:
            return
        
        metrics = self.strategy_metrics[strategy_name]
        
        # Basic trade statistics
        metrics.total_trades = len(strategy_trades)
        metrics.winning_trades = sum(1 for t in strategy_trades if t.pnl > 0)
        metrics.losing_trades = sum(1 for t in strategy_trades if t.pnl < 0)
        
        # Win rate and returns
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        metrics.total_pnl = sum(t.pnl for t in strategy_trades)
        metrics.return_pct = metrics.total_pnl / self.initial_capital * 100
        
        # Average trade metrics
        winning_trades = [t for t in strategy_trades if t.pnl > 0]
        losing_trades = [t for t in strategy_trades if t.pnl < 0]
        
        metrics.avg_winning_trade = (
            sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades else 0.0
        )
        metrics.avg_losing_trade = (
            sum(t.pnl for t in losing_trades) / len(losing_trades)
            if losing_trades else 0.0
        )
        
        # Largest trades
        metrics.largest_winner = max((t.pnl for t in strategy_trades), default=0.0)
        metrics.largest_loser = min((t.pnl for t in strategy_trades), default=0.0)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        metrics.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )
        
        # Average holding period
        total_holding_time = sum(
            (t.exit_time - t.entry_time for t in strategy_trades),
            timedelta()
        )
        metrics.avg_holding_period = total_holding_time / metrics.total_trades
        
        # Calculate returns for ratio metrics
        returns = [t.return_pct for t in strategy_trades]
        
        # Sharpe ratio (assuming daily returns)
        excess_returns = np.mean(returns) - 0.02/252  # Assuming 2% risk-free rate
        metrics.sharpe_ratio = (
            float(excess_returns / (np.std(returns) + 1e-10) * np.sqrt(252))
        )
        
        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0
        metrics.sortino_ratio = (
            float(excess_returns / (downside_std + 1e-10) * np.sqrt(252))
        )
        
        # Calmar ratio
        strategy_drawdowns = [
            dd for t, dd in self.drawdown_curve
            if any(st.entry_time <= t <= st.exit_time for st in strategy_trades)
        ]
        metrics.max_drawdown = max(strategy_drawdowns, default=0.0)
        metrics.calmar_ratio = (
            metrics.return_pct / (metrics.max_drawdown * 100)
            if metrics.max_drawdown > 0 else 0.0
        )
    
    def get_metrics(self, strategy_name: Optional[str] = None) -> PerformanceMetrics:
        """Get performance metrics.
        
        Args:
            strategy_name: Optional strategy name to get specific metrics
            
        Returns:
            Performance metrics
        """
        if strategy_name and strategy_name in self.strategy_metrics:
            return self.strategy_metrics[strategy_name]
        return self.metrics
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as a DataFrame.
        
        Returns:
            DataFrame with timestamp and equity value
        """
        return pd.DataFrame(
            self.equity_curve,
            columns=['timestamp', 'equity']
        ).set_index('timestamp')
    
    def get_drawdown_curve(self) -> pd.DataFrame:
        """Get drawdown curve as a DataFrame.
        
        Returns:
            DataFrame with timestamp and drawdown percentage
        """
        return pd.DataFrame(
            self.drawdown_curve,
            columns=['timestamp', 'drawdown']
        ).set_index('timestamp')
    
    def get_trade_history(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Get trade history as a DataFrame.
        
        Args:
            strategy_name: Optional strategy name to filter trades
            
        Returns:
            DataFrame with trade details
        """
        trades = (
            [t for t in self.trades if t.strategy_name == strategy_name]
            if strategy_name else self.trades
        )
        
        return pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size': t.position_size,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'holding_period': t.holding_period,
                'max_drawdown': t.max_drawdown,
                'max_runup': t.max_runup,
                'risk_reward_ratio': t.risk_reward_ratio,
                'strategy': t.strategy_name
            }
            for t in trades
        ]) 