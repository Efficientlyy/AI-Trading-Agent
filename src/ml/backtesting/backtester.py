"""Backtesting framework for enhanced price prediction strategies.

This module provides a comprehensive backtesting framework that simulates
trading with realistic conditions including transaction costs, slippage,
and market impact.
"""

from typing import Dict, List, Optional, TypedDict, Union
from typing_extensions import NotRequired
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TransactionCosts:
    """Transaction cost model parameters."""
    commission_rate: float = 0.001  # 0.1% commission
    slippage_factor: float = 0.0001  # 1 basis point slippage
    market_impact: float = 0.00005  # 0.5 basis point market impact per $1M

@dataclass
class Position:
    """Trading position information."""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradeResult(TypedDict):
    """Result of a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    costs: float
    holding_period: float
    type: str  # 'long' or 'short'
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'

class BacktestResult(TypedDict):
    """Complete backtest results."""
    trades: List[TradeResult]
    equity_curve: NDArray[np.float64]
    returns: NDArray[np.float64]
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    avg_trade: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_consecutive_losses: int
    total_costs: float
    total_slippage: float
    total_market_impact: float

class BacktestMetrics(TypedDict):
    """Real-time backtest metrics."""
    current_drawdown: float
    running_sharpe: float
    win_rate: float
    avg_trade_pnl: float
    total_pnl: float
    num_trades: int
    equity: float
    costs: float

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_costs: Optional[TransactionCosts] = None,
        risk_free_rate: float = 0.02,
        max_position_size: float = 0.1,  # 10% of capital
        max_leverage: float = 1.0
    ):
        """Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_costs: Transaction cost model parameters
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            max_position_size: Maximum position size as fraction of capital
            max_leverage: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        
        # State variables
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeResult] = []
        self.equity_curve: List[float] = [initial_capital]
        self.metrics = self._init_metrics()
    
    def _init_metrics(self) -> BacktestMetrics:
        """Initialize backtest metrics."""
        return {
            "current_drawdown": 0.0,
            "running_sharpe": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "total_pnl": 0.0,
            "num_trades": 0,
            "equity": self.initial_capital,
            "costs": 0.0
        }
    
    def _calculate_transaction_costs(
        self,
        price: float,
        size: float,
        volume: float
    ) -> float:
        """Calculate total transaction costs including commission, slippage, and market impact.
        
        Args:
            price: Asset price
            size: Position size in units
            volume: Trading volume for market impact calculation
        
        Returns:
            Total transaction costs
        """
        notional = abs(price * size)
        commission = notional * self.transaction_costs.commission_rate
        slippage = notional * self.transaction_costs.slippage_factor
        market_impact = (notional * self.transaction_costs.market_impact * 
                        (notional / 1_000_000))  # Scale with position size
        return commission + slippage + market_impact
    
    def _calculate_position_size(
        self,
        price: float,
        volatility: float,
        confidence: float
    ) -> float:
        """Calculate position size based on volatility and model confidence.
        
        Args:
            price: Current asset price
            volatility: Asset volatility (annualized)
            confidence: Model confidence score (0-1)
        
        Returns:
            Position size in units
        """
        # Kelly criterion with safety fraction
        kelly_fraction = confidence - (1 - confidence)
        safety_fraction = min(kelly_fraction * 0.5, self.max_position_size)
        
        # Scale by volatility
        vol_scalar = 0.2 / max(volatility, 0.05)  # Target 20% annualized vol
        position_value = self.capital * safety_fraction * vol_scalar
        
        return position_value / price
    
    def _update_metrics(self, trade: TradeResult) -> None:
        """Update real-time backtest metrics.
        
        Args:
            trade: Completed trade result
        """
        self.metrics["num_trades"] += 1
        self.metrics["total_pnl"] += trade["pnl"]
        self.metrics["avg_trade_pnl"] = (
            self.metrics["total_pnl"] / self.metrics["num_trades"]
        )
        
        wins = sum(t["pnl"] > 0 for t in self.trades)
        self.metrics["win_rate"] = wins / len(self.trades)
        
        # Update equity and drawdown
        self.metrics["equity"] = self.capital
        peak = max(self.equity_curve)
        self.metrics["current_drawdown"] = (peak - self.capital) / peak
        
        # Update costs
        self.metrics["costs"] += trade["costs"]
        
        # Update Sharpe ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1:
            excess_returns = returns - self.risk_free_rate / 252  # Daily
            self.metrics["running_sharpe"] = (
                np.mean(excess_returns) / np.std(excess_returns, ddof=1)
                * np.sqrt(252)  # Annualize
            )
    
    def execute_trade(
        self,
        symbol: str,
        direction: int,  # 1 for long, -1 for short
        price: float,
        timestamp: datetime,
        volume: float,
        volatility: float,
        confidence: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[TradeResult]:
        """Execute a trade with position sizing and risk management.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (1=long, -1=short)
            price: Current price
            timestamp: Trade timestamp
            volume: Trading volume
            volatility: Asset volatility
            confidence: Model confidence
            stop_loss: Optional stop-loss price
            take_profit: Optional take-profit price
        
        Returns:
            Trade result if a position was closed
        """
        # Check if we have an existing position to close
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate PnL
            exit_price = price
            pnl = (exit_price - position.entry_price) * position.size
            if position.size < 0:  # Short position
                pnl = -pnl
            
            # Calculate costs
            costs = self._calculate_transaction_costs(
                exit_price,
                position.size,
                volume
            )
            
            # Update capital
            self.capital += pnl - costs
            
            # Record trade
            trade: TradeResult = {
                "symbol": symbol,
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "return_pct": pnl / (abs(position.size * position.entry_price)),
                "costs": costs,
                "holding_period": (timestamp - position.entry_time).days,
                "type": "long" if position.size > 0 else "short",
                "exit_reason": "signal"
            }
            
            self.trades.append(trade)
            self.equity_curve.append(self.capital)
            self._update_metrics(trade)
            
            # Clear position
            del self.positions[symbol]
            
            # Return trade result
            return trade
        
        # Open new position if signal is non-zero
        elif direction != 0:
            # Calculate position size
            size = self._calculate_position_size(price, volatility, confidence)
            if direction < 0:
                size = -size
            
            # Check leverage constraints
            total_exposure = sum(
                abs(p.size * price) for p in self.positions.values()
            )
            if (total_exposure + abs(size * price)) / self.capital > self.max_leverage:
                return None
            
            # Calculate entry costs
            costs = self._calculate_transaction_costs(price, size, volume)
            
            # Check if we have enough capital
            if costs > self.capital * 0.001:  # Max 0.1% per trade on costs
                return None
            
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                size=size,
                entry_price=price,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Update capital and metrics
            self.capital -= costs
            self.metrics["costs"] += costs
            self.equity_curve.append(self.capital)
        
        return None
    
    def check_stops(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        volume: float
    ) -> Optional[TradeResult]:
        """Check and execute stop-loss and take-profit orders.
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Current timestamp
            volume: Trading volume
        
        Returns:
            Trade result if a position was closed
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Check stop-loss
        if (position.stop_loss is not None and
            ((position.size > 0 and price <= position.stop_loss) or
             (position.size < 0 and price >= position.stop_loss))):
            
            # Calculate PnL
            pnl = (price - position.entry_price) * position.size
            if position.size < 0:
                pnl = -pnl
            
            # Calculate costs
            costs = self._calculate_transaction_costs(price, position.size, volume)
            
            # Update capital
            self.capital += pnl - costs
            
            # Record trade
            trade: TradeResult = {
                "symbol": symbol,
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "entry_price": position.entry_price,
                "exit_price": price,
                "size": position.size,
                "pnl": pnl,
                "return_pct": pnl / (abs(position.size * position.entry_price)),
                "costs": costs,
                "holding_period": (timestamp - position.entry_time).days,
                "type": "long" if position.size > 0 else "short",
                "exit_reason": "stop_loss"
            }
            
            self.trades.append(trade)
            self.equity_curve.append(self.capital)
            self._update_metrics(trade)
            
            # Clear position
            del self.positions[symbol]
            
            return trade
        
        # Check take-profit
        if (position.take_profit is not None and
            ((position.size > 0 and price >= position.take_profit) or
             (position.size < 0 and price <= position.take_profit))):
            
            # Calculate PnL
            pnl = (price - position.entry_price) * position.size
            if position.size < 0:
                pnl = -pnl
            
            # Calculate costs
            costs = self._calculate_transaction_costs(price, position.size, volume)
            
            # Update capital
            self.capital += pnl - costs
            
            # Record trade
            trade: TradeResult = {
                "symbol": symbol,
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "entry_price": position.entry_price,
                "exit_price": price,
                "size": position.size,
                "pnl": pnl,
                "return_pct": pnl / (abs(position.size * position.entry_price)),
                "costs": costs,
                "holding_period": (timestamp - position.entry_time).days,
                "type": "long" if position.size > 0 else "short",
                "exit_reason": "take_profit"
            }
            
            self.trades.append(trade)
            self.equity_curve.append(self.capital)
            self._update_metrics(trade)
            
            # Clear position
            del self.positions[symbol]
            
            return trade
        
        return None
    
    def get_results(self) -> BacktestResult:
        """Calculate and return final backtest results.
        
        Returns:
            Complete backtest results and statistics
        """
        if not self.trades:
            raise ValueError("No trades executed in backtest")
        
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate metrics
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_costs = sum(t["costs"] for t in self.trades)
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] <= 0]
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate consecutive losses
        pnl_series = [t["pnl"] for t in self.trades]
        max_consecutive_losses = 0
        current_consecutive = 0
        for pnl in pnl_series:
            if pnl <= 0:
                current_consecutive += 1
                max_consecutive_losses = max(
                    max_consecutive_losses,
                    current_consecutive
                )
            else:
                current_consecutive = 0
        
        # Prepare results
        results: BacktestResult = {
            "trades": self.trades,
            "equity_curve": equity_curve,
            "returns": returns,
            "sharpe_ratio": self.metrics["running_sharpe"],
            "max_drawdown": max_drawdown,
            "profit_factor": (
                abs(sum(t["pnl"] for t in winning_trades)) /
                abs(sum(t["pnl"] for t in losing_trades))
                if losing_trades else float('inf')
            ),
            "win_rate": len(winning_trades) / len(self.trades),
            "avg_trade": total_pnl / len(self.trades),
            "avg_winning_trade": (
                sum(t["pnl"] for t in winning_trades) / len(winning_trades)
                if winning_trades else 0
            ),
            "avg_losing_trade": (
                sum(t["pnl"] for t in losing_trades) / len(losing_trades)
                if losing_trades else 0
            ),
            "max_consecutive_losses": max_consecutive_losses,
            "total_costs": total_costs,
            "total_slippage": sum(
                t["costs"] * self.transaction_costs.slippage_factor
                for t in self.trades
            ),
            "total_market_impact": sum(
                t["costs"] * self.transaction_costs.market_impact
                for t in self.trades
            )
        }
        
        return results 