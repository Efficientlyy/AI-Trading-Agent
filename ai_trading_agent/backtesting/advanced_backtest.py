"""
Advanced Backtesting Framework

This module provides a comprehensive backtesting framework for evaluating
the performance of the enhanced Technical Analysis Agent components.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ..agent.indicator_engine import IndicatorEngine
from ..common.utils import get_logger


class BacktestResult:
    """Container for backtesting results."""
    
    def __init__(self):
        """Initialize backtesting results container."""
        self.trades = []
        self.equity_curve = None
        self.performance_metrics = {}
        self.signals = []
        self.rejected_signals = []
        self.market_regimes = {}
        self.parameters_used = {}
    
    def add_trade(self, trade: Dict[str, Any]):
        """Add a trade to the results."""
        self.trades.append(trade)
        
    def add_signal(self, signal: Dict[str, Any], is_valid: bool = True):
        """Add a signal to the results."""
        if is_valid:
            self.signals.append(signal)
        else:
            self.rejected_signals.append(signal)
    
    def set_equity_curve(self, equity_curve: pd.DataFrame):
        """Set the equity curve for the backtest."""
        self.equity_curve = equity_curve
        
    def set_performance_metrics(self, metrics: Dict[str, Any]):
        """Set performance metrics."""
        self.performance_metrics = metrics
        
    def add_market_regime(self, date: datetime, regime: Dict[str, Any]):
        """Add market regime classification for a specific date."""
        self.market_regimes[date] = regime
        
    def add_parameters(self, date: datetime, parameters: Dict[str, Any]):
        """Add parameters used for a specific date."""
        self.parameters_used[date] = parameters
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the backtest results."""
        return {
            "total_trades": len(self.trades),
            "total_signals": len(self.signals),
            "rejected_signals": len(self.rejected_signals),
            "performance_metrics": self.performance_metrics,
            "market_regimes": {
                "trending": sum(1 for r in self.market_regimes.values() if r["regime"] == "trending"),
                "ranging": sum(1 for r in self.market_regimes.values() if r["regime"] == "ranging"),
                "volatile": sum(1 for r in self.market_regimes.values() if r["regime"] == "volatile"),
                "unknown": sum(1 for r in self.market_regimes.values() if r["regime"] == "unknown")
            }
        }
    
    def save(self, file_path: str) -> bool:
        """Save backtest results to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "trades": self.trades,
                "signals": self.signals,
                "rejected_signals": self.rejected_signals,
                "performance_metrics": self.performance_metrics,
                "market_regimes": {
                    k.isoformat() if isinstance(k, datetime) else str(k): v 
                    for k, v in self.market_regimes.items()
                },
                "parameters_used": {
                    k.isoformat() if isinstance(k, datetime) else str(k): v 
                    for k, v in self.parameters_used.items()
                }
            }
            
            # Save equity curve separately as CSV if it exists
            if self.equity_curve is not None:
                equity_csv_path = os.path.splitext(file_path)[0] + "_equity.csv"
                self.equity_curve.to_csv(equity_csv_path)
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            logger = get_logger("BacktestResult")
            logger.error(f"Error saving backtest results: {str(e)}")
            return False


class AdvancedBacktester:
    """
    Advanced backtesting framework for the Technical Analysis Agent.
    
    This class provides functionality for backtesting the advanced technical
    analysis components, including multi-timeframe analysis, ML signal validation,
    and adaptive parameter tuning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the backtester.
        
        Args:
            config: Configuration dictionary with backtesting parameters
                - initial_capital: Starting capital for the backtest
                - position_sizing: Method for sizing positions
                - slippage_model: Model for simulating execution slippage
                - commission_model: Model for simulating trading commissions
                - agent_config: Configuration for the AdvancedTechnicalAnalysisAgent
        """
        self.logger = get_logger("AdvancedBacktester")
        self.config = config or {}
        
        # Extract configuration
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.position_sizing = self.config.get("position_sizing", "fixed")
        self.position_size = self.config.get("position_size", 0.1)  # 10% of capital
        self.slippage_model = self.config.get("slippage_model", "fixed")
        self.slippage_amount = self.config.get("slippage_amount", 0.001)  # 0.1%
        self.commission_model = self.config.get("commission_model", "percentage")
        self.commission_amount = self.config.get("commission_amount", 0.001)  # 0.1%
        
        # Initialize components
        agent_config = self.config.get("agent_config", {})
        self.agent = AdvancedTechnicalAnalysisAgent(agent_config)
        
        # Initialize performance tracking
        self.results = BacktestResult()
        
        self.logger.info(
            f"Initialized Advanced Backtester with "
            f"initial_capital={self.initial_capital}, "
            f"position_sizing={self.position_sizing}, "
            f"slippage_model={self.slippage_model}, "
            f"commission_model={self.commission_model}"
        )
    
    def run_backtest(
        self, 
        market_data: Dict[str, Dict[str, pd.DataFrame]], 
        symbols: List[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run a backtest over the provided market data.
        
        Args:
            market_data: Dictionary mapping symbols to timeframe-specific market data
                Format: {symbol: {timeframe: DataFrame}}
            symbols: List of symbols to backtest, or None for all
            start_date: Start date for the backtest, or None for all data
            end_date: End date for the backtest, or None for all data
            
        Returns:
            BacktestResult with backtest performance data
        """
        self.logger.info(
            f"Starting backtest with {len(market_data)} symbols "
            f"from {start_date} to {end_date}"
        )
        
        # Use all symbols if none specified
        if symbols is None:
            symbols = list(market_data.keys())
            
        # Initialize portfolio and tracking
        portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "equity": self.initial_capital,
            "trades": [],
            "history": []
        }
        
        # Get primary timeframe for date iteration
        primary_timeframe = self.agent.timeframes[0] if self.agent.timeframes else "1d"
        
        # Prepare date range for iteration
        all_dates = set()
        for symbol in symbols:
            if symbol in market_data and primary_timeframe in market_data[symbol]:
                df = market_data[symbol][primary_timeframe]
                all_dates.update(df.index)
        
        date_range = sorted(all_dates)
        
        # Filter date range if specified
        if start_date:
            date_range = [d for d in date_range if d >= start_date]
        if end_date:
            date_range = [d for d in date_range if d <= end_date]
            
        if not date_range:
            self.logger.warning("No dates in range for backtest")
            return self.results
            
        self.logger.info(f"Backtesting {len(date_range)} days from {date_range[0]} to {date_range[-1]}")
        
        # Create a sliding window for analysis
        lookback = 100  # Use 100 days of data for analysis
        
        for i, current_date in enumerate(tqdm(date_range)):
            # Skip initial lookback period
            if i < lookback:
                continue
                
            # Prepare data window for this date
            window_data = {}
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                    
                window_data[symbol] = {}
                for tf, df in market_data[symbol].items():
                    # Filter data up to current date
                    window_df = df[df.index <= current_date].tail(lookback)
                    if not window_df.empty:
                        window_data[symbol][tf] = window_df
            
            # Get primary timeframe data for this date
            current_data = {}
            for symbol in symbols:
                if symbol in window_data and primary_timeframe in window_data[symbol]:
                    df = window_data[symbol][primary_timeframe]
                    current_row = df[df.index == current_date]
                    if not current_row.empty:
                        current_data[symbol] = current_row.iloc[0]
            
            # Skip if no data for this date
            if not current_data:
                continue
                
            # Get signals from the agent
            signals = self.agent.analyze(window_data, symbols)
            
            # Track rejected signals for analysis
            rejected_count = self.agent.metrics["signals_rejected"]
            generated_count = self.agent.metrics["signals_generated"]
            
            # Process each signal
            for signal in signals:
                symbol = signal["payload"]["symbol"]
                signal_value = signal["payload"]["signal"]
                price = signal["payload"]["price_at_signal"]
                
                # Add to results for analysis
                self.results.add_signal(signal)
                
                # Track market regime
                if "market_regime" in signal["payload"]:
                    self.results.add_market_regime(
                        current_date, 
                        signal["payload"]["market_regime"]
                    )
                
                # Skip if price is not available
                if price <= 0:
                    continue
                
                # Determine trade direction
                if signal_value > 0:  # Buy signal
                    # Check if we're already long
                    if symbol in portfolio["positions"] and portfolio["positions"][symbol]["quantity"] > 0:
                        continue
                        
                    # Calculate position size
                    position_value = self._calculate_position_size(portfolio, symbol, price)
                    
                    # Skip if no cash or too small position
                    if position_value <= 0 or portfolio["cash"] < position_value:
                        continue
                        
                    # Calculate quantity
                    quantity = position_value / price
                    
                    # Apply slippage to execution price
                    execution_price = self._apply_slippage(price, "buy")
                    
                    # Calculate commission
                    commission = self._calculate_commission(position_value)
                    
                    # Execute the trade
                    cost = execution_price * quantity + commission
                    
                    # Update portfolio
                    portfolio["cash"] -= cost
                    
                    if symbol in portfolio["positions"]:
                        # Update existing position
                        avg_price = (
                            portfolio["positions"][symbol]["avg_price"] * 
                            portfolio["positions"][symbol]["quantity"] +
                            execution_price * quantity
                        ) / (portfolio["positions"][symbol]["quantity"] + quantity)
                        
                        portfolio["positions"][symbol]["quantity"] += quantity
                        portfolio["positions"][symbol]["avg_price"] = avg_price
                    else:
                        # Create new position
                        portfolio["positions"][symbol] = {
                            "quantity": quantity,
                            "avg_price": execution_price,
                            "entry_date": current_date
                        }
                    
                    # Track the trade
                    trade = {
                        "symbol": symbol,
                        "direction": "buy",
                        "quantity": quantity,
                        "price": execution_price,
                        "commission": commission,
                        "date": current_date,
                        "signal": signal,
                        "position_value": position_value
                    }
                    
                    portfolio["trades"].append(trade)
                    self.results.add_trade(trade)
                    
                elif signal_value < 0:  # Sell signal
                    # Check if we have a position
                    if symbol not in portfolio["positions"] or portfolio["positions"][symbol]["quantity"] <= 0:
                        continue
                        
                    # Get position details
                    position = portfolio["positions"][symbol]
                    quantity = position["quantity"]
                    avg_price = position["avg_price"]
                    
                    # Apply slippage to execution price
                    execution_price = self._apply_slippage(price, "sell")
                    
                    # Calculate value and commission
                    position_value = execution_price * quantity
                    commission = self._calculate_commission(position_value)
                    
                    # Execute the trade
                    proceeds = position_value - commission
                    
                    # Update portfolio
                    portfolio["cash"] += proceeds
                    portfolio["positions"][symbol]["quantity"] = 0
                    
                    # Calculate profit/loss
                    pl = (execution_price - avg_price) * quantity - commission
                    pl_percent = (execution_price / avg_price - 1) * 100
                    
                    # Track the trade
                    trade = {
                        "symbol": symbol,
                        "direction": "sell",
                        "quantity": quantity,
                        "price": execution_price,
                        "commission": commission,
                        "date": current_date,
                        "signal": signal,
                        "pl": pl,
                        "pl_percent": pl_percent,
                        "position_value": position_value,
                        "holding_period": (current_date - position["entry_date"]).days
                    }
                    
                    portfolio["trades"].append(trade)
                    self.results.add_trade(trade)
            
            # Update portfolio value
            portfolio_value = portfolio["cash"]
            for symbol, position in portfolio["positions"].items():
                if position["quantity"] > 0 and symbol in current_data:
                    price = current_data[symbol]["close"]
                    portfolio_value += price * position["quantity"]
            
            # Track portfolio history
            history_entry = {
                "date": current_date,
                "cash": portfolio["cash"],
                "equity": portfolio_value,
                "positions": {
                    s: {
                        "quantity": p["quantity"],
                        "value": p["quantity"] * current_data[s]["close"] if s in current_data else 0
                    }
                    for s, p in portfolio["positions"].items() if p["quantity"] > 0
                }
            }
            
            portfolio["history"].append(history_entry)
            
            # Track parameters used
            for strategy_name, strategy in self.agent.strategies.items():
                if hasattr(strategy, 'config'):
                    self.results.add_parameters(current_date, {
                        strategy_name: strategy.config
                    })
        
        # Prepare equity curve
        equity_curve = pd.DataFrame([
            {"date": entry["date"], "equity": entry["equity"]}
            for entry in portfolio["history"]
        ])
        
        if not equity_curve.empty:
            equity_curve.set_index("date", inplace=True)
            self.results.set_equity_curve(equity_curve)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(equity_curve, portfolio["trades"])
            self.results.set_performance_metrics(metrics)
            
            self.logger.info(
                f"Backtest completed with {len(portfolio['trades'])} trades. "
                f"Final equity: {equity_curve['equity'].iloc[-1]:.2f}, "
                f"Return: {metrics['total_return']:.2f}%, "
                f"Sharpe: {metrics['sharpe_ratio']:.2f}"
            )
        else:
            self.logger.warning("No equity curve generated - no trades executed")
        
        return self.results
    
    def _calculate_position_size(
        self, 
        portfolio: Dict[str, Any], 
        symbol: str, 
        price: float
    ) -> float:
        """Calculate position size based on position sizing model."""
        if self.position_sizing == "fixed_percent":
            return portfolio["equity"] * self.position_size
            
        elif self.position_sizing == "fixed_dollar":
            return min(self.position_size, portfolio["cash"])
            
        elif self.position_sizing == "kelly":
            # Simple Kelly criterion implementation
            # In a real system, this would use historical win rate and reward/risk
            win_rate = 0.5  # Default assumption
            reward_risk = 1.5  # Default assumption
            
            # Get historical performance if available
            if portfolio["trades"]:
                # Calculate win rate
                wins = sum(1 for t in portfolio["trades"] if t.get("pl", 0) > 0)
                win_rate = wins / len(portfolio["trades"])
                
                # Calculate average reward/risk
                avg_win = np.mean([t["pl_percent"] for t in portfolio["trades"] if t.get("pl", 0) > 0]) if wins > 0 else 1
                avg_loss = abs(np.mean([t["pl_percent"] for t in portfolio["trades"] if t.get("pl", 0) < 0])) if len(portfolio["trades"]) - wins > 0 else 1
                reward_risk = avg_win / max(0.1, avg_loss)  # Avoid division by zero
            
            # Kelly formula: f = W - (1 - W) / R where W = win rate, R = reward/risk
            kelly_pct = max(0, min(1, win_rate - (1 - win_rate) / reward_risk))
            
            # Use half Kelly for safety
            half_kelly = kelly_pct * 0.5
            
            return portfolio["equity"] * half_kelly
            
        else:  # Default to fixed percent
            return portfolio["equity"] * self.position_size
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage to execution price."""
        if self.slippage_model == "fixed":
            # Fixed percentage slippage
            factor = (1 + self.slippage_amount) if direction == "buy" else (1 - self.slippage_amount)
            return price * factor
            
        elif self.slippage_model == "variable":
            # Variable slippage based on volatility (simplified)
            base_slippage = self.slippage_amount
            variable_component = np.random.uniform(0, base_slippage)
            total_slippage = base_slippage + variable_component
            
            factor = (1 + total_slippage) if direction == "buy" else (1 - total_slippage)
            return price * factor
            
        else:  # No slippage
            return price
    
    def _calculate_commission(self, position_value: float) -> float:
        """Calculate trading commission."""
        if self.commission_model == "percentage":
            return position_value * self.commission_amount
            
        elif self.commission_model == "fixed":
            return self.commission_amount
            
        else:  # No commission
            return 0.0
    
    def _calculate_performance_metrics(
        self, 
        equity_curve: pd.DataFrame, 
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate performance metrics from equity curve and trades."""
        if equity_curve.empty:
            return {}
            
        # Basic metrics
        start_equity = equity_curve["equity"].iloc[0]
        end_equity = equity_curve["equity"].iloc[-1]
        total_return = (end_equity / start_equity - 1) * 100
        
        # Calculate daily returns
        equity_curve["daily_return"] = equity_curve["equity"].pct_change()
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days / 365, 0.01)  # Avoid division by zero
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        # Volatility and risk metrics
        daily_volatility = equity_curve["daily_return"].std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = 0.0
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return / 100) / annualized_volatility
        
        # Drawdown analysis
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["peak"] - 1) * 100
        max_drawdown = abs(equity_curve["drawdown"].min())
        
        # Trade metrics
        if trades:
            # Win rate
            winning_trades = [t for t in trades if t.get("pl", 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Average profit/loss
            avg_profit = np.mean([t["pl"] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t["pl"] for t in trades if t.get("pl", 0) <= 0])) if len(trades) - len(winning_trades) > 0 else 1
            
            # Profit factor
            total_profit = sum(t["pl"] for t in winning_trades)
            total_loss = abs(sum(t["pl"] for t in trades if t.get("pl", 0) <= 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Average holding period
            avg_holding_period = np.mean([t.get("holding_period", 0) for t in trades if "holding_period" in t])
            
            trade_metrics = {
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(trades) - len(winning_trades),
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_holding_period": avg_holding_period
            }
        else:
            trade_metrics = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "avg_holding_period": 0
            }
        
        # Combine all metrics
        return {
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility * 100),  # Convert to percentage
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "days": int(days),
            **trade_metrics
        }
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot the equity curve from backtest results."""
        if self.results.equity_curve is None or self.results.equity_curve.empty:
            self.logger.warning("No equity curve available to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.results.equity_curve.index, self.results.equity_curve["equity"])
        plt.title("Backtest Equity Curve")
        plt.ylabel("Equity")
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        plt.fill_between(
            self.results.equity_curve.index, 
            self.results.equity_curve["drawdown"], 
            0, 
            where=self.results.equity_curve["drawdown"] < 0,
            color="red", 
            alpha=0.3
        )
        plt.title("Drawdowns")
        plt.ylabel("Drawdown %")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved equity curve plot to {save_path}")
        
        plt.show()
    
    def plot_trade_analysis(self, save_path: Optional[str] = None):
        """Plot trade analysis charts."""
        if not self.results.trades:
            self.logger.warning("No trades available to analyze")
            return
            
        plt.figure(figsize=(12, 10))
        
        # Extract trade data
        trade_dates = [t["date"] for t in self.results.trades if "date" in t]
        pnl = [t.get("pl", 0) for t in self.results.trades]
        pnl_percent = [t.get("pl_percent", 0) for t in self.results.trades]
        holding_periods = [t.get("holding_period", 0) for t in self.results.trades if "holding_period" in t]
        
        # Plot trade P&L
        plt.subplot(2, 2, 1)
        plt.bar(range(len(pnl)), pnl, color=["green" if p > 0 else "red" for p in pnl])
        plt.title("Trade P&L")
        plt.ylabel("Profit/Loss")
        plt.xlabel("Trade Number")
        plt.grid(True)
        
        # Plot P&L distribution
        plt.subplot(2, 2, 2)
        plt.hist(pnl_percent, bins=20, color="blue", alpha=0.7)
        plt.title("P&L Distribution (%)")
        plt.ylabel("Frequency")
        plt.xlabel("P&L %")
        plt.grid(True)
        
        # Plot holding period distribution
        plt.subplot(2, 2, 3)
        plt.hist(holding_periods, bins=20, color="purple", alpha=0.7)
        plt.title("Holding Period Distribution (Days)")
        plt.ylabel("Frequency")
        plt.xlabel("Holding Period")
        plt.grid(True)
        
        # Plot cumulative P&L
        plt.subplot(2, 2, 4)
        plt.plot(np.cumsum(pnl))
        plt.title("Cumulative P&L")
        plt.ylabel("Cumulative Profit/Loss")
        plt.xlabel("Trade Number")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved trade analysis plot to {save_path}")
            
        plt.show()
    
    def save_results(self, directory: str) -> bool:
        """
        Save backtest results to a directory.
        
        Args:
            directory: Directory to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save main results
            results_path = os.path.join(directory, "backtest_results.json")
            self.results.save(results_path)
            
            # Save plots
            equity_plot_path = os.path.join(directory, "equity_curve.png")
            self.plot_equity_curve(equity_plot_path)
            
            trade_plot_path = os.path.join(directory, "trade_analysis.png")
            self.plot_trade_analysis(trade_plot_path)
            
            # Save agent state
            agent_state_dir = os.path.join(directory, "agent_state")
            self.agent.save_state(agent_state_dir)
            
            self.logger.info(f"Saved backtest results to {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
            return False
