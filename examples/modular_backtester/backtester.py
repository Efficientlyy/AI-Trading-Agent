"""
Backtesting framework.

This module provides classes for backtesting trading strategies with historical data,
measuring performance, and optimizing parameters.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from .models import Signal, SignalType, CandleData, Position, TimeFrame
from .strategies import Strategy

# Set up logging
logger = logging.getLogger("backtester")


class BacktestMetrics:
    """
    Class to calculate and store performance metrics for a backtest.
    
    Metrics include:
    - Total profit/loss
    - Win rate
    - Max drawdown
    - Sharpe ratio
    - Profit factor
    """
    
    def __init__(self):
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.volatility = 0.0
        self.avg_trade_duration = 0.0
        self.avg_win_pnl = 0.0
        self.avg_loss_pnl = 0.0
        self.max_consecutive_losses = 0
        self.trades_history = []
        self.equity_curve = []
    
    def calculate_from_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Calculate metrics from a list of completed trades."""
        if not trades:
            return
        
        self.trades_history = trades
        self.total_trades = len(trades)
        
        # Initialize tracking variables
        gross_profit = 0.0
        gross_loss = 0.0
        win_pnl_sum = 0.0
        loss_pnl_sum = 0.0
        durations = []
        equity = [0.0]  # Start with 0
        peak_equity = 0.0
        consecutive_losses = 0
        max_consecutive_losses = 0
        returns = []
        
        for trade in trades:
            pnl = trade.get('pnl', 0.0)
            self.total_pnl += pnl
            equity.append(equity[-1] + pnl)
            
            # Track returns for Sharpe ratio
            if equity[-2] != 0:
                returns.append(pnl / abs(equity[-2]) if abs(equity[-2]) > 0 else 0)
            
            # Update peak equity and calculate drawdown
            if equity[-1] > peak_equity:
                peak_equity = equity[-1]
            else:
                drawdown = peak_equity - equity[-1]
                drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0
                self.max_drawdown = max(self.max_drawdown, drawdown)
                self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown_pct)
            
            # Win/loss stats
            if pnl > 0:
                self.winning_trades += 1
                gross_profit += pnl
                win_pnl_sum += pnl
                consecutive_losses = 0
            elif pnl < 0:
                self.losing_trades += 1
                gross_loss += abs(pnl)
                loss_pnl_sum += abs(pnl)
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Calculate duration if timestamps available
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                
                duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        # Store the equity curve
        self.equity_curve = equity[1:]  # Remove the initial 0
        
        # Calculate final metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        self.avg_win_pnl = win_pnl_sum / self.winning_trades if self.winning_trades > 0 else 0.0
        self.avg_loss_pnl = loss_pnl_sum / self.losing_trades if self.losing_trades > 0 else 0.0
        self.avg_trade_duration = sum(durations) / len(durations) if durations else 0.0
        self.max_consecutive_losses = max_consecutive_losses
        
        # Calculate Sharpe ratio (annualized)
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1.0
            self.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            self.volatility = std_return * np.sqrt(252)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            'total_pnl': round(self.total_pnl, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.winning_trades / self.total_trades * 100, 2) if self.total_trades > 0 else 0.0,
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'volatility': round(self.volatility * 100, 2),
            'avg_trade_duration': round(self.avg_trade_duration, 2),
            'avg_win_pnl': round(self.avg_win_pnl, 2),
            'avg_loss_pnl': round(self.avg_loss_pnl, 2),
            'max_consecutive_losses': self.max_consecutive_losses
        }
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """Plot the equity curve for the backtest."""
        if not self.equity_curve:
            logger.warning("No equity curve data available to plot")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Equity Curve')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add drawdown shading
        equity_array = np.array(self.equity_curve)
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = max_equity - equity_array
        drawdown_pct = drawdown / max_equity
        drawdown_pct[np.isnan(drawdown_pct)] = 0
        
        plt.fill_between(range(len(equity_array)), equity_array, max_equity, 
                         where=drawdown>0, color='red', alpha=0.3, label='Drawdown')
        
        plt.title('Backtest Equity Curve')
        plt.xlabel('Trades')
        plt.ylabel('Cumulative Profit/Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_trade_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot the distribution of trade outcomes."""
        if not self.trades_history:
            logger.warning("No trade history available to plot")
            return
        
        pnl_values = [trade.get('pnl', 0.0) for trade in self.trades_history]
        
        plt.figure(figsize=(12, 6))
        plt.hist(pnl_values, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        
        plt.title('Trade P&L Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Trade distribution saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class StrategyBacktester:
    """
    Backtester for evaluating trading strategies.
    
    Features:
    - Process historical data and generate trade signals
    - Track positions and calculate P&L
    - Generate performance metrics and reports
    - Risk management and position sizing
    - Parameter optimization
    """
    
    def __init__(self, initial_capital: float = 10000.0, position_size: float = 1.0, 
                 use_stop_loss: bool = False, stop_loss_pct: float = 0.05,
                 use_take_profit: bool = False, take_profit_pct: float = 0.1,
                 enable_trailing_stop: bool = False, trailing_stop_pct: float = 0.03,
                 max_open_positions: int = 1, commission_pct: float = 0.001):
        """
        Initialize the backtester with configuration parameters.
        
        Args:
            initial_capital: Starting capital for the backtest
            position_size: Default position size as percentage of capital (0.0-1.0)
            use_stop_loss: Whether to use stop loss orders
            stop_loss_pct: Stop loss percentage from entry price
            use_take_profit: Whether to use take profit orders
            take_profit_pct: Take profit percentage from entry price
            enable_trailing_stop: Whether to use trailing stops
            trailing_stop_pct: Trailing stop percentage
            max_open_positions: Maximum number of positions to hold simultaneously
            commission_pct: Commission percentage for trades
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.use_take_profit = use_take_profit
        self.take_profit_pct = take_profit_pct
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.max_open_positions = max_open_positions
        self.commission_pct = commission_pct
        
        self.open_positions: Dict[str, Position] = {}  # Symbol -> Position
        self.closed_positions: List[Dict[str, Any]] = []
        self.historical_data: Dict[str, List[CandleData]] = {}  # Symbol -> candles
        self.strategy: Optional[Strategy] = None
        self.metrics = BacktestMetrics()
        
        # Capital and equity tracking
        self.daily_balance: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        
        # Parameter optimization results
        self.optimization_results = []
    
    def set_strategy(self, strategy: Strategy) -> None:
        """Set the strategy to be used for backtesting."""
        self.strategy = strategy
    
    def add_historical_data(self, symbol: str, candles: List[CandleData]) -> None:
        """
        Add historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            candles: List of CandleData objects in chronological order
        """
        # Sort candles by timestamp to ensure chronological order
        sorted_candles = sorted(candles, key=lambda x: x.timestamp)
        self.historical_data[symbol] = sorted_candles
    
    def _apply_position_sizing(self, signal: Signal, candle: CandleData) -> float:
        """
        Calculate position size based on volatility and risk settings.
        
        Returns the number of units to trade.
        """
        # Default fixed position sizing
        base_size = self.current_capital * self.position_size
        
        # Check if signal has volatility-based position sizing information
        vol_based_size = signal.metadata.get('recommended_size', None)
        if vol_based_size is not None:
            # Scale by volatility but don't exceed max position size
            adjusted_size = min(self.position_size, vol_based_size) * self.current_capital
            return adjusted_size / candle.close if candle.close > 0 else 0
        
        # Fall back to default position sizing
        return base_size / candle.close if candle.close > 0 else 0
    
    def _check_stop_orders(self, candle: CandleData) -> None:
        """Check and execute stop loss and take profit orders."""
        symbols_to_close = []
        
        for symbol, position in self.open_positions.items():
            if symbol != candle.symbol:
                continue  # Skip positions for other symbols
            
            updated = False
            
            # Check for stop loss
            if self.use_stop_loss and position.stop_price is not None and candle.low <= position.stop_price:
                # Stop loss triggered - close at stop price
                position.close_price = position.stop_price
                position.exit_time = candle.timestamp
                symbols_to_close.append(symbol)
                updated = True
                logger.info(f"Stop loss triggered for {symbol} at {position.stop_price}")
            
            # Check for take profit
            elif self.use_take_profit and position.target_price is not None and candle.high >= position.target_price:
                # Take profit triggered - close at target price
                position.close_price = position.target_price
                position.exit_time = candle.timestamp
                symbols_to_close.append(symbol)
                updated = True
                logger.info(f"Take profit triggered for {symbol} at {position.target_price}")
            
            # Check for trailing stop if not already updated
            elif not updated and self.enable_trailing_stop and position.is_long:
                # Update trailing stop for long positions
                new_stop = candle.close * (1 - self.trailing_stop_pct)
                if position.stop_price is None or new_stop > position.stop_price:
                    position.stop_price = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} to {new_stop}")
            elif not updated and self.enable_trailing_stop and not position.is_long:
                # Update trailing stop for short positions
                new_stop = candle.close * (1 + self.trailing_stop_pct)
                if position.stop_price is None or new_stop < position.stop_price:
                    position.stop_price = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} to {new_stop}")
        
        # Close positions outside the loop to avoid modifying during iteration
        for symbol in symbols_to_close:
            self._close_position(symbol)
    
    def _open_position(self, signal: Signal, candle: CandleData) -> None:
        """Open a new position based on a signal."""
        symbol = signal.symbol
        
        # Check if we can open a new position (max positions limit)
        if len(self.open_positions) >= self.max_open_positions:
            logger.info(f"Max positions limit reached, skipping signal for {symbol}")
            return
        
        # Skip if we already have an open position for this symbol
        if symbol in self.open_positions:
            logger.info(f"Already have an open position for {symbol}, skipping signal")
            return
        
        # Determine position direction
        is_long = signal.signal_type == SignalType.BUY
        
        # Calculate position size
        size = self._apply_position_sizing(signal, candle)
        
        # Apply commission to entry
        commission = candle.close * size * self.commission_pct
        
        # Calculate stop loss and take profit prices
        stop_price = None
        target_price = None
        
        if self.use_stop_loss:
            if is_long:
                stop_price = candle.close * (1 - self.stop_loss_pct)
            else:
                stop_price = candle.close * (1 + self.stop_loss_pct)
        
        if self.use_take_profit:
            if is_long:
                target_price = candle.close * (1 + self.take_profit_pct)
            else:
                target_price = candle.close * (1 - self.take_profit_pct)
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_price=candle.close,
            entry_time=candle.timestamp,
            size=size,
            is_long=is_long,
            stop_price=stop_price,
            target_price=target_price,
            metadata={
                'signal_confidence': signal.confidence,
                'entry_commission': commission
            }
        )
        
        # Store the position
        self.open_positions[symbol] = position
        
        logger.info(f"{'Long' if is_long else 'Short'} position opened for {symbol} at {candle.close} "
                   f"with size {size:.4f}, stop: {stop_price}, target: {target_price}")
    
    def _close_position(self, symbol: str, close_price: Optional[float] = None, timestamp: Optional[datetime] = None) -> None:
        """Close an open position and record the result."""
        if symbol not in self.open_positions:
            logger.warning(f"Attempted to close non-existent position for {symbol}")
            return
        
        position = self.open_positions.pop(symbol)
        
        # Use provided close price and timestamp or existing values
        if close_price is not None:
            position.close_price = close_price
        
        if timestamp is not None:
            position.exit_time = timestamp
        
        # Apply commission to exit
        commission = position.close_price * position.size * self.commission_pct
        total_commission = commission + position.metadata.get('entry_commission', 0)
        
        # Calculate P&L
        if position.is_long:
            pnl = (position.close_price - position.entry_price) * position.size - total_commission
        else:
            pnl = (position.entry_price - position.close_price) * position.size - total_commission
        
        # Update capital
        self.current_capital += pnl
        
        # Record closed position
        closed_position = {
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'close_price': position.close_price,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'is_long': position.is_long,
            'size': position.size,
            'pnl': pnl,
            'commission': total_commission,
            'duration_hours': (position.exit_time - position.entry_time).total_seconds() / 3600 if position.exit_time and position.entry_time else 0
        }
        
        self.closed_positions.append(closed_position)
        
        logger.info(f"Position closed for {symbol} at {position.close_price} with P&L: ${pnl:.2f}")
        
        # Update daily balance if this is a new day
        if not self.daily_balance or self.daily_balance[-1][0].date() != position.exit_time.date():
            self.daily_balance.append((position.exit_time, self.current_capital))
    
    def run_backtest(self) -> BacktestMetrics:
        """
        Run the backtest with current settings.
        
        Returns:
            BacktestMetrics: Performance metrics for the backtest
        """
        if not self.strategy:
            raise ValueError("No strategy set for backtest")
        
        if not self.historical_data:
            raise ValueError("No historical data provided for backtest")
        
        # Reset backtest state
        self.open_positions = {}
        self.closed_positions = []
        self.current_capital = self.initial_capital
        self.daily_balance = [(datetime.now(), self.initial_capital)]
        
        # Process each symbol's data
        for symbol, candles in self.historical_data.items():
            for candle in candles:
                # First check stop orders for existing positions
                self._check_stop_orders(candle)
                
                # Process candle with strategy
                if self.strategy:
                    signal = self.strategy.process_candle(candle)
                    
                    if signal:
                        # Open position based on signal
                        self._open_position(signal, candle)
                    
                    # Check for exit signals (sell for long positions, buy for short positions)
                    if signal and symbol in self.open_positions:
                        position = self.open_positions[symbol]
                        
                        if (position.is_long and signal.signal_type == SignalType.SELL) or \
                           (not position.is_long and signal.signal_type == SignalType.BUY):
                            # Close position on opposite signal
                            self._close_position(symbol, candle.close, candle.timestamp)
        
        # Close any remaining open positions at the end of the backtest
        for symbol in list(self.open_positions.keys()):
            last_candle = self.historical_data[symbol][-1] if symbol in self.historical_data and self.historical_data[symbol] else None
            if last_candle:
                self._close_position(symbol, last_candle.close, last_candle.timestamp)
            else:
                self._close_position(symbol)
        
        # Calculate metrics
        self.metrics.calculate_from_trades(self.closed_positions)
        
        return self.metrics
    
    def optimize_parameters(self, parameter_ranges: Dict[str, List[Any]], 
                           optimization_metric: str = 'sharpe_ratio',
                           max_iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search or random search.
        
        Args:
            parameter_ranges: Dictionary of parameter names and possible values
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_pnl', etc.)
            max_iterations: Maximum number of iterations for optimization
        
        Returns:
            Dict containing best parameters and performance
        """
        logger.info(f"Starting parameter optimization with {max_iterations} iterations")
        
        if not self.strategy:
            raise ValueError("No strategy set for optimization")
        
        # Store original parameters to restore later
        original_params = {}
        for param_name in parameter_ranges.keys():
            if hasattr(self.strategy, param_name):
                original_params[param_name] = getattr(self.strategy, param_name)
        
        # Prepare results storage
        self.optimization_results = []
        best_result = None
        best_score = float('-inf')
        
        # Determine if we should use grid search or random search
        total_combinations = 1
        for values in parameter_ranges.values():
            total_combinations *= len(values)
        
        use_random_search = total_combinations > max_iterations
        
        if use_random_search:
            # Random search
            import random
            logger.info(f"Using random search due to large parameter space ({total_combinations} combinations)")
            for i in range(max_iterations):
                # Select random parameter combination
                params = {}
                for param_name, values in parameter_ranges.items():
                    params[param_name] = random.choice(values)
                
                result = self._evaluate_parameters(params, optimization_metric)
                self.optimization_results.append(result)
                
                # Update best result
                score = result['metrics'].get(optimization_metric, float('-inf'))
                if score > best_score:
                    best_score = score
                    best_result = result
                
                logger.info(f"Iteration {i+1}/{max_iterations}: "
                           f"Score={score:.4f}, Best={best_score:.4f}")
        else:
            # Grid search - iterate through all combinations
            logger.info(f"Using grid search for {total_combinations} parameter combinations")
            from itertools import product
            
            # Generate all parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            for i, combination in enumerate(product(*param_values)):
                if i >= max_iterations:
                    break
                
                # Create parameter dictionary
                params = {param_names[j]: combination[j] for j in range(len(param_names))}
                
                result = self._evaluate_parameters(params, optimization_metric)
                self.optimization_results.append(result)
                
                # Update best result
                score = result['metrics'].get(optimization_metric, float('-inf'))
                if score > best_score:
                    best_score = score
                    best_result = result
                
                logger.info(f"Combination {i+1}/{min(total_combinations, max_iterations)}: "
                           f"Score={score:.4f}, Best={best_score:.4f}")
        
        # Restore original parameters
        for param_name, value in original_params.items():
            setattr(self.strategy, param_name, value)
        
        return best_result
    
    def _evaluate_parameters(self, params: Dict[str, Any], 
                            optimization_metric: str) -> Dict[str, Any]:
        """Evaluate a set of parameters by running a backtest."""
        # Apply parameters to strategy
        for param_name, value in params.items():
            setattr(self.strategy, param_name, value)
        
        # Run backtest
        metrics = self.run_backtest()
        metrics_dict = metrics.to_dict()
        
        return {
            'parameters': params.copy(),
            'metrics': metrics_dict
        }
    
    def generate_report(self, report_dir: str = "reports") -> str:
        """
        Generate a detailed backtest report.
        
        Args:
            report_dir: Directory to save the report
        
        Returns:
            Path to the generated report file
        """
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate timestamp for report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"backtest_report_{timestamp}.txt"
        report_path = os.path.join(report_dir, report_filename)
        
        # Generate equity curve chart
        chart_filename = f"equity_curve_{timestamp}.png"
        chart_path = os.path.join(report_dir, chart_filename)
        self.metrics.plot_equity_curve(chart_path)
        
        # Generate trade distribution chart
        dist_filename = f"trade_distribution_{timestamp}.png"
        dist_path = os.path.join(report_dir, dist_filename)
        self.metrics.plot_trade_distribution(dist_path)
        
        # Create report content
        with open(report_path, 'w') as f:
            f.write("=== BACKTEST REPORT ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- STRATEGY SETTINGS ---\n")
            if self.strategy:
                f.write(f"Strategy: {self.strategy.name}\n")
                # Write strategy parameters
                for attr in dir(self.strategy):
                    if not attr.startswith('_') and not callable(getattr(self.strategy, attr)):
                        value = getattr(self.strategy, attr)
                        if not isinstance(value, (dict, list)) and attr not in ['signals']:
                            f.write(f"  {attr}: {value}\n")
            
            f.write("\n--- BACKTEST SETTINGS ---\n")
            f.write(f"Initial Capital: ${self.initial_capital:.2f}\n")
            f.write(f"Position Size: {self.position_size*100:.1f}%\n")
            f.write(f"Stop Loss: {'Enabled' if self.use_stop_loss else 'Disabled'} ({self.stop_loss_pct*100:.1f}%)\n")
            f.write(f"Take Profit: {'Enabled' if self.use_take_profit else 'Disabled'} ({self.take_profit_pct*100:.1f}%)\n")
            f.write(f"Trailing Stop: {'Enabled' if self.enable_trailing_stop else 'Disabled'} ({self.trailing_stop_pct*100:.1f}%)\n")
            f.write(f"Max Open Positions: {self.max_open_positions}\n")
            f.write(f"Commission: {self.commission_pct*100:.3f}%\n")
            
            f.write("\n--- PERFORMANCE METRICS ---\n")
            metrics = self.metrics.to_dict()
            f.write(f"Total Profit/Loss: ${metrics['total_pnl']:.2f}\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Annualized Volatility: {metrics['volatility']:.2f}%\n")
            f.write(f"Average Trade Duration: {metrics['avg_trade_duration']:.2f} hours\n")
            f.write(f"Average Win: ${metrics['avg_win_pnl']:.2f}\n")
            f.write(f"Average Loss: ${metrics['avg_loss_pnl']:.2f}\n")
            f.write(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}\n")
            
            f.write("\n--- DETAILED TRADE LIST ---\n")
            for i, trade in enumerate(self.closed_positions):
                f.write(f"Trade {i+1}: {trade['symbol']} {'LONG' if trade['is_long'] else 'SHORT'}\n")
                f.write(f"  Entry: {trade['entry_price']:.4f} at {trade['entry_time']}\n")
                f.write(f"  Exit:  {trade['close_price']:.4f} at {trade['exit_time']}\n")
                f.write(f"  P&L:   ${trade['pnl']:.2f}\n")
                f.write(f"  Duration: {trade['duration_hours']:.1f} hours\n")
                f.write("\n")
            
            f.write(f"\nEquity curve chart saved to: {chart_path}\n")
            f.write(f"Trade distribution chart saved to: {dist_path}\n")
        
        logger.info(f"Backtest report generated at {report_path}")
        return report_path
    
    def save_results(self, filepath: str) -> None:
        """Save backtest results to a JSON file."""
        # Convert results to serializable format
        results = {
            'strategy': self.strategy.name if self.strategy else 'Unknown',
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'metrics': self.metrics.to_dict(),
            'trades': []
        }
        
        # Convert trades to serializable format
        for trade in self.closed_positions:
            serializable_trade = {}
            for key, value in trade.items():
                if isinstance(value, datetime):
                    serializable_trade[key] = value.isoformat()
                else:
                    serializable_trade[key] = value
            results['trades'].append(serializable_trade)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Backtest results saved to {filepath}") 