"""
Technical Strategy Backtester for the AI Crypto Trading System.

This module provides a comprehensive backtesting framework for technical indicator
based trading strategies, allowing for:
- Testing of all implemented technical indicator strategies
- Historical data replay with configurable timeframes
- Custom indicator parameter optimization
- Performance metrics calculation and visualization
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set, Type, cast
from dataclasses import dataclass, field
import itertools
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta

from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.base_strategy import Strategy
from src.backtesting import BacktestStats, BacktestMode, TimeFrame as BTimeFrame

# Configure simple logger
logger = logging.getLogger("technical_backtester")

# Performance metrics
@dataclass
class BacktestPerformanceMetrics:
    """Performance metrics for a backtest run."""
    
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # Profit/loss metrics
    total_profit_loss: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    largest_profit: float = 0.0
    largest_loss: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    
    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    max_drawdown_duration: int = 0
    
    # Risk-adjusted return metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade metrics
    avg_trade_duration: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Equity curve data
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trade_points: List[Tuple[int, float, bool]] = field(default_factory=list)  # (index, price, is_buy)
    
    def calculate_ratios(self):
        """Calculate derived ratios based on collected metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            
        if self.total_loss != 0:
            self.profit_factor = abs(self.total_profit / self.total_loss) if self.total_loss != 0 else float('inf')
            
        if self.average_loss != 0:
            self.risk_reward_ratio = abs(self.average_profit / self.average_loss) if self.average_loss != 0 else float('inf')
            
        # Calculate Sharpe, Sortino, and Calmar ratios if we have equity data
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                self.sharpe_ratio = float(mean_return / std_return if std_return != 0 else 0)
                
                # Sortino ratio (only considers negative returns for risk calculation)
                negative_returns = returns[returns < 0]
                downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
                self.sortino_ratio = float(mean_return / downside_deviation if downside_deviation != 0 else 0)
                
                # Calmar ratio (return / max drawdown)
                self.calmar_ratio = float(mean_return / self.max_drawdown_percent if self.max_drawdown_percent != 0 else 0)

@dataclass
class OptimizationResult:
    """Result of a parameter optimization run."""
    
    parameters: Dict[str, Any]
    metrics: BacktestPerformanceMetrics
    equity_curve: List[float]
    parameter_key: str = ""
    
    def __post_init__(self):
        """Generate a key from parameters for easy identification."""
        self.parameter_key = "_".join([f"{k}={v}" for k, v in self.parameters.items()])

class StrategyBacktester:
    """Backtester for technical indicator strategies.
    
    This class provides functionality to:
    1. Run backtests of technical strategies on historical data
    2. Optimize strategy parameters
    3. Calculate performance metrics
    4. Visualize backtest results
    """
    
    def __init__(
        self,
        strategy_class: Type[Strategy],
        data_directory: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the technical strategy backtester.
        
        Args:
            strategy_class: The strategy class to backtest
            data_directory: Directory containing historical data files
            config_override: Optional configuration overrides for the strategy
            logger: Optional logger instance
        """
        self.strategy_class = strategy_class
        self.strategy_id = strategy_class.__name__.lower()
        self.data_directory = data_directory or os.path.join(os.getcwd(), "data", "historical")
        self.config_override = config_override or {}
        self.logger = logger or logger
        
        # Performance tracking
        self.metrics = BacktestPerformanceMetrics()
        self.optimization_results: List[OptimizationResult] = []
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_equity = 1000.0  # Starting capital
        self.position_size = 0.0
        self.in_position = False
        self.position_entry_price = 0.0
        self.position_entry_time: Optional[datetime] = None
        
        # Tracking for metrics calculation
        self.equity_curve = [self.current_equity]
        self.trade_history: List[Dict[str, Any]] = []
        self.drawdowns: List[float] = []
        self.current_drawdown = 0.0
        self.max_equity = self.current_equity
        
        # Parameter optimization settings
        self.param_grid: Dict[str, List[Any]] = {}
        self.optimization_target = "total_profit_loss"  # Default optimization target
    
    def load_data(self, symbol: str, timeframe: Union[TimeFrame, str], 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load historical data for backtesting.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTC-USD')
            timeframe: The candle timeframe
            start_date: Optional start date for data filtering
            end_date: Optional end date for data filtering
            
        Returns:
            DataFrame containing the historical data
        """
        # Convert timeframe to string if it's an enum
        tf_str = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        
        # Build file path
        file_path = os.path.join(self.data_directory, f"{symbol}_{tf_str}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical data file not found: {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Ensure we have a datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Store the data
        self.historical_data[symbol] = df
        
        return df
    
    def prepare_strategy(self, **kwargs) -> Strategy:
        """Prepare a strategy instance for backtesting.
        
        Args:
            **kwargs: Additional parameters to pass to the strategy
            
        Returns:
            An initialized strategy instance
        """
        # Create a new strategy instance
        strategy = self.strategy_class(strategy_id=self.strategy_id)
        
        # Override configuration with provided values
        for key, value in {**self.config_override, **kwargs}.items():
            strategy_key = key
            if not key.startswith("strategies."):
                strategy_key = f"strategies.{self.strategy_id}.{key}"
            
            # Try to set the attribute directly based on the key name
            attr_name = key.split('.')[-1]
            if hasattr(strategy, attr_name):
                setattr(strategy, attr_name, value)
        
        return strategy
    
    def run_backtest(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 1000.0,
        position_sizing: float = 1.0,
        **strategy_params
    ) -> BacktestPerformanceMetrics:
        """Run a backtest for the strategy on historical data.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            start_date: Optional start date for the backtest
            end_date: Optional end date for the backtest
            initial_capital: Initial capital for the backtest
            position_sizing: Position size as a fraction of capital (0.0-1.0)
            **strategy_params: Additional parameters to pass to the strategy
            
        Returns:
            Performance metrics for the backtest
        """
        # Reset metrics and tracking variables
        self.metrics = BacktestPerformanceMetrics()
        self.current_equity = initial_capital
        self.position_size = 0.0
        self.in_position = False
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.equity_curve = [self.current_equity]
        self.trade_history = []
        self.drawdowns = []
        self.current_drawdown = 0.0
        self.max_equity = self.current_equity
        
        # Load data if needed
        if symbol not in self.historical_data:
            self.load_data(symbol, timeframe, start_date, end_date)
        
        df = self.historical_data[symbol]
        if df.empty:
            self.logger.error(f"No data available for {symbol} with timeframe {timeframe}")
            return self.metrics
        
        # Prepare the strategy
        strategy = self.prepare_strategy(**strategy_params)
        
        # Run the backtest
        self.logger.info(f"Starting backtest for {symbol} with {strategy.__class__.__name__}")
        
        # Reset DataFrame index to make sure it's a proper datetime index
        df = df.reset_index()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # If no timestamp column, we need to create one
            df['timestamp'] = pd.to_datetime(df.index)
        
        # Now process each row
        for i, row in df.iterrows():
            timestamp = row['timestamp'].to_pydatetime()
            
            # Create a candle data object
            candle = CandleData(
                symbol=symbol,
                exchange="backtest",
                timestamp=timestamp,
                open=float(row.get('open', 0.0)),
                high=float(row.get('high', 0.0)),
                low=float(row.get('low', 0.0)),
                close=float(row.get('close', 0.0)),
                volume=float(row.get('volume', 0.0)),
                timeframe=timeframe if isinstance(timeframe, TimeFrame) 
                          else TimeFrame(timeframe)
            )
            
            # Process candle with strategy and get signals
            signals = self._process_candle_with_strategy(strategy, candle)
            
            # Handle signals and execute trades
            self._process_signals(signals, candle, position_sizing)
            
            # Update equity curve
            current_price = candle.close
            if self.in_position:
                # Calculate current value of position plus remaining cash
                position_value = self.position_size * current_price
                self.current_equity = position_value
            
            self.equity_curve.append(self.current_equity)
            
            # Update drawdown tracking
            if self.current_equity > self.max_equity:
                self.max_equity = self.current_equity
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
                
            self.drawdowns.append(self.current_drawdown)
            if self.current_drawdown > self.metrics.max_drawdown_percent:
                self.metrics.max_drawdown_percent = self.current_drawdown
                self.metrics.max_drawdown = self.max_equity - self.current_equity
        
        # Close any open position at the end of the backtest
        if self.in_position and len(df) > 0:
            # Get the last row for closing position
            last_row = df.iloc[-1]
            last_timestamp = last_row['timestamp'].to_pydatetime()
            
            last_candle = CandleData(
                symbol=symbol,
                exchange="backtest",
                timestamp=last_timestamp,
                open=float(last_row.get('open', 0.0)),
                high=float(last_row.get('high', 0.0)),
                low=float(last_row.get('low', 0.0)),
                close=float(last_row.get('close', 0.0)),
                volume=float(last_row.get('volume', 0.0)),
                timeframe=timeframe if isinstance(timeframe, TimeFrame) 
                          else TimeFrame(timeframe)
            )
            self._close_position(last_candle)
        
        # Calculate final metrics
        self._calculate_performance_metrics()
        
        return self.metrics
    
    def _process_candle_with_strategy(self, strategy: Strategy, candle: CandleData) -> List[Signal]:
        """Process a candle with the strategy and return signals.
        
        This handles both synchronous and asynchronous strategy implementations.
        
        Args:
            strategy: The strategy instance
            candle: The candle data to process
            
        Returns:
            List of signals generated by the strategy
        """
        # Call the strategy's process_candle method
        result = strategy.process_candle(candle)
        
        # Handle both sync and async implementations
        if isinstance(result, list):
            return result
        else:
            # For simplicity in backtesting, we'll return an empty list if not a List[Signal]
            self.logger.warning("Strategy returned non-list result. Expected List[Signal]")
            return []
    
    def _process_signals(self, signals: List[Signal], candle: CandleData, position_sizing: float):
        """Process strategy signals and execute trades.
        
        Args:
            signals: List of signals from the strategy
            candle: Current candle data
            position_sizing: Position size as a fraction of capital
        """
        for signal in signals:
            # Extract signal properties safely
            signal_type = getattr(signal, "type", None)
            signal_side = getattr(signal, "side", "")
            
            if signal_type == SignalType.ENTRY and not self.in_position:
                # Enter a new position
                self._open_position(candle, signal_side == "buy", position_sizing)
            
            elif signal_type == SignalType.EXIT and self.in_position:
                # Close the current position
                self._close_position(candle)
    
    def _open_position(self, candle: CandleData, is_long: bool, position_sizing: float):
        """Open a new trading position.
        
        Args:
            candle: Current candle data
            is_long: Whether this is a long (True) or short (False) position
            position_sizing: Position size as a fraction of capital
        """
        self.in_position = True
        self.position_entry_price = candle.close
        self.position_entry_time = candle.timestamp
        
        # Calculate position size based on current equity
        position_value = self.current_equity * position_sizing
        self.position_size = position_value / candle.close
        
        # Record trade entry
        entry = {
            'entry_time': candle.timestamp,
            'entry_price': candle.close,
            'direction': 'long' if is_long else 'short',
            'size': self.position_size,
            'value': position_value
        }
        
        self.metrics.trade_points.append((len(self.equity_curve) - 1, candle.close, True))
        self.logger.debug(f"Opened position: {entry}")
    
    def _close_position(self, candle: CandleData):
        """Close an open trading position.
        
        Args:
            candle: Current candle data
        """
        if not self.in_position or self.position_entry_time is None:
            return
            
        # Calculate profit/loss
        exit_price = candle.close
        entry_price = self.position_entry_price
        
        # Calculate position value
        position_value = self.position_size * exit_price
        profit_loss = position_value - (self.position_size * entry_price)
        
        # Update metrics
        self.metrics.total_trades += 1
        if profit_loss > 0:
            self.metrics.winning_trades += 1
            self.metrics.total_profit += profit_loss
            if profit_loss > self.metrics.largest_profit:
                self.metrics.largest_profit = profit_loss
        elif profit_loss < 0:
            self.metrics.losing_trades += 1
            self.metrics.total_loss += abs(profit_loss)
            if abs(profit_loss) > self.metrics.largest_loss:
                self.metrics.largest_loss = abs(profit_loss)
        else:
            self.metrics.break_even_trades += 1
        
        self.metrics.total_profit_loss += profit_loss
        
        # Update current equity
        self.current_equity = position_value
        
        # Calculate trade duration
        duration = (candle.timestamp - self.position_entry_time).total_seconds()
        
        # Record completed trade
        trade = {
            'entry_time': self.position_entry_time,
            'entry_price': self.position_entry_price,
            'exit_time': candle.timestamp,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'duration': duration
        }
        self.trade_history.append(trade)
        
        # Reset position tracking
        self.in_position = False
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        
        self.metrics.trade_points.append((len(self.equity_curve) - 1, candle.close, False))
        self.logger.debug(f"Closed position: {trade}")
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics based on the completed backtest."""
        # Basic metrics already tracked during backtest
        
        # Average profit/loss
        if self.metrics.winning_trades > 0:
            self.metrics.average_profit = self.metrics.total_profit / self.metrics.winning_trades
        
        if self.metrics.losing_trades > 0:
            self.metrics.average_loss = self.metrics.total_loss / self.metrics.losing_trades
        
        # Average trade duration
        if self.metrics.total_trades > 0 and self.trade_history:
            total_duration = sum(trade['duration'] for trade in self.trade_history)
            self.metrics.avg_trade_duration = total_duration / self.metrics.total_trades
        
        # Find max consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in self.trade_history:
            if trade['profit_loss'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
            elif trade['profit_loss'] < 0:
                consecutive_losses += 1
                consecutive_wins = 0
            
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        self.metrics.max_consecutive_wins = max_consecutive_wins
        self.metrics.max_consecutive_losses = max_consecutive_losses
        
        # Calculate drawdown duration
        in_drawdown = False
        current_drawdown_start = 0
        
        for i, dd in enumerate(self.drawdowns):
            if not in_drawdown and dd > 0:
                in_drawdown = True
                current_drawdown_start = i
            elif in_drawdown and dd == 0:
                drawdown_duration = i - current_drawdown_start
                if drawdown_duration > self.metrics.max_drawdown_duration:
                    self.metrics.max_drawdown_duration = drawdown_duration
                in_drawdown = False
        
        # Store equity and drawdown curves
        self.metrics.equity_curve = self.equity_curve
        self.metrics.drawdown_curve = self.drawdowns
        
        # Calculate ratio metrics
        self.metrics.calculate_ratios()
    
    def optimize_parameters(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        param_grid: Dict[str, List[Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 1000.0,
        position_sizing: float = 1.0,
        optimization_target: str = "total_profit_loss",
        parallel: bool = True
    ) -> List[OptimizationResult]:
        """Optimize strategy parameters using grid search.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            param_grid: Dictionary of parameter names and lists of values to test
            start_date: Optional start date for the backtest
            end_date: Optional end date for the backtest
            initial_capital: Initial capital for the backtest
            position_sizing: Position size as a fraction of capital
            optimization_target: Metric to optimize ('total_profit_loss', 'sharpe_ratio', etc.)
            parallel: Whether to run optimizations in parallel
            
        Returns:
            List of optimization results sorted by the target metric
        """
        self.param_grid = param_grid
        self.optimization_target = optimization_target
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Running parameter optimization with {len(param_combinations)} combinations")
        
        # Load data once before optimization
        if symbol not in self.historical_data:
            self.load_data(symbol, timeframe, start_date, end_date)
        
        results = []
        
        if parallel and len(param_combinations) > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Create parameter dictionaries for each combination
                param_dicts = []
                for combo in param_combinations:
                    params = dict(zip(param_names, combo))
                    param_dicts.append(params)
                
                # Run backtests in parallel
                future_to_params = {
                    executor.submit(
                        self._run_single_backtest, 
                        symbol, 
                        timeframe, 
                        start_date, 
                        end_date, 
                        initial_capital, 
                        position_sizing,
                        params
                    ): params for params in param_dicts
                }
                
                for future in concurrent.futures.as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        metrics, equity_curve = future.result()
                        results.append(OptimizationResult(
                            parameters=params,
                            metrics=metrics,
                            equity_curve=equity_curve
                        ))
                    except Exception as e:
                        self.logger.error(f"Error in parameter optimization: {e}")
        else:
            # Run sequentially
            for combo in param_combinations:
                params = dict(zip(param_names, combo))
                try:
                    metrics = self.run_backtest(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        position_sizing=position_sizing,
                        **params
                    )
                    
                    results.append(OptimizationResult(
                        parameters=params,
                        metrics=metrics,
                        equity_curve=self.equity_curve.copy()
                    ))
                except Exception as e:
                    self.logger.error(f"Error in parameter optimization: {e}")
        
        # Sort results by the optimization target
        self.optimization_results = sorted(
            results,
            key=lambda x: getattr(x.metrics, optimization_target, 0),
            reverse=True  # Assuming higher values are better
        )
        
        # Log the best results
        if self.optimization_results:
            best = self.optimization_results[0]
            self.logger.info(f"Best parameters: {best.parameters}")
            self.logger.info(f"Best {optimization_target}: {getattr(best.metrics, optimization_target)}")
        
        return self.optimization_results
    
    def _run_single_backtest(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        initial_capital: float,
        position_sizing: float,
        params: Dict[str, Any]
    ) -> Tuple[BacktestPerformanceMetrics, List[float]]:
        """Run a single backtest with specific parameters (for parallel execution).
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            start_date: Optional start date for the backtest
            end_date: Optional end date for the backtest
            initial_capital: Initial capital for the backtest
            position_sizing: Position size as a fraction of capital
            params: Strategy parameters to test
            
        Returns:
            Tuple of (performance metrics, equity curve)
        """
        # Create a new backtester instance to avoid shared state in parallel processing
        backtester = StrategyBacktester(
            strategy_class=self.strategy_class,
            data_directory=self.data_directory,
            config_override=self.config_override
        )
        
        # Load the data
        backtester.load_data(symbol, timeframe, start_date, end_date)
        
        # Run the backtest
        metrics = backtester.run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            position_sizing=position_sizing,
            **params
        )
        
        return metrics, backtester.equity_curve
    
    def plot_equity_curve(self, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """Plot the equity curve from a backtest.
        
        Args:
            title: Optional title for the plot
            figsize: Figure size as (width, height)
            
        Returns:
            The matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(self.metrics.equity_curve, label='Equity', color='blue')
        ax1.set_title(title or f'Equity Curve - {self.strategy_id.capitalize()}')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot buy/sell points
        for point_idx, price, is_buy in self.metrics.trade_points:
            marker = '^' if is_buy else 'v'
            color = 'green' if is_buy else 'red'
            ax1.scatter(point_idx, self.metrics.equity_curve[point_idx], marker=marker, color=color, s=100)
        
        # Plot drawdown
        ax2.fill_between(range(len(self.metrics.drawdown_curve)), 0, 
                          [d * 100 for d in self.metrics.drawdown_curve], color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_optimization_results(self, param_names: Optional[List[str]] = None, 
                                  top_n: int = 10) -> Figure:
        """Plot the results of parameter optimization.
        
        Args:
            param_names: List of parameter names to include in the plot
            top_n: Number of top results to show
            
        Returns:
            The matplotlib Figure object
        """
        if not self.optimization_results:
            self.logger.warning("No optimization results to plot")
            return cast(Figure, None)
        
        # Default to all parameters if none specified
        if not param_names:
            param_names = list(self.optimization_results[0].parameters.keys())
        
        # Limit to top N results
        results_to_plot = self.optimization_results[:top_n]
        
        # Create a figure with an axis for each parameter plus performance
        fig, axes = plt.subplots(len(param_names) + 1, 1, figsize=(12, 2 * (len(param_names) + 1)))
        
        # Handle single axis case
        if len(param_names) == 0:
            axes = [axes]
        
        # Plot performance metric
        performance_values = [getattr(r.metrics, self.optimization_target) for r in results_to_plot]
        axes[0].bar(range(len(results_to_plot)), performance_values)
        axes[0].set_title(f'Top {top_n} Results - {self.optimization_target}')
        axes[0].set_ylabel(self.optimization_target)
        axes[0].set_xticks([])
        
        # Plot parameter values for each result
        for i, param in enumerate(param_names):
            ax = axes[i + 1]
            param_values = [r.parameters.get(param) for r in results_to_plot]
            
            ax.bar(range(len(results_to_plot)), param_values)
            ax.set_ylabel(param)
            
            if i == len(param_names) - 1:
                ax.set_xlabel('Parameter Set')
            else:
                ax.set_xticks([])
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """Generate a detailed backtest report.
        
        Args:
            output_dir: Optional directory to save the report
            
        Returns:
            Path to the generated report
        """
        output_dir = output_dir or os.path.join(os.getcwd(), "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"{self.strategy_id}_report_{timestamp}.html")
        
        # Create HTML report
        with open(report_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Backtest Report - {self.strategy_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-value {{ font-weight: bold; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                    .section {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>Backtest Report - {self.strategy_id}</h1>
                <div class="section">
                    <h2>Performance Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Profit/Loss</td>
                            <td class="metric-value {'good' if self.metrics.total_profit_loss > 0 else 'bad'}">
                                {self.metrics.total_profit_loss:.2f}
                            </td>
                        </tr>
                        <tr>
                            <td>Total Trades</td>
                            <td class="metric-value">{self.metrics.total_trades}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td class="metric-value">{self.metrics.win_rate:.2%}</td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td class="metric-value">{self.metrics.profit_factor:.2f}</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td class="metric-value bad">{self.metrics.max_drawdown_percent:.2%}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td class="metric-value {'good' if self.metrics.sharpe_ratio > 1 else 'bad'}">
                                {self.metrics.sharpe_ratio:.2f}
                            </td>
                        </tr>
                        <tr>
                            <td>Sortino Ratio</td>
                            <td class="metric-value {'good' if self.metrics.sortino_ratio > 1 else 'bad'}">
                                {self.metrics.sortino_ratio:.2f}
                            </td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Trade Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Winning Trades</td>
                            <td>{self.metrics.winning_trades}</td>
                        </tr>
                        <tr>
                            <td>Losing Trades</td>
                            <td>{self.metrics.losing_trades}</td>
                        </tr>
                        <tr>
                            <td>Average Profit</td>
                            <td class="good">{self.metrics.average_profit:.2f}</td>
                        </tr>
                        <tr>
                            <td>Average Loss</td>
                            <td class="bad">{self.metrics.average_loss:.2f}</td>
                        </tr>
                        <tr>
                            <td>Largest Profit</td>
                            <td class="good">{self.metrics.largest_profit:.2f}</td>
                        </tr>
                        <tr>
                            <td>Largest Loss</td>
                            <td class="bad">{self.metrics.largest_loss:.2f}</td>
                        </tr>
                        <tr>
                            <td>Max Consecutive Wins</td>
                            <td>{self.metrics.max_consecutive_wins}</td>
                        </tr>
                        <tr>
                            <td>Max Consecutive Losses</td>
                            <td>{self.metrics.max_consecutive_losses}</td>
                        </tr>
                        <tr>
                            <td>Average Trade Duration</td>
                            <td>{timedelta(seconds=self.metrics.avg_trade_duration)}</td>
                        </tr>
                    </table>
                </div>
            </body>
            </html>
            """)
        
        # Save equity curve plot
        equity_fig = self.plot_equity_curve()
        if equity_fig:
            plot_path = os.path.join(output_dir, f"{self.strategy_id}_equity_{timestamp}.png")
            equity_fig.savefig(plot_path)
            plt.close(equity_fig)
        
        # Save optimization plot if results exist
        if self.optimization_results:
            opt_fig = self.plot_optimization_results()
            if opt_fig:
                opt_path = os.path.join(output_dir, f"{self.strategy_id}_optimization_{timestamp}.png")
                opt_fig.savefig(opt_path)
                plt.close(opt_fig)
        
        self.logger.info(f"Backtest report generated at {report_path}")
        return report_path 