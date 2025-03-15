"""Regime-based trading strategy module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import matplotlib.pyplot as plt

from ..detection import RegimeDetectorFactory, BaseRegimeDetector


class RegimeStrategy:
    """
    Regime-based trading strategy.
    
    This class implements a trading strategy based on market regime detection.
    It uses different regime detection algorithms to identify market regimes
    and applies different trading rules for each regime.
    """
    
    def __init__(
        self,
        detector_method: str = 'trend',
        detector_params: Optional[Dict[str, Any]] = None,
        regime_rules: Optional[Dict[int, Dict[str, Any]]] = None,
        initial_capital: float = 10000.0,
        position_sizing: str = 'fixed',
        max_position_size: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        """
        Initialize the regime strategy.
        
        Args:
            detector_method: Method for regime detection (default: 'trend')
            detector_params: Parameters for the detector (default: None)
            regime_rules: Trading rules for each regime (default: None)
            initial_capital: Initial capital for backtesting (default: 10000.0)
            position_sizing: Position sizing method ('fixed', 'percent', 'kelly') (default: 'fixed')
            max_position_size: Maximum position size as a fraction of capital (default: 1.0)
            stop_loss_pct: Stop loss percentage (default: None)
            take_profit_pct: Take profit percentage (default: None)
        """
        self.detector_method = detector_method
        self.detector_params = detector_params or {}
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Create detector
        factory = RegimeDetectorFactory()
        self.detector = factory.create(detector_method, **self.detector_params)
        
        # Set default regime rules if not provided
        if regime_rules is None:
            # Default rules based on detector method
            if detector_method == 'trend':
                self.regime_rules = {
                    0: {'action': 'sell', 'allocation': 0.0},  # Downtrend
                    1: {'action': 'hold', 'allocation': 0.5},  # Sideways
                    2: {'action': 'buy', 'allocation': 1.0}    # Uptrend
                }
            elif detector_method == 'volatility':
                self.regime_rules = {
                    0: {'action': 'buy', 'allocation': 1.0},   # Low volatility
                    1: {'action': 'hold', 'allocation': 0.5},  # Medium volatility
                    2: {'action': 'sell', 'allocation': 0.0}   # High volatility
                }
            elif detector_method == 'momentum':
                self.regime_rules = {
                    0: {'action': 'sell', 'allocation': 0.0},  # Negative momentum
                    1: {'action': 'hold', 'allocation': 0.5},  # Neutral momentum
                    2: {'action': 'buy', 'allocation': 1.0}    # Positive momentum
                }
            else:
                # Generic rules for any detector
                self.regime_rules = {
                    i: {'action': 'buy' if i % 2 == 0 else 'sell', 'allocation': 1.0 if i % 2 == 0 else 0.0}
                    for i in range(self.detector.n_regimes)
                }
        else:
            self.regime_rules = regime_rules
        
        # Initialize backtest results
        self.results = None
        self.equity_curve = None
        self.trades = None
        self.performance_metrics = None
    
    def backtest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Dictionary containing backtest results
        """
        # Detect regimes
        labels = self.detector.fit_predict(data)
        
        # Create DataFrame for backtest
        df = pd.DataFrame({
            'date': data['dates'],
            'price': data['prices'],
            'regime': labels
        })
        
        # Initialize columns
        df['position'] = 0.0
        df['cash'] = self.initial_capital
        df['equity'] = self.initial_capital
        df['trade'] = False
        df['trade_type'] = ''
        df['trade_price'] = 0.0
        df['trade_size'] = 0.0
        df['trade_value'] = 0.0
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        
        # Apply trading rules
        position = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(1, len(df)):
            # Get current regime and previous position
            regime = df.loc[i, 'regime']
            prev_position = df.loc[i-1, 'position']
            prev_cash = df.loc[i-1, 'cash']
            prev_equity = df.loc[i-1, 'equity']
            
            # Get trading rule for current regime
            rule = self.regime_rules.get(regime, {'action': 'hold', 'allocation': 0.0})
            action = rule['action']
            allocation = rule['allocation']
            
            # Check stop loss and take profit
            if prev_position != 0 and (self.stop_loss_pct is not None or self.take_profit_pct is not None):
                current_price = df.loc[i, 'price']
                
                # Check stop loss
                if self.stop_loss_pct is not None and prev_position > 0 and current_price <= stop_loss:
                    action = 'sell'
                    allocation = 0.0
                elif self.stop_loss_pct is not None and prev_position < 0 and current_price >= stop_loss:
                    action = 'buy'
                    allocation = 0.0
                
                # Check take profit
                if self.take_profit_pct is not None and prev_position > 0 and current_price >= take_profit:
                    action = 'sell'
                    allocation = 0.0
                elif self.take_profit_pct is not None and prev_position < 0 and current_price <= take_profit:
                    action = 'buy'
                    allocation = 0.0
            
            # Calculate position size
            if action == 'buy' and prev_position <= 0:
                # Calculate position size based on position sizing method
                if self.position_sizing == 'fixed':
                    position_size = allocation * self.max_position_size
                elif self.position_sizing == 'percent':
                    position_size = allocation * self.max_position_size * prev_equity / df.loc[i, 'price']
                elif self.position_sizing == 'kelly':
                    # Simple Kelly criterion implementation
                    win_rate = 0.5  # Default win rate
                    win_loss_ratio = 1.0  # Default win/loss ratio
                    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
                    position_size = max(0, kelly_fraction) * allocation * self.max_position_size
                else:
                    position_size = allocation * self.max_position_size
                
                # Calculate trade details
                trade_size = position_size - prev_position
                trade_price = df.loc[i, 'price']
                trade_value = trade_size * trade_price
                
                # Update position and cash
                position = position_size
                cash = prev_cash - trade_value
                
                # Set stop loss and take profit
                if self.stop_loss_pct is not None:
                    stop_loss = trade_price * (1 - self.stop_loss_pct)
                if self.take_profit_pct is not None:
                    take_profit = trade_price * (1 + self.take_profit_pct)
                
                # Record trade
                df.loc[i, 'trade'] = True
                df.loc[i, 'trade_type'] = 'buy'
                df.loc[i, 'trade_price'] = trade_price
                df.loc[i, 'trade_size'] = trade_size
                df.loc[i, 'trade_value'] = trade_value
                df.loc[i, 'stop_loss'] = stop_loss
                df.loc[i, 'take_profit'] = take_profit
                
            elif action == 'sell' and prev_position >= 0:
                # Calculate position size based on position sizing method
                if self.position_sizing == 'fixed':
                    position_size = -allocation * self.max_position_size
                elif self.position_sizing == 'percent':
                    position_size = -allocation * self.max_position_size * prev_equity / df.loc[i, 'price']
                elif self.position_sizing == 'kelly':
                    # Simple Kelly criterion implementation
                    win_rate = 0.5  # Default win rate
                    win_loss_ratio = 1.0  # Default win/loss ratio
                    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
                    position_size = -max(0, kelly_fraction) * allocation * self.max_position_size
                else:
                    position_size = -allocation * self.max_position_size
                
                # Calculate trade details
                trade_size = position_size - prev_position
                trade_price = df.loc[i, 'price']
                trade_value = trade_size * trade_price
                
                # Update position and cash
                position = position_size
                cash = prev_cash - trade_value  # Negative trade_value for selling
                
                # Set stop loss and take profit
                if self.stop_loss_pct is not None:
                    stop_loss = trade_price * (1 + self.stop_loss_pct)
                if self.take_profit_pct is not None:
                    take_profit = trade_price * (1 - self.take_profit_pct)
                
                # Record trade
                df.loc[i, 'trade'] = True
                df.loc[i, 'trade_type'] = 'sell'
                df.loc[i, 'trade_price'] = trade_price
                df.loc[i, 'trade_size'] = trade_size
                df.loc[i, 'trade_value'] = trade_value
                df.loc[i, 'stop_loss'] = stop_loss
                df.loc[i, 'take_profit'] = take_profit
                
            elif action == 'hold':
                # Keep current position
                position = prev_position
                cash = prev_cash
                
            else:
                # Default: keep current position
                position = prev_position
                cash = prev_cash
            
            # Update position and cash
            df.loc[i, 'position'] = position
            df.loc[i, 'cash'] = cash
            
            # Calculate equity
            df.loc[i, 'equity'] = cash + position * df.loc[i, 'price']
        
        # Calculate returns
        df['return'] = df['equity'].pct_change().fillna(0)
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1
        
        # Extract trades
        trades = df[df['trade']].copy()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(df)
        
        # Store results
        self.results = df
        self.equity_curve = df[['date', 'equity']]
        self.trades = trades
        self.performance_metrics = performance_metrics
        
        # Return results
        return {
            'results': df,
            'equity_curve': self.equity_curve,
            'trades': trades,
            'performance_metrics': performance_metrics
        }
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            results: DataFrame containing backtest results
            
        Returns:
            Dictionary containing performance metrics
        """
        # Extract returns
        returns = results['return'].values
        
        # Calculate metrics
        total_return = results['equity'].iloc[-1] / results['equity'].iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        
        # Calculate volatility
        daily_volatility = np.std(returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.0  # Assume zero risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate drawdown
        equity_curve = results['equity'].values
        max_drawdown = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        trades = results[results['trade']].copy()
        if len(trades) > 0:
            trades['profit'] = trades['trade_value'] * -1  # Negative trade_value means profit for sell
            trades['win'] = trades['profit'] > 0
            win_rate = trades['win'].mean()
        else:
            win_rate = 0
        
        # Return metrics
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size (default: (12, 6))
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(self.results['date'], self.results['equity'], label='Equity')
        
        # Plot buy and sell signals
        buys = self.results[self.results['trade_type'] == 'buy']
        sells = self.results[self.results['trade_type'] == 'sell']
        
        ax.scatter(buys['date'], buys['equity'], color='green', marker='^', s=100, label='Buy')
        ax.scatter(sells['date'], sells['equity'], color='red', marker='v', s=100, label='Sell')
        
        # Add regime background
        unique_regimes = sorted(set(self.results['regime']))
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(unique_regimes))]
        
        for i, regime in enumerate(unique_regimes):
            mask = self.results['regime'] == regime
            ax.fill_between(self.results['date'], ax.get_ylim()[0], ax.get_ylim()[1], 
                           where=mask, color=colors[i], alpha=0.1, 
                           label=f'Regime {regime}')
        
        # Add labels and title
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance metrics as text
        if self.performance_metrics:
            metrics_text = '\n'.join([
                f"Total Return: {self.performance_metrics['total_return']:.2%}",
                f"Annual Return: {self.performance_metrics['annual_return']:.2%}",
                f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}",
                f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}",
                f"Win Rate: {self.performance_metrics['win_rate']:.2%}",
                f"Trades: {self.performance_metrics['num_trades']}"
            ])
            
            plt.figtext(0.01, 0.01, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_regime_performance(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot performance by regime.
        
        Args:
            figsize: Figure size (default: (12, 6))
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        # Calculate returns by regime
        regime_returns = {}
        
        for regime in sorted(set(self.results['regime'])):
            regime_mask = self.results['regime'] == regime
            regime_returns[regime] = self.results.loc[regime_mask, 'return'].mean()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regime returns
        regimes = list(regime_returns.keys())
        returns = list(regime_returns.values())
        
        ax.bar(regimes, returns, color='blue', alpha=0.7)
        
        # Add labels and title
        ax.set_title('Average Daily Return by Regime')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Average Daily Return')
        ax.grid(True, alpha=0.3)
        
        # Add regime names if available
        if hasattr(self.detector, 'regime_names'):
            regime_names = self.detector.regime_names
            ax.set_xticks(regimes)
            ax.set_xticklabels([regime_names[r] if r < len(regime_names) else f"Regime {r}" for r in regimes])
        
        plt.tight_layout()
        return fig 