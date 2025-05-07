"""
Simple Portfolio Manager module for the AI Trading Agent.

This module provides a simplified portfolio management implementation
that tracks positions, calculates performance metrics, and handles
position sizing based on different methods.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ..common import logger


class SimplePortfolioManager:
    """
    Simple Portfolio Manager that tracks positions and handles position sizing.
    
    Supports different position sizing methods:
    - equal: Equal position sizes for all assets
    - fixed: Fixed percentage of portfolio per position
    - volatility: Position size based on volatility
    - risk_parity: Position size inversely proportional to risk
    - kelly: Kelly criterion for optimal position sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the portfolio manager.
        
        Args:
            config: Configuration dictionary
        """
        self.name = "SimplePortfolioManager"
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.position_sizing_method = config.get('position_sizing_method', 'risk_parity')
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)
        self.max_position_size = config.get('max_position_size', 0.2)
        
        # Initialize portfolio
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'timestamp': datetime.now().isoformat()
        }
        
        # Track performance
        self.portfolio_history = [self.portfolio.copy()]
        self.trade_history = []
        self.performance_metrics = {
            'initial_value': self.initial_capital,
            'current_value': self.initial_capital,
            'return_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        logger.info(f"Initialized {self.name} with initial_capital={self.initial_capital}, "
                   f"position_sizing_method={self.position_sizing_method}")
    
    def get_current_portfolio(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.
        
        Returns:
            Portfolio dictionary
        """
        return self.portfolio
    
    def update_portfolio(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the portfolio based on a trade.
        
        Args:
            trade: Trade dictionary with symbol, side, quantity, price, etc.
        
        Returns:
            Updated portfolio dictionary
        """
        logger.info(f"Updating portfolio with trade: {trade}")
        
        symbol = trade['symbol']
        side = trade['side'].lower()
        quantity = float(trade['quantity'])
        price = float(trade['price'])
        commission = float(trade.get('commission', 0.0))
        
        # Calculate trade value
        trade_value = price * quantity
        total_cost = trade_value + commission
        
        # Update portfolio based on trade side
        if side == 'buy':
            # Deduct cash
            if self.portfolio['cash'] < total_cost:
                logger.warning(f"Insufficient cash for trade: {total_cost} > {self.portfolio['cash']}")
                return self.portfolio
            
            self.portfolio['cash'] -= total_cost
            
            # Update position
            if symbol not in self.portfolio['positions']:
                self.portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'current_price': price,
                    'cost_basis': trade_value,
                    'market_value': trade_value,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'realized_pnl': 0.0
                }
            else:
                # Update existing position
                position = self.portfolio['positions'][symbol]
                total_quantity = position['quantity'] + quantity
                total_cost_basis = position['cost_basis'] + trade_value
                
                # Calculate new average price
                position['avg_price'] = total_cost_basis / total_quantity if total_quantity > 0 else 0
                position['quantity'] = total_quantity
                position['cost_basis'] = total_cost_basis
                position['market_value'] = total_quantity * price
                position['current_price'] = price
                
                # Update unrealized P&L
                position['unrealized_pnl'] = position['market_value'] - position['cost_basis']
                position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / position['cost_basis']) * 100 if position['cost_basis'] > 0 else 0
        
        elif side == 'sell':
            # Check if position exists
            if symbol not in self.portfolio['positions']:
                logger.warning(f"Position {symbol} not found for sell trade")
                return self.portfolio
            
            position = self.portfolio['positions'][symbol]
            
            # Check if enough quantity
            if position['quantity'] < quantity:
                logger.warning(f"Insufficient quantity for sell trade: {quantity} > {position['quantity']}")
                return self.portfolio
            
            # Calculate realized P&L
            sell_value = trade_value - commission
            cost_basis_per_unit = position['cost_basis'] / position['quantity'] if position['quantity'] > 0 else 0
            cost_basis_sold = cost_basis_per_unit * quantity
            realized_pnl = sell_value - cost_basis_sold
            
            # Update position
            position['realized_pnl'] += realized_pnl
            position['quantity'] -= quantity
            
            # Update cost basis and market value
            if position['quantity'] > 0:
                position['cost_basis'] = cost_basis_per_unit * position['quantity']
                position['market_value'] = position['quantity'] * price
                position['current_price'] = price
                
                # Update unrealized P&L
                position['unrealized_pnl'] = position['market_value'] - position['cost_basis']
                position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / position['cost_basis']) * 100 if position['cost_basis'] > 0 else 0
            else:
                # Remove position if quantity is zero
                self.portfolio['positions'].pop(symbol)
            
            # Add cash
            self.portfolio['cash'] += sell_value
            
            # Add trade to history with P&L
            trade['realized_pnl'] = realized_pnl
            trade['realized_pnl_pct'] = (realized_pnl / cost_basis_sold) * 100 if cost_basis_sold > 0 else 0
        
        # Update total portfolio value
        self._update_portfolio_value()
        
        # Add trade to history
        self.trade_history.append(trade)
        
        # Add portfolio snapshot to history
        self.portfolio_history.append(self.portfolio.copy())
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"Portfolio updated: cash={self.portfolio['cash']:.2f}, "
                   f"total_value={self.portfolio['total_value']:.2f}")
        
        return self.portfolio
    
    def calculate_position_size(self, symbol: str, signal: Dict[str, Any], current_price: float) -> float:
        """
        Calculate position size based on the configured method.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary with signal_strength, confidence_score, etc.
            current_price: Current price of the asset
        
        Returns:
            Position size in quantity
        """
        # Get signal strength and confidence
        signal_strength = signal.get('signal_strength', 0.0)
        confidence = signal.get('confidence_score', 0.5)
        
        # Skip if signal is not strong enough
        if abs(signal_strength) < 0.2:
            return 0.0
        
        # Calculate available capital
        available_capital = self.portfolio['cash']
        
        # Calculate position value based on method
        if self.position_sizing_method == 'equal':
            # Equal position sizing
            position_value = available_capital * self.max_position_size
        
        elif self.position_sizing_method == 'fixed':
            # Fixed percentage of portfolio
            position_value = self.portfolio['total_value'] * self.max_position_size * abs(signal_strength)
        
        elif self.position_sizing_method == 'volatility':
            # Volatility-based position sizing
            volatility = signal.get('metadata', {}).get('volatility', 0.02)
            position_value = (self.portfolio['total_value'] * self.max_position_size) / (volatility * 10)
        
        elif self.position_sizing_method == 'risk_parity':
            # Risk parity position sizing
            risk_factor = signal.get('metadata', {}).get('risk_factor', 1.0)
            position_value = (self.portfolio['total_value'] * self.max_position_size) / risk_factor
        
        elif self.position_sizing_method == 'kelly':
            # Kelly criterion
            win_rate = signal.get('metadata', {}).get('win_rate', 0.55)
            win_loss_ratio = signal.get('metadata', {}).get('win_loss_ratio', 1.5)
            
            # Kelly formula: f* = (p * b - q) / b
            # where p = win rate, q = 1 - p, b = win/loss ratio
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Limit kelly fraction to avoid excessive risk
            kelly_fraction = min(kelly_fraction, self.max_position_size)
            
            position_value = self.portfolio['total_value'] * kelly_fraction * confidence
        
        else:
            # Default to fixed percentage
            position_value = self.portfolio['total_value'] * self.max_position_size * abs(signal_strength)
        
        # Limit to available capital
        position_value = min(position_value, available_capital)
        
        # Calculate quantity
        quantity = position_value / current_price if current_price > 0 else 0
        
        # Round to appropriate precision based on price
        if current_price > 1000:
            quantity = round(quantity, 4)  # BTC precision
        elif current_price > 100:
            quantity = round(quantity, 3)
        elif current_price > 10:
            quantity = round(quantity, 2)
        elif current_price > 1:
            quantity = round(quantity, 1)
        else:
            quantity = round(quantity)
        
        logger.info(f"Calculated position size for {symbol}: {quantity} units "
                   f"(value: ${position_value:.2f}, method: {self.position_sizing_method})")
        
        return quantity
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Returns:
            List of trade dictionaries
        """
        return self.trade_history
    
    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """
        Get portfolio history.
        
        Returns:
            List of portfolio snapshots
        """
        return self.portfolio_history
    
    def _update_portfolio_value(self) -> None:
        """Update the total portfolio value."""
        # Calculate total value of positions
        positions_value = sum(position['market_value'] for position in self.portfolio['positions'].values())
        
        # Update total value
        self.portfolio['total_value'] = self.portfolio['cash'] + positions_value
        
        # Update timestamp
        self.portfolio['timestamp'] = datetime.now().isoformat()
    
    def _update_performance_metrics(self) -> None:
        """Update portfolio performance metrics."""
        # Get initial and current value
        initial_value = self.portfolio_history[0]['total_value']
        current_value = self.portfolio['total_value']
        
        # Calculate return
        return_pct = ((current_value / initial_value) - 1) * 100
        
        # Calculate drawdown
        portfolio_values = [snapshot['total_value'] for snapshot in self.portfolio_history]
        peak = max(portfolio_values)
        drawdown = ((peak - current_value) / peak) * 100 if peak > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdown *= 100  # Convert to percentage
        
        # Calculate Sharpe ratio if we have enough history
        if len(self.portfolio_history) > 2:
            returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
                returns.append(daily_return)
            
            mean_return = np.mean(returns) if returns else 0
            std_return = np.std(returns) if returns else 1
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate win rate and profit factor
        winning_trades = [t for t in self.trade_history if t.get('side') == 'sell' and t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('side') == 'sell' and t.get('realized_pnl', 0) < 0]
        
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = sum(t.get('realized_pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('realized_pnl', 0) for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Update metrics
        self.performance_metrics = {
            'initial_value': initial_value,
            'current_value': current_value,
            'return_pct': return_pct,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Rebalance the portfolio to match the target allocation.
        
        Args:
            target_allocation: Dictionary mapping symbols to target allocation percentages
        
        Returns:
            List of trades required for rebalancing
        """
        logger.info(f"Rebalancing portfolio to target allocation: {target_allocation}")
        
        # Calculate current allocation
        current_allocation = {}
        for symbol, position in self.portfolio['positions'].items():
            current_allocation[symbol] = position['market_value'] / self.portfolio['total_value']
        
        # Calculate trades needed for rebalancing
        rebalance_trades = []
        
        for symbol, target_pct in target_allocation.items():
            current_pct = current_allocation.get(symbol, 0.0)
            
            # Check if rebalancing is needed
            if abs(current_pct - target_pct) > self.rebalance_threshold:
                # Calculate target value
                target_value = self.portfolio['total_value'] * target_pct
                
                if symbol in self.portfolio['positions']:
                    # Existing position
                    position = self.portfolio['positions'][symbol]
                    current_value = position['market_value']
                    current_price = position['current_price']
                    
                    # Calculate trade quantity
                    trade_value = target_value - current_value
                    trade_quantity = trade_value / current_price if current_price > 0 else 0
                    
                    # Create trade
                    if abs(trade_quantity) > 0.00001:  # Minimum trade size
                        trade = {
                            'symbol': symbol,
                            'side': 'buy' if trade_quantity > 0 else 'sell',
                            'quantity': abs(trade_quantity),
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'rebalance'
                        }
                        rebalance_trades.append(trade)
                
                elif target_pct > 0:
                    # New position
                    # Estimate price (would come from market data in a real system)
                    estimated_price = 100.0  # Default price for testing
                    
                    # Calculate quantity
                    quantity = target_value / estimated_price if estimated_price > 0 else 0
                    
                    # Create trade
                    if quantity > 0.00001:  # Minimum trade size
                        trade = {
                            'symbol': symbol,
                            'side': 'buy',
                            'quantity': quantity,
                            'price': estimated_price,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'rebalance'
                        }
                        rebalance_trades.append(trade)
        
        logger.info(f"Rebalancing requires {len(rebalance_trades)} trades")
        
        return rebalance_trades
