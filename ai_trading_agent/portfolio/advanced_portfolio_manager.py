"""
Advanced Portfolio Manager for AI Trading Agent.

This module provides a comprehensive portfolio management system that combines
correlation-based allocation, risk-adjusted allocation, and intelligent rebalancing
to optimize portfolio performance across various market conditions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import uuid

from ..common import logger
from .correlation_allocator import CorrelationAllocator
from .risk_adjusted_allocator import RiskAdjustedAllocator

class AdvancedPortfolioManager:
    """
    Advanced Portfolio Manager that implements sophisticated portfolio optimization techniques.
    
    This class integrates correlation-based allocation, risk targeting, dynamic rebalancing,
    and multi-asset portfolio optimization to maximize risk-adjusted returns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced portfolio manager.
        
        Args:
            config: Configuration dictionary with parameters:
                - initial_capital: Initial portfolio capital
                - target_risk: Target portfolio volatility
                - correlation_based: Whether to use correlation-based allocation
                - risk_adjusted: Whether to use risk-adjusted allocation
                - rebalance_threshold: Threshold for portfolio rebalancing
                - optimization_method: Method for portfolio optimization
                - market_regime_adaptive: Whether to adapt to market regimes
                - drawdown_protection: Whether to use drawdown protection
                - volatility_targeting: Whether to use volatility targeting
        """
        self.name = "AdvancedPortfolioManager"
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.target_risk = config.get('target_risk', 0.15)  # 15% annualized volatility
        self.correlation_based = config.get('correlation_based', True)
        self.risk_adjusted = config.get('risk_adjusted', True)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5% threshold
        self.optimization_method = config.get('optimization_method', 'min_variance')
        self.market_regime_adaptive = config.get('market_regime_adaptive', True)
        self.drawdown_protection = config.get('drawdown_protection', True)
        self.volatility_targeting = config.get('volatility_targeting', True)
        self.max_single_asset_allocation = config.get('max_single_asset_allocation', 0.30)
        
        # Initialize component allocators
        correlation_config = {
            'min_allocation': config.get('min_allocation', 0.01),
            'max_allocation': self.max_single_asset_allocation,
            'optimization_method': self.optimization_method,
            'correlation_lookback': config.get('correlation_lookback', 90)
        }
        self.correlation_allocator = CorrelationAllocator(correlation_config)
        
        risk_config = {
            'target_risk': self.target_risk,
            'max_drawdown_limit': config.get('max_drawdown_limit', 0.20),
            'vol_lookback': config.get('vol_lookback', 63),
            'risk_measure': config.get('risk_measure', 'volatility'),
            'dynamic_allocation': self.market_regime_adaptive
        }
        self.risk_allocator = RiskAdjustedAllocator(risk_config)
        
        # Initialize portfolio state
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'timestamp': datetime.now().isoformat()
        }
        
        # Historical data
        self.portfolio_history = [self.portfolio.copy()]
        self.trade_history = []
        self.allocation_history = []
        self.performance_metrics = self._initialize_performance_metrics()
        
        # Market regime tracking
        self.current_market_regime = 'normal'
        self.market_regime_history = []
        
        logger.info(f"Initialized {self.name} with initial_capital={self.initial_capital}, "
                   f"target_risk={self.target_risk*100:.1f}%, "
                   f"optimization_method={self.optimization_method}")
    
    def _initialize_performance_metrics(self) -> Dict[str, Any]:
        """Initialize performance metrics dictionary."""
        return {
            'initial_value': self.initial_capital,
            'current_value': self.initial_capital,
            'return_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'correlation_score': 0.0,
            'diversification_ratio': 1.0
        }
    
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
        trade_id = trade.get('id', str(uuid.uuid4()))
        timestamp = trade.get('timestamp', datetime.now().isoformat())
        
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
                position['avg_price'] = ((position['avg_price'] * position['quantity']) + 
                                        (price * quantity)) / total_quantity
                position['quantity'] = total_quantity
                position['cost_basis'] += trade_value
                position['market_value'] = position['quantity'] * price
                position['current_price'] = price
                
                # Update unrealized P&L
                position['unrealized_pnl'] = position['market_value'] - position['cost_basis']
                if position['cost_basis'] > 0:
                    position['unrealized_pnl_pct'] = position['unrealized_pnl'] / position['cost_basis']
                else:
                    position['unrealized_pnl_pct'] = 0.0
        
        elif side == 'sell':
            # Verify position exists
            if symbol not in self.portfolio['positions']:
                logger.warning(f"Cannot sell {symbol}, position does not exist")
                return self.portfolio
            
            position = self.portfolio['positions'][symbol]
            
            # Verify sufficient quantity
            if position['quantity'] < quantity:
                logger.warning(f"Insufficient quantity to sell: {quantity} > {position['quantity']}")
                return self.portfolio
            
            # Add cash
            self.portfolio['cash'] += trade_value - commission
            
            # Calculate realized P&L
            realized_pnl = (price - position['avg_price']) * quantity
            position['realized_pnl'] += realized_pnl
            
            # Update position
            position['quantity'] -= quantity
            position['market_value'] = position['quantity'] * price
            position['current_price'] = price
            
            # Remove position if quantity is zero
            if position['quantity'] <= 0:
                del self.portfolio['positions'][symbol]
            else:
                # Update unrealized P&L
                position['unrealized_pnl'] = position['market_value'] - (position['avg_price'] * position['quantity'])
                if position['cost_basis'] > 0:
                    position['unrealized_pnl_pct'] = position['unrealized_pnl'] / position['cost_basis']
                else:
                    position['unrealized_pnl_pct'] = 0.0
        
        # Add trade to history
        trade_record = {
            'id': trade_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission,
            'timestamp': timestamp,
            'realized_pnl': realized_pnl if side == 'sell' else 0.0
        }
        self.trade_history.append(trade_record)
        
        # Update portfolio value and metrics
        self._update_portfolio_value()
        self._update_performance_metrics()
        
        # Save portfolio snapshot
        self.portfolio_history.append(self.portfolio.copy())
        
        return self.portfolio
    
    def _update_portfolio_value(self):
        """Update the total portfolio value."""
        market_value = sum(p['market_value'] for p in self.portfolio['positions'].values())
        self.portfolio['total_value'] = self.portfolio['cash'] + market_value
        self.portfolio['timestamp'] = datetime.now().isoformat()
    
    def _update_performance_metrics(self):
        """Update portfolio performance metrics."""
        # Basic metrics
        initial_value = self.performance_metrics['initial_value']
        current_value = self.portfolio['total_value']
        
        # Calculate return
        return_pct = (current_value / initial_value) - 1 if initial_value > 0 else 0
        
        # Calculate drawdown
        peak_value = max([p['total_value'] for p in self.portfolio_history] + [initial_value])
        drawdown = (current_value / peak_value) - 1
        
        # Calculate max drawdown
        max_drawdown = min(self.performance_metrics['max_drawdown'], drawdown)
        
        # Calculate volatility if enough history
        volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        
        if len(self.portfolio_history) > 2:
            # Calculate daily returns
            values = [p['total_value'] for p in self.portfolio_history]
            returns = pd.Series([values[i] / values[i-1] - 1 for i in range(1, len(values))])
            
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
            
            # Calculate annualized return
            ann_return = return_pct * (252 / len(returns)) if len(returns) > 0 else 0.0
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_ratio = ann_return / volatility if volatility > 0 else 0.0
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
            sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calculate win rate
        completed_trades = [t for t in self.trade_history if t['side'] == 'sell']
        total_trades = len(completed_trades)
        
        winning_trades = [t for t in completed_trades if t.get('realized_pnl', 0) > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.get('realized_pnl', 0) for t in winning_trades)
        losing_trades = [t for t in completed_trades if t.get('realized_pnl', 0) <= 0]
        gross_loss = abs(sum(t.get('realized_pnl', 0) for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate correlation score
        correlation_score = 0.0
        if len(self.portfolio['positions']) > 1:
            correlation_score = self._calculate_correlation_score()
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio()
        
        # Update metrics
        self.performance_metrics = {
            'initial_value': initial_value,
            'current_value': current_value,
            'return_pct': return_pct,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'correlation_score': correlation_score,
            'diversification_ratio': diversification_ratio
        }
    
    def _calculate_correlation_score(self) -> float:
        """
        Calculate a score representing the average correlation between assets.
        Lower score is better (less correlation).
        
        Returns:
            Correlation score (0-1 range)
        """
        # For a real implementation, this would use actual return correlations
        # Here we use a placeholder value
        return 0.3  # Moderate correlation
    
    def _calculate_diversification_ratio(self) -> float:
        """
        Calculate diversification ratio based on position weights.
        Higher ratio is better (more diversified).
        
        Returns:
            Diversification ratio (>=1.0)
        """
        # Get position weights
        total_value = self.portfolio['total_value']
        if total_value <= 0:
            return 1.0
        
        weights = []
        for symbol, position in self.portfolio['positions'].items():
            weight = position['market_value'] / total_value
            weights.append(weight)
        
        # Add cash weight
        cash_weight = self.portfolio['cash'] / total_value
        weights.append(cash_weight)
        
        # Herfindahl-Hirschman Index (HHI) - measure of concentration
        hhi = sum(w**2 for w in weights)
        
        # Convert to diversification ratio (1/HHI)
        div_ratio = 1.0 / hhi if hhi > 0 else 1.0
        
        return div_ratio
    
    def optimize_allocations(self, 
                           returns: pd.DataFrame, 
                           market_regime: Optional[str] = None, 
                           risk_factors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize portfolio allocations based on historical returns and current market regime.
        
        Args:
            returns: DataFrame of asset returns
            market_regime: Current market regime
            risk_factors: Asset-specific risk factors
            
        Returns:
            Dictionary of optimized allocations
        """
        logger.info(f"Optimizing allocations with market_regime={market_regime}")
        
        # Store market regime
        if market_regime:
            self.current_market_regime = market_regime
            self.market_regime_history.append({
                'regime': market_regime,
                'timestamp': datetime.now().isoformat()
            })
        
        # Step 1: Use correlation-based allocation if enabled
        if self.correlation_based:
            allocations = self.correlation_allocator.calculate_allocations(returns)
        else:
            # Equal weight fallback
            allocations = {asset: 1.0 / len(returns.columns) for asset in returns.columns}
        
        # Step 2: Apply risk-adjusted allocation if enabled
        if self.risk_adjusted:
            risk_allocations = self.risk_allocator.calculate_allocations(
                returns, market_regime, risk_factors)
            
            # Blend allocations (simple average)
            allocations = {
                asset: (allocations.get(asset, 0) + risk_allocations.get(asset, 0)) / 2
                for asset in set(list(allocations.keys()) + list(risk_allocations.keys()))
            }
        
        # Step 3: Apply volatility targeting if enabled
        if self.volatility_targeting:
            allocations = self.risk_allocator.scale_portfolio_to_risk_target(allocations, returns)
        
        # Step 4: Apply drawdown protection if enabled and in drawdown
        if self.drawdown_protection and self.performance_metrics.get('drawdown', 0) < -0.1:
            # In significant drawdown, increase cash allocation
            drawdown = abs(self.performance_metrics.get('drawdown', 0))
            cash_increase = min(0.5, drawdown)  # Up to 50% cash based on drawdown severity
            
            # Adjust allocations
            scaled_allocations = {}
            for asset, allocation in allocations.items():
                if asset != 'CASH':
                    scaled_allocations[asset] = allocation * (1 - cash_increase)
            
            # Set cash allocation
            cash_allocation = allocations.get('CASH', 0) + cash_increase
            scaled_allocations['CASH'] = cash_allocation
            
            allocations = scaled_allocations
        
        # Store allocation history
        self.allocation_history.append({
            'allocations': allocations.copy(),
            'timestamp': datetime.now().isoformat(),
            'market_regime': self.current_market_regime
        })
        
        return allocations
    
    def get_rebalance_trades(self, target_allocations: Dict[str, float],
                           current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance the portfolio to target allocations.
        
        Args:
            target_allocations: Target asset allocations
            current_prices: Current prices for each asset
            
        Returns:
            List of trades required for rebalancing
        """
        logger.info(f"Calculating rebalance trades to match target allocations")
        
        # Calculate current allocations
        current_allocations = {}
        for symbol, position in self.portfolio['positions'].items():
            current_allocations[symbol] = position['market_value'] / self.portfolio['total_value']
        
        # Add cash allocation
        current_allocations['CASH'] = self.portfolio['cash'] / self.portfolio['total_value']
        
        # Calculate required trades
        rebalance_trades = []
        
        # Handle non-cash assets
        for symbol, target_pct in target_allocations.items():
            if symbol == 'CASH':
                continue
                
            current_pct = current_allocations.get(symbol, 0.0)
            
            # Check if rebalancing is needed
            if abs(current_pct - target_pct) > self.rebalance_threshold:
                # Get current price
                if symbol not in current_prices:
                    logger.warning(f"Missing price for {symbol}, skipping rebalance")
                    continue
                
                price = current_prices[symbol]
                
                # Calculate target value
                target_value = self.portfolio['total_value'] * target_pct
                
                if symbol in self.portfolio['positions']:
                    # Existing position
                    position = self.portfolio['positions'][symbol]
                    current_value = position['market_value']
                    
                    # Calculate trade
                    trade_value = target_value - current_value
                    trade_quantity = abs(trade_value / price) if price > 0 else 0
                    
                    # Create trade if significant
                    if trade_quantity * price > 10.0:  # Minimum $10 trade
                        trade = {
                            'symbol': symbol,
                            'side': 'buy' if trade_value > 0 else 'sell',
                            'quantity': trade_quantity,
                            'price': price,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'rebalance'
                        }
                        rebalance_trades.append(trade)
                
                elif target_pct > 0:
                    # New position
                    quantity = target_value / price if price > 0 else 0
                    
                    # Create trade if significant
                    if quantity * price > 10.0:  # Minimum $10 trade
                        trade = {
                            'symbol': symbol,
                            'side': 'buy',
                            'quantity': quantity,
                            'price': price,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'rebalance'
                        }
                        rebalance_trades.append(trade)
        
        logger.info(f"Rebalancing requires {len(rebalance_trades)} trades")
        return rebalance_trades
    
    def should_rebalance(self, target_allocations: Dict[str, float]) -> bool:
        """
        Determine if portfolio rebalancing is needed.
        
        Args:
            target_allocations: Target asset allocations
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        # Calculate current allocations
        current_allocations = {}
        for symbol, position in self.portfolio['positions'].items():
            current_allocations[symbol] = position['market_value'] / self.portfolio['total_value']
        
        # Add cash allocation
        current_allocations['CASH'] = self.portfolio['cash'] / self.portfolio['total_value']
        
        # Check deviation from target
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0.0)
            if abs(current_pct - target_pct) > self.rebalance_threshold:
                return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics
    
    def get_current_portfolio(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.
        
        Returns:
            Portfolio dictionary
        """
        return self.portfolio
    
    def update_prices(self, current_prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
        """
        for symbol, position in self.portfolio['positions'].items():
            if symbol in current_prices:
                price = current_prices[symbol]
                position['current_price'] = price
                position['market_value'] = position['quantity'] * price
                position['unrealized_pnl'] = position['market_value'] - position['cost_basis']
                if position['cost_basis'] > 0:
                    position['unrealized_pnl_pct'] = position['unrealized_pnl'] / position['cost_basis']
        
        # Update portfolio value and metrics
        self._update_portfolio_value()
        self._update_performance_metrics()
    
    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """
        Get allocation history.
        
        Returns:
            List of allocation history records
        """
        return self.allocation_history
    
    def get_market_regime_history(self) -> List[Dict[str, Any]]:
        """
        Get market regime history.
        
        Returns:
            List of market regime records
        """
        return self.market_regime_history
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio summary
        """
        # Asset allocation summary
        asset_allocation = {}
        for symbol, position in self.portfolio['positions'].items():
            asset_allocation[symbol] = position['market_value'] / self.portfolio['total_value']
        
        # Add cash allocation
        asset_allocation['CASH'] = self.portfolio['cash'] / self.portfolio['total_value']
        
        # Risk metrics
        risk_metrics = {
            'volatility': self.performance_metrics.get('volatility', 0.0),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0),
            'correlation_score': self.performance_metrics.get('correlation_score', 0.0),
            'diversification_ratio': self.performance_metrics.get('diversification_ratio', 1.0)
        }
        
        # Return metrics
        return_metrics = {
            'return_pct': self.performance_metrics.get('return_pct', 0.0),
            'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0.0),
            'sortino_ratio': self.performance_metrics.get('sortino_ratio', 0.0)
        }
        
        # Trading metrics
        trading_metrics = {
            'win_rate': self.performance_metrics.get('win_rate', 0.0),
            'profit_factor': self.performance_metrics.get('profit_factor', 0.0),
            'total_trades': self.performance_metrics.get('total_trades', 0)
        }
        
        # Compile summary
        return {
            'total_value': self.portfolio['total_value'],
            'cash': self.portfolio['cash'],
            'cash_pct': self.portfolio['cash'] / self.portfolio['total_value'],
            'positions_count': len(self.portfolio['positions']),
            'asset_allocation': asset_allocation,
            'risk_metrics': risk_metrics,
            'return_metrics': return_metrics,
            'trading_metrics': trading_metrics,
            'market_regime': self.current_market_regime,
            'timestamp': datetime.now().isoformat()
        }
