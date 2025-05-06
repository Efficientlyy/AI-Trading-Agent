"""
Portfolio manager module for the AI Trading Agent.

This module handles portfolio allocation, position sizing, and portfolio rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..common import logger


class PortfolioManagerABC:
    """
    Abstract base class for portfolio managers.
    """
    
    def __init__(self, name: str):
        """
        Initialize the portfolio manager.
        
        Args:
            name: Name of the portfolio manager
        """
        self.name = name
    
    def calculate_position_sizes(self, signals: Dict[str, int], 
                                current_portfolio: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on signals and current portfolio.
        
        Args:
            signals: Dictionary mapping symbols to signal values (-1, 0, 1)
            current_portfolio: Current portfolio state
            market_data: Current market data
        
        Returns:
            Dictionary mapping symbols to target position sizes (as fractions of portfolio)
        """
        raise NotImplementedError("Subclasses must implement calculate_position_sizes")
    
    def rebalance_portfolio(self, current_positions: Dict[str, float],
                           target_positions: Dict[str, float],
                           threshold: float = 0.1) -> Dict[str, float]:
        """
        Determine position adjustments needed to rebalance the portfolio.
        
        Args:
            current_positions: Current positions as fractions of portfolio
            target_positions: Target positions as fractions of portfolio
            threshold: Minimum difference to trigger rebalancing
        
        Returns:
            Dictionary mapping symbols to position adjustments
        """
        raise NotImplementedError("Subclasses must implement rebalance_portfolio")


class SimplePortfolioManager(PortfolioManagerABC):
    """
    Simple portfolio manager implementing basic position sizing methods.
    """
    
    def __init__(self, name: str = "SimplePortfolioManager", 
                initial_capital: float = 10000.0,
                position_sizing_method: str = "equal",
                max_position_size: float = 0.2,
                max_open_positions: int = 5):
        """
        Initialize the simple portfolio manager.
        
        Args:
            name: Name of the portfolio manager
            initial_capital: Initial capital amount
            position_sizing_method: Method for position sizing ('equal', 'risk_parity', 'kelly')
            max_position_size: Maximum position size as fraction of portfolio
            max_open_positions: Maximum number of open positions
        """
        super().__init__(name)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_sizing_method = position_sizing_method
        self.max_position_size = max_position_size
        self.max_open_positions = max_open_positions
        self.positions = {}  # Current positions
        self.portfolio_history = []  # History of portfolio values
        
        logger.info(f"Initialized {self.name} with {initial_capital} capital")
    
    def calculate_position_sizes(self, signals: Dict[str, int], 
                                current_portfolio: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate position sizes based on signals and current portfolio.
        
        Args:
            signals: Dictionary mapping symbols to signal values (-1, 0, 1)
            current_portfolio: Current portfolio state
            market_data: Current market data
        
        Returns:
            Dictionary mapping symbols to target position sizes (as fractions of portfolio)
        """
        # Update current capital if provided
        if current_portfolio and 'total_value' in current_portfolio:
            self.current_capital = current_portfolio['total_value']
        
        # Filter symbols with non-zero signals
        active_symbols = {symbol: signal for symbol, signal in signals.items() if signal != 0}
        
        # Limit to max number of open positions
        if len(active_symbols) > self.max_open_positions:
            # Sort by signal strength (absolute value) if available
            if all('signal_strength' in market_data.get(symbol, {}) for symbol in active_symbols):
                sorted_symbols = sorted(
                    active_symbols.keys(),
                    key=lambda s: abs(market_data[s].get('signal_strength', 0)),
                    reverse=True
                )
                active_symbols = {s: active_symbols[s] for s in sorted_symbols[:self.max_open_positions]}
            else:
                # Just take the first N symbols
                active_symbols = dict(list(active_symbols.items())[:self.max_open_positions])
        
        # Calculate position sizes based on the selected method
        if self.position_sizing_method == 'equal':
            return self._equal_position_sizing(active_symbols)
        elif self.position_sizing_method == 'risk_parity':
            return self._risk_parity_position_sizing(active_symbols, market_data)
        elif self.position_sizing_method == 'kelly':
            return self._kelly_position_sizing(active_symbols, market_data)
        else:
            logger.warning(f"Unknown position sizing method: {self.position_sizing_method}. Using equal sizing.")
            return self._equal_position_sizing(active_symbols)
    
    def _equal_position_sizing(self, active_symbols: Dict[str, Any]) -> Dict[str, float]:
        """
        Allocate equal position sizes to all active symbols.
        
        Args:
            active_symbols: Dictionary mapping symbols to signal values
                           (either simple int values or dictionaries with signal_strength)
        
        Returns:
            Dictionary mapping symbols to target position sizes
        """
        if not active_symbols:
            return {}
        
        # Calculate equal position size (respecting max position size)
        position_size = min(1.0 / len(active_symbols), self.max_position_size)
        
        # Assign position sizes (negative for short positions)
        position_sizes = {}
        for symbol, signal in active_symbols.items():
            # Handle both simple int signals and dictionary signals
            if isinstance(signal, dict) and 'signal_strength' in signal:
                signal_value = signal['signal_strength']
            else:
                signal_value = signal
                
            # Convert signal to direction (-1, 0, 1)
            if isinstance(signal_value, (int, float)):
                if signal_value > 0.2:
                    direction = 1
                elif signal_value < -0.2:
                    direction = -1
                else:
                    direction = 0
            else:
                # Default to hold if signal is invalid
                logger.warning(f"Invalid signal type for {symbol}: {type(signal_value)}")
                direction = 0
                
            position_sizes[symbol] = position_size * direction
        
        return position_sizes
    
    def _risk_parity_position_sizing(self, active_symbols: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Allocate position sizes based on risk parity (inverse volatility weighting).
        
        Args:
            active_symbols: Dictionary mapping symbols to signal values
                           (either simple int values or dictionaries with signal_strength)
            market_data: Current market data including volatility
        
        Returns:
            Dictionary mapping symbols to target position sizes
        """
        if not active_symbols:
            return {}
        
        # Extract volatilities
        volatilities = {}
        for symbol in active_symbols:
            if symbol in market_data and 'volatility' in market_data[symbol]:
                volatilities[symbol] = market_data[symbol]['volatility']
            else:
                # Default volatility if not available
                volatilities[symbol] = 0.2  # 20% annualized volatility as default
                logger.warning(f"Volatility not available for {symbol}, using default value")
        
        # Calculate inverse volatility weights
        inv_vols = {symbol: 1.0 / vol if vol > 0 else 0.0 for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol == 0:
            return self._equal_position_sizing(active_symbols)
        
        # Normalize weights
        weights = {symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_vols.items()}
        
        # Apply maximum position size constraint
        weights = {symbol: min(weight, self.max_position_size) for symbol, weight in weights.items()}
        
        # Normalize again if needed
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        # Apply signal direction
        position_sizes = {}
        for symbol, signal in active_symbols.items():
            # Handle both simple int signals and dictionary signals
            if isinstance(signal, dict) and 'signal_strength' in signal:
                signal_value = signal['signal_strength']
            else:
                signal_value = signal
                
            # Convert signal to direction (-1, 0, 1)
            if isinstance(signal_value, (int, float)):
                if signal_value > 0.2:
                    direction = 1
                elif signal_value < -0.2:
                    direction = -1
                else:
                    direction = 0
            else:
                # Default to hold if signal is invalid
                logger.warning(f"Invalid signal type for {symbol}: {type(signal_value)}")
                direction = 0
                
            position_sizes[symbol] = weights[symbol] * direction
        
        return position_sizes
    
    def _kelly_position_sizing(self, active_symbols: Dict[str, Any], 
                              market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Allocate position sizes based on the Kelly criterion.
        
        Args:
            active_symbols: Dictionary mapping symbols to signal values (-1, 0, 1)
            market_data: Current market data including win rate and payoff ratio
        
        Returns:
            Dictionary mapping symbols to target position sizes
        """
        if not active_symbols:
            return {}
        
        position_sizes = {}
        
        for symbol, signal in active_symbols.items():
            # Extract win rate and payoff ratio from market data
            if symbol in market_data:
                win_rate = market_data[symbol].get('win_rate', 0.5)  # Default 50%
                payoff_ratio = market_data[symbol].get('payoff_ratio', 1.0)  # Default 1.0
            else:
                win_rate = 0.5
                payoff_ratio = 1.0
                logger.warning(f"Win rate and payoff ratio not available for {symbol}, using default values")
            
            # Calculate Kelly fraction
            # f* = (p * b - q) / b where p = win rate, q = 1-p, b = payoff ratio
            q = 1.0 - win_rate
            kelly_fraction = (win_rate * payoff_ratio - q) / payoff_ratio if payoff_ratio > 0 else 0.0
            
            # Apply half-Kelly for more conservative sizing
            kelly_fraction = kelly_fraction * 0.5
            
            # Apply maximum position size constraint
            position_size = min(max(0.0, kelly_fraction), self.max_position_size)
            
            # Handle both simple int signals and dictionary signals
            if isinstance(signal, dict) and 'signal_strength' in signal:
                signal_value = signal['signal_strength']
            else:
                signal_value = signal
                
            # Convert signal to direction (-1, 0, 1)
            if isinstance(signal_value, (int, float)):
                if signal_value > 0.2:
                    direction = 1
                elif signal_value < -0.2:
                    direction = -1
                else:
                    direction = 0
            else:
                # Default to hold if signal is invalid
                logger.warning(f"Invalid signal type for {symbol}: {type(signal_value)}")
                direction = 0
                
            # Apply signal direction
            position_sizes[symbol] = position_size * direction
        
        # Normalize if total exceeds 1.0
        total_position_size = sum(abs(size) for size in position_sizes.values())
        if total_position_size > 1.0:
            position_sizes = {
                symbol: size / total_position_size for symbol, size in position_sizes.items()
            }
        
        return position_sizes
    
    def rebalance_portfolio(self, current_positions: Dict[str, float],
                           target_positions: Dict[str, float],
                           threshold: float = 0.1) -> Dict[str, float]:
        """
        Determine position adjustments needed to rebalance the portfolio.
        
        Args:
            current_positions: Current positions as fractions of portfolio
            target_positions: Target positions as fractions of portfolio
            threshold: Minimum difference to trigger rebalancing
        
        Returns:
            Dictionary mapping symbols to position adjustments
        """
        adjustments = {}
        
        # Calculate adjustments for existing positions
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())
        
        for symbol in all_symbols:
            current_size = current_positions.get(symbol, 0.0)
            target_size = target_positions.get(symbol, 0.0)
            
            # Calculate difference
            diff = target_size - current_size
            
            # Apply threshold to avoid small adjustments
            if abs(diff) >= threshold:
                adjustments[symbol] = diff
        
        return adjustments
    
    def update_portfolio_value(self, positions: Dict[str, float], 
                              prices: Dict[str, float],
                              timestamp: Optional[datetime] = None) -> float:
        """
        Update the portfolio value based on current positions and prices.
        
        Args:
            positions: Current positions in units
            prices: Current prices
            timestamp: Current timestamp
        
        Returns:
            Updated portfolio value
        """
        # Calculate portfolio value
        portfolio_value = 0.0
        position_values = {}
        
        for symbol, units in positions.items():
            if symbol in prices:
                position_value = units * prices[symbol]
                position_values[symbol] = position_value
                portfolio_value += position_value
        
        # Add cash
        cash = self.current_capital - sum(position_values.values())
        portfolio_value += cash
        
        # Update current capital
        self.current_capital = portfolio_value
        
        # Record portfolio history
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate drawdown
        max_capital = self.initial_capital
        if self.portfolio_history:
            max_capital = max(max_capital, max(h.get('total_value', 0.0) for h in self.portfolio_history))
        
        drawdown = (max_capital - portfolio_value) / max_capital if max_capital > 0 else 0.0
        
        portfolio_snapshot = {
            'timestamp': timestamp,
            'total_value': portfolio_value,
            'cash': cash,
            'positions': positions.copy(),
            'position_values': position_values.copy(),
            'drawdown': drawdown
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        return portfolio_value
    
    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """
        Get the portfolio history.
        
        Returns:
            List of portfolio snapshots
        """
        return self.portfolio_history
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get the current positions.
        
        Returns:
            Dictionary mapping symbols to position sizes
        """
        return self.positions
    
    def get_current_portfolio(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.
        
        Returns:
            Dictionary with current portfolio state
        """
        if not self.portfolio_history:
            return {
                'total_value': self.current_capital,
                'cash': self.current_capital,
                'positions': {},
                'position_values': {}
            }
        
        return self.portfolio_history[-1]
