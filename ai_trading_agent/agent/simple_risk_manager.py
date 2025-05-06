"""
Simple risk manager module for the AI Trading Agent.

This module provides a simple risk manager implementation that handles
position sizing, stop-loss management, and risk constraints.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..common import logger


class SimpleRiskManager:
    """
    Simple risk manager implementation with position sizing and stop-loss management.
    """
    
    def __init__(self, name: str = "RiskManager",
                max_position_size: float = 0.2,
                max_risk_per_trade: float = 0.02,
                stop_loss_atr_multiple: float = 2.0):
        """
        Initialize the simple risk manager.
        
        Args:
            name: Name of the risk manager
            max_position_size: Maximum position size as fraction of portfolio
            max_risk_per_trade: Maximum risk per trade as fraction of portfolio
            stop_loss_atr_multiple: Stop loss as multiple of ATR
        """
        self.name = name
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.stop_losses = {}  # Dictionary to store stop-loss levels for each position
        
        logger.info(f"Initialized {name} with max_position_size={max_position_size*100}%, "
                   f"max_risk_per_trade={max_risk_per_trade*100}%, "
                   f"stop_loss_atr_multiple={stop_loss_atr_multiple}")
    
    def apply_risk_constraints(self, signals: Dict[str, int], 
                              current_portfolio: Dict[str, Any],
                              market_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Apply risk constraints to trading signals.
        
        Args:
            signals: Dictionary mapping symbols to signal values (-1, 0, 1)
            current_portfolio: Current portfolio state
            market_data: Current market data
        
        Returns:
            Dictionary mapping symbols to risk-adjusted signal values
        """
        # Check stop-losses first
        stop_loss_signals = self.generate_stop_loss_signals(current_portfolio, market_data)
        
        # Merge stop-loss signals with strategy signals (stop-loss takes precedence)
        risk_adjusted_signals = signals.copy()
        for symbol, signal in stop_loss_signals.items():
            if signal != 0:  # If stop-loss is triggered
                risk_adjusted_signals[symbol] = signal
                logger.info(f"Stop-loss signal for {symbol}: {signal} overrides strategy signal: {signals.get(symbol, 0)}")
        
        # Apply portfolio-level risk constraints
        total_portfolio_value = current_portfolio.get('total_value', 0)
        if total_portfolio_value <= 0:
            logger.warning("Invalid portfolio value, cannot apply risk constraints")
            return risk_adjusted_signals
        
        # Calculate current exposure
        current_positions = current_portfolio.get('positions', {})
        current_position_values = current_portfolio.get('position_values', {})
        
        current_exposure = sum(abs(value) for value in current_position_values.values()) / total_portfolio_value if current_position_values else 0
        
        # If current exposure is already at maximum, don't allow new positions
        if current_exposure >= 1.0:
            logger.warning("Maximum portfolio exposure reached, no new positions allowed")
            # Only allow closing positions
            for symbol, signal in list(risk_adjusted_signals.items()):
                current_position = current_positions.get(symbol, 0)
                if (signal > 0 and current_position >= 0) or (signal < 0 and current_position <= 0):
                    risk_adjusted_signals[symbol] = 0
                    logger.info(f"Signal for {symbol} removed due to maximum exposure")
        
        return risk_adjusted_signals
    
    def generate_stop_loss_signals(self, portfolio_state: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Check open positions against stop-loss criteria and generate signals to close them.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
        
        Returns:
            Dictionary mapping symbols to stop-loss signals
        """
        signals = {}
        positions = portfolio_state.get('positions', {})
        
        for symbol, quantity in positions.items():
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}, cannot check stop-loss")
                continue
            
            if quantity == 0:
                continue
            
            # Get current price
            current_data = market_data[symbol]
            if isinstance(current_data, dict):
                current_price = current_data.get('close', 0)
            elif isinstance(current_data, pd.Series):
                current_price = current_data.get('close', 0)
            else:
                current_price = float(current_data)
            
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}, cannot check stop-loss")
                continue
            
            # Check if we have a stop-loss level for this symbol
            if symbol in self.stop_losses:
                stop_loss_level = self.stop_losses[symbol]
                
                # Check stop-loss
                if quantity > 0:  # Long position
                    if current_price <= stop_loss_level:
                        logger.info(f"STOP-LOSS triggered for LONG {symbol}: Price {current_price:.4f} <= Stop Level {stop_loss_level:.4f}")
                        signals[symbol] = -1  # Signal to sell
                elif quantity < 0:  # Short position
                    if current_price >= stop_loss_level:
                        logger.info(f"STOP-LOSS triggered for SHORT {symbol}: Price {current_price:.4f} >= Stop Level {stop_loss_level:.4f}")
                        signals[symbol] = 1  # Signal to buy-to-cover
            
        return signals
    
    def calculate_atr(self, symbol: str, historical_data: Dict[str, pd.DataFrame], 
                     period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for a symbol.
        
        Args:
            symbol: Trading symbol
            historical_data: Dictionary mapping symbols to historical data
            period: ATR period
        
        Returns:
            ATR value
        """
        if symbol not in historical_data:
            logger.warning(f"No historical data for {symbol}, cannot calculate ATR")
            return 0.0
        
        data = historical_data[symbol]
        
        if len(data) < period + 1:
            logger.warning(f"Insufficient historical data for {symbol}, cannot calculate ATR")
            return 0.0
        
        # Calculate true range
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Handle the case where we don't have enough previous data
        if len(high) <= 1 or len(low) <= 1 or len(close) <= 1:
            # Use a default ATR based on price volatility
            return close[-1] * 0.02  # Default to 2% of price
        
        # Calculate true range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR
        atr = np.mean(true_range[-period:])
        
        return atr
    
    def set_stop_loss(self, symbol: str, entry_price: float, position_type: str, 
                     atr: Optional[float] = None) -> float:
        """
        Set stop-loss level for a position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: Position type ('long' or 'short')
            atr: ATR value (if None, use default)
        
        Returns:
            Stop-loss level
        """
        if atr is None or atr <= 0:
            atr = entry_price * 0.02  # Default to 2% of price
        
        if position_type.lower() == 'long':
            stop_loss_level = entry_price - (atr * self.stop_loss_atr_multiple)
        else:  # short
            stop_loss_level = entry_price + (atr * self.stop_loss_atr_multiple)
        
        self.stop_losses[symbol] = stop_loss_level
        
        logger.info(f"Set stop-loss for {symbol} {position_type}: Entry {entry_price:.4f}, Stop {stop_loss_level:.4f}")
        
        return stop_loss_level
    
    def get_position_size(self, symbol: str, signal: int, entry_price: float, 
                         stop_loss: float, portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            signal: Signal value (-1, 0, 1)
            entry_price: Entry price
            stop_loss: Stop-loss level
            portfolio_value: Portfolio value
        
        Returns:
            Position size as fraction of portfolio
        """
        if signal == 0 or entry_price <= 0 or portfolio_value <= 0:
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            logger.warning(f"Invalid risk per unit for {symbol}, using default")
            risk_per_unit = entry_price * 0.02  # Default to 2% of price
        
        # Calculate risk amount
        risk_amount = portfolio_value * self.max_risk_per_trade
        
        # Calculate position size based on risk
        position_size = risk_amount / risk_per_unit
        
        # Apply maximum position size constraint
        max_position = portfolio_value * self.max_position_size
        position_size = min(position_size, max_position)
        
        # Convert to fraction of portfolio
        position_fraction = position_size / portfolio_value
        
        logger.info(f"Calculated position size for {symbol}: {position_fraction:.2%} of portfolio")
        
        return position_fraction
