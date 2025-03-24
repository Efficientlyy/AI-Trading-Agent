"""Risk management strategies for trading.

This module provides various risk management strategies for trading,
including stop loss, take profit, trailing stops, and time-based exits.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
from enum import Enum


class ExitType(Enum):
    """Types of position exits."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    SIGNAL_EXIT = "signal_exit"
    CLOSE_ALL = "close_all"


class RiskManager:
    """Base class for risk management strategies."""
    
    def __init__(self):
        """Initialize the risk manager."""
        pass
    
    def should_exit(self, 
                    position_data: Dict[str, Any], 
                    current_price: float, 
                    current_time: datetime,
                    **kwargs: Any) -> Tuple[bool, Optional[ExitType]]:
        """
        Determine if a position should be exited.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_exit, exit_type)
        """
        raise NotImplementedError("Subclasses must implement should_exit")
    
    def update_stop_levels(self, 
                           position_data: Dict[str, Any], 
                           current_price: float, 
                           current_time: datetime,
                           **kwargs: Any) -> Dict[str, Any]:
        """
        Update stop loss and take profit levels.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters
            
        Returns:
            Updated position data
        """
        return position_data


class BasicRiskManager(RiskManager):
    """Basic risk manager with fixed stop loss and take profit levels."""
    
    def __init__(self, 
                 stop_loss_pct: Optional[float] = None, 
                 take_profit_pct: Optional[float] = None,
                 max_holding_days: Optional[int] = None):
        """
        Initialize the basic risk manager.
        
        Args:
            stop_loss_pct: Stop loss percentage (e.g., 0.05 = 5%)
            take_profit_pct: Take profit percentage (e.g., 0.1 = 10%)
            max_holding_days: Maximum number of days to hold a position
        """
        super().__init__()
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
    
    def should_exit(self, 
                    position_data: Dict[str, Any], 
                    current_price: float, 
                    current_time: datetime,
                    **kwargs: Any) -> Tuple[bool, Optional[ExitType]]:
        """
        Determine if a position should be exited based on stop loss, take profit, or time.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_exit, exit_type)
        """
        # Extract position information
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', current_time)
        position_type = position_data.get('type', 'long')  # 'long' or 'short'
        
        # Check stop loss
        if self.stop_loss_pct is not None:
            if position_type == 'long' and current_price <= entry_price * (1 - self.stop_loss_pct):
                return True, ExitType.STOP_LOSS
            elif position_type == 'short' and current_price >= entry_price * (1 + self.stop_loss_pct):
                return True, ExitType.STOP_LOSS
        
        # Check take profit
        if self.take_profit_pct is not None:
            if position_type == 'long' and current_price >= entry_price * (1 + self.take_profit_pct):
                return True, ExitType.TAKE_PROFIT
            elif position_type == 'short' and current_price <= entry_price * (1 - self.take_profit_pct):
                return True, ExitType.TAKE_PROFIT
        
        # Check time stop
        if self.max_holding_days is not None:
            max_exit_time = entry_time + timedelta(days=self.max_holding_days)
            if current_time >= max_exit_time:
                return True, ExitType.TIME_STOP
        
        # Check for signal exit
        signal_exit = kwargs.get('signal_exit', False)
        if signal_exit:
            return True, ExitType.SIGNAL_EXIT
        
        # Check for force close
        force_close = kwargs.get('force_close', False)
        if force_close:
            return True, ExitType.CLOSE_ALL
        
        return False, None


class TrailingStopManager(RiskManager):
    """Risk manager with trailing stop loss."""
    
    def __init__(self, 
                 initial_stop_pct: float = 0.05, 
                 trailing_pct: float = 0.02,
                 take_profit_pct: Optional[float] = None,
                 max_holding_days: Optional[int] = None):
        """
        Initialize the trailing stop manager.
        
        Args:
            initial_stop_pct: Initial stop loss percentage (e.g., 0.05 = 5%)
            trailing_pct: Trailing stop percentage (e.g., 0.02 = 2%)
            take_profit_pct: Take profit percentage (e.g., 0.1 = 10%)
            max_holding_days: Maximum number of days to hold a position
        """
        super().__init__()
        self.initial_stop_pct = initial_stop_pct
        self.trailing_pct = trailing_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
    
    def should_exit(self, 
                    position_data: Dict[str, Any], 
                    current_price: float, 
                    current_time: datetime,
                    **kwargs: Any) -> Tuple[bool, Optional[ExitType]]:
        """
        Determine if a position should be exited based on trailing stop, take profit, or time.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_exit, exit_type)
        """
        # Extract position information
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', current_time)
        position_type = position_data.get('type', 'long')  # 'long' or 'short'
        
        # Get current stop level
        stop_level = position_data.get('stop_level')
        
        # Check trailing stop
        if stop_level is not None:
            if position_type == 'long' and current_price <= stop_level:
                return True, ExitType.TRAILING_STOP
            elif position_type == 'short' and current_price >= stop_level:
                return True, ExitType.TRAILING_STOP
        
        # Check take profit
        if self.take_profit_pct is not None:
            if position_type == 'long' and current_price >= entry_price * (1 + self.take_profit_pct):
                return True, ExitType.TAKE_PROFIT
            elif position_type == 'short' and current_price <= entry_price * (1 - self.take_profit_pct):
                return True, ExitType.TAKE_PROFIT
        
        # Check time stop
        if self.max_holding_days is not None:
            max_exit_time = entry_time + timedelta(days=self.max_holding_days)
            if current_time >= max_exit_time:
                return True, ExitType.TIME_STOP
        
        # Check for signal exit
        signal_exit = kwargs.get('signal_exit', False)
        if signal_exit:
            return True, ExitType.SIGNAL_EXIT
        
        # Check for force close
        force_close = kwargs.get('force_close', False)
        if force_close:
            return True, ExitType.CLOSE_ALL
        
        return False, None
    
    def update_stop_levels(self, 
                           position_data: Dict[str, Any], 
                           current_price: float, 
                           current_time: datetime,
                           **kwargs: Any) -> Dict[str, Any]:
        """
        Update trailing stop level based on current price.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters
            
        Returns:
            Updated position data
        """
        # Make a copy of the position data to avoid modifying the original
        updated_data = position_data.copy()
        
        # Extract position information
        entry_price = updated_data.get('entry_price', current_price)
        position_type = updated_data.get('type', 'long')  # 'long' or 'short'
        
        # Get current stop level and highest/lowest seen price
        stop_level = updated_data.get('stop_level')
        highest_price = updated_data.get('highest_price', entry_price)
        lowest_price = updated_data.get('lowest_price', entry_price)
        
        # Initialize stop level if not set
        if stop_level is None:
            if position_type == 'long':
                stop_level = entry_price * (1 - self.initial_stop_pct)
            else:  # short
                stop_level = entry_price * (1 + self.initial_stop_pct)
            updated_data["stop_level"] = stop_level
        
        # Update highest/lowest seen price
        if position_type == 'long':
            if current_price > highest_price:
                highest_price = current_price
                # Update trailing stop
                new_stop = highest_price * (1 - self.trailing_pct)
                if new_stop > stop_level:
                    updated_data["stop_level"] = new_stop
            updated_data["highest_price"] = highest_price
        else:  # short
            if current_price < lowest_price:
                lowest_price = current_price
                # Update trailing stop
                new_stop = lowest_price * (1 + self.trailing_pct)
                if new_stop < stop_level:
                    updated_data["stop_level"] = new_stop
            updated_data["lowest_price"] = lowest_price
        
        return updated_data


class VolatilityBasedRiskManager(RiskManager):
    """Risk manager with volatility-based stop loss using Average True Range (ATR)."""
    
    def __init__(self, 
                 atr_multiplier: float = 2.0,
                 lookback_periods: int = 14,
                 take_profit_atr_multiplier: Optional[float] = None,
                 max_holding_days: Optional[int] = None):
        """
        Initialize the volatility-based risk manager.
        
        Args:
            atr_multiplier: Multiplier for ATR to set stop loss
            lookback_periods: Number of periods to look back for ATR calculation
            take_profit_atr_multiplier: Multiplier for ATR to set take profit (optional)
            max_holding_days: Maximum number of days to hold a position
        """
        super().__init__()
        self.atr_multiplier = atr_multiplier
        self.lookback_periods = lookback_periods
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.max_holding_days = max_holding_days
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """
        Calculate Average True Range (ATR).
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            
        Returns:
            ATR value
        """
        if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
            return 0.0
        
        # Calculate True Range
        tr1 = np.abs(highs[1:] - lows[1:])  # Current high - current low
        tr2 = np.abs(highs[1:] - closes[:-1])  # Current high - previous close
        tr3 = np.abs(lows[1:] - closes[:-1])  # Current low - previous close
        
        # True Range is the maximum of the three
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR (simple moving average of TR)
        periods = min(len(tr), self.lookback_periods)
        atr = np.mean(tr[-periods:])
        
        return float(atr)
    
    def should_exit(self, 
                    position_data: Dict[str, Any], 
                    current_price: float, 
                    current_time: datetime,
                    **kwargs: Any) -> Tuple[bool, Optional[ExitType]]:
        """
        Determine if a position should be exited based on ATR-based stop loss, take profit, or time.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters including price history for ATR calculation
            
        Returns:
            Tuple of (should_exit, exit_type)
        """
        # Extract position information
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', current_time)
        position_type = position_data.get('type', 'long')  # 'long' or 'short'
        
        # Get stop loss and take profit levels
        stop_loss = position_data.get('stop_loss', None)
        take_profit = position_data.get('take_profit', None)
        
        # Check stop loss
        if stop_loss is not None:
            if position_type == 'long' and current_price <= stop_loss:
                return True, ExitType.STOP_LOSS
            elif position_type == 'short' and current_price >= stop_loss:
                return True, ExitType.STOP_LOSS
        
        # Check take profit
        if take_profit is not None:
            if position_type == 'long' and current_price >= take_profit:
                return True, ExitType.TAKE_PROFIT
            elif position_type == 'short' and current_price <= take_profit:
                return True, ExitType.TAKE_PROFIT
        
        # Check max holding time
        if self.max_holding_days is not None:
            max_exit_time = entry_time + timedelta(days=self.max_holding_days)
            if current_time >= max_exit_time:
                return True, ExitType.TIME_STOP
        
        # Check for signal exit
        signal_exit = kwargs.get('signal_exit', False)
        if signal_exit:
            return True, ExitType.SIGNAL_EXIT
        
        # Check for close all positions
        close_all = kwargs.get('close_all', False)
        if close_all:
            return True, ExitType.CLOSE_ALL
        
        return False, None
    
    def update_stop_levels(self, 
                           position_data: Dict[str, Any], 
                           current_price: float, 
                           current_time: datetime,
                           **kwargs: Any) -> Dict[str, Any]:
        """
        Update stop loss and take profit levels based on ATR.
        
        Args:
            position_data: Dictionary containing position information
            current_price: Current price of the asset
            current_time: Current time
            **kwargs: Additional parameters including price history for ATR calculation
            
        Returns:
            Updated position data
        """
        # Get price history for ATR calculation
        highs = kwargs.get('highs', None)
        lows = kwargs.get('lows', None)
        closes = kwargs.get('closes', None)
        
        # If we don't have price history, return unchanged
        if highs is None or lows is None or closes is None:
            return position_data
        
        # Calculate ATR
        atr = self.calculate_atr(highs, lows, closes)
        
        # Extract position information
        entry_price = position_data.get('entry_price', current_price)
        position_type = position_data.get('type', 'long')  # 'long' or 'short'
        
        # Update stop loss if not already set
        if 'stop_loss' not in position_data:
            if position_type == 'long':
                stop_loss = entry_price - (atr * self.atr_multiplier)
            else:  # short
                stop_loss = entry_price + (atr * self.atr_multiplier)
            
            position_data["stop_loss"] = stop_loss
        
        # Update take profit if not already set and take_profit_atr_multiplier is specified
        if 'take_profit' not in position_data and self.take_profit_atr_multiplier is not None:
            if position_type == 'long':
                take_profit = entry_price + (atr * self.take_profit_atr_multiplier)
            else:  # short
                take_profit = entry_price - (atr * self.take_profit_atr_multiplier)
            
            position_data["take_profit"] = take_profit
        
        return position_data


class RiskManagerFactory:
    """Factory for creating risk managers."""
    
    @staticmethod
    def create_risk_manager(method: str, **kwargs: Any) -> RiskManager:
        """
        Create a risk manager based on the specified method.
        
        Args:
            method: Risk management method ('basic', 'trailing_stop', 'volatility')
            **kwargs: Additional parameters for the risk manager
            
        Returns:
            Risk manager instance
        
        Raises:
            ValueError: If the method is not supported
        """
        if method == 'basic':
            stop_loss_pct = kwargs.get('stop_loss_pct', None)
            take_profit_pct = kwargs.get('take_profit_pct', None)
            max_holding_days = kwargs.get('max_holding_days', None)
            return BasicRiskManager(stop_loss_pct, take_profit_pct, max_holding_days)
        
        elif method == 'trailing_stop':
            initial_stop_pct = kwargs.get('initial_stop_pct', 0.05)
            trailing_pct = kwargs.get('trailing_pct', 0.02)
            take_profit_pct = kwargs.get('take_profit_pct', None)
            max_holding_days = kwargs.get('max_holding_days', None)
            return TrailingStopManager(initial_stop_pct, trailing_pct, take_profit_pct, max_holding_days)
        
        elif method == 'volatility':
            atr_multiplier = kwargs.get('atr_multiplier', 2.0)
            lookback_periods = kwargs.get('lookback_periods', 14)
            take_profit_atr_multiplier = kwargs.get('take_profit_atr_multiplier', None)
            max_holding_days = kwargs.get('max_holding_days', None)
            return VolatilityBasedRiskManager(atr_multiplier, lookback_periods, take_profit_atr_multiplier, max_holding_days)
        
        else:
            raise ValueError(f"Unsupported risk management method: {method}") 