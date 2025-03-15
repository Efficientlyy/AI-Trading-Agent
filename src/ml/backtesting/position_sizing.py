"""Position sizing strategies for trading.

This module provides various position sizing algorithms for trading strategies,
including fixed, percent-of-equity, Kelly criterion, and volatility-adjusted methods.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable


class PositionSizer:
    """Base class for position sizing strategies."""
    
    def __init__(self, max_position_size: float = 1.0):
        """
        Initialize the position sizer.
        
        Args:
            max_position_size: Maximum position size as a fraction of available capital (0.0 to 1.0)
        """
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, 
                                equity: float, 
                                price: float, 
                                **kwargs: Any) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            equity: Current equity value
            price: Current price of the asset
            **kwargs: Additional parameters specific to the position sizing method
            
        Returns:
            Position size in units of the asset
        """
        raise NotImplementedError("Subclasses must implement calculate_position_size")


class FixedPositionSizer(PositionSizer):
    """Fixed fraction position sizer.
    
    Allocates a fixed fraction of the maximum position size for each trade.
    """
    
    def __init__(self, fraction: float = 1.0, max_position_size: float = 1.0):
        """
        Initialize the fixed position sizer.
        
        Args:
            fraction: Fraction of the maximum position size to use (0.0 to 1.0)
            max_position_size: Maximum position size as a fraction of available capital (0.0 to 1.0)
        """
        super().__init__(max_position_size)
        self.fraction = min(max(fraction, 0.0), 1.0)  # Ensure fraction is between 0 and 1
    
    def calculate_position_size(self, 
                                equity: float, 
                                price: float, 
                                **kwargs: Any) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            equity: Current equity value
            price: Current price of the asset
            **kwargs: Additional parameters (not used for this sizer)
            
        Returns:
            Position size in units of the asset
        """
        # Calculate units based on fixed fraction of equity
        capital_to_use = equity * self.max_position_size * self.fraction
        return capital_to_use / price if price > 0 else 0.0


class PercentPositionSizer(PositionSizer):
    """Percent of equity position sizer.
    
    Allocates a percentage of equity for each trade.
    """
    
    def __init__(self, percent: float = 0.1, max_position_size: float = 1.0):
        """
        Initialize the percent position sizer.
        
        Args:
            percent: Percentage of equity to use for each trade (0.0 to 1.0)
            max_position_size: Maximum position size as a fraction of available capital (0.0 to 1.0)
        """
        super().__init__(max_position_size)
        self.percent = min(max(percent, 0.0), 1.0)  # Ensure percent is between 0 and 1
    
    def calculate_position_size(self, 
                                equity: float, 
                                price: float, 
                                **kwargs: Any) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            equity: Current equity value
            price: Current price of the asset
            **kwargs: Additional parameters (not used for this sizer)
            
        Returns:
            Position size in units of the asset
        """
        # Calculate units based on percentage of equity
        percent_to_use = min(self.percent, self.max_position_size)
        capital_to_use = equity * percent_to_use
        return capital_to_use / price if price > 0 else 0.0


class KellyPositionSizer(PositionSizer):
    """Kelly criterion position sizer.
    
    Allocates position size based on the Kelly criterion formula.
    """
    
    def __init__(self, 
                 win_rate: Optional[float] = None, 
                 win_loss_ratio: Optional[float] = None,
                 fraction: float = 1.0,  # Kelly fraction (usually < 1.0 to reduce volatility)
                 max_position_size: float = 1.0,
                 lookback_trades: int = 20):
        """
        Initialize the Kelly position sizer.
        
        Args:
            win_rate: Estimated win rate (0.0 to 1.0)
            win_loss_ratio: Ratio of average win to average loss
            fraction: Fraction of Kelly to use (0.0 to 1.0, typically 0.5 for half-Kelly)
            max_position_size: Maximum position size as a fraction of available capital (0.0 to 1.0)
            lookback_trades: Number of past trades to consider for win rate calculation
        """
        super().__init__(max_position_size)
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.fraction = min(max(fraction, 0.0), 1.0)  # Ensure fraction is between 0 and 1
        self.lookback_trades = lookback_trades
    
    def calculate_position_size(self, 
                                equity: float, 
                                price: float, 
                                **kwargs: Any) -> float:
        """
        Calculate the position size for a trade using the Kelly criterion.
        
        Args:
            equity: Current equity value
            price: Current price of the asset
            **kwargs: Additional parameters
                - win_rate: Override the stored win rate
                - win_loss_ratio: Override the stored win/loss ratio
                - trades: List of past trades for calculating win rate and ratio
            
        Returns:
            Position size in units of the asset
        """
        # Get win rate and win/loss ratio from kwargs or use stored values
        win_rate = kwargs.get('win_rate', self.win_rate)
        win_loss_ratio = kwargs.get('win_loss_ratio', self.win_loss_ratio)
        
        # Calculate from past trades if provided and rates not specified
        trades = kwargs.get('trades', None)
        if trades is not None and (win_rate is None or win_loss_ratio is None):
            # Use the most recent trades up to lookback_trades
            recent_trades = trades[-self.lookback_trades:] if len(trades) > self.lookback_trades else trades
            
            if len(recent_trades) > 0:
                # Calculate win rate
                winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
                win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0.5
                
                # Calculate win/loss ratio
                avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 1.0
                losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
                avg_loss = np.mean([-t.get('pnl', 0) for t in losing_trades]) if losing_trades else 1.0
                
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Default values if still None
        win_rate = win_rate if win_rate is not None else 0.5
        win_loss_ratio = win_loss_ratio if win_loss_ratio is not None else 1.0
        
        # Calculate Kelly percentage
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fraction and ensure it's positive and within max_position_size
        kelly_pct = max(0.0, float(kelly_pct * self.fraction))
        kelly_pct = min(float(kelly_pct), self.max_position_size)
        
        # Calculate units based on Kelly percentage of equity
        capital_to_use = equity * kelly_pct
        return capital_to_use / price if price > 0 else 0.0


class VolatilityPositionSizer(PositionSizer):
    """Volatility-adjusted position sizer.
    
    Adjusts position size based on asset volatility to target a specific level of risk.
    """
    
    def __init__(self, 
                 target_risk_pct: float = 0.01,  # Target 1% daily risk
                 volatility_lookback: int = 20,
                 max_position_size: float = 1.0):
        """
        Initialize the volatility position sizer.
        
        Args:
            target_risk_pct: Target percentage risk per trade (e.g., 0.01 = 1%)
            volatility_lookback: Number of periods to look back for volatility calculation
            max_position_size: Maximum position size as a fraction of available capital (0.0 to 1.0)
        """
        super().__init__(max_position_size)
        self.target_risk_pct = target_risk_pct
        self.volatility_lookback = volatility_lookback
    
    def calculate_position_size(self, 
                                equity: float, 
                                price: float, 
                                **kwargs: Any) -> float:
        """
        Calculate the position size based on volatility.
        
        Args:
            equity: Current equity value
            price: Current price of the asset
            **kwargs: Additional parameters
                - volatility: Asset volatility (std dev of returns)
                - returns: Historical returns for calculating volatility if not provided
                - stop_loss_pct: Stop loss percentage (optional)
            
        Returns:
            Position size in units of the asset
        """
        # Get volatility from kwargs or calculate from returns
        volatility = kwargs.get('volatility', None)
        
        if volatility is None:
            returns = kwargs.get('returns', None)
            if returns is not None:
                # Calculate volatility from recent returns
                recent_returns = returns[-self.volatility_lookback:] if len(returns) > self.volatility_lookback else returns
                volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
            else:
                # Default volatility if no data provided
                volatility = 0.01
        
        # Get stop loss percentage
        stop_loss_pct = kwargs.get('stop_loss_pct', None)
        
        # Calculate position size based on target risk and volatility
        if stop_loss_pct is not None and stop_loss_pct > 0:
            # If stop loss is provided, size based on stop distance
            risk_amount = equity * self.target_risk_pct
            position_size = risk_amount / (price * stop_loss_pct)
        else:
            # Otherwise, size based on volatility (N x volatility = risk)
            risk_amount = equity * self.target_risk_pct
            position_size = risk_amount / (price * volatility)
        
        # Apply maximum position size constraint
        max_size = (equity * self.max_position_size) / price
        return min(float(position_size), max_size)


class PositionSizerFactory:
    """Factory for creating position sizers."""
    
    @staticmethod
    def create_position_sizer(method: str, **kwargs: Any) -> PositionSizer:
        """
        Create a position sizer based on the specified method.
        
        Args:
            method: Position sizing method ('fixed', 'percent', 'kelly', 'volatility')
            **kwargs: Additional parameters for the position sizer
            
        Returns:
            Position sizer instance
        
        Raises:
            ValueError: If the method is not supported
        """
        if method == 'fixed':
            fraction = kwargs.get('fraction', 1.0)
            max_position_size = kwargs.get('max_position_size', 1.0)
            return FixedPositionSizer(fraction, max_position_size)
        
        elif method == 'percent':
            percent = kwargs.get('percent', 0.1)
            max_position_size = kwargs.get('max_position_size', 1.0)
            return PercentPositionSizer(percent, max_position_size)
        
        elif method == 'kelly':
            win_rate = kwargs.get('win_rate', None)
            win_loss_ratio = kwargs.get('win_loss_ratio', None)
            fraction = kwargs.get('fraction', 0.5)  # Half-Kelly is more conservative
            max_position_size = kwargs.get('max_position_size', 1.0)
            lookback_trades = kwargs.get('lookback_trades', 20)
            return KellyPositionSizer(win_rate, win_loss_ratio, fraction, max_position_size, lookback_trades)
        
        elif method == 'volatility':
            target_risk_pct = kwargs.get('target_risk_pct', 0.01)
            volatility_lookback = kwargs.get('volatility_lookback', 20)
            max_position_size = kwargs.get('max_position_size', 1.0)
            return VolatilityPositionSizer(target_risk_pct, volatility_lookback, max_position_size)
        
        else:
            raise ValueError(f"Unsupported position sizing method: {method}") 