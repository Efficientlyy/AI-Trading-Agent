"""
Slippage Models for Order Execution

This module implements various slippage models to simulate realistic execution
prices in different market conditions and order sizes.
"""

import logging
import math
import random
from typing import Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np

from ..common.enums import OrderSide

logger = logging.getLogger(__name__)


class SlippageModelType(Enum):
    """Types of slippage models available."""
    FIXED = "fixed"                     # Fixed percentage slippage
    VOLUME_BASED = "volume_based"       # Slippage based on order size relative to volume
    MARKET_IMPACT = "market_impact"     # Square-root market impact model
    VOLATILITY_ADJUSTED = "volatility"  # Volatility-scaled slippage
    CUSTOM = "custom"                   # Custom function-based slippage


class SlippageModel:
    """
    Base class for slippage models.
    
    Slippage models simulate the difference between expected execution price
    and actual fill price due to market conditions, order size, and other factors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the slippage model.
        
        Args:
            config: Configuration parameters for the model
        """
        self.name = "BaseSlippageModel"
        self.config = config
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Optional market data for context
            
        Returns:
            Execution price after slippage
        """
        # Base implementation does not modify price
        return price


class FixedSlippageModel(SlippageModel):
    """
    Fixed percentage slippage model.
    
    Applies a fixed percentage to the price based on order side:
    - Buy orders: price * (1 + slippage_pct)
    - Sell orders: price * (1 - slippage_pct)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fixed slippage model.
        
        Args:
            config: Configuration with slippage_pct parameter
        """
        super().__init__(config)
        self.name = "FixedSlippageModel"
        self.slippage_pct = config.get('slippage_pct', 0.001)  # Default 0.1%
        logger.info(f"Initialized {self.name} with {self.slippage_pct*100:.3f}% slippage")
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage based on fixed percentage.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Not used in this model
            
        Returns:
            Execution price after slippage
        """
        if side == OrderSide.BUY:
            # Buy orders get worse prices (higher)
            return price * (1 + self.slippage_pct)
        else:
            # Sell orders get worse prices (lower)
            return price * (1 - self.slippage_pct)


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model.
    
    Slippage increases with order size relative to market volume:
    slippage = base_slippage * (order_size / avg_volume)^volume_impact
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the volume-based slippage model.
        
        Args:
            config: Configuration with:
                - base_slippage: Base slippage percentage
                - volume_impact: Exponent for volume impact (typically 0.5-1.0)
                - default_volume: Default volume if not provided in market data
        """
        super().__init__(config)
        self.name = "VolumeBasedSlippageModel"
        self.base_slippage = config.get('base_slippage', 0.001)  # 0.1% base
        self.volume_impact = config.get('volume_impact', 0.6)    # Square root-ish
        self.default_volume = config.get('default_volume', 100.0)  # Default volume
        logger.info(f"Initialized {self.name} with base_slippage={self.base_slippage*100:.3f}%, "
                   f"volume_impact={self.volume_impact}")
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage based on order volume.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Should contain 'volume' key with recent trading volume
            
        Returns:
            Execution price after slippage
        """
        # Get market volume from data or use default
        market_volume = self.default_volume
        if market_data and 'volume' in market_data:
            market_volume = market_data['volume']
        
        # Calculate volume ratio and slippage
        volume_ratio = (quantity * price) / market_volume
        slippage_pct = self.base_slippage * (volume_ratio ** self.volume_impact)
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)


class MarketImpactSlippageModel(SlippageModel):
    """
    Square-root market impact model based on academic research.
    
    Uses the formula: slippage = sigma * sqrt(order_size / daily_volume) * market_impact_factor
    Where sigma is market volatility and market_impact_factor is a scaling constant.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market impact slippage model.
        
        Args:
            config: Configuration with:
                - market_impact_factor: Scaling factor for market impact (typically 0.1-1.0)
                - default_volatility: Default volatility if not in market data
                - default_volume: Default volume if not in market data
        """
        super().__init__(config)
        self.name = "MarketImpactSlippageModel"
        self.market_impact_factor = config.get('market_impact_factor', 0.3)
        self.default_volatility = config.get('default_volatility', 0.02)  # 2% daily volatility
        self.default_volume = config.get('default_volume', 100.0)
        logger.info(f"Initialized {self.name} with impact_factor={self.market_impact_factor}")
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage based on market impact model.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Should contain 'volatility' and 'volume' if available
            
        Returns:
            Execution price after slippage
        """
        # Get volatility and volume from market data or use defaults
        volatility = self.default_volatility
        volume = self.default_volume
        
        if market_data:
            if 'volatility' in market_data:
                volatility = market_data['volatility']
            if 'volume' in market_data:
                volume = market_data['volume']
        
        # Calculate order size in base currency
        order_size = quantity * price
        
        # Calculate market impact (square root model)
        # This is a common model in academic literature
        market_impact = volatility * math.sqrt(order_size / volume) * self.market_impact_factor
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            return price * (1 + market_impact)
        else:
            return price * (1 - market_impact)


class VolatilityAdjustedSlippageModel(SlippageModel):
    """
    Volatility-scaled slippage model.
    
    Slippage increases with market volatility:
    slippage = base_slippage * (current_volatility / reference_volatility)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the volatility-adjusted slippage model.
        
        Args:
            config: Configuration with:
                - base_slippage: Base slippage percentage
                - reference_volatility: Reference volatility level
                - max_slippage: Maximum allowed slippage
                - volatility_scaling: How strongly volatility affects slippage
        """
        super().__init__(config)
        self.name = "VolatilityAdjustedSlippageModel"
        self.base_slippage = config.get('base_slippage', 0.001)  # 0.1% base
        self.reference_volatility = config.get('reference_volatility', 0.02)  # 2% reference
        self.max_slippage = config.get('max_slippage', 0.01)  # 1% max slippage
        self.volatility_scaling = config.get('volatility_scaling', 1.0)  # Linear scaling
        logger.info(f"Initialized {self.name} with base={self.base_slippage*100:.3f}%, "
                   f"max={self.max_slippage*100:.3f}%")
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage based on market volatility.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Should contain 'volatility' if available
            
        Returns:
            Execution price after slippage
        """
        # Get volatility from market data or use reference
        volatility = self.reference_volatility
        if market_data and 'volatility' in market_data:
            volatility = market_data['volatility']
        
        # Calculate volatility-adjusted slippage
        volatility_ratio = (volatility / self.reference_volatility) ** self.volatility_scaling
        slippage_pct = min(self.base_slippage * volatility_ratio, self.max_slippage)
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)


class CustomSlippageModel(SlippageModel):
    """
    Custom slippage model that accepts a function for maximum flexibility.
    
    Allows implementing any custom logic for slippage calculation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the custom slippage model.
        
        Args:
            config: Configuration with a slippage_function that takes the same
                   parameters as calculate_slippage and returns an execution price.
        """
        super().__init__(config)
        self.name = "CustomSlippageModel"
        # In a real implementation, this would be a callable
        # For demo, we implement a simple random-walk slippage
        self.min_slippage = config.get('min_slippage', 0.0005)  # 0.05% min
        self.max_slippage = config.get('max_slippage', 0.003)   # 0.3% max
        logger.info(f"Initialized {self.name} with random slippage range "
                   f"[{self.min_slippage*100:.3f}%, {self.max_slippage*100:.3f}%]")
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide,
                          market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate slippage using custom function.
        
        Args:
            price: Current market price
            quantity: Order quantity
            side: Order side (buy/sell)
            market_data: Any market data needed for calculation
            
        Returns:
            Execution price after slippage
        """
        # Demo implementation: random slippage within range
        slippage_pct = self.min_slippage + random.random() * (self.max_slippage - self.min_slippage)
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)


def create_slippage_model(model_type: SlippageModelType, config: Dict[str, Any]) -> SlippageModel:
    """
    Factory function to create slippage models.
    
    Args:
        model_type: Type of slippage model to create
        config: Configuration dictionary for the model
        
    Returns:
        Configured slippage model instance
    """
    if model_type == SlippageModelType.FIXED:
        return FixedSlippageModel(config)
    elif model_type == SlippageModelType.VOLUME_BASED:
        return VolumeBasedSlippageModel(config)
    elif model_type == SlippageModelType.MARKET_IMPACT:
        return MarketImpactSlippageModel(config)
    elif model_type == SlippageModelType.VOLATILITY_ADJUSTED:
        return VolatilityAdjustedSlippageModel(config)
    elif model_type == SlippageModelType.CUSTOM:
        return CustomSlippageModel(config)
    else:
        # Default to fixed slippage
        logger.warning(f"Unknown slippage model type: {model_type}, defaulting to FIXED")
        return FixedSlippageModel(config)


# Example usage:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test all slippage models
    models = [
        create_slippage_model(SlippageModelType.FIXED, {'slippage_pct': 0.001}),
        create_slippage_model(SlippageModelType.VOLUME_BASED, {'base_slippage': 0.001, 'volume_impact': 0.6}),
        create_slippage_model(SlippageModelType.MARKET_IMPACT, {'market_impact_factor': 0.3}),
        create_slippage_model(SlippageModelType.VOLATILITY_ADJUSTED, {'base_slippage': 0.001, 'max_slippage': 0.01}),
        create_slippage_model(SlippageModelType.CUSTOM, {'min_slippage': 0.0005, 'max_slippage': 0.003})
    ]
    
    # Test data
    price = 50000.0  # BTC price in USD
    quantity = 0.1  # 0.1 BTC
    side = OrderSide.BUY
    market_data = {
        'volume': 1000.0,  # $1M volume
        'volatility': 0.03  # 3% volatility
    }
    
    # Print results
    print(f"Testing slippage models with price={price}, quantity={quantity}, side={side.value}")
    print(f"Market data: {market_data}")
    print("-" * 60)
    
    for model in models:
        execution_price = model.calculate_slippage(price, quantity, side, market_data)
        slippage_amount = execution_price - price if side == OrderSide.BUY else price - execution_price
        slippage_pct = slippage_amount / price * 100
        
        print(f"{model.name}:")
        print(f"  Execution price: ${execution_price:.2f}")
        print(f"  Slippage: ${slippage_amount:.2f} ({slippage_pct:.4f}%)")
        print("-" * 60)
