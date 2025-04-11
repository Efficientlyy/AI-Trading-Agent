"""
Trading Engine Package

Contains core components for order management, execution, and agent definition.
"""

from .models import (
    Order,
    Trade,
    Position,
    Portfolio,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide
)
from .order_manager import OrderManager
from .base_agent import BaseTradingAgent

# Potentially add ExecutionEngine or Matcher later

__all__ = [
    # Models
    'Order',
    'Trade',
    'Position',
    'Portfolio',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'PositionSide',
    # Components
    'OrderManager',
    'BaseTradingAgent',
]
