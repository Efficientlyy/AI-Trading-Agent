"""
Trading Engine Package

Contains core components for order management, execution, and portfolio management.
"""

# Import Enums directly from the enums module
from .enums import (
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide
)

# Import Models directly from the models module
from .models import (
    Order,
    Trade,
    Position,
    Portfolio
)

# Import other components
from .order_manager import OrderManager
from .base_agent import BaseTradingAgent
# Attempt to import other known components, handling potential ImportError
try:
    from .execution_handler import ExecutionHandler
except ImportError:
    ExecutionHandler = None # Or raise an error if it's critical

try:
    from .portfolio_manager import PortfolioManager
except ImportError:
    PortfolioManager = None # Or raise an error if it's critical


# Define what gets imported when using 'from ai_trading_agent.trading_engine import *'
__all__ = [
    # Enums
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'PositionSide',
    # Models
    'Order',
    'Trade',
    'Position',
    'Portfolio',
    # Components
    'OrderManager',
    'BaseTradingAgent',
] 

# Conditionally add ExecutionHandler and PortfolioManager to __all__ if they were imported
if ExecutionHandler:
    __all__.append('ExecutionHandler')
if PortfolioManager:
    __all__.append('PortfolioManager')
