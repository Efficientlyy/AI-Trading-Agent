"""
Defines the abstract base class for all trading agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal
import pandas as pd
from datetime import datetime

# Fix imports to not reference 'src' directly
from ..data_acquisition.data_service import DataService
from .order_manager import OrderManager
from .models import Order, Trade, Portfolio
from ..common import logger

class BaseTradingAgent(ABC):
    """Abstract Base Class for trading agent implementations."""

    def __init__(self, data_service: DataService, portfolio: Portfolio, config: Dict[str, Any]):
        """
        Initializes the BaseTradingAgent.

        Args:
            data_service: Instance of DataService for market data access.
            portfolio: Instance of the Portfolio model holding account state.
            config: Agent-specific configuration dictionary.
        """
        self.data_service = data_service
        self.portfolio = portfolio
        self.order_manager = OrderManager(portfolio) # Each agent gets its own OrderManager instance linked to the portfolio
        self.config = config
        self.agent_name = self.config.get('name', self.__class__.__name__)
        self.is_initialized = False
        self.is_running = False
        logger.info(f"Initializing agent: {self.agent_name}")

    @abstractmethod
    async def initialize(self):
        """
        Perform any initialization required before the agent starts trading.
        E.g., loading historical data, pre-calculating indicators, subscribing to streams.
        """
        logger.info(f"[{self.agent_name}] Initializing...")
        # Implementation specific initialization
        self.is_initialized = True
        pass

    @abstractmethod
    async def on_data(self, market_data: Dict[str, pd.DataFrame]):
        """
        Called when new market data (e.g., OHLCV bars, ticks) is available.
        This is the primary method where trading logic resides.

        Args:
            market_data: A dictionary where keys are symbols and values are
                         pandas DataFrames containing the latest market data.
                         The structure of the DataFrame depends on the DataService.
        """
        if not self.is_initialized or not self.is_running:
            return
        # Implementation specific trading logic
        pass

    @abstractmethod
    async def on_order_update(self, order: Order):
        """
        Called when there is an update to one of the agent's orders.
        E.g., order filled, partially filled, canceled, rejected.

        Args:
            order: The updated Order object.
        """
        if not self.is_initialized or not self.is_running:
            return
        logger.info(f"[{self.agent_name}] Received order update: {order.order_id} Status: {order.status} Filled: {order.filled_quantity}/{order.quantity}")
        # Implementation specific logic based on order status
        pass

    @abstractmethod
    async def on_trade(self, trade: Trade):
        """
        Called when one of the agent's orders results in a trade (fill).

        Args:
            trade: The Trade object representing the fill.
        """
        if not self.is_initialized or not self.is_running:
            return
        logger.info(f"[{self.agent_name}] Received trade: {trade.trade_id} for Order {trade.order_id}: {trade.side} {trade.quantity} {trade.symbol} @ {trade.price}")
        # Implementation specific logic based on trade execution
        # Note: Portfolio update based on trade often happens in the main loop/execution engine
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Perform any cleanup operations before the agent stops.
        E.g., canceling open orders, saving state.
        """
        logger.info(f"[{self.agent_name}] Shutting down...")
        self.is_running = False
        # Implementation specific cleanup
        pass

    async def start(self):
        """
        Starts the agent's operation after initialization.
        """
        if not self.is_initialized:
            logger.warning(f"[{self.agent_name}] Agent not initialized. Call initialize() first.")
            return
        logger.info(f"[{self.agent_name}] Starting...")
        self.is_running = True
        # Potentially start internal loops or background tasks if needed

    async def stop(self):
        """
        Stops the agent's operation and calls shutdown.
        """
        logger.info(f"[{self.agent_name}] Stopping...")
        await self.shutdown()

    # --- Helper methods for placing orders --- 
    async def create_market_order(self, symbol: str, side: Literal['buy', 'sell'], quantity: float) -> Optional[Order]:
        """Helper method to create a market order."""
        if not self.is_running:
            logger.warning(f"[{self.agent_name}] Agent not running. Cannot create order.")
            return None
        logger.info(f"[{self.agent_name}] Attempting to create MARKET order: {side} {quantity} {symbol}")
        return self.order_manager.create_order(symbol, side, 'market', quantity)

    async def create_limit_order(self, symbol: str, side: Literal['buy', 'sell'], quantity: float, price: float) -> Optional[Order]:
        """Helper method to create a limit order."""
        if not self.is_running:
            logger.warning(f"[{self.agent_name}] Agent not running. Cannot create order.")
            return None
        logger.info(f"[{self.agent_name}] Attempting to create LIMIT order: {side} {quantity} {symbol} @ {price}")
        return self.order_manager.create_order(symbol, side, 'limit', quantity, price=price)

    async def cancel_order(self, order_id: str) -> bool:
        """Helper method to cancel an order."""
        if not self.is_running:
            logger.warning(f"[{self.agent_name}] Agent not running. Cannot cancel order.")
            return False
        logger.info(f"[{self.agent_name}] Attempting to cancel order: {order_id}")
        # In a real system, this would likely return a confirmation object or task
        return self.order_manager.cancel_order(order_id)
