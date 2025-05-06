"""
Execution handler module for the AI Trading Agent.

This module handles order execution, either simulated or live.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd 
import numpy as np
from datetime import datetime
import uuid

from ..common import logger


class Order:
    """Simple Order class for representing trading orders."""
    
    def __init__(self, symbol: str, quantity: float, order_type: str = 'MARKET',
                limit_price: Optional[float] = None, stop_price: Optional[float] = None):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (positive for buy, negative for sell)
            order_type: Order type ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
        """
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.status = 'PENDING'
        self.created_timestamp = datetime.now()
        self.executed_timestamp = None
        self.executed_price = None
        self.executed_quantity = None
        self.commission = 0.0


class ExecutionHandlerABC(ABC):
    """
    Abstract base class for execution handlers.
    """
    
    def __init__(self, name: str):
        """
        Initialize the execution handler.
        
        Args:
            name: Name of the execution handler
        """
        self.name = name
    
    @abstractmethod
    def execute_order(self, order: Order, market_data: Dict[str, Any]) -> Order:
        """
        Execute an order based on market data.
        
        Args:
            order: Order to execute
            market_data: Current market data
        
        Returns:
            Updated order with execution details
        """
        raise NotImplementedError("Subclasses must implement execute_order")
    
    @abstractmethod
    def submit_orders(self, orders: List[Order], market_data: Dict[str, Any]) -> List[Order]:
        """
        Submit multiple orders for execution.
        
        Args:
            orders: List of orders to execute
            market_data: Current market data
        
        Returns:
            List of updated orders with execution details
        """
        raise NotImplementedError("Subclasses must implement submit_orders")


class SimulatedExecutionHandler(ExecutionHandlerABC):
    """
    Simulated execution handler for backtesting and paper trading.
    """
    
    def __init__(self, name: str = "SimulatedExecution", 
                slippage: float = 0.001,
                commission: float = 0.001):
        """
        Initialize the simulated execution handler.
        
        Args:
            name: Name of the execution handler
            slippage: Slippage as fraction of price
            commission: Commission as fraction of trade value
        """
        super().__init__(name)
        self.slippage = slippage
        self.commission = commission
        self.orders = []  # Historical orders
        
        logger.info(f"Initialized {self.name} with slippage={slippage}, commission={commission}")
    
    def execute_order(self, order: Order, market_data: Dict[str, Any]) -> Order:
        """
        Execute an order based on market data.
        
        Args:
            order: Order to execute
            market_data: Current market data
        
        Returns:
            Updated order with execution details
        """
        symbol = order.symbol
        
        # Check if we have market data for this symbol
        if symbol not in market_data:
            logger.warning(f"No market data available for {symbol}. Order {order.order_id} remains pending.")
            return order
        
        # Get current price
        current_data = market_data[symbol]
        if isinstance(current_data, pd.Series):
            # If market data is a pandas Series
            if 'close' in current_data:
                execution_price = current_data['close']
            else:
                # Use the first numeric value in the series
                for col in current_data.index:
                    if isinstance(current_data[col], (int, float)) and not pd.isna(current_data[col]):
                        execution_price = current_data[col]
                        break
                else:
                    logger.warning(f"No suitable price found in market data for {symbol}. Order {order.order_id} remains pending.")
                    return order
        elif isinstance(current_data, dict):
            # If market data is a dictionary
            if 'close' in current_data:
                execution_price = current_data['close']
            elif 'price' in current_data:
                execution_price = current_data['price']
            else:
                logger.warning(f"No suitable price found in market data for {symbol}. Order {order.order_id} remains pending.")
                return order
        else:
            # If market data is a scalar
            execution_price = float(current_data)
        
        # Apply slippage
        if order.quantity > 0:  # Buy order
            execution_price *= (1 + self.slippage)
        else:  # Sell order
            execution_price *= (1 - self.slippage)
        
        # Calculate commission
        commission = abs(order.quantity * execution_price * self.commission)
        
        # Update order
        order.status = 'FILLED'
        order.executed_price = execution_price
        order.executed_quantity = order.quantity
        order.executed_timestamp = datetime.now()
        order.commission = commission
        
        # Add to order history
        self.orders.append(order)
        
        logger.info(f"Executed order {order.order_id}: {order.quantity} {symbol} @ {execution_price:.4f} (Commission: {commission:.2f})")
        
        return order
    
    def submit_orders(self, orders: List[Order], market_data: Dict[str, Any]) -> List[Order]:
        """
        Submit multiple orders for execution.
        
        Args:
            orders: List of orders to execute
            market_data: Current market data
        
        Returns:
            List of updated orders with execution details
        """
        executed_orders = []
        
        for order in orders:
            executed_order = self.execute_order(order, market_data)
            executed_orders.append(executed_order)
        
        return executed_orders
    
    def get_order_history(self) -> List[Order]:
        """
        Get the order history.
        
        Returns:
            List of historical orders
        """
        return self.orders


class LiveExecutionHandler(ExecutionHandlerABC):
    """
    Live execution handler for real trading.
    """
    
    def __init__(self, name: str = "LiveExecution", 
                api_config: Dict[str, Any] = None,
                dry_run: bool = True):
        """
        Initialize the live execution handler.
        
        Args:
            name: Name of the execution handler
            api_config: API configuration for the exchange
            dry_run: If True, orders will not be sent to the exchange
        """
        super().__init__(name)
        self.api_config = api_config or {}
        self.dry_run = dry_run
        self.orders = []  # Historical orders
        self.exchange = None  # Exchange API client
        
        logger.info(f"Initialized {self.name} with dry_run={dry_run}")
        
        # Initialize exchange client if not in dry run mode
        if not dry_run:
            self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the exchange API client."""
        try:
            import ccxt
            
            exchange_id = self.api_config.get('name', 'binance')
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_config.get('api_key', ''),
                'secret': self.api_config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Use testnet if specified
            if self.api_config.get('testnet', False):
                self.exchange.set_sandbox_mode(True)
            
            logger.info(f"Connected to exchange: {exchange_id}")
            
        except ImportError:
            logger.error("ccxt library not installed. Please install it with 'pip install ccxt'.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def execute_order(self, order: Order, market_data: Dict[str, Any]) -> Order:
        """
        Execute an order based on market data.
        
        Args:
            order: Order to execute
            market_data: Current market data
        
        Returns:
            Updated order with execution details
        """
        if self.dry_run:
            # Simulate execution in dry run mode
            logger.info(f"[DRY RUN] Would execute order {order.order_id}: {order.quantity} {order.symbol}")
            
            # Use simulated execution
            sim_handler = SimulatedExecutionHandler()
            return sim_handler.execute_order(order, market_data)
        
        if self.exchange is None:
            logger.error("Exchange client not initialized")
            order.status = 'REJECTED'
            return order
        
        try:
            symbol = order.symbol
            side = 'buy' if order.quantity > 0 else 'sell'
            amount = abs(order.quantity)
            
            # Determine order type and parameters
            order_type = order.order_type.lower()
            params = {}
            
            if order_type == 'limit':
                price = order.limit_price
            elif order_type == 'stop':
                order_type = 'market'
                params['stopPrice'] = order.stop_price
            elif order_type == 'stop_limit':
                order_type = 'limit'
                price = order.limit_price
                params['stopPrice'] = order.stop_price
            else:
                # Default to market order
                order_type = 'market'
                price = None
            
            # Execute order
            logger.info(f"Executing order {order.order_id}: {side} {amount} {symbol}")
            
            if order_type == 'market':
                result = self.exchange.create_order(symbol, order_type, side, amount)
            else:
                result = self.exchange.create_order(symbol, order_type, side, amount, price, params)
            
            # Update order with execution details
            order.status = 'FILLED'  # Simplification, should check actual status
            order.executed_timestamp = datetime.now()
            
            # Extract execution details from result
            if 'price' in result:
                order.executed_price = float(result['price'])
            elif 'average' in result and result['average']:
                order.executed_price = float(result['average'])
            
            if 'filled' in result:
                order.executed_quantity = float(result['filled'])
            
            if 'fee' in result and result['fee']:
                order.commission = float(result['fee']['cost'])
            
            # Add to order history
            self.orders.append(order)
            
            logger.info(f"Order executed: {order.order_id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = 'REJECTED'
            return order
    
    def submit_orders(self, orders: List[Order], market_data: Dict[str, Any]) -> List[Order]:
        """
        Submit multiple orders for execution.
        
        Args:
            orders: List of orders to execute
            market_data: Current market data
        
        Returns:
            List of updated orders with execution details
        """
        executed_orders = []
        
        for order in orders:
            executed_order = self.execute_order(order, market_data)
            executed_orders.append(executed_order)
        
        return executed_orders
    
    def get_order_history(self) -> List[Order]:
        """
        Get the order history.
        
        Returns:
            List of historical orders
        """
        return self.orders
