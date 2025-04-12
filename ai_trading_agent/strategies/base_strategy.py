"""
Base Strategy module for AI Trading Agent.

This module provides the BaseStrategy class that defines the interface
for all trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

from ..trading_engine.models import Order, Trade, Position, Portfolio
from ..trading_engine.enums import OrderSide, OrderType, OrderStatus, PositionSide
from ..common import logger


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    Strategies are responsible for generating trading signals based on
    market data and current portfolio state.
    """
    
    def __init__(
        self,
        symbols: List[str],
        parameters: Dict[str, Any] = None,
        name: str = None,
    ):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols to trade
            parameters: Dictionary of strategy parameters
            name: Name of the strategy (defaults to class name)
        """
        self.symbols = symbols
        self.parameters = parameters or {}
        self.name = name or self.__class__.__name__
        
        # Initialize state
        self.initialized = False
        self.last_update_time = None
        
        logger.info(f"Initialized strategy {self.name} for symbols {self.symbols}")
    
    def initialize(self, data: Dict[str, pd.DataFrame], portfolio: Portfolio) -> None:
        """
        Initialize the strategy with historical data and portfolio.
        
        This method should be called before the first call to generate_signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            portfolio: Current portfolio state
        """
        self._validate_data(data)
        self._initialize_indicators(data)
        self.initialized = True
        self.last_update_time = datetime.now()
        
        logger.info(f"Strategy {self.name} initialized with {len(data)} symbols")
    
    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate the input data.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
        
        Raises:
            ValueError: If data is invalid
        """
        # Check that all required symbols are present
        missing_symbols = [symbol for symbol in self.symbols if symbol not in data]
        if missing_symbols:
            raise ValueError(f"Missing data for symbols: {missing_symbols}")
        
        # Check that all dataframes have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for symbol, df in data.items():
            if symbol in self.symbols:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns {missing_columns} for symbol {symbol}")
    
    @abstractmethod
    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize technical indicators and other strategy-specific data.
        
        This method should be implemented by subclasses to initialize
        any indicators or other data structures needed by the strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
        """
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> List[Order]:
        """
        Generate trading signals based on market data and portfolio state.
        
        This is the main method that must be implemented by all strategies.
        It should analyze the market data and current portfolio state to
        generate trading signals in the form of Order objects.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            portfolio: Current portfolio state
            timestamp: Current timestamp (optional)
            
        Returns:
            List[Order]: List of orders to execute
        """
        pass
    
    def update(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> List[Order]:
        """
        Update the strategy with new data and generate signals.
        
        This method should be called periodically to update the strategy
        with new market data and generate trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            portfolio: Current portfolio state
            timestamp: Current timestamp (optional)
            
        Returns:
            List[Order]: List of orders to execute
        """
        if not self.initialized:
            self.initialize(data, portfolio)
        
        # Update last update time
        self.last_update_time = datetime.now()
        
        # Generate signals
        return self.generate_signals(data, portfolio, timestamp)
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Create a new order.
        
        This is a helper method to create Order objects with a unique ID.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY or SELL)
            quantity: Order quantity
            order_type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT)
            limit_price: Limit price (required for LIMIT and STOP_LIMIT orders)
            stop_price: Stop price (required for STOP and STOP_LIMIT orders)
            
        Returns:
            Order: New order object
        """
        # Generate a unique order ID
        order_id = f"{self.name}-{uuid.uuid4().hex[:8]}"
        
        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            type=order_type,
            price=limit_price,
        )
        
        # Set the order ID
        order.order_id = order_id
        
        return order
    
    def calculate_position_size(
        self,
        symbol: str,
        portfolio: Portfolio,
        price: float,
        risk_pct: float = 0.02,
        max_position_pct: float = 0.2,
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        This is a helper method to calculate the appropriate position size
        based on the current portfolio value and risk parameters.
        
        Args:
            symbol: Symbol to trade
            portfolio: Current portfolio state
            price: Current price of the symbol
            risk_pct: Percentage of portfolio to risk on this trade
            max_position_pct: Maximum percentage of portfolio for this position
            
        Returns:
            float: Position size in units of the symbol
        """
        # Calculate position size based on risk
        risk_amount = portfolio.total_value * risk_pct
        
        # Calculate maximum position size based on portfolio percentage
        max_position_value = portfolio.total_value * max_position_pct
        max_position_size = max_position_value / price
        
        # Get current position size
        current_position_size = 0
        if symbol in portfolio.positions:
            current_position_size = portfolio.positions[symbol].quantity
        
        # Calculate available position size
        available_position_size = max_position_size - current_position_size
        
        # Calculate position size based on risk
        risk_position_size = risk_amount / price
        
        # Return the smaller of the two
        return min(risk_position_size, available_position_size)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the strategy parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of strategy parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the strategy parameters.
        
        Args:
            parameters: Dictionary of strategy parameters
        """
        self.parameters.update(parameters)
        
        # Reset initialization flag to force re-initialization
        self.initialized = False
        
        logger.info(f"Updated parameters for strategy {self.name}: {parameters}")
    
    def __str__(self) -> str:
        """
        Get a string representation of the strategy.
        
        Returns:
            str: String representation
        """
        return f"{self.name} (symbols: {self.symbols}, parameters: {self.parameters})"
