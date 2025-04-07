"""
Execution Handler module for AI Trading Agent.

This module provides functionality for simulating trade execution
with realistic market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import uuid

from src.trading_engine.models import Order, Trade
from src.trading_engine.enums import OrderSide, OrderType, OrderStatus
from src.common import logger


class ExecutionHandler:
    """
    Execution Handler class for simulating trade execution.
    
    This class handles the execution of orders with realistic market conditions,
    including slippage, partial fills, and rejection scenarios.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_model: str = "fixed",
        slippage_params: Dict[str, Any] = None,
        enable_partial_fills: bool = False,
        rejection_probability: float = 0.0,
    ):
        """
        Initialize the execution handler.
        
        Args:
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage_model: Slippage model to use ('fixed', 'normal', 'proportional')
            slippage_params: Parameters for the slippage model
            enable_partial_fills: Whether to enable partial fills
            rejection_probability: Probability of order rejection (0.0 to 1.0)
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_params = slippage_params or {}
        self.enable_partial_fills = enable_partial_fills
        self.rejection_probability = rejection_probability
        
        # Set default slippage parameters if not provided
        if "fixed" not in self.slippage_params:
            self.slippage_params["fixed"] = 0.0
        if "normal" not in self.slippage_params:
            self.slippage_params["normal"] = {"mean": 0.0, "std": 0.001}
        if "proportional" not in self.slippage_params:
            self.slippage_params["proportional"] = 0.001
        
        logger.info(f"Initialized execution handler with commission_rate={commission_rate}, "
                   f"slippage_model={slippage_model}")
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> List[Trade]:
        """
        Execute an order using the current market data.
        
        Args:
            order: Order to execute
            market_data: Market data for the symbol
            timestamp: Current timestamp
            
        Returns:
            List[Trade]: List of trades resulting from the order execution
        """
        # Check if order should be rejected
        if np.random.random() < self.rejection_probability:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} rejected due to random rejection")
            return []
        
        # Get current market data
        current_bar = market_data.loc[market_data.index <= timestamp].iloc[-1]
        
        # Calculate execution price based on order type
        execution_price = self._calculate_execution_price(order, current_bar)
        
        # If execution price is None, order cannot be executed
        if execution_price is None:
            return []
        
        # Apply slippage
        execution_price = self._apply_slippage(order, execution_price)
        
        # Determine fill quantity
        fill_quantity = self._determine_fill_quantity(order)
        
        # Create trade
        trade = Trade(
            symbol=order.symbol,
            order_id=order.order_id,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=timestamp,
        )
        
        # Update order with fill
        order.add_fill(
            fill_quantity=fill_quantity,
            fill_price=execution_price,
            commission=fill_quantity * execution_price * self.commission_rate,
        )
        
        # Update order status
        if order.remaining_quantity < 1e-8:  # Effectively zero
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        logger.info(f"Executed order {order.order_id}: {order.side.value} {fill_quantity} {order.symbol} @ {execution_price:.4f}")
        
        return [trade]
    
    def _calculate_execution_price(self, order: Order, bar: pd.Series) -> Optional[float]:
        """
        Calculate execution price based on order type and current bar.
        
        Args:
            order: Order to execute
            bar: Current price bar
            
        Returns:
            float: Execution price, or None if order cannot be executed
        """
        if order.order_type == OrderType.MARKET:
            # Market orders execute at the open of the next bar
            return bar['open']
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and bar['low'] <= order.limit_price:
                # Buy limit orders execute at limit price if low <= limit price
                return min(bar['open'], order.limit_price)
            elif order.side == OrderSide.SELL and bar['high'] >= order.limit_price:
                # Sell limit orders execute at limit price if high >= limit price
                return max(bar['open'], order.limit_price)
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and bar['high'] >= order.stop_price:
                # Buy stop orders execute at stop price if high >= stop price
                return max(bar['open'], order.stop_price)
            elif order.side == OrderSide.SELL and bar['low'] <= order.stop_price:
                # Sell stop orders execute at stop price if low <= stop price
                return min(bar['open'], order.stop_price)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY and bar['high'] >= order.stop_price:
                # Buy stop-limit orders trigger at stop price, then execute as limit orders
                if bar['low'] <= order.limit_price:
                    return min(max(bar['open'], order.stop_price), order.limit_price)
            elif order.side == OrderSide.SELL and bar['low'] <= order.stop_price:
                # Sell stop-limit orders trigger at stop price, then execute as limit orders
                if bar['high'] >= order.limit_price:
                    return max(min(bar['open'], order.stop_price), order.limit_price)
        
        # Order cannot be executed
        return None
    
    def _apply_slippage(self, order: Order, price: float) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            order: Order being executed
            price: Raw execution price
            
        Returns:
            float: Price with slippage applied
        """
        if self.slippage_model == "fixed":
            # Fixed slippage in price units
            slippage_amount = self.slippage_params["fixed"]
            if order.side == OrderSide.BUY:
                return price * (1 + slippage_amount)
            else:  # SELL
                return price * (1 - slippage_amount)
        
        elif self.slippage_model == "normal":
            # Normal distribution slippage
            mean = self.slippage_params["normal"]["mean"]
            std = self.slippage_params["normal"]["std"]
            slippage_factor = np.random.normal(mean, std)
            if order.side == OrderSide.BUY:
                return price * (1 + abs(slippage_factor))
            else:  # SELL
                return price * (1 - abs(slippage_factor))
        
        elif self.slippage_model == "proportional":
            # Proportional to volatility
            slippage_factor = self.slippage_params["proportional"]
            if order.side == OrderSide.BUY:
                return price * (1 + slippage_factor)
            else:  # SELL
                return price * (1 - slippage_factor)
        
        # Default: no slippage
        return price
    
    def _determine_fill_quantity(self, order: Order) -> float:
        """
        Determine the fill quantity for an order.
        
        Args:
            order: Order being executed
            
        Returns:
            float: Fill quantity
        """
        remaining_quantity = order.quantity - order.filled_quantity
        
        if not self.enable_partial_fills:
            # Fill the entire remaining quantity
            return remaining_quantity
        
        # Simulate partial fills
        # Randomly fill between 50% and 100% of the remaining quantity
        fill_percentage = 0.5 + 0.5 * np.random.random()
        return remaining_quantity * fill_percentage


class SimulatedExchange:
    """
    Simulated Exchange class for backtesting.
    
    This class simulates a trading exchange with order book, market data,
    and execution capabilities.
    """
    
    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        commission_rate: float = 0.001,
        slippage_model: str = "fixed",
        slippage_params: Dict[str, Any] = None,
        enable_partial_fills: bool = False,
        rejection_probability: float = 0.0,
    ):
        """
        Initialize the simulated exchange.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrames with OHLCV data
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage_model: Slippage model to use ('fixed', 'normal', 'proportional')
            slippage_params: Parameters for the slippage model
            enable_partial_fills: Whether to enable partial fills
            rejection_probability: Probability of order rejection (0.0 to 1.0)
        """
        self.market_data = market_data
        self.symbols = list(market_data.keys())
        
        # Initialize execution handler
        self.execution_handler = ExecutionHandler(
            commission_rate=commission_rate,
            slippage_model=slippage_model,
            slippage_params=slippage_params,
            enable_partial_fills=enable_partial_fills,
            rejection_probability=rejection_probability,
        )
        
        # Initialize order book
        self.order_book = {}
        for symbol in self.symbols:
            self.order_book[symbol] = {
                "bids": [],  # List of (price, quantity) tuples
                "asks": [],  # List of (price, quantity) tuples
            }
        
        # Initialize current timestamp
        self.current_timestamp = None
        
        logger.info(f"Initialized simulated exchange with {len(self.symbols)} symbols")
    
    def place_order(self, order: Order) -> None:
        """
        Place an order on the exchange.
        
        Args:
            order: Order to place
        """
        # Validate order
        if order.symbol not in self.symbols:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} rejected: Symbol {order.symbol} not available")
            return
        
        # Set order status to submitted
        order.status = OrderStatus.SUBMITTED
        
        logger.info(f"Placed order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
    
    def execute_orders(self, orders: List[Order], timestamp: pd.Timestamp) -> List[Trade]:
        """
        Execute a list of orders at the given timestamp.
        
        Args:
            orders: List of orders to execute
            timestamp: Current timestamp
            
        Returns:
            List[Trade]: List of trades resulting from the order execution
        """
        self.current_timestamp = timestamp
        trades = []
        
        for order in orders:
            # Skip orders that are not submitted
            if order.status != OrderStatus.SUBMITTED:
                continue
            
            # Get market data for the symbol
            if order.symbol not in self.market_data:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order.order_id} rejected: No market data for {order.symbol}")
                continue
            
            market_data = self.market_data[order.symbol]
            
            # Execute the order
            order_trades = self.execution_handler.execute_order(
                order=order,
                market_data=market_data,
                timestamp=timestamp,
            )
            
            trades.extend(order_trades)
        
        return trades
    
    def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for all symbols.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping symbols to price data
        """
        prices = {}
        
        for symbol, data in self.market_data.items():
            if self.current_timestamp is None:
                # Use the latest data
                bar = data.iloc[-1]
            else:
                # Use data up to current timestamp
                bars = data[data.index <= self.current_timestamp]
                if len(bars) == 0:
                    continue
                bar = bars.iloc[-1]
            
            prices[symbol] = {
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
        
        return prices
