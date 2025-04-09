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
from src.trading_engine.exceptions import ExecutionError, TradingEngineError


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
        try:
            # Check if order should be rejected
            if np.random.random() < self.rejection_probability:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected", order_id=order.order_id, reason="random rejection")
                return []
            
            # Get current market data
            current_bar = market_data.loc[market_data.index <= timestamp].iloc[-1]
            
            # Calculate execution price based on order type
            execution_price = self._calculate_execution_price(order, current_bar)
            
            # If execution price is None, order cannot be executed
            if execution_price is None:
                logger.info("Order not executed (price condition not met)", order_id=order.order_id)
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
            
            logger.info(f"Executed order", order_id=order.order_id, side=order.side.value,
                        quantity=fill_quantity, price=execution_price, status=order.status.value)
            
            return [trade]
        
        except Exception as e:
            logger.error(f"Execution error for order {order.order_id}: {e}", exc_info=True)
            raise ExecutionError(f"Execution failed for order {order.order_id}") from e
    
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
            # Market orders execute at current price
            if order.side == OrderSide.BUY:
                return bar["close"]
            else:  # SELL
                return bar["close"]
                
        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute if price is favorable
            if order.side == OrderSide.BUY:
                # Buy limit: execute if current price <= limit price
                if bar["close"] <= order.price:
                    return bar["close"]
            else:  # SELL
                # Sell limit: execute if current price >= limit price
                if bar["close"] >= order.price:
                    return bar["close"]
                    
        elif order.order_type == OrderType.STOP:
            # Stop orders execute if price crosses the stop price
            if order.side == OrderSide.BUY:
                # Buy stop: execute if current price >= stop price
                if bar["close"] >= order.stop_price:
                    return bar["close"]
            else:  # SELL
                # Sell stop: execute if current price <= stop price
                if bar["close"] <= order.stop_price:
                    return bar["close"]
                    
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit orders: first check if stop price is triggered
            if order.side == OrderSide.BUY:
                # Buy stop-limit: trigger if current price >= stop price
                if bar["close"] >= order.stop_price:
                    # Then check if limit price is favorable
                    if bar["close"] <= order.price:
                        return bar["close"]
            else:  # SELL
                # Sell stop-limit: trigger if current price <= stop price
                if bar["close"] <= order.stop_price:
                    # Then check if limit price is favorable
                    if bar["close"] >= order.price:
                        return bar["close"]
        
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
        
        # Set order status to open
        order.status = OrderStatus.OPEN
        
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
            # Skip orders that are not open
            if order.status != OrderStatus.OPEN:
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
    
    def generate_order_book(self, symbol: str, timestamp: pd.Timestamp) -> Dict[str, List[Tuple[float, float]]]:
        """
        Generate a simulated order book for a symbol at a given timestamp.
        
        Args:
            symbol: Symbol to generate order book for
            timestamp: Timestamp for the order book
            
        Returns:
            Dict with 'bids' and 'asks' lists, each containing (price, quantity) tuples
        """
        if symbol not in self.market_data:
            logger.warning(f"Cannot generate order book: No market data for {symbol}")
            return {"bids": [], "asks": []}
            
        # Get current price data
        data = self.market_data[symbol]
        bars = data[data.index <= timestamp]
        if len(bars) == 0:
            logger.warning(f"Cannot generate order book: No data for {symbol} at {timestamp}")
            return {"bids": [], "asks": []}
            
        bar = bars.iloc[-1]
        mid_price = bar["close"]
        
        # Generate random order book with realistic depth
        bids = []
        asks = []
        
        # Number of price levels
        num_levels = np.random.randint(5, 15)
        
        # Price step as a percentage of mid price
        price_step_pct = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2%
        
        # Generate bids (buy orders) - prices below mid price
        for i in range(num_levels):
            # Price decreases as we go down the order book
            price_level = mid_price * (1 - (i + 1) * price_step_pct)
            # Quantity tends to increase for lower prices
            quantity = np.random.lognormal(mean=1.0, sigma=0.5) * (1 + i * 0.1)
            bids.append((price_level, quantity))
        
        # Generate asks (sell orders) - prices above mid price
        for i in range(num_levels):
            # Price increases as we go up the order book
            price_level = mid_price * (1 + (i + 1) * price_step_pct)
            # Quantity tends to increase for higher prices
            quantity = np.random.lognormal(mean=1.0, sigma=0.5) * (1 + i * 0.1)
            asks.append((price_level, quantity))
        
        # Sort bids in descending order (highest price first)
        bids.sort(key=lambda x: x[0], reverse=True)
        
        # Sort asks in ascending order (lowest price first)
        asks.sort(key=lambda x: x[0])
        
        return {
            "bids": bids,
            "asks": asks
        }
