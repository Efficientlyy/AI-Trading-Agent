"""
Paper Trading Manager Module

Provides simulated trading functionality using real-time market data.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import pandas as pd
import numpy as np
import uuid
from decimal import Decimal, ROUND_DOWN

from ..agent.data_manager import RealTimeDataManager
from ..common.enums import OrderType, OrderSide, OrderStatus, TimeInForce
from ..common.models import Order, Position, Trade, Balance, Portfolio
from ..common.utils import calculate_order_fees

logger = logging.getLogger(__name__)

class PaperTradingManager:
    """
    Simulated trading environment using real-time market data.
    
    Emulates exchange behavior for placing and executing orders without
    using real funds. Tracks portfolio, positions, balances, and order history.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the paper trading manager.
        
        Args:
            config: Configuration containing:
                - initial_balance: Dictionary mapping currency to initial balance amount
                - leverage: Maximum leverage allowed (default: 1.0 for spot trading)
                - fee_rate: Trading fee percentage as decimal (e.g., 0.001 for 0.1%)
                - slippage: Simulated slippage percentage as decimal (e.g., 0.0005 for 0.05%)
                - data_manager: Instance of RealTimeDataManager (or None, will be set later)
        """
        self.initial_balance = config.get('initial_balance', {'USDT': 10000.0})
        self.leverage = config.get('leverage', 1.0)
        self.fee_rate = config.get('fee_rate', 0.001)  # 0.1% default
        self.slippage = config.get('slippage', 0.0005)  # 0.05% default
        
        # Data source for real-time market data
        self.data_manager: Optional[RealTimeDataManager] = config.get('data_manager')
        
        # Internal state
        self.portfolio = Portfolio(
            balances={currency: Balance(currency=currency, free=amount, locked=0.0) 
                      for currency, amount in self.initial_balance.items()},
            positions={},
            order_history=[],
            trade_history=[],
            realized_pnl=0.0,
            unrealized_pnl=0.0
        )
        
        # Latest market prices per symbol
        self.latest_prices: Dict[str, float] = {}
        
        # Active orders awaiting execution
        self.open_orders: Dict[str, Order] = {}  # order_id -> Order
        
        # Order execution lock to prevent race conditions
        self.order_lock = asyncio.Lock()
        
        # Processing flag
        self.is_running = False
        
        # Task for updating market data
        self.update_task = None
        
        logger.info("Paper Trading Manager initialized with balance: "
                   f"{', '.join([f'{amt} {ccy}' for ccy, amt in self.initial_balance.items()])}")
    
    async def start(self):
        """Start the paper trading manager."""
        if self.is_running:
            return
            
        if not self.data_manager:
            raise ValueError("Cannot start paper trading: data manager is not set")
        
        self.is_running = True
        
        # Start data updates
        self.update_task = asyncio.create_task(self._update_loop())
        
        logger.info("Paper Trading Manager started")
        
    async def stop(self):
        """Stop the paper trading manager."""
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None
        
        logger.info("Paper Trading Manager stopped")
    
    async def _update_loop(self):
        """Background task for updating market data and processing orders."""
        while self.is_running:
            try:
                # Update market data from the data manager
                has_updates = await self.data_manager.update()
                
                if has_updates:
                    # Get current market data
                    market_data = self.data_manager.get_current_data()
                    
                    if market_data:
                        # Update latest prices
                        for symbol, data in market_data.items():
                            if 'close' in data and pd.notna(data['close']):
                                self.latest_prices[symbol] = float(data['close'])
                        
                        # Update unrealized PnL
                        await self._update_portfolio_values()
                        
                        # Process pending orders
                        await self._process_orders()
                
                # Sleep to avoid excessive CPU usage
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in paper trading update loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Longer delay after error
    
    async def place_order(self, order: Order) -> Tuple[bool, str]:
        """
        Place a new order in the paper trading environment.
        
        Args:
            order: Order to place
        
        Returns:
            Tuple of (success, message)
        """
        async with self.order_lock:
            try:
                # Validate the order
                valid, message = self._validate_order(order)
                if not valid:
                    return False, message
                
                # Generate order ID if not provided
                if not order.order_id:
                    order.order_id = str(uuid.uuid4())
                
                # Set order status to OPEN
                order.status = OrderStatus.OPEN
                order.created_time = datetime.now()
                
                # Reserve funds (lock balance)
                self._reserve_funds_for_order(order)
                
                # Store the order
                self.open_orders[order.order_id] = order
                self.portfolio.order_history.append(order)
                
                logger.info(f"Placed order: {order.order_id}, {order.symbol}, "
                           f"{order.side.name}, {order.quantity} @ {order.price}")
                
                # For market orders, try to execute immediately
                if order.type == OrderType.MARKET:
                    await self._execute_market_order(order)
                
                return True, f"Order placed: {order.order_id}"
                
            except Exception as e:
                logger.error(f"Error placing order: {e}", exc_info=True)
                return False, f"Error placing order: {e}"
    
    async def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of order to cancel
        
        Returns:
            Tuple of (success, message)
        """
        async with self.order_lock:
            try:
                if order_id not in self.open_orders:
                    return False, f"Order {order_id} not found or already completed"
                
                order = self.open_orders[order_id]
                
                # Update order status
                order.status = OrderStatus.CANCELED
                order.updated_time = datetime.now()
                
                # Release locked funds
                self._release_reserved_funds(order)
                
                # Remove from open orders
                del self.open_orders[order_id]
                
                logger.info(f"Canceled order: {order_id}")
                
                return True, f"Order {order_id} canceled"
                
            except Exception as e:
                logger.error(f"Error canceling order: {e}", exc_info=True)
                return False, f"Error canceling order: {e}"
    
    def _validate_order(self, order: Order) -> Tuple[bool, str]:
        """
        Validate an order before placing it.
        
        Args:
            order: Order to validate
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check required fields
        if not order.symbol:
            return False, "Symbol is required"
        
        if order.quantity <= 0:
            return False, "Quantity must be greater than 0"
        
        if order.type == OrderType.LIMIT and (order.price is None or order.price <= 0):
            return False, "Limit orders require a valid price"
        
        # Check if symbol exists
        base_ccy, quote_ccy = self._parse_symbol(order.symbol)
        if not base_ccy or not quote_ccy:
            return False, f"Invalid symbol format: {order.symbol}"
        
        # Check if we have price data for this symbol
        if order.symbol not in self.latest_prices and order.type == OrderType.MARKET:
            return False, f"No price data available for {order.symbol}"
        
        # For sell orders, check if we have enough of the base currency
        if order.side == OrderSide.SELL:
            if base_ccy not in self.portfolio.balances:
                return False, f"No balance for {base_ccy}"
                
            available = self.portfolio.balances[base_ccy].free
            if available < order.quantity:
                return False, f"Insufficient {base_ccy} balance: have {available}, need {order.quantity}"
        
        # For buy orders, check if we have enough of the quote currency
        elif order.side == OrderSide.BUY:
            if quote_ccy not in self.portfolio.balances:
                return False, f"No balance for {quote_ccy}"
                
            # Calculate required amount
            price = order.price if order.type == OrderType.LIMIT else self.latest_prices.get(order.symbol, 0)
            if price <= 0:
                return False, "Invalid price for calculation"
                
            required_amount = order.quantity * price * (1 + self.fee_rate)  # Include fees
            available = self.portfolio.balances[quote_ccy].free
            
            if available < required_amount:
                return False, f"Insufficient {quote_ccy} balance: have {available}, need {required_amount}"
        
        return True, "Order validated"
    
    def _reserve_funds_for_order(self, order: Order):
        """
        Lock funds for an order.
        
        Args:
            order: Order to reserve funds for
        """
        base_ccy, quote_ccy = self._parse_symbol(order.symbol)
        
        if order.side == OrderSide.SELL:
            # Lock base currency (e.g., BTC in BTC/USDT)
            balance = self.portfolio.balances.get(base_ccy)
            if balance:
                balance.free -= order.quantity
                balance.locked += order.quantity
                
        elif order.side == OrderSide.BUY:
            # Lock quote currency (e.g., USDT in BTC/USDT)
            price = order.price if order.type == OrderType.LIMIT else self.latest_prices.get(order.symbol, 0)
            required_amount = order.quantity * price * (1 + self.fee_rate)  # Include fees
            
            balance = self.portfolio.balances.get(quote_ccy)
            if balance:
                balance.free -= required_amount
                balance.locked += required_amount
    
    def _release_reserved_funds(self, order: Order):
        """
        Release locked funds for a canceled order.
        
        Args:
            order: Order to release funds for
        """
        base_ccy, quote_ccy = self._parse_symbol(order.symbol)
        
        # Calculate unfilled quantity
        filled_qty = order.filled_quantity if order.filled_quantity else 0
        unfilled_qty = order.quantity - filled_qty
        
        if unfilled_qty <= 0:
            return  # Nothing to release
            
        if order.side == OrderSide.SELL:
            # Release base currency
            balance = self.portfolio.balances.get(base_ccy)
            if balance:
                balance.locked -= unfilled_qty
                balance.free += unfilled_qty
                
        elif order.side == OrderSide.BUY:
            # Release quote currency
            price = order.price if order.type == OrderType.LIMIT else self.latest_prices.get(order.symbol, 0)
            release_amount = unfilled_qty * price * (1 + self.fee_rate)
            
            balance = self.portfolio.balances.get(quote_ccy)
            if balance:
                balance.locked -= release_amount
                balance.free += release_amount
    
    async def _process_orders(self):
        """Process all open orders against current market prices."""
        if not self.open_orders:
            return
            
        async with self.order_lock:
            # Make a copy to avoid modification during iteration
            orders_to_process = list(self.open_orders.values())
            
            for order in orders_to_process:
                if order.type == OrderType.LIMIT:
                    await self._process_limit_order(order)
    
    async def _process_limit_order(self, order: Order):
        """
        Check if a limit order can be executed based on current prices.
        
        Args:
            order: Limit order to process
        """
        if order.symbol not in self.latest_prices:
            return
            
        current_price = self.latest_prices[order.symbol]
        
        # Check if limit price is reached
        can_execute = False
        
        if order.side == OrderSide.BUY and current_price <= order.price:
            can_execute = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            can_execute = True
            
        if can_execute:
            await self._execute_order(order, current_price)
    
    async def _execute_market_order(self, order: Order):
        """
        Execute a market order immediately.
        
        Args:
            order: Market order to execute
        """
        if order.symbol not in self.latest_prices:
            # Can't execute without a price
            order.status = OrderStatus.REJECTED
            order.updated_time = datetime.now()
            order.status_message = "No current price available for market execution"
            # Release funds
            self._release_reserved_funds(order)
            # Remove from open orders
            if order.order_id in self.open_orders:
                del self.open_orders[order.order_id]
            return
            
        current_price = self.latest_prices[order.symbol]
        
        # Add slippage to simulate real market execution
        execution_price = self._apply_slippage(current_price, order.side)
        
        await self._execute_order(order, execution_price)
    
    async def _execute_order(self, order: Order, execution_price: float):
        """
        Execute an order at the specified price.
        
        Args:
            order: Order to execute
            execution_price: Price to execute at
        """
        base_ccy, quote_ccy = self._parse_symbol(order.symbol)
        
        # Calculate executed amount with slippage
        executed_price = self._round_price(execution_price, 8)  # Round to 8 decimal places
        executed_quantity = order.quantity
        executed_amount = executed_price * executed_quantity
        
        # Calculate fees
        fee_currency = quote_ccy  # Fees in quote currency by default
        fee_amount = executed_amount * self.fee_rate
        
        # Generate trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=executed_quantity,
            price=executed_price,
            amount=executed_amount,
            fee=fee_amount,
            fee_currency=fee_currency,
            timestamp=datetime.now(),
            is_maker=False  # Always assume taker for paper trading
        )
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = executed_quantity
        order.average_price = executed_price
        order.updated_time = trade.timestamp
        
        # Update balances and positions
        if order.side == OrderSide.BUY:
            # Add base currency
            if base_ccy not in self.portfolio.balances:
                self.portfolio.balances[base_ccy] = Balance(currency=base_ccy, free=0.0, locked=0.0)
            self.portfolio.balances[base_ccy].free += executed_quantity
            
            # Deduct quote currency (already locked)
            if quote_ccy in self.portfolio.balances:
                total_cost = executed_amount + fee_amount
                # Just reduce the locked amount (funds were locked during order placement)
                self.portfolio.balances[quote_ccy].locked -= total_cost
                
            # Update or create position
            self._update_position(order.symbol, order.side, executed_quantity, executed_price)
                
        elif order.side == OrderSide.SELL:
            # Deduct base currency (already locked)
            if base_ccy in self.portfolio.balances:
                self.portfolio.balances[base_ccy].locked -= executed_quantity
                
            # Add quote currency
            if quote_ccy not in self.portfolio.balances:
                self.portfolio.balances[quote_ccy] = Balance(currency=quote_ccy, free=0.0, locked=0.0)
            received_amount = executed_amount - fee_amount
            self.portfolio.balances[quote_ccy].free += received_amount
            
            # Update or create position
            self._update_position(order.symbol, order.side, executed_quantity, executed_price)
        
        # Add trade to history
        self.portfolio.trade_history.append(trade)
        
        # Remove from open orders
        if order.order_id in self.open_orders:
            del self.open_orders[order.order_id]
            
        logger.info(f"Executed order {order.order_id}: {order.symbol}, {order.side.name}, "
                   f"{executed_quantity} @ {executed_price}, fee: {fee_amount} {fee_currency}")
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: float, price: float):
        """
        Update a position based on a trade.
        
        Args:
            symbol: Trading pair
            side: Buy or sell
            quantity: Trade quantity
            price: Execution price
        """
        if symbol not in self.portfolio.positions:
            # Create new position if it doesn't exist
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                entry_price=0.0,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
            
        position = self.portfolio.positions[symbol]
        
        if side == OrderSide.BUY:
            # Adding to position
            if position.quantity == 0:
                # New position
                position.quantity = quantity
                position.entry_price = price
            else:
                # Average entry price
                old_value = position.quantity * position.entry_price
                new_value = quantity * price
                position.entry_price = (old_value + new_value) / (position.quantity + quantity)
                position.quantity += quantity
                
        elif side == OrderSide.SELL:
            if position.quantity >= quantity:
                # Reducing position
                realized_pnl = (price - position.entry_price) * quantity
                position.realized_pnl += realized_pnl
                self.portfolio.realized_pnl += realized_pnl
                
                position.quantity -= quantity
                # Entry price remains the same unless fully closed
                
                if position.quantity == 0:
                    # Reset entry price when position is fully closed
                    position.entry_price = 0.0
                    position.unrealized_pnl = 0.0
            else:
                # This shouldn't happen with proper validation, but handle it anyway
                logger.error(f"Attempting to sell more than position size for {symbol}: "
                           f"have {position.quantity}, selling {quantity}")
        
        # Update current price
        position.current_price = price
        # Update unrealized PnL
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
    
    async def _update_portfolio_values(self):
        """Update portfolio values and unrealized PnL based on current market prices."""
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.portfolio.positions.items():
            if symbol in self.latest_prices and position.quantity > 0:
                # Update current price
                current_price = self.latest_prices[symbol]
                position.current_price = current_price
                
                # Calculate unrealized PnL
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                total_unrealized_pnl += position.unrealized_pnl
        
        # Update portfolio totals
        self.portfolio.unrealized_pnl = total_unrealized_pnl
    
    def _parse_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse trading pair symbol into base and quote currencies.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Tuple of (base_currency, quote_currency)
        """
        if '/' in symbol:
            parts = symbol.split('/')
            if len(parts) == 2:
                return parts[0], parts[1]
        
        return None, None
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply slippage to price based on order side.
        
        Args:
            price: Current market price
            side: Order side (buy/sell)
            
        Returns:
            Price with slippage applied
        """
        if side == OrderSide.BUY:
            # Buy price is worse (higher) with slippage
            return price * (1 + self.slippage)
        else:
            # Sell price is worse (lower) with slippage
            return price * (1 - self.slippage)
    
    def _round_price(self, price: float, decimals: int = 8) -> float:
        """
        Round price to specified number of decimal places.
        
        Args:
            price: Price to round
            decimals: Number of decimal places
            
        Returns:
            Rounded price
        """
        factor = 10 ** decimals
        return float(Decimal(price * factor).quantize(Decimal('1'), rounding=ROUND_DOWN) / Decimal(factor))
    
    def get_portfolio(self) -> Portfolio:
        """
        Get the current portfolio state.
        
        Returns:
            Portfolio object with current balances, positions, and trades
        """
        return self.portfolio
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order object if found, None otherwise
        """
        # Check open orders first
        if order_id in self.open_orders:
            return self.open_orders[order_id]
            
        # Check order history
        for order in self.portfolio.order_history:
            if order.order_id == order_id:
                return order
                
        return None