"""
Manages the lifecycle of orders within the trading engine.

Responsibilities:
- Creating new orders.
- Tracking active orders.
- Updating order status based on fills or cancellations.
- Interacting with the Portfolio model.
"""

from typing import Dict, Optional, Literal
from .models import Order, Trade, Portfolio, OrderStatus
from common import logger

class OrderManager:
    """Handles order creation, tracking, and updates."""

    def __init__(self, portfolio: Portfolio):
        """Initializes the OrderManager with a reference to the Portfolio."""
        self.portfolio = portfolio # Uses the Portfolio's order dict directly
        logger.info("OrderManager initialized.")

    def create_order(self,
                       symbol: str,
                       side: Literal['buy', 'sell'],
                       order_type: Literal['market', 'limit'],
                       quantity: float,
                       price: Optional[float] = None,
                       client_order_id: Optional[str] = None
                       ) -> Optional[Order]:
        """
        Creates a new order and adds it to the portfolio's order dictionary.
        Performs basic validation before creation.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT').
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            quantity: Amount to buy/sell.
            price: Required for limit orders.
            client_order_id: Optional user-defined identifier.

        Returns:
            The created Order object if successful, None otherwise.
        """
        # --- Correctly indented block ---
        try:
            # Basic validation (more can be added, e.g., balance check)
            if quantity <= 0:
                logger.error(f"Order creation failed: Quantity must be positive ({quantity}).")
                return None # Indented under if
            if order_type == 'limit' and (price is None or price <= 0):
                logger.error(f"Order creation failed: Limit orders require a positive price (got {price}).")
                return None # Indented under if

            # Check available balance (simplistic for now, assumes base currency)
            required_balance = 0
            if order_type == 'limit' and side == 'buy':
                required_balance = quantity * price
            # Market buy estimate is harder without current price - skip check for now or require pre-funding
            # Add check for short selling availability later if needed

            # if side == 'buy' and self.portfolio.current_balance < required_balance:
            #     logger.error(f"Order creation failed: Insufficient balance. Need {required_balance}, have {self.portfolio.current_balance}")
            #     return None

            order = Order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
                status='new' # Status becomes 'open' when sent to exchange/matcher
            )
            self.portfolio.orders[order.order_id] = order
            logger.info(f"Created Order {order.order_id}: {side} {quantity} {symbol} @ {order_type} {price or 'N/A'}")
            return order # Indented under try

        except ValueError as e: # Correctly indented except
            logger.error(f"Order creation failed due to validation error: {e}")
            return None # Indented under except
        except Exception as e: # Correctly indented except
            logger.error(f"Unexpected error during order creation: {e}", exc_info=True)
            return None # Indented under except

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieves an order by its internal ID."""
        return self.portfolio.orders.get(order_id)

    def update_order_status(self, order_id: str, new_status: OrderStatus):
        """Updates the status of a specific order."""
        order = self.get_order(order_id)
        if order:
            old_status = order.status
            order.update_status(new_status)
            logger.info(f"Updated Order {order_id} status from {old_status} to {new_status}")
            # Remove from active orders if it's a final state
            if new_status in ['filled', 'canceled', 'rejected', 'expired']:
                # Potentially move to a separate historical orders dict
                pass
        else:
            logger.warning(f"Attempted to update status for non-existent order ID: {order_id}")

    def cancel_order(self, order_id: str) -> bool:
        """Marks an order as canceled if it's in a cancelable state."""
        order = self.get_order(order_id)
        if order:
            if order.status in ['new', 'open', 'partially_filled']:
                # In a real system, this would trigger a cancel request to the exchange
                self.update_order_status(order_id, 'canceled')
                logger.info(f"Order {order_id} marked as canceled.")
                return True
            else:
                logger.warning(f"Cannot cancel order {order_id} in status {order.status}")
                return False
        else:
            logger.warning(f"Attempted to cancel non-existent order ID: {order_id}")
            return False

    def process_trade(self, trade: Trade, current_market_prices: Dict[str, float]):
        """
        Processes an executed trade, updating the corresponding order's state
        and the portfolio.

        Args:
            trade: The Trade object representing a fill.
            current_market_prices: Dict mapping symbol to current market price.
        """
        logger.info(f"process_trade called with trade {trade.trade_id} and current_market_prices: {current_market_prices}")
        
        order = self.get_order(trade.order_id)
        if order:
            if order.status in ['filled', 'canceled', 'rejected', 'expired']:
                 logger.warning(f"Received trade {trade.trade_id} for already finalized order {order.order_id} (status: {order.status}). Ignoring portfolio update for this trade.")
                 logger.info(f"Skipping fill update for finalized order {order.order_id}")
                 return # Don't update portfolio again

            logger.info(f"Processing Trade {trade.trade_id} for Order {order.order_id}: {trade.side} {trade.quantity} {trade.symbol} @ {trade.price}")
            try:
                logger.info(f"About to call order.add_fill with quantity={trade.quantity}, price={trade.price}")
                order.add_fill(
                    fill_quantity=trade.quantity,
                    fill_price=trade.price,
                    commission=trade.commission,
                    commission_asset=trade.commission_asset
                )
                logger.info(f"Order {order.order_id} status after fill: {order.status}, Filled Qty: {order.filled_quantity}/{order.quantity}")
                
                # --- Update Portfolio State ---
                logger.info(f"About to call portfolio.update_from_trade with trade={trade.trade_id}")
                self.portfolio.update_from_trade(trade, current_market_prices)
                logger.info(f"Portfolio updated for trade {trade.trade_id}. New balance: {self.portfolio.current_balance:.2f}")
                # Log position change
                position = self.portfolio.get_position(trade.symbol)
                if position:
                    logger.info(f"Position {trade.symbol}: Side={position.side}, Qty={position.quantity:.4f}, Entry={position.entry_price:.2f}, UPL={position.unrealized_pnl:.2f}")
                else:
                    logger.info(f"Position {trade.symbol} closed.")
                    
            except ValueError as e:
                logger.error(f"Error processing trade {trade.trade_id} for order {order.order_id}: {e}")
                # Decide if we should reject the trade or handle differently
            except Exception as e:
                logger.error(f"Unexpected error in process_trade: {e}", exc_info=True)
                raise

        else:
            logger.error(f"Received trade {trade.trade_id} for unknown order ID: {trade.order_id}")

    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Order]:
        """Returns a dictionary of open orders, optionally filtered by symbol."""
        open_statuses = ['new', 'open', 'partially_filled']
        open_orders = {oid: o for oid, o in self.portfolio.orders.items() if o.status in open_statuses}
        if symbol:
            return {oid: o for oid, o in open_orders.items() if o.symbol == symbol}
        return open_orders
