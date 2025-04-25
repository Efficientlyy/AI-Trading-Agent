"""
Data models for the trading engine using Pydantic.

Defines structures for Orders, Trades, and Positions.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
from typing import Optional, Dict, List, Any, ClassVar
from datetime import datetime, timezone
import uuid
from decimal import Decimal, InvalidOperation
import logging

# Import enums from the enums module using absolute import instead of relative
from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus, PositionSide

# --- Helper Functions ---
from ..common.time_utils import to_utc_naive

def utcnow() -> datetime:
    """Return naive UTC timestamp."""
    return to_utc_naive(datetime.now(timezone.utc))

def decimal_zero() -> Decimal:
    """Return Decimal('0') for default factory."""
    return Decimal('0')

def calculate_position_pnl(position: 'Position', current_market_price: Decimal) -> None:
    """Calculates unrealized PnL for a given position object."""
    price_dec = Decimal(str(current_market_price)) if not isinstance(current_market_price, Decimal) else current_market_price
    if position.quantity == Decimal('0'):
        position.unrealized_pnl = Decimal('0')
    elif position.side == PositionSide.LONG:
        position.unrealized_pnl = (price_dec - position.entry_price) * position.quantity
    else: # Short position
        position.unrealized_pnl = (position.entry_price - price_dec) * position.quantity
    position.last_update_time = utcnow()

# --- Core Models ---

class Order(BaseModel):
    """
    Represents a trading order.
    
    An Order tracks the lifecycle of a trading instruction from creation to execution.
    It includes validation for order quantity and price, and manages the state of
    partial fills and order status.
    
    Attributes:
        order_id: Unique identifier for the order
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        side: Buy or sell
        type: Order type (market, limit, etc.)
        quantity: Order quantity
        price: Order price (required for limit orders)
        stop_price: Order stop price (required for stop and stop-limit orders)
        status: Current order status
        filled_quantity: Amount of the order that has been filled
        remaining_quantity: Amount of the order that remains to be filled
        fills: List of fill records with quantity, price, and timestamp
        created_at: Timestamp when the order was created
        updated_at: Timestamp of the last update to the order
    """
    order_id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4()}")
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float = Field(gt=0) # Order quantity must be positive
    price: Optional[float] = None # Required for limit orders, validated in model_validator
    stop_price: Optional[float] = None # Required for stop and stop-limit orders
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    remaining_quantity: float = Field(default=0.0) # Will be set to quantity in __init__
    fills: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    
    # Aliases for backward compatibility
    order_type: Optional[OrderType] = None
    limit_price: Optional[float] = None
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
    
    def __init__(self, **data):
        # Handle backward compatibility
        if 'order_type' in data and 'type' not in data:
            data['type'] = data['order_type']
            
        if 'limit_price' in data and 'price' not in data:
            data['price'] = data['limit_price']
        
        # Convert string literals to enum values
        if 'side' in data and isinstance(data['side'], str):
            try:
                data['side'] = OrderSide[data['side']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in OrderSide:
                    if enum_val.name.lower() == data['side'].lower():
                        data['side'] = enum_val
                        break
        
        if 'type' in data and isinstance(data['type'], str):
            try:
                data['type'] = OrderType[data['type']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in OrderType:
                    if enum_val.name.lower() == data['type'].lower():
                        data['type'] = enum_val
                        break
        
        if 'status' in data and isinstance(data['status'], str):
            try:
                data['status'] = OrderStatus[data['status']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in OrderStatus:
                    if enum_val.name.lower() == data['status'].lower():
                        data['status'] = enum_val
                        break
            
        super().__init__(**data)
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
            
        # Set aliases for backward compatibility
        self.order_type = self.type
        self.limit_price = self.price

    @field_validator('quantity')
    @classmethod
    def quantity_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Order quantity must be positive")
        return v

    @model_validator(mode='after')
    def ensure_timestamps_consistent(self) -> 'Order':
        """Ensure updated_at is the same as created_at when order is first created."""
        if self.status == OrderStatus.NEW and self.filled_quantity == 0.0:
            self.updated_at = self.created_at
        return self
        
    @model_validator(mode='after')
    def validate_limit_order_price(self) -> 'Order':
        """Validate that limit orders have a valid price."""
        if self.type == OrderType.LIMIT:
            if self.price is None:
                raise ValueError("Price is required for limit orders")
            if self.price <= 0:
                raise ValueError("Limit price must be positive")
        return self
        
    @model_validator(mode='after')
    def validate_stop_order_price(self) -> 'Order':
        """Validate that stop orders have a valid stop price."""
        if self.type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None:
                raise ValueError("Stop price is required for stop orders")
            if self.stop_price <= 0:
                raise ValueError("Stop price must be positive")
                
            # For stop-limit orders, also validate the limit price
            if self.type == OrderType.STOP_LIMIT and self.price is None:
                raise ValueError("Limit price is required for stop-limit orders")
        return self

    @model_validator(mode='after')
    def set_price_from_limit_price(self) -> 'Order':
        """
        For backward compatibility, handle the case where price is passed directly
        instead of limit_price.
        """
        # If we have a price but no limit_price, set limit_price to price
        if hasattr(self, 'price') and self.price is not None:
            # Already set in the model
            pass
        return self

    def update_status(self, new_status: OrderStatus):
        """
        Update the order status and timestamp.
        
        Args:
            new_status: The new status to set
        """
        self.status = new_status
        self.updated_at = utcnow()
        
    def get_average_fill_price(self) -> Optional[float]:
        """
        Calculate the average fill price across all fills.
        
        Returns:
            The weighted average fill price, or None if no fills
        """
        if not self.fills or self.filled_quantity == 0:
            return None
            
        total_value = sum(fill['price'] * fill['quantity'] for fill in self.fills)
        return total_value / self.filled_quantity

    @property
    def average_fill_price(self) -> Optional[float]:
        """
        Property that returns the average fill price.
        
        Returns:
            The weighted average fill price, or None if no fills
        """
        return self.get_average_fill_price()

    def add_fill(self, fill_quantity: Optional[float] = None, fill_price: Optional[float] = None, 
             commission: Optional[float] = None, 
             commission_asset: Optional[str] = None, 
             exchange_order_id: Optional[str] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Add a fill to the order and update its state.
        
        Args:
            fill_quantity: Amount filled in this execution
            fill_price: Price at which the fill occurred
            commission: Optional commission amount
            commission_asset: Optional asset in which commission was paid
            exchange_order_id: Optional exchange order ID
            
        Returns:
            The fill record that was added
            
        Raises:
            ValueError: If fill quantity is invalid
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle keyword arguments for backward compatibility
        if 'quantity' in kwargs and fill_quantity is None:
            fill_quantity = kwargs['quantity']
        if 'price' in kwargs and fill_price is None:
            fill_price = kwargs['price']
        if 'timestamp' in kwargs:
            # Ignore timestamp as we'll use utcnow()
            pass
            
        logger.info(f"Adding fill to order {self.order_id}: {fill_quantity} @ {fill_price}")
        logger.info(f"Before fill - Status: {self.status}, Filled: {self.filled_quantity}, Remaining: {self.remaining_quantity}")
        
        if fill_quantity is None or fill_price is None:
            raise ValueError("Both fill_quantity and fill_price must be provided (not None)")
        
        if fill_quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if fill_quantity > self.remaining_quantity:
            raise ValueError(f"Fill quantity {fill_quantity} exceeds remaining quantity {self.remaining_quantity}")
        
        fill_record = {
            "quantity": fill_quantity,
            "price": fill_price,
            "timestamp": utcnow(),
        }
        if commission is not None:
            fill_record["commission"] = commission
        if commission_asset is not None:
            fill_record["commission_asset"] = commission_asset
        if exchange_order_id is not None:
            fill_record["exchange_order_id"] = exchange_order_id
            
        self.fills.append(fill_record)
        self.filled_quantity += fill_quantity
        self.remaining_quantity -= fill_quantity
        
        # Update order status
        if self.remaining_quantity < 1e-8:  # Effectively zero
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
            
        # Calculate and log the average fill price
        total_value = sum(fill["quantity"] * fill["price"] for fill in self.fills)
        avg_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        logger.info(f"After fill - Status: {self.status}, Filled: {self.filled_quantity}, Remaining: {self.remaining_quantity}")
        logger.info(f"Average fill price: {avg_price}")
        
        self.updated_at = utcnow()
        return fill_record


class Trade(BaseModel):
    """
    Represents an executed trade (fill).
    
    A Trade is created when an Order is executed, either fully or partially.
    It records the details of the execution including price, quantity, and timestamp.
    
    Attributes:
        trade_id: Unique identifier for the trade
        order_id: ID of the order that generated this trade
        exchange_trade_id: Optional ID from the exchange
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        side: Side of the original order (BUY or SELL)
        quantity: Executed quantity
        price: Execution price
        timestamp: When the trade occurred
        commission: Optional commission amount
        commission_asset: Optional asset in which commission was paid
        is_maker: Whether this trade was a maker (vs taker)
    """
    trade_id: str = Field(default_factory=lambda: f"trd_{uuid.uuid4()}")
    order_id: str # Link back to the order that generated this trade
    exchange_trade_id: Optional[str] = None # ID from the exchange
    symbol: str
    side: OrderSide # Side of the original order
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime = Field(default_factory=utcnow)
    commission: Optional[float] = None
    commission_asset: Optional[str] = None
    is_maker: Optional[bool] = None # Useful for fee calculation
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
    
    def __init__(self, **data):
        # Convert string literals to enum values
        if 'side' in data and isinstance(data['side'], str):
            try:
                data['side'] = OrderSide[data['side']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in OrderSide:
                    if enum_val.name.lower() == data['side'].lower():
                        data['side'] = enum_val
                        break
        
        super().__init__(**data)


class Fill(BaseModel):
    """
    Represents a fill (execution) of an order.
    
    A Fill records the details of a specific execution event, which may be a partial
    or complete fill of an order. It includes information about the quantity filled,
    the price at which it was filled, and any associated costs.
    
    Attributes:
        fill_id: Unique identifier for the fill
        order_id: ID of the order that was filled
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        side: Buy or sell
        quantity: Quantity filled in this execution
        price: Price at which the fill occurred
        timestamp: When the fill occurred
        commission: Optional commission amount
        commission_asset: Optional asset in which commission was paid
        exchange_order_id: Optional exchange order ID
    """
    fill_id: str = Field(default_factory=lambda: f"fill_{uuid.uuid4()}")
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)  # Fill quantity must be positive
    price: float = Field(gt=0)  # Fill price must be positive
    timestamp: datetime = Field(default_factory=utcnow)
    commission: Optional[float] = None
    commission_asset: Optional[str] = None
    exchange_order_id: Optional[str] = None
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
    
    def __init__(self, **data):
        # Convert string literals to enum values
        if 'side' in data and isinstance(data['side'], str):
            try:
                data['side'] = OrderSide[data['side']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in OrderSide:
                    if enum_val.name.lower() == data['side'].lower():
                        data['side'] = enum_val
                        break
        
        super().__init__(**data)
    
    @property
    def value(self) -> float:
        """
        Calculate the total value of this fill.
        
        Returns:
            The quantity * price
        """
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """
        Calculate the net value of this fill after commission.
        
        Returns:
            The value minus commission (if commission is in the same asset)
        """
        if self.commission is None or self.commission_asset != self.symbol:
            return self.value
        
        if self.side == OrderSide.BUY:
            return self.value - self.commission
        else:  # SELL
            return self.value - self.commission


class Position(BaseModel):
    """
    Represents a current position in a trading symbol.
    
    A Position tracks the current holdings of an asset, including quantity,
    entry price, and profit/loss calculations. It supports both long and short positions
    and handles position updates from trades.
    
    Attributes:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        side: Long or short position direction
        quantity: Position size (absolute value)
        entry_price: Average entry price for the position
        unrealized_pnl: Current unrealized profit/loss
        realized_pnl: Accumulated realized profit/loss from partial closes
        last_update_time: Timestamp of the last position update
    """
    symbol: str
    side: PositionSide
    quantity: Decimal = Field(default=Decimal('0')) # Position quantity (absolute value)
    entry_price: Decimal = Field(default=Decimal('0')) # Average entry price
    unrealized_pnl: Decimal = Field(default=Decimal('0'))
    realized_pnl: Decimal = Field(default=Decimal('0'))
    last_update_time: datetime = Field(default_factory=utcnow)
    # Add more fields as needed: leverage, margin, liquidation price, etc.
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }
    
    def __init__(self, **data):
        # Convert string literals to enum values
        if 'side' in data and isinstance(data['side'], str):
            try:
                data['side'] = PositionSide[data['side']]
            except KeyError:
                # Try case-insensitive match
                for enum_val in PositionSide:
                    if enum_val.name.lower() == data['side'].lower():
                        data['side'] = enum_val
                        break
        
        super().__init__(**data)

    @field_validator('quantity')
    @classmethod
    def quantity_non_negative(cls, v: Decimal) -> Decimal:
        # Internal quantity should always be positive; side determines direction
        if v < Decimal('0'):
            raise ValueError("Position quantity must be non-negative.")
        return v
        
    @model_validator(mode='after')
    def validate_entry_price(self) -> 'Position':
        """Validate that entry_price is positive when position has quantity."""
        if self.quantity > Decimal('0') and self.entry_price <= Decimal('0'):
            raise ValueError("Entry price must be positive for non-zero positions")
        return self

    def update_market_price(self, current_price: Decimal) -> None:
        """
        Update the position's unrealized profit/loss based on the current market price.
        
        Args:
            current_price: The current market price of the asset
        """
        if self.quantity <= Decimal('0'):
            self.unrealized_pnl = Decimal('0')
            return
            
        # Calculate unrealized P&L
        current_price_dec = Decimal(str(current_price)) if not isinstance(current_price, Decimal) else current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price_dec - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price_dec) * self.quantity
            
        self.last_update_time = utcnow()

    def update_position(self, trade_qty: Optional[Decimal], trade_price: Optional[Decimal], trade_side: OrderSide, current_market_price: Decimal) -> None:
        """
        Updates the position based on a new trade and current market price.
        
        This method handles various scenarios:
        - Opening a new position
        - Increasing an existing position
        - Reducing an existing position
        - Closing a position
        - Flipping from long to short or vice versa
        
        Args:
            trade_qty: Quantity of the trade
            trade_price: Price at which the trade executed
            trade_side: Buy or sell
            current_market_price: Current market price for PnL calculation
        """
        if trade_qty is None or trade_qty == Decimal('0'):
            logging.warning("Attempted to update position with zero or None quantity.")
            return
        if trade_price is None or trade_price <= Decimal('0'):
             logging.warning("Attempted to update position with zero, negative or None price.")
             return

        # Ensure inputs are Decimal
        trade_qty_dec = Decimal(str(trade_qty)) if not isinstance(trade_qty, Decimal) else trade_qty
        trade_price_dec = Decimal(str(trade_price)) if not isinstance(trade_price, Decimal) else trade_price
        
        if self.quantity == Decimal('0'): # Opening a new position
            self.side = PositionSide.LONG if trade_side == OrderSide.BUY else PositionSide.SHORT
            self.entry_price = trade_price_dec
            self.quantity = abs(trade_qty_dec)
            self.realized_pnl = Decimal('0') # Reset realized PnL for new position

        elif (self.side == PositionSide.LONG and trade_side == OrderSide.BUY) or \
             (self.side == PositionSide.SHORT and trade_side == OrderSide.SELL): # Increasing position size
            # Update average entry price
            current_value = self.entry_price * self.quantity
            trade_value = trade_price_dec * abs(trade_qty_dec)
            self.quantity += abs(trade_qty_dec)
            self.entry_price = (current_value + trade_value) / self.quantity

        else: # Reducing or closing position (or flipping)
            reduction_qty = min(self.quantity, abs(trade_qty_dec))
            trade_pnl = Decimal('0')
            if self.side == PositionSide.LONG:
                trade_pnl = (trade_price_dec - self.entry_price) * reduction_qty
            else: # Short position
                trade_pnl = (self.entry_price - trade_price_dec) * reduction_qty

            self.realized_pnl += trade_pnl
            self.quantity -= reduction_qty

            if self.quantity == Decimal('0'): # Position closed
                self.entry_price = Decimal('0')

            if self.quantity.is_zero() or abs(self.quantity) < Decimal('1e-10'):
                self.quantity = Decimal('0')
                self.entry_price = Decimal('0')

            if abs(trade_qty_dec) > reduction_qty: # Position flipped
                self.side = PositionSide.SHORT if self.side == PositionSide.LONG else PositionSide.LONG
                self.entry_price = trade_price_dec
                self.quantity = abs(trade_qty_dec) - reduction_qty # Remaining quantity after flip

        # Update unrealized P&L based on current market price AFTER position changes
        current_market_price_dec = Decimal(str(current_market_price)) if not isinstance(current_market_price, Decimal) else current_market_price
        self.update_market_price(current_market_price_dec)
        self.last_update_time = utcnow()

    @property
    def value(self) -> Decimal:
        """
        Calculate the current market value of the position.
        
        Returns:
            The quantity * entry_price
        """
        return self.quantity * self.entry_price

    def get_position_value(self, current_market_price: Optional[Decimal] = None) -> Decimal:
        """Calculate the current market value of the position."""
        price = current_market_price if current_market_price is not None else self.entry_price
        price_dec = Decimal(str(price)) if not isinstance(price, Decimal) else price # Ensure Decimal
        if price_dec is None or price_dec <= Decimal('0'):
            return Decimal('0') # Cannot value if price is unknown or zero
        return self.quantity * price_dec

    @property
    def total_pnl(self) -> Decimal:
        """
        Get the total profit/loss (realized + unrealized).
        
        Returns:
            The sum of realized and unrealized PnL
        """
        return self.realized_pnl + self.unrealized_pnl


# --- Portfolio State --- 
class Portfolio(BaseModel):
    """
    Represents the overall state of the trading account.
    
    The Portfolio is the central model that tracks all positions, orders, trades,
    and account balances. It provides methods to update the portfolio state based on
    executed trades and calculate various performance metrics.
    
    Attributes:
        account_id: Identifier for the trading account
        starting_balance: Initial cash balance when the portfolio was created
        current_balance: Current cash balance
        positions: Dictionary of open positions keyed by symbol
        orders: Dictionary of orders keyed by order_id
        trades: List of all executed trades
        timestamp: Last update timestamp
        total_realized_portfolio_pnl: Accumulated realized PnL from closed positions
    """
    account_id: str = "default_account"
    starting_balance: Decimal
    current_balance: Decimal # Cash balance
    positions: Dict[str, Position] = Field(default_factory=dict) # Keyed by symbol
    orders: Dict[str, Order] = Field(default_factory=dict) # Keyed by internal order_id
    trades: List[Trade] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)
    total_realized_portfolio_pnl: Decimal = Field(default_factory=decimal_zero)
    total_value: Decimal = Field(default_factory=decimal_zero)
    # Add equity, margin used, etc. later

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
        "arbitrary_types_allowed": True,  # Allow arbitrary types like pandas Timestamp
    }
    
    def __init__(self, **data):
        initial_capital = data.pop('initial_capital', None)
        # Handle aliases for backward compatibility
        if initial_capital is not None and 'starting_balance' not in data:
            initial_capital_dec = Decimal(str(initial_capital)) if not isinstance(initial_capital, Decimal) else initial_capital
            data['starting_balance'] = initial_capital_dec
            data['current_balance'] = initial_capital_dec
            
        super().__init__(**data)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get a position by symbol if it exists.
        
        Args:
            symbol: Trading pair symbol to look up
            
        Returns:
            The Position object if found, None otherwise
        """
        return self.positions.get(symbol)

    def update_from_trade(self, trade: Trade, current_market_prices: Dict[str, Decimal]) -> None:
        """
        Updates portfolio state based on an executed trade.
        
        This method:
        1. Updates or creates the relevant position
        2. Updates the cash balance
        3. Updates unrealized PnL for all positions
        4. Adds the trade to history
        
        Args:
            trade: The Trade object representing the fill.
            current_market_prices: Dictionary of current market prices keyed by symbol
        """
        if trade.quantity == Decimal('0'): # Ignore zero quantity trades
            return
        
        # Ensure trade timestamp is a Python datetime object
        from ..common.time_utils import to_utc_naive
        try:
            trade_copy = trade.model_copy()
            trade_copy.timestamp = to_utc_naive(trade.timestamp)
            trade = trade_copy
        except Exception:
            pass
            
        symbol = trade.symbol
        current_price = current_market_prices.get(symbol)
        
        # Ensure trade quantity and price are Decimal
        trade_qty_dec = Decimal(str(trade.quantity)) if not isinstance(trade.quantity, Decimal) else trade.quantity
        trade_price_dec = Decimal(str(trade.price)) if not isinstance(trade.price, Decimal) else trade.price
        trade_value = trade_qty_dec * trade_price_dec # Decimal * Decimal
        
        if current_price is None:
             # Cannot update PnL without current price, log warning?
             # Or use trade price as approximation for this update cycle?
             self.logger.warning(f"Missing current market price for {symbol} in update_from_trade. Using trade price {trade.price} as fallback.")
             current_price_dec = trade_price_dec # Already Decimal
        else:
            current_price_dec = Decimal(str(current_price)) if not isinstance(current_price, Decimal) else current_price
        
        # 1. Update/Create Position
        position = self.positions.get(symbol)
        
        if position:
            self.logger.info(f"Calling position.update_position for {symbol} with current_price: {current_price}")
            position.update_position(trade_qty_dec, trade_price_dec, trade.side, current_price_dec)
            if position.quantity == Decimal('0'):
                # Accumulate realized PnL from the closed position before deleting
                self.total_realized_portfolio_pnl += position.realized_pnl
                del self.positions[symbol] # Remove closed position
        else:
            # Create new position if it doesn't exist (should only happen if it was closed before)
            
            # If the trade closes a non-existent position (e.g. selling when no long position),
            # it effectively opens a position in the opposite direction.
            new_side: PositionSide = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
            new_position = Position(
                symbol=symbol,
                side=new_side,
                quantity=trade_qty_dec, # Ensured to be Decimal
                entry_price=trade_price_dec, # Ensured to be Decimal
            )
            self.logger.info(f"Calling calculate_position_pnl for new {symbol} with current_price: {current_price_dec}")
            calculate_position_pnl(new_position, current_price_dec)
 
            # Now add the potentially updated position to the portfolio
            self.positions[symbol] = new_position
            
        # 2. Update Cash Balance
        if trade.side == OrderSide.BUY:
            self.current_balance -= trade_value # Decrease cash on buy
        else: # SELL
            self.current_balance += trade_value # Increase cash on sell
        
        # Handle commission deduction if necessary (assuming commission is in quote asset)
        if trade.commission is not None and trade.commission > 0:
            commission_dec = Decimal(str(trade.commission)) if not isinstance(trade.commission, Decimal) else trade.commission
            self.current_balance -= commission_dec
 
        # 3. Update PnL for all positions (using potentially new market prices)
        self.update_all_unrealized_pnl(current_market_prices)
        
        # 4. Add trade to history
        self.trades.append(trade)

        # 5. Update Portfolio Timestamp
        self.timestamp = utcnow()

    def update_all_unrealized_pnl(self, current_market_prices: Dict[str, Decimal]) -> None:
        """
        Updates the unrealized PnL for all open positions based on current prices.
        
        Args:
            current_market_prices: Dictionary of current market prices keyed by symbol 
                                      (expected as Decimal or float/str convertible to Decimal).
        """
        for symbol, position in self.positions.items():
            current_price = current_market_prices.get(symbol)
            if current_price is not None:
                current_price_dec = Decimal(str(current_price)) if not isinstance(current_price, Decimal) else current_price
                calculate_position_pnl(position, current_price_dec)
            else:
                # Keep existing PnL if no current price is available? Log warning?
                self.logger.warning(f"Missing current market price for {symbol} in update_all_unrealized_pnl. Skipping PnL update for this position.")
 
    @property
    def total_equity(self) -> Decimal:
        """
        Calculates total equity (Cash + Value of Positions).
        
        Returns:
            The total portfolio equity value
        """
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values()) 
        return self.current_balance + total_unrealized # Decimal + Decimal
 
    @property
    def total_realized_pnl(self) -> Decimal:
        """
        Calculates total realized PnL across all positions.
        
        This includes PnL from currently open positions and accumulated PnL from closed positions.
        
        Returns:
            The total realized profit/loss
        """
        pnl_from_open_positions = sum(pos.realized_pnl for pos in self.positions.values())
        return self.total_realized_portfolio_pnl + pnl_from_open_positions
 
    def get_open_positions_count(self) -> int:
        """
        Get the number of open positions.
        
        Returns:
            The count of open positions
        """
        return len(self.positions)
        
    def get_position_exposure(self, symbol: str, current_market_price: Decimal) -> Decimal:
        """
        Calculate the exposure for a specific position.
        
        Args:
            symbol: The trading pair symbol
            current_market_price: Current market price for the symbol
            
        Returns:
            The position exposure as a percentage of total equity (0-1)
        """
        current_price_dec = Decimal(str(current_market_price)) if not isinstance(current_market_price, Decimal) else current_market_price
        position = self.get_position(symbol)
        total_equity = self.total_equity
        if not position or position.quantity == Decimal('0') or total_equity == Decimal('0'):
            return Decimal('0')
            
        position_value = position.quantity * current_price_dec
        return position_value / total_equity # Decimal / Decimal
 
    def update_total_value(self, current_market_prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate and update the total portfolio value.
        
        This includes cash balance plus the value of all open positions.
        
        Args:
            current_market_prices: Dictionary of current market prices keyed by symbol 
                                      (expected as Decimal or float/str convertible to Decimal).
        
        Returns:
            The updated total portfolio value
        """
        # Start with cash balance
        current_total_value = self.current_balance
        
        # Add value of all positions
        for symbol, position in self.positions.items():
            if position.quantity > Decimal('0'):
                current_price = current_market_prices.get(symbol)
                if current_price is not None:
                    current_price_dec = Decimal(str(current_price)) if not isinstance(current_price, Decimal) else current_price
                    position_value = position.get_position_value(current_price_dec)
                    current_total_value += position_value
                else:
                    # If no current price is available, use the entry price as a fallback
                    position_value = position.get_position_value()
                    current_total_value += position_value
                    self.logger.warning(f"No current market price for {symbol} in update_total_value. Using entry price as fallback.")
        
        # Store the total value
        self.total_value = current_total_value
        
        return current_total_value
 
    @property
    def cash(self) -> Decimal:
        """
        Property that returns the current cash balance.
        
        This is an alias for current_balance for backward compatibility.
        
        Returns:
            The current cash balance
        """
        return self.current_balance
