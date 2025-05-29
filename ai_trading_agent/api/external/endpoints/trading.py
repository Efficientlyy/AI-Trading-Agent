"""
Trading API endpoints for External API Gateway.

This module implements the trading endpoints for external partners,
providing access to order placement, execution, and management.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Import authentication
from ..auth import APIKeyAuth, JWTAuth
from ..config import PartnerTier

# Import trading services (placeholder - would be implemented elsewhere)
from ....execution.trading import TradingService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/trading", tags=["Trading"])


# Enums and data models
class OrderType(str, Enum):
    """Types of trading orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good 'til canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good 'til date


class OrderStatus(str, Enum):
    """Order statuses."""
    NEW = "new"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PENDING = "pending"
    EXPIRED = "expired"


class OrderRequest(BaseModel):
    """Request model for placing an order."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL', 'BTC-USD')")
    side: OrderSide = Field(..., description="Order side (buy or sell)")
    type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    
    # Optional fields depending on order type
    limit_price: Optional[float] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    extended_hours: bool = Field(False, description="Whether to allow trading in extended hours")
    
    # For trailing stop orders
    trail_percent: Optional[float] = Field(None, description="Trail percentage for trailing stop orders")
    trail_price: Optional[float] = Field(None, description="Trail price for trailing stop orders")
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        """Validate limit price is provided for relevant order types."""
        if 'type' in values and values['type'] in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError(f"Limit price is required for {values['type']} orders")
        return v

    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Validate stop price is provided for relevant order types."""
        if 'type' in values and values['type'] in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError(f"Stop price is required for {values['type']} orders")
        return v

    @validator('trail_percent', 'trail_price')
    def validate_trailing_stop(cls, v, values):
        """Validate trailing stop parameters."""
        if 'type' in values and values['type'] == OrderType.TRAILING_STOP:
            if values.get('trail_percent') is None and values.get('trail_price') is None:
                raise ValueError("Either trail_percent or trail_price is required for trailing stop orders")
        return v


class Order(BaseModel):
    """Model for a trading order."""
    id: UUID = Field(..., description="Order ID")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    created_at: datetime = Field(..., description="Order creation timestamp")
    updated_at: datetime = Field(..., description="Order last update timestamp")
    submitted_at: Optional[datetime] = Field(None, description="Order submission timestamp")
    filled_at: Optional[datetime] = Field(None, description="Order fill timestamp")
    expired_at: Optional[datetime] = Field(None, description="Order expiration timestamp")
    canceled_at: Optional[datetime] = Field(None, description="Order cancellation timestamp")
    
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    time_in_force: TimeInForce = Field(..., description="Time in force")
    status: OrderStatus = Field(..., description="Order status")
    
    quantity: float = Field(..., description="Order quantity")
    filled_quantity: float = Field(0, description="Filled quantity")
    filled_average_price: Optional[float] = Field(None, description="Filled average price")
    
    limit_price: Optional[float] = Field(None, description="Limit price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    trail_percent: Optional[float] = Field(None, description="Trail percentage")
    trail_price: Optional[float] = Field(None, description="Trail price")
    
    extended_hours: bool = Field(False, description="Extended hours trading")
    
    fees: float = Field(0, description="Trading fees")
    commission: float = Field(0, description="Commission fees")
    
    reject_reason: Optional[str] = Field(None, description="Rejection reason if applicable")
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            UUID: lambda v: str(v)
        }


class Position(BaseModel):
    """Model for a trading position."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Current position quantity")
    side: str = Field(..., description="Position side (long or short)")
    entry_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    cost_basis: float = Field(..., description="Cost basis")
    unrealized_pl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pl_percent: float = Field(..., description="Unrealized profit/loss percentage")
    realized_pl: float = Field(..., description="Realized profit/loss for this position")


class Account(BaseModel):
    """Model for a trading account."""
    id: UUID = Field(..., description="Account ID")
    buying_power: float = Field(..., description="Available buying power")
    cash: float = Field(..., description="Cash balance")
    equity: float = Field(..., description="Total equity")
    portfolio_value: float = Field(..., description="Total portfolio value")
    initial_margin: float = Field(..., description="Initial margin requirement")
    maintenance_margin: float = Field(..., description="Maintenance margin requirement")
    day_trade_count: int = Field(0, description="Number of day trades")
    account_blocked: bool = Field(False, description="Account blocked status")
    trading_blocked: bool = Field(False, description="Trading blocked status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    status: str = Field(..., description="Account status")


# Service instance
trading_service = TradingService()


# Endpoints
@router.post("/orders", response_model=Order, status_code=201)
async def place_order(
    order_request: OrderRequest,
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Place a new trading order.
    
    This endpoint allows placing various types of trading orders,
    including market, limit, stop, and trailing stop orders.
    
    Args:
        order_request: Order details
        auth: Authentication information
        
    Returns:
        Order object with status and details
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    # Check if this order type is allowed for this tier
    complex_order_types = [OrderType.TRAILING_STOP, OrderType.STOP_LIMIT]
    if order_request.type in complex_order_types and partner_tier != PartnerTier.ENTERPRISE.value:
        raise HTTPException(
            status_code=403,
            detail=f"{order_request.type} orders require Enterprise subscription"
        )
    
    # Check extended hours trading permission
    if order_request.extended_hours and partner_tier != PartnerTier.ENTERPRISE.value:
        raise HTTPException(
            status_code=403,
            detail="Extended hours trading requires Enterprise subscription"
        )
    
    try:
        # Place the order
        order = await trading_service.place_order(
            symbol=order_request.symbol,
            side=order_request.side.value,
            type=order_request.type.value,
            quantity=order_request.quantity,
            limit_price=order_request.limit_price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force.value,
            client_order_id=order_request.client_order_id,
            extended_hours=order_request.extended_hours,
            trail_percent=order_request.trail_percent,
            trail_price=order_request.trail_price,
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        return order
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to place order: {str(e)}"
        )


@router.get("/orders", response_model=List[Order])
async def list_orders(
    status: Optional[OrderStatus] = Query(None, description="Filter by order status"),
    side: Optional[OrderSide] = Query(None, description="Filter by order side"),
    symbols: Optional[List[str]] = Query(None, description="Filter by symbols"),
    after: Optional[datetime] = Query(None, description="Filter by orders after this time"),
    before: Optional[datetime] = Query(None, description="Filter by orders before this time"),
    limit: int = Query(100, description="Maximum number of orders to return"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    List trading orders.
    
    This endpoint returns a list of orders, optionally filtered
    by status, side, symbols, and time range.
    
    Args:
        status: Filter by order status
        side: Filter by order side
        symbols: Filter by symbols
        after: Filter by orders after this time
        before: Filter by orders before this time
        limit: Maximum number of orders to return
        auth: Authentication information
        
    Returns:
        List of order objects
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    # Limit order history based on tier
    max_days = {
        PartnerTier.PREMIUM.value: 30,   # 30 days
        PartnerTier.ENTERPRISE.value: 90  # 90 days
    }.get(partner_tier, 7)  # Default to 7 days
    
    # Calculate default 'after' date if not provided
    if after is None:
        after = datetime.utcnow() - timedelta(days=max_days)
    
    # Enforce limits based on tier
    if partner_tier == PartnerTier.PREMIUM.value:
        earliest_allowed = datetime.utcnow() - timedelta(days=max_days)
        if after < earliest_allowed:
            after = earliest_allowed
    
    try:
        orders = await trading_service.list_orders(
            status=status.value if status else None,
            side=side.value if side else None,
            symbols=symbols,
            after=after,
            before=before,
            limit=limit,
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        return orders
    except Exception as e:
        logger.error(f"Error listing orders: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list orders: {str(e)}"
        )


@router.get("/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: UUID = Path(..., description="Order ID"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get a specific order by ID.
    
    This endpoint returns details of a specific order by its ID.
    
    Args:
        order_id: Order ID
        auth: Authentication information
        
    Returns:
        Order object
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    try:
        order = await trading_service.get_order(
            order_id=order_id,
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        if not order:
            raise HTTPException(
                status_code=404,
                detail=f"Order not found: {order_id}"
            )
        
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch order: {str(e)}"
        )


@router.delete("/orders/{order_id}", response_model=Order)
async def cancel_order(
    order_id: UUID = Path(..., description="Order ID"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Cancel a specific order by ID.
    
    This endpoint cancels a specific order by its ID.
    
    Args:
        order_id: Order ID
        auth: Authentication information
        
    Returns:
        Canceled order object
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    try:
        canceled_order = await trading_service.cancel_order(
            order_id=order_id,
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        if not canceled_order:
            raise HTTPException(
                status_code=404,
                detail=f"Order not found or already canceled: {order_id}"
            )
        
        return canceled_order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling order: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel order: {str(e)}"
        )


@router.get("/positions", response_model=List[Position])
async def list_positions(
    symbols: Optional[List[str]] = Query(None, description="Filter by symbols"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    List current trading positions.
    
    This endpoint returns a list of current trading positions,
    optionally filtered by symbols.
    
    Args:
        symbols: Filter by symbols
        auth: Authentication information
        
    Returns:
        List of position objects
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    try:
        positions = await trading_service.list_positions(
            symbols=symbols,
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        return positions
    except Exception as e:
        logger.error(f"Error listing positions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list positions: {str(e)}"
        )


@router.get("/account", response_model=Account)
async def get_account(
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get trading account information.
    
    This endpoint returns information about the trading account,
    including balances, buying power, and status.
    
    Args:
        auth: Authentication information
        
    Returns:
        Account object
    """
    # Check partner tier for trading access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to trading
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Trading requires Premium or Enterprise subscription"
        )
    
    try:
        account = await trading_service.get_account(
            partner_id=auth.get("partner_id")  # Pass partner ID for tracking
        )
        
        return account
    except Exception as e:
        logger.error(f"Error fetching account: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch account: {str(e)}"
        )
