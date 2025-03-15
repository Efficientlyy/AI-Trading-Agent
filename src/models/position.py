"""Position models for the trading system.

This module defines the Position class, which represents an open trading position.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PositionStatus(str, Enum):
    """Status of a trading position."""
    PENDING = "pending"  # Position has been created but not yet filled
    OPEN = "open"        # Position is currently open
    CLOSED = "closed"    # Position has been closed
    CANCELLED = "cancelled"  # Position was cancelled before filling
    PARTIALLY_FILLED = "partially_filled"  # Position is partially filled


class PositionSide(str, Enum):
    """Side of a trading position."""
    LONG = "long"    # Long position (buy)
    SHORT = "short"  # Short position (sell)


class Position(BaseModel):
    """A trading position.
    
    This class represents a trading position, which can be either a long or short position.
    It tracks the position's status, entry and exit prices, and calculates profit and loss.
    
    Attributes:
        id: A unique identifier for this position
        exchange: The exchange where this position was taken
        symbol: The trading pair symbol (e.g., BTC/USDT)
        side: Whether this is a long or short position
        entry_price: The price at which this position was entered
        amount: The amount of the base asset in this position
        status: The current status of this position (pending, open, closed, etc.)
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        leverage: The leverage used for this position (default: 1.0)
        created_at: When this position was created (default: current time)
        opened_at: When this position was opened (set by open())
        closed_at: When this position was closed (set by close())
        exit_price: The price at which this position was closed
        realized_pnl: The realized profit or loss after closing
        fee_paid: Trading fees paid for this position
        metadata: Additional data about this position
        tags: Tags associated with this position
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str
    symbol: str
    side: PositionSide
    entry_price: float
    amount: float
    status: PositionStatus = PositionStatus.PENDING
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    fee_paid: float = 0.0
    metadata: Dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate the unrealized profit/loss for this position.
        
        Args:
            current_price: The current price of the asset
            
        Returns:
            float: The unrealized profit/loss
        """
        if self.status != PositionStatus.OPEN:
            return 0.0
        
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.amount * self.leverage
        else:  # SHORT
            return (self.entry_price - current_price) * self.amount * self.leverage
    
    def calculate_roi(self, current_price: float) -> float:
        """Calculate the return on investment (ROI) as a percentage.
        
        Args:
            current_price: The current price of the asset
            
        Returns:
            float: The ROI as a percentage (e.g., 5.2 for 5.2%)
        """
        if self.status != PositionStatus.OPEN or self.entry_price == 0:
            return 0.0
        
        invested = self.entry_price * self.amount
        if invested == 0:
            return 0.0
        
        pnl = self.calculate_unrealized_pnl(current_price)
        return (pnl / invested) * 100.0
    
    def is_stop_loss_triggered(self, current_price: float) -> bool:
        """Check if the stop loss for this position is triggered.
        
        Args:
            current_price: The current price of the asset
            
        Returns:
            bool: True if stop loss is triggered, False otherwise
        """
        if self.status != PositionStatus.OPEN or self.stop_loss is None:
            return False
        
        if self.side == PositionSide.LONG:
            return current_price <= self.stop_loss
        else:  # SHORT
            return current_price >= self.stop_loss
    
    def is_take_profit_triggered(self, current_price: float) -> bool:
        """Check if the take profit for this position is triggered.
        
        Args:
            current_price: The current price of the asset
            
        Returns:
            bool: True if take profit is triggered, False otherwise
        """
        if self.status != PositionStatus.OPEN or self.take_profit is None:
            return False
        
        if self.side == PositionSide.LONG:
            return current_price >= self.take_profit
        else:  # SHORT
            return current_price <= self.take_profit
    
    def close(self, exit_price: float, close_time: Optional[datetime] = None) -> None:
        """Close the position.
        
        Args:
            exit_price: The price at which the position was closed
            close_time: The time when the position was closed (defaults to now)
        """
        if self.status != PositionStatus.OPEN:
            return
        
        self.status = PositionStatus.CLOSED
        self.exit_price = exit_price
        self.closed_at = close_time or datetime.now(timezone.utc)
        
        # Calculate realized PnL
        if self.side == PositionSide.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.amount * self.leverage
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.amount * self.leverage
        
        # Subtract fees if tracked
        if self.fee_paid > 0:
            self.realized_pnl -= self.fee_paid
    
    def open(self, entry_time: Optional[datetime] = None) -> None:
        """Mark the position as open.
        
        Args:
            entry_time: The time when the position was opened (defaults to now)
        """
        if self.status not in (PositionStatus.PENDING, PositionStatus.PARTIALLY_FILLED):
            return
        
        self.status = PositionStatus.OPEN
        self.opened_at = entry_time or datetime.now(timezone.utc)