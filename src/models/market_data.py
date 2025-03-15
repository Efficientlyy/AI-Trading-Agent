"""Models for market data used throughout the system."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TimeFrame(str, Enum):
    """Standard timeframes for candlestick data."""
    
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    ONE_HOUR = "1h"


class CandleData(BaseModel):
    """Model for candlestick data."""
    
    symbol: str
    exchange: str
    timestamp: datetime
    timeframe: TimeFrame
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True
    trades: Optional[int] = None
    vwap: Optional[float] = None
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value):
        """Validate that the symbol is in the correct format."""
        if not isinstance(value, str) or "/" not in value:
            raise ValueError(f"Symbol must be in format 'BASE/QUOTE', got {value}")
        return value
    
    @field_validator("high", "low", "open", "close")
    @classmethod
    def validate_prices(cls, value):
        """Validate that prices are positive."""
        if value <= 0:
            raise ValueError(f"Price must be positive, got {value}")
        return value
    
    @field_validator("volume")
    @classmethod
    def validate_volume(cls, value):
        """Validate that volume is non-negative."""
        if value < 0:
            raise ValueError(f"Volume must be non-negative, got {value}")
        return value
    
    @property
    def range(self) -> float:
        """Calculate the range (high - low) of the candle."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Calculate the body (abs(close - open)) of the candle."""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """Check if the candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if the candle is bearish (close < open)."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if the candle is a doji (open approximately equals close)."""
        return abs(self.close - self.open) <= 0.0001 * self.open


class OrderBookData(BaseModel):
    """Model for order book data."""
    
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[Dict[str, float]] = Field(..., description="List of bid prices and sizes")
    asks: List[Dict[str, float]] = Field(..., description="List of ask prices and sizes")
    
    @field_validator("bids", "asks")
    @classmethod
    def validate_orders(cls, value):
        """Validate that orders have price and size."""
        for order in value:
            if "price" not in order or "size" not in order:
                raise ValueError("Each order must have 'price' and 'size'")
            if order["price"] <= 0 or order["size"] <= 0:
                raise ValueError("Price and size must be positive")
        return value
    
    @property
    def best_bid(self) -> float:
        """Get the best bid price."""
        if not self.bids:
            return 0.0
        return max(order["price"] for order in self.bids)
    
    @property
    def best_ask(self) -> float:
        """Get the best ask price."""
        if not self.asks:
            return 0.0
        return min(order["price"] for order in self.asks)
    
    @property
    def spread(self) -> float:
        """Calculate the spread (best_ask - best_bid)."""
        return self.best_ask - self.best_bid
    
    @property
    def spread_percent(self) -> float:
        """Calculate the spread as a percentage of the best bid."""
        if self.best_bid == 0:
            return 0.0
        return (self.spread / self.best_bid) * 100


class TradeData(BaseModel):
    """Model for individual trade data."""
    
    symbol: str
    exchange: str
    timestamp: datetime
    price: float
    size: float
    side: str = Field(..., description="'buy' or 'sell'")
    trade_id: Optional[str] = None
    
    @field_validator("side")
    @classmethod
    def validate_side(cls, value):
        """Validate that the side is either 'buy' or 'sell'."""
        if value.lower() not in ["buy", "sell"]:
            raise ValueError(f"Side must be 'buy' or 'sell', got {value}")
        return value.lower()
    
    @field_validator("price", "size")
    @classmethod
    def validate_positive(cls, value):
        """Validate that price and size are positive."""
        if value <= 0:
            raise ValueError(f"Value must be positive, got {value}")
        return value