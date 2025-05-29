"""
Market data API models for pydantic validation.

This module provides pydantic models used by the market data API endpoints
to validate request and response data.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class AssetType(str, Enum):
    """Enumeration of asset types."""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"

class TimeFrame(str, Enum):
    """Enumeration of time frames for historical data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class MarketStatus(str, Enum):
    """Enumeration of market status."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"

class HistoricalBar(BaseModel):
    """Model for historical price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class MarketRegime(str, Enum):
    """Enumeration of market regimes."""
    UNKNOWN = "unknown"
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    VOLATILE_BULL = "volatile_bull"
    VOLATILE_BEAR = "volatile_bear"
    RECOVERY = "recovery"
    BREAKDOWN = "breakdown"
    TRENDING = "trending"

class VolatilityRegime(str, Enum):
    """Enumeration of volatility regimes."""
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"
    CRISIS = "crisis"

class Asset(BaseModel):
    """Model for asset data."""
    symbol: str
    name: str
    type: str
    price: float
    change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    source: Optional[str] = None
    confidence: Optional[int] = None
    # New fields for enhanced asset info
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    volatility: Optional[str] = None
    market_cap: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    total_supply: Optional[float] = None
    current_regime: Optional[MarketRegime] = None
    volatility_regime: Optional[VolatilityRegime] = None

class AssetsResponse(BaseModel):
    """Response model for assets list."""
    assets: List[Asset]

class AssetDetailResponse(BaseModel):
    """Response model for detailed asset information."""
    asset: Asset
    market_status: MarketStatus = MarketStatus.OPEN
    last_updated: datetime
    historical_data: Optional[List[HistoricalBar]] = None
    related_assets: Optional[List[Asset]] = None
    market_sentiment: Optional[float] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    
class MarketOverviewResponse(BaseModel):
    """Response model for market overview."""
    timestamp: datetime
    market_status: MarketStatus = MarketStatus.OPEN
    top_gainers: List[Asset]
    top_losers: List[Asset]
    most_volatile: List[Asset]
    market_sentiment: Optional[float] = None
    trading_volume: Optional[float] = None
    
class HistoricalDataRequest(BaseModel):
    """Request model for historical data."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USD')")
    timeframe: TimeFrame = Field(TimeFrame.DAY_1, description="Timeframe for historical data")
    limit: int = Field(100, description="Number of data points to return", ge=1, le=1000)
    start_date: Optional[datetime] = Field(None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    
class HistoricalDataResponse(BaseModel):
    """Response model for historical data."""
    symbol: str
    timeframe: TimeFrame
    data: List[HistoricalBar]
    currency: str = "USD"
    
class TechnicalIndicatorsResponse(BaseModel):
    """Response model for technical indicators."""
    symbol: str
    timestamp: datetime
    indicators: Dict[str, Any]
