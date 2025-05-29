"""
Market Data API endpoints for External API Gateway.

This module implements the market data endpoints for external partners,
providing access to historical and real-time market data.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import authentication and rate limiting
from ..auth import APIKeyAuth, JWTAuth
from ..config import PartnerTier

# Import data services
from ....data_sources.market_data import MarketDataService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/market-data", tags=["Market Data"])


# Data models
class HistoricalDataRequest(BaseModel):
    """Request model for historical data."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL', 'BTC-USD')")
    start_date: datetime = Field(..., description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    interval: str = Field("1d", description="Data interval ('1m', '5m', '15m', '1h', '1d', '1w')")
    include_extended_hours: bool = Field(False, description="Include extended hours data")
    adjust_for_splits: bool = Field(True, description="Adjust for stock splits")
    adjust_for_dividends: bool = Field(True, description="Adjust for dividends")


class BarData(BaseModel):
    """Model for OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None


class HistoricalDataResponse(BaseModel):
    """Response model for historical data."""
    symbol: str
    interval: str
    bars: List[BarData]
    currency: str
    timezone: str = "UTC"


class RealTimeQuote(BaseModel):
    """Model for real-time quote data."""
    symbol: str
    price: float
    change: float
    change_percent: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: int
    timestamp: datetime


class SymbolMetadata(BaseModel):
    """Model for symbol metadata."""
    symbol: str
    name: str
    exchange: str
    asset_class: str
    currency: str
    timezone: str
    is_tradable: bool
    is_shortable: Optional[bool] = None
    min_tick_size: Optional[float] = None


# Market data service instance
# This would normally be injected, but we'll create it here for simplicity
market_data_service = MarketDataService()


# Endpoints
@router.get("/symbols", response_model=List[str])
async def get_available_symbols(
    exchange: Optional[str] = None,
    asset_class: Optional[str] = None,
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get a list of available trading symbols.
    
    This endpoint returns a list of available trading symbols,
    optionally filtered by exchange and asset class.
    
    Args:
        exchange: Optional filter by exchange
        asset_class: Optional filter by asset class
        auth: Authentication information
        
    Returns:
        List of symbol strings
    """
    try:
        symbols = await market_data_service.get_available_symbols(
            exchange=exchange,
            asset_class=asset_class
        )
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch symbols: {str(e)}"
        )


@router.get("/symbols/{symbol}", response_model=SymbolMetadata)
async def get_symbol_metadata(
    symbol: str = Path(..., description="Trading symbol"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get metadata for a specific trading symbol.
    
    This endpoint returns detailed metadata for a specific
    trading symbol, including exchange, asset class, and
    tradability information.
    
    Args:
        symbol: Trading symbol
        auth: Authentication information
        
    Returns:
        Symbol metadata
    """
    try:
        metadata = await market_data_service.get_symbol_metadata(symbol)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol not found: {symbol}"
            )
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching symbol metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch symbol metadata: {str(e)}"
        )


@router.post("/historical", response_model=HistoricalDataResponse)
async def get_historical_data(
    request: HistoricalDataRequest,
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get historical market data for a symbol.
    
    This endpoint returns historical OHLCV (Open, High, Low, Close, Volume)
    data for a specified trading symbol and time range.
    
    The data is returned in chronological order.
    
    Args:
        request: Historical data request parameters
        auth: Authentication information
        
    Returns:
        Historical OHLCV data
    """
    # Check partner tier for historical data access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to historical data
    if partner_tier == PartnerTier.PUBLIC.value:
        raise HTTPException(
            status_code=403,
            detail="Historical data access requires a paid subscription"
        )
    
    # Set end date to now if not provided
    end_date = request.end_date or datetime.utcnow()
    
    # Limit historical data range based on tier
    max_days = {
        PartnerTier.BASIC.value: 365,    # 1 year
        PartnerTier.PREMIUM.value: 1825, # 5 years
        PartnerTier.ENTERPRISE.value: None  # Unlimited
    }.get(partner_tier, 30)  # Default to 30 days
    
    if max_days is not None:
        min_date = end_date - timedelta(days=max_days)
        if request.start_date < min_date:
            if partner_tier == PartnerTier.BASIC.value:
                raise HTTPException(
                    status_code=403,
                    detail=f"Basic tier limited to {max_days} days of historical data"
                )
            # For other tiers, just limit the range
            request.start_date = min_date
    
    try:
        bars = await market_data_service.get_historical_bars(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=end_date,
            interval=request.interval,
            include_extended_hours=request.include_extended_hours,
            adjust_for_splits=request.adjust_for_splits,
            adjust_for_dividends=request.adjust_for_dividends
        )
        
        # Get currency for the symbol
        metadata = await market_data_service.get_symbol_metadata(request.symbol)
        currency = metadata.get("currency", "USD") if metadata else "USD"
        
        return HistoricalDataResponse(
            symbol=request.symbol,
            interval=request.interval,
            bars=bars,
            currency=currency
        )
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch historical data: {str(e)}"
        )


@router.get("/quotes", response_model=List[RealTimeQuote])
async def get_real_time_quotes(
    symbols: List[str] = Query(..., description="List of trading symbols"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get real-time quotes for multiple symbols.
    
    This endpoint returns real-time market quotes for a list of
    trading symbols.
    
    Args:
        symbols: List of trading symbols
        auth: Authentication information
        
    Returns:
        List of real-time quotes
    """
    # Check partner tier for real-time data access
    partner_tier = auth.get("tier", PartnerTier.PUBLIC.value)
    
    # Verify tier has access to real-time data
    if partner_tier in [PartnerTier.PUBLIC.value, PartnerTier.BASIC.value]:
        raise HTTPException(
            status_code=403,
            detail="Real-time data access requires Premium or Enterprise subscription"
        )
    
    # Limit number of symbols based on tier
    max_symbols = {
        PartnerTier.PREMIUM.value: 20,
        PartnerTier.ENTERPRISE.value: 100
    }.get(partner_tier, 5)
    
    if len(symbols) > max_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {max_symbols} symbols allowed for {partner_tier} tier"
        )
    
    try:
        quotes = await market_data_service.get_real_time_quotes(symbols)
        return quotes
    except Exception as e:
        logger.error(f"Error fetching real-time quotes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch real-time quotes: {str(e)}"
        )


@router.get("/market-hours", response_model=Dict[str, Any])
async def get_market_hours(
    markets: Optional[List[str]] = Query(None, description="List of markets to check"),
    date: Optional[datetime] = Query(None, description="Date to check (defaults to today)"),
    auth: Dict[str, Any] = Depends(APIKeyAuth()),
):
    """
    Get market hours information.
    
    This endpoint returns the opening and closing times for different
    markets, including special holidays and early closings.
    
    Args:
        markets: List of markets to check (e.g., 'NASDAQ', 'NYSE')
        date: Date to check (defaults to today)
        auth: Authentication information
        
    Returns:
        Dictionary of market hours information
    """
    try:
        # Default to today if date not provided
        query_date = date or datetime.utcnow().date()
        
        # Default markets if not provided
        query_markets = markets or ["NASDAQ", "NYSE", "OTC"]
        
        market_hours = await market_data_service.get_market_hours(
            markets=query_markets,
            date=query_date
        )
        return market_hours
    except Exception as e:
        logger.error(f"Error fetching market hours: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch market hours: {str(e)}"
        )
