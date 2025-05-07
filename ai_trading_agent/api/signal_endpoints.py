"""
Trading Signals API Endpoints

This module provides FastAPI endpoints for retrieving trading signals
from the signal integration service.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

# Local imports
from ai_trading_agent.signal_generation.signal_integration_service import (
    SignalIntegrationService,
    SignalType,
    SignalSource,
    get_signals_for_symbol,
    get_signals_for_multiple_symbols
)
from ai_trading_agent.data_collectors.twelve_data_client import TwelveDataClient

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/signals",
    tags=["signals"],
    responses={404: {"description": "Not found"}},
)

# Signal service instance
_signal_service = None


def get_signal_service() -> SignalIntegrationService:
    """
    Get or create a SignalIntegrationService instance
    
    Returns:
        SignalIntegrationService: The signal service instance
    """
    global _signal_service
    if _signal_service is None:
        # Get API keys from environment variables
        alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Create service with default parameters
        _signal_service = SignalIntegrationService(
            sentiment_weight=0.3,
            technical_weight=0.7,
            sentiment_threshold=0.2,
            rsi_overbought=70,
            rsi_oversold=30,
            bollinger_band_threshold=0.05,
            macd_signal_threshold=0.0,
            alpha_vantage_api_key=alpha_vantage_api_key
        )
    
    return _signal_service


# Data client instance
_data_client = None


def get_data_client() -> TwelveDataClient:
    """
    Get or create a TwelveDataClient instance
    
    Returns:
        TwelveDataClient: The data client instance
    """
    global _data_client
    if _data_client is None:
        # Get API key from environment variable
        twelve_data_api_key = os.getenv("TWELVE_DATA_API_KEY")
        
        # Create client
        _data_client = TwelveDataClient(api_key=twelve_data_api_key)
    
    return _data_client


# Pydantic models for request and response
class SignalResponse(BaseModel):
    """Model for a trading signal response"""
    id: str
    symbol: str
    timestamp: str
    type: str
    source: str
    strength: float
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalsResponse(BaseModel):
    """Model for a trading signals response"""
    technical: List[SignalResponse] = Field(default_factory=list)
    sentiment: List[SignalResponse] = Field(default_factory=list)
    combined: List[SignalResponse] = Field(default_factory=list)


class MultiSymbolSignalsResponse(BaseModel):
    """Model for a multi-symbol trading signals response"""
    signals: Dict[str, SignalsResponse] = Field(default_factory=dict)


@router.get("/{symbol}", response_model=SignalsResponse)
async def get_signals(
    symbol: str,
    timeframe: str = Query("1d", description="Timeframe for price data (e.g., 1h, 4h, 1d)"),
    days: int = Query(30, description="Number of days of historical data to use"),
    force_refresh: bool = Query(False, description="Force refresh of cached signals"),
    service: SignalIntegrationService = Depends(get_signal_service),
    data_client: TwelveDataClient = Depends(get_data_client)
):
    """
    Get trading signals for a symbol
    
    Args:
        symbol: Trading symbol (e.g., BTC/USDT)
        timeframe: Timeframe for price data (e.g., 1h, 4h, 1d)
        days: Number of days of historical data to use
        force_refresh: Force refresh of cached signals
        service: SignalIntegrationService instance
        data_client: TwelveDataClient instance
        
    Returns:
        SignalsResponse: Trading signals for the symbol
    """
    try:
        # Normalize symbol format (e.g., BTC/USDT -> BTC/USDT)
        normalized_symbol = symbol.upper()
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch historical price data
        price_data = await data_client.get_historical_prices(
            symbol=normalized_symbol,
            interval=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        
        # Get signals
        signals = await get_signals_for_symbol(
            symbol=normalized_symbol,
            price_data=df,
            service=service,
            force_refresh=force_refresh
        )
        
        # Convert to response model
        response = SignalsResponse(
            technical=[SignalResponse(**signal) for signal in signals["technical"]],
            sentiment=[SignalResponse(**signal) for signal in signals["sentiment"]],
            combined=[SignalResponse(**signal) for signal in signals["combined"]]
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=MultiSymbolSignalsResponse)
async def get_signals_for_symbols(
    symbols: List[str] = Query(..., description="List of trading symbols"),
    timeframe: str = Query("1d", description="Timeframe for price data"),
    days: int = Query(30, description="Number of days of historical data to use"),
    force_refresh: bool = Query(False, description="Force refresh of cached signals"),
    service: SignalIntegrationService = Depends(get_signal_service),
    data_client: TwelveDataClient = Depends(get_data_client)
):
    """
    Get trading signals for multiple symbols
    
    Args:
        symbols: List of trading symbols
        timeframe: Timeframe for price data
        days: Number of days of historical data to use
        force_refresh: Force refresh of cached signals
        service: SignalIntegrationService instance
        data_client: TwelveDataClient instance
        
    Returns:
        MultiSymbolSignalsResponse: Trading signals for multiple symbols
    """
    try:
        # Normalize symbols
        normalized_symbols = [symbol.upper() for symbol in symbols]
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch historical price data for all symbols
        price_data_dict = {}
        for symbol in normalized_symbols:
            try:
                # Fetch historical price data
                price_data = await data_client.get_historical_prices(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(price_data)
                
                # Add to dictionary
                price_data_dict[symbol] = df
            
            except Exception as e:
                logger.error(f"Error fetching price data for {symbol}: {str(e)}")
                # Continue with other symbols
        
        # Get signals for all symbols
        signals_dict = await get_signals_for_multiple_symbols(
            symbols=normalized_symbols,
            price_data_dict=price_data_dict,
            service=service,
            force_refresh=force_refresh
        )
        
        # Convert to response model
        response = MultiSymbolSignalsResponse(
            signals={
                symbol: SignalsResponse(
                    technical=[SignalResponse(**signal) for signal in signals["technical"]],
                    sentiment=[SignalResponse(**signal) for signal in signals["sentiment"]],
                    combined=[SignalResponse(**signal) for signal in signals["combined"]]
                )
                for symbol, signals in signals_dict.items()
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting signals for multiple symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{symbol}", response_model=List[SignalResponse])
async def get_latest_signals(
    symbol: str,
    source: Optional[str] = Query(None, description="Signal source (TECHNICAL, SENTIMENT, COMBINED)"),
    limit: int = Query(10, description="Maximum number of signals to return"),
    service: SignalIntegrationService = Depends(get_signal_service),
    data_client: TwelveDataClient = Depends(get_data_client)
):
    """
    Get the latest trading signals for a symbol
    
    Args:
        symbol: Trading symbol
        source: Signal source (TECHNICAL, SENTIMENT, COMBINED)
        limit: Maximum number of signals to return
        service: SignalIntegrationService instance
        data_client: TwelveDataClient instance
        
    Returns:
        List[SignalResponse]: Latest trading signals for the symbol
    """
    try:
        # Normalize symbol
        normalized_symbol = symbol.upper()
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        # Fetch historical price data
        price_data = await data_client.get_historical_prices(
            symbol=normalized_symbol,
            interval="1d",  # Daily data for latest signals
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        
        # Get signals
        signals = await get_signals_for_symbol(
            symbol=normalized_symbol,
            price_data=df,
            service=service,
            force_refresh=True  # Always get fresh signals for latest
        )
        
        # Filter by source if specified
        if source:
            source = source.upper()
            if source in ["TECHNICAL", "SENTIMENT", "COMBINED"]:
                all_signals = signals[source.lower()]
            else:
                # Invalid source, return all signals
                all_signals = (
                    signals["technical"] +
                    signals["sentiment"] +
                    signals["combined"]
                )
        else:
            # No source specified, return all signals
            all_signals = (
                signals["technical"] +
                signals["sentiment"] +
                signals["combined"]
            )
        
        # Sort by timestamp (newest first)
        all_signals.sort(
            key=lambda s: datetime.fromisoformat(s["timestamp"]),
            reverse=True
        )
        
        # Limit the number of signals
        all_signals = all_signals[:limit]
        
        # Convert to response model
        response = [SignalResponse(**signal) for signal in all_signals]
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting latest signals for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/websocket-feed", response_model=List[SignalResponse])
async def get_websocket_feed_data(
    symbols: List[str] = Query(..., description="List of trading symbols to monitor"),
    limit: int = Query(20, description="Maximum number of signals to return"),
    service: SignalIntegrationService = Depends(get_signal_service),
    data_client: TwelveDataClient = Depends(get_data_client)
):
    """
    Get initial data for the WebSocket feed
    
    Args:
        symbols: List of trading symbols to monitor
        limit: Maximum number of signals to return
        service: SignalIntegrationService instance
        data_client: TwelveDataClient instance
        
    Returns:
        List[SignalResponse]: Initial trading signals for the WebSocket feed
    """
    try:
        # Normalize symbols
        normalized_symbols = [symbol.upper() for symbol in symbols]
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        # Fetch historical price data for all symbols
        price_data_dict = {}
        for symbol in normalized_symbols:
            try:
                # Fetch historical price data
                price_data = await data_client.get_historical_prices(
                    symbol=symbol,
                    interval="1d",  # Daily data for WebSocket feed
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(price_data)
                
                # Add to dictionary
                price_data_dict[symbol] = df
            
            except Exception as e:
                logger.error(f"Error fetching price data for {symbol}: {str(e)}")
                # Continue with other symbols
        
        # Get signals for all symbols
        signals_dict = await get_signals_for_multiple_symbols(
            symbols=normalized_symbols,
            price_data_dict=price_data_dict,
            service=service,
            force_refresh=True  # Always get fresh signals for WebSocket feed
        )
        
        # Combine all signals
        all_signals = []
        for symbol, signals in signals_dict.items():
            all_signals.extend(signals["technical"])
            all_signals.extend(signals["sentiment"])
            all_signals.extend(signals["combined"])
        
        # Sort by timestamp (newest first)
        all_signals.sort(
            key=lambda s: datetime.fromisoformat(s["timestamp"]),
            reverse=True
        )
        
        # Limit the number of signals
        all_signals = all_signals[:limit]
        
        # Convert to response model
        response = [SignalResponse(**signal) for signal in all_signals]
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting WebSocket feed data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
