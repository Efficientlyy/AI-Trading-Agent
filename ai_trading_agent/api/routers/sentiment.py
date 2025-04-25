"""
Sentiment API Router

This module provides FastAPI endpoints for accessing sentiment data from Alpha Vantage.
It's designed to work within the rate limits of the free tier while providing
the necessary data for the trading system's dashboard.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ...sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector
from ..sentiment_api import SentimentAPI

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Define response models
class SentimentSignalModel(BaseModel):
    symbol: str
    signal: str
    strength: float
    score: float
    trend: float
    volatility: float
    timestamp: str

class SentimentSummaryModel(BaseModel):
    sentimentData: dict[str, SentimentSignalModel]
    timestamp: str

class HistoricalSentimentModel(BaseModel):
    timestamp: str
    score: float
    raw_score: float

# Dependency for getting SentimentAPI
def get_sentiment_api(use_mock: bool = False):
    """Dependency to get SentimentAPI instance"""
    # In production, you might want to use a more sophisticated
    # dependency injection system that caches the API instance
    return SentimentAPI(use_mock=use_mock)

# Define endpoints
@router.get("/summary", response_model=SentimentSummaryModel)
async def get_sentiment_summary(
    symbols: Optional[List[str]] = Query(None, description="List of symbols to get sentiment for"),
    sentiment_api: SentimentAPI = Depends(get_sentiment_api)
):
    """
    Get sentiment summary for multiple symbols.
    
    This endpoint provides a summary of sentiment signals for the specified symbols,
    which can be used to generate trading signals or display on a dashboard.
    
    If no symbols are provided, a default set of crypto symbols will be used.
    """
    try:
        # Default to crypto symbols if none provided
        symbols_to_use = symbols or ["BTC", "ETH", "XRP", "ADA", "SOL", "DOGE"]
        
        # Log request for debugging
        logger.info(f"Getting sentiment summary for {symbols_to_use}")
        
        # Get sentiment summary
        result = sentiment_api.get_sentiment_summary(symbols_to_use)
        
        # Return result
        return result
    except Exception as e:
        logger.error(f"Error getting sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical", response_model=List[HistoricalSentimentModel])
async def get_historical_sentiment(
    symbol: str = Query(..., description="Symbol to get historical sentiment for"),
    timeframe: str = Query("1M", description="Timeframe ('1D', '1W', '1M', '3M', '1Y')"),
    sentiment_api: SentimentAPI = Depends(get_sentiment_api)
):
    """
    Get historical sentiment data for a specific symbol.
    
    This endpoint provides time series sentiment data for the specified symbol,
    which can be used for charting sentiment trends or backtesting sentiment-based strategies.
    """
    try:
        # Log request for debugging
        logger.info(f"Getting historical sentiment for {symbol} ({timeframe})")
        
        # Get historical sentiment
        result = sentiment_api.get_historical_sentiment(symbol, timeframe)
        
        # Return result
        return result
    except Exception as e:
        logger.error(f"Error getting historical sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-mock")
async def generate_mock_data(
    symbols: Optional[List[str]] = Query(None, description="List of symbols to generate mock data for"),
    sentiment_api: SentimentAPI = Depends(get_sentiment_api)
):
    """
    Generate mock data for all endpoints and symbols.
    
    This endpoint is useful for development and testing without hitting the Alpha Vantage API.
    It generates mock data for both the sentiment summary and historical sentiment endpoints.
    
    WARNING: This endpoint will make real API calls to generate the mock data initially.
    """
    try:
        # Default to crypto symbols if none provided
        symbols_to_use = symbols or ["BTC", "ETH"]
        
        # Log request for debugging
        logger.info(f"Generating mock data for {symbols_to_use}")
        
        # Generate mock data
        result = sentiment_api.generate_mock_data(symbols_to_use)
        
        # Return result
        return result
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))