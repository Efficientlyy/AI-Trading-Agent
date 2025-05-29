from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Import authentication if needed, or mock user
from backend.security.auth import get_mock_user_override # Assuming mock user for now

logger = logging.getLogger(__name__)

alpha_vantage_router = APIRouter(prefix="/api/alpha-vantage", tags=["alpha-vantage"])

# Models
class SentimentDataPoint(BaseModel):
    time_published: str
    title: str
    url: str
    source: str
    summary: str
    overall_sentiment_score: float
    overall_sentiment_label: str
    ticker_sentiment: List[Dict[str, Any]] # Simplified for mock

class AlphaVantageSentimentResponse(BaseModel):
    feed: List[SentimentDataPoint]
    items: str # e.g., "50"
    sentiment_score_definition: str
    relevance_score_definition: str


@alpha_vantage_router.get("/sentiment", response_model=AlphaVantageSentimentResponse)
async def get_alpha_vantage_sentiment(
    topic: Optional[str] = Query(None, description="A specific topic to filter sentiment for (e.g., blockchain)"),
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers (e.g., COIN,MSFT)"),
    current_user: Dict[str, Any] = Depends(get_mock_user_override) # Or your actual auth
):
    """
    Mock endpoint for Alpha Vantage news & sentiment.
    In a real implementation, this would call the Alpha Vantage API.
    """
    logger.info(f"Fetching Alpha Vantage sentiment for topic: {topic}, tickers: {tickers}")

    # Mock response structure based on Alpha Vantage documentation
    mock_feed_item = SentimentDataPoint(
        time_published="20240510T013222",
        title="Mock Alpha Vantage News Title about " + (topic or "general market"),
        url="https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=IBM&apikey=demo",
        source="Alpha Vantage Mock News",
        summary="This is a mock summary of the news article. " + (f"Focusing on {topic}." if topic else ""),
        overall_sentiment_score=0.123,
        overall_sentiment_label="Neutral",
        ticker_sentiment=[
            {"ticker": "IBM", "relevance_score": "0.5", "ticker_sentiment_score": "0.1", "ticker_sentiment_label": "Neutral"},
            {"ticker": "MSFT", "relevance_score": "0.3", "ticker_sentiment_score": "-0.2", "ticker_sentiment_label": "Somewhat-Bearish"}
        ]
    )
    
    return AlphaVantageSentimentResponse(
        items="1", # Mocking one item
        sentiment_score_definition="A score from -1 (most negative) to 1 (most positive).",
        relevance_score_definition="A score from 0 to 1, indicating relevance.",
        feed=[mock_feed_item]
    )