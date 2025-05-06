"""
API Routers Package

This package contains FastAPI routers for different API endpoints.
"""

# Export routers
from .sentiment import router as sentiment_analysis
from .trading_signals import router as trading_signals
from .simple_paper_trading import router as simple_paper_trading

# Health check router
from fastapi import APIRouter
health = APIRouter(prefix="/health", tags=["health"])

@health.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

__all__ = ["sentiment_analysis", "trading_signals", "simple_paper_trading", "health"]