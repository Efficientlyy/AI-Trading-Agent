"""
API Package

This package provides API endpoints for accessing data from the trading system.
"""

from fastapi import APIRouter

# Import routers
from .routers.sentiment import router as sentiment_router

# Create main API router
api_router = APIRouter(prefix="/api")

# Include sub-routers
api_router.include_router(sentiment_router)

# Export routers
__all__ = ["api_router"]