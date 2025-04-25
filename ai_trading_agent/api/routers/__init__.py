"""
API Routers Package

This package contains FastAPI routers for different API endpoints.
"""

# Export routers
from .sentiment import router as sentiment_router

__all__ = ["sentiment_router"]