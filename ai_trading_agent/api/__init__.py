"""
API Package

This package provides API endpoints for accessing data from the trading system.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# Import routers directly
from ai_trading_agent.api.routers.sentiment import router as sentiment_router
from ai_trading_agent.api.routers.trading_signals import router as trading_signals_router
from ai_trading_agent.api.routers.paper_trading import router as paper_trading_router
from ai_trading_agent.api.websocket import router as websocket_router

# Create main API router
api_router = APIRouter(prefix="/api")

# Add custom response class to ensure proper JSON serialization
class CustomJSONResponse(JSONResponse):
    def render(self, content):
        # Ensure content is directly serialized without any wrapping
        return super().render(content)

# Include sub-routers
api_router.include_router(sentiment_router)
api_router.include_router(trading_signals_router)
api_router.include_router(paper_trading_router)

# Export routers
__all__ = ["api_router", "websocket_router"]