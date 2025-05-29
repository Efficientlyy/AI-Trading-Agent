"""
API Endpoints package for the External API Gateway.

This package contains the domain-specific API endpoints for the
External API Gateway, organized by functional area.
"""

from .market_data import router as market_data_router
from .trading import router as trading_router
from .analytics import router as analytics_router
from .signals import router as signals_router

# Export all routers
routers = [
    market_data_router,
    trading_router,
    analytics_router,
    signals_router
]
