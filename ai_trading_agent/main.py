"""
AI Trading Agent FastAPI Application

This module sets up the FastAPI application and its routers.
"""

import logging
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api import router as api_router
from .api.websocket import router as ws_router
from .api.crypto_websocket import router as crypto_ws_router, initialize as init_crypto_ws
from .event_bus import EventBus
from .common import get_logger

# Configure logging
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Trading Agent",
    description="API for AI Trading Agent",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create event bus
event_bus = EventBus()

# Add routers
app.include_router(api_router)
app.include_router(ws_router)
app.include_router(crypto_ws_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting AI Trading Agent application")
    
    # Initialize crypto WebSocket
    crypto_initialized = await init_crypto_ws()
    if crypto_initialized:
        logger.info("Crypto WebSocket initialized successfully")
    else:
        logger.warning("Failed to initialize Crypto WebSocket")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down AI Trading Agent application")
    
    # Import shutdown function here to avoid circular imports
    from .api.crypto_websocket import shutdown as shutdown_crypto_ws
    await shutdown_crypto_ws()

if __name__ == "__main__":
    uvicorn.run("ai_trading_agent.main:app", host="0.0.0.0", port=8000, reload=True)
