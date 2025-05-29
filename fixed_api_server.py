#!/usr/bin/env python
"""
Fixed API Server for AI Trading Agent

This script starts a standalone API server that properly handles agent control
commands and works around connectivity issues.
"""

import os
import sys
import logging
import time
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fixed-api-server")

# Set environment variables
os.environ['USE_MOCK_DATA'] = 'true'  # Force mock data
os.environ['ALLOW_CORS'] = 'true'     # Enable CORS

# Adjust Python path to find the modules
api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ai_trading_agent', 'api'))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Also add project root to path if not already there
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create a new FastAPI application
app = FastAPI(title="AI Trading Agent Fixed API", version="1.0")

# Configure CORS
origins = [
    "http://localhost:3000",         # React development server
    "http://localhost:8000",         # API server (for docs)
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
try:
    # Force setting USE_MOCK_DATA to True in the system_control module
    from ai_trading_agent.api.system_control import system_control_router, USE_MOCK_DATA
    import ai_trading_agent.api.system_control as system_control
    if not hasattr(system_control, 'USE_MOCK_DATA') or not system_control.USE_MOCK_DATA:
        logger.warning("Forcing USE_MOCK_DATA to True in system_control module")
        system_control.USE_MOCK_DATA = True
        
    # Include the system control router
    app.include_router(system_control_router)
    logger.info("Successfully loaded system_control_router")
    
    # Also import the data feed manager to ensure it's properly initialized
    from ai_trading_agent.api.data_feed_manager import data_feed_manager
    if hasattr(data_feed_manager, 'force_connected'):
        data_feed_manager.force_connected = True
        logger.info("Set data_feed_manager.force_connected = True")
except ImportError as e:
    logger.error(f"Failed to import router or modules: {e}")
    sys.exit(1)

# Debug endpoints to verify API is working
@app.get("/healthcheck")
async def healthcheck():
    """Simple healthcheck endpoint to verify API is running."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api-debug")
async def api_debug():
    """Debug endpoint to check API configuration."""
    config = {
        "USE_MOCK_DATA": getattr(system_control, 'USE_MOCK_DATA', None),
        "Python Path": sys.path,
        "Environment Variables": {
            "USE_MOCK_DATA": os.environ.get('USE_MOCK_DATA'),
            "ALLOW_CORS": os.environ.get('ALLOW_CORS'),
        }
    }
    return config

# Add all other routers from the api directory
# This could be expanded to include all API endpoints
# But for now we're focused on the system control functionality

if __name__ == "__main__":
    logger.info("Starting fixed API server...")
    logger.info("Using USE_MOCK_DATA=true to ensure reliable operation")
    
    # Print startup information
    logger.info(f"API Directory: {api_dir}")
    logger.info(f"Project Root: {project_root}")
    
    # Start the server
    uvicorn.run(
        "fixed_api_server:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False,
        log_level="info"
    )
