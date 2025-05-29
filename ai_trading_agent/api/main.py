"""
Main API server for the AI Trading Agent.
# Fix Python path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



This module provides the main FastAPI server for the AI Trading Agent,
including endpoints for data, trading, and dashboard functionality.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Use absolute imports with fallback for development environment
try:
    from ai_trading_agent.common import logger
except ImportError:
    # Create a basic logger if the common module is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Using basic logger due to import error")

# Import session management API
try:
    from ai_trading_agent.api.session_management_api import include_session_management_api
except ImportError:
    # Create a fallback function if the session management API is not available
    logger.warning("Session management API not available, using mock implementation")
    def include_session_management_api(app):
        logger.info("Mock session management API registered")
        pass

# Import paper trading router directly
try:
    from ai_trading_agent.api import paper_trading_api
    paper_trading_router = paper_trading_api.router
    has_paper_trading_api = True
    logger.info("Registered paper_trading_api router via absolute import.")
except ImportError as abs_err:
    try:
        from . import paper_trading_api
        paper_trading_router = paper_trading_api.router
        has_paper_trading_api = True
        logger.info("Registered paper_trading_api router via relative import.")
    except ImportError as rel_err:
        logger.error(f"Failed to import paper_trading_api.py: Absolute error: {abs_err} | Relative error: {rel_err}")
        has_paper_trading_api = False

try:
    from ai_trading_agent.api.agent_visualization_api import router as agent_visualization_router
    has_agent_visualization_api = True
except ImportError:
    try:
        # Fallback to relative import for development environment
        from .agent_visualization_api import router as agent_visualization_router
        has_agent_visualization_api = True
    except ImportError:
        logger.warning("Agent visualization API not available, endpoints will not be registered")
        has_agent_visualization_api = False

try:
    from ai_trading_agent.api.websocket_api import router as websocket_router
    has_websocket_api = True
except ImportError:
    try:
        # Fallback to relative import for development environment
        from .websocket_api import router as websocket_router
        has_websocket_api = True
    except ImportError:
        logger.warning("WebSocket API not available, endpoints will not be registered")
        has_websocket_api = False

try:
    from ai_trading_agent.api.routers.agent import router as agent_router
    has_agent_api = True
except ImportError:
    try:
        # Fallback to relative import for development environment
        from .routers.agent import router as agent_router
        has_agent_api = True
    except ImportError:
        logger.warning("Agent API not available, endpoints will not be registered")
        has_agent_api = False

# Import system control router
try:
    from ai_trading_agent.api.system_control import system_control_router
    has_system_control_api = True
except ImportError:
    logger.warning("System Control API not available, endpoints will not be registered")
    has_system_control_api = False
    
# Import data feed manager
try:
    from ai_trading_agent.api.data_feed_manager import data_feed_manager
    has_data_feed_manager = True
except ImportError:
    logger.warning("Data Feed Manager not available, data feed connection may not work properly")
    has_data_feed_manager = False

# Import sentiment pipeline API
try:
    from ai_trading_agent.api.sentiment_pipeline_api import router as sentiment_pipeline_router
    has_sentiment_pipeline_api = True
except ImportError:
    logger.warning("Sentiment Pipeline API not available, endpoints will not be registered")
    has_sentiment_pipeline_api = False

# Import technical analysis API
try:
    from ai_trading_agent.api.technical_analysis_api import router as technical_analysis_router
    has_technical_analysis_api = True
except ImportError:
    try:
        # Fallback to relative import for development environment
        from .technical_analysis_api import router as technical_analysis_router
        has_technical_analysis_api = True
    except ImportError:
        logger.warning("Technical Analysis API not available, endpoints will not be registered")
        has_technical_analysis_api = False

# Create FastAPI app
app = FastAPI(
    title="AI Trading Agent API",
    description="API for the AI Trading Agent system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=86400  # Cache preflight requests for 24 hours
)

# Include routers if available
if has_paper_trading_api:
    logger.info("Paper trading API endpoints registered")
    app.include_router(paper_trading_router, prefix="/api", tags=["paper-trading"])

if has_agent_visualization_api:
    app.include_router(agent_visualization_router)
    logger.info("Agent visualization API endpoints registered")
    
if has_websocket_api:
    app.include_router(websocket_router)
    logger.info("WebSocket API endpoints registered")
    
    # Register WebSocket startup and shutdown handlers
    try:
        from ai_trading_agent.api.websocket_api import startup_event, shutdown_event
        
        @app.on_event("startup")
        async def start_mexc_streams():
            await startup_event()
            
        @app.on_event("shutdown")
        async def stop_mexc_streams():
            await shutdown_event()
            
        logger.info("WebSocket API startup/shutdown handlers registered")
    except Exception as e:
        logger.warning(f"Failed to register WebSocket API event handlers: {e}")
    
    # Include MEXC WebSocket router
    try:
        from ai_trading_agent.api.mexc_websocket import router as mexc_ws_router
        app.include_router(mexc_ws_router)
        logger.info("MEXC WebSocket API endpoints registered at /ws/mexc/")
        
        # Import MEXC connector but don't crash if connection fails
        try:
            from ai_trading_agent.config.mexc_config import MEXC_CONFIG
            logger.info(f"MEXC API Key configured: {'Yes' if MEXC_CONFIG.get('API_KEY') else 'No'}")
            logger.info(f"MEXC API Secret configured: {'Yes' if MEXC_CONFIG.get('API_SECRET') else 'No'}")
        except Exception as e:
            logger.warning(f"Error loading MEXC configuration: {e}")
    except ImportError:
        try:
            # Fallback to relative import for development environment
            from .mexc_websocket import router as mexc_ws_router
            app.include_router(mexc_ws_router)
            logger.info("MEXC WebSocket API endpoints registered at /ws/mexc/")
        except ImportError:
            logger.warning("MEXC WebSocket API not available, endpoints will not be registered")

if has_agent_api:
    app.include_router(agent_router)
    logger.info("Agent API endpoints registered")

if has_system_control_api:
    app.include_router(system_control_router, prefix="/api")
    logger.info("System Control API endpoints registered at /api/system/")

if has_sentiment_pipeline_api:
    app.include_router(sentiment_pipeline_router)
    logger.info("Sentiment Pipeline API endpoints registered at /api/sentiment/")

if has_technical_analysis_api:
    app.include_router(technical_analysis_router)
    logger.info("Technical Analysis API endpoints registered")

@app.on_event("startup")
async def log_routes():
    route_list = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            route_list.append(f"{route.path} - {route.methods}")
    logger.info("Registered routes:\n" + "\n".join(route_list))


# Include session management API
try:
    include_session_management_api(app)
    logger.info("Session management API endpoints registered")
except Exception as e:
    logger.error(f"Error registering session management API: {str(e)}")

# Serve static files for dashboard
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard", "build")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")


@app.get("/")
async def root():
    """Root endpoint that redirects to the dashboard."""
    return {"message": "AI Trading Agent API", "docs_url": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Serve dashboard index.html for all dashboard routes
@app.get("/dashboard/{full_path:path}")
async def serve_dashboard(full_path: str):
    """Serve the dashboard for any dashboard route."""
    index_path = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.on_event("startup")
async def log_routes():
    route_list = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            route_list.append(f"{route.path} - {route.methods}")
    logger.info("Registered routes:\n" + "\n".join(route_list))


# Enable direct execution of this file
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    print(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "ai_trading_agent.api.main:app", 
        host=host,
        port=port,
        log_level=log_level
    )
