"""
Main API server for the AI Trading Agent.

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
    import sys
    import os
    
    # Add the current directory to the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import the direct_paper_trading module using an absolute import
    import direct_paper_trading
    paper_trading_router = direct_paper_trading.router
    has_paper_trading_api = True
    logger.info("Using direct paper trading API with absolute import")
except ImportError as e:
    logger.error(f"Error importing direct paper trading API: {str(e)}")
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

if has_agent_api:
    app.include_router(agent_router)
    logger.info("Agent API endpoints registered")

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
