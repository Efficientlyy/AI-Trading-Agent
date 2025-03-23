"""
Start Sentiment Dashboard Example.

This script demonstrates how to run the sentiment analysis dashboard.
"""

import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the sentiment dashboard."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the sentiment analysis dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    args = parser.parse_args()
    
    try:
        # Import required packages
        try:
            import uvicorn
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            from fastapi.templating import Jinja2Templates
            import pandas as pd
            import numpy as np
        except ImportError as e:
            logger.error(f"Error importing required packages: {e}")
            logger.error("Please make sure you have installed all required packages:")
            logger.error("pip install fastapi uvicorn jinja2 starlette pandas numpy")
            sys.exit(1)
            
        # Try to import dashboard components
        try:
            from src.dashboard import sentiment_router
        except ImportError as e:
            logger.error(f"Error importing sentiment dashboard: {e}")
            logger.error("Make sure you have run the install script or installed the package")
            sys.exit(1)
            
        # Create a FastAPI app
        app = FastAPI(title="Sentiment Analysis Dashboard")
        
        # Mount sentiment dashboard
        app.include_router(sentiment_router)
        
        # Add a redirect from root to sentiment dashboard
        @app.get("/")
        async def redirect_to_sentiment():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/sentiment")
        
        # Log startup information
        logger.info(f"Starting sentiment dashboard on http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Run the server
        uvicorn.run(
            app, 
            host=args.host,
            port=args.port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Error running sentiment dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()