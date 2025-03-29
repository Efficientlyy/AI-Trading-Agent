#!/usr/bin/env python
"""
Standalone Continuous Improvement Dashboard for Sentiment Analysis.

This script runs the continuous improvement dashboard as a standalone application,
allowing users to monitor and manage the automated improvement process.
"""

import asyncio
import logging
import argparse
import os
import sys
import json
import time
from datetime import datetime, timedelta

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from src.analysis_agents.sentiment.continuous_improvement import continuous_improvement_manager
from src.analysis_agents.sentiment.ab_testing import ab_testing_framework, ExperimentType
from src.dashboard.components.continuous_improvement_dashboard import create_layout
from src.common.config import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/continuous_improvement_dashboard.log")
    ]
)

logger = logging.getLogger("continuous_improvement_dashboard")


async def initialize_dependencies():
    """Initialize required dependencies."""
    try:
        # Initialize the A/B testing framework
        await ab_testing_framework.initialize()
        
        # Initialize the continuous improvement manager
        await continuous_improvement_manager.initialize()
        
        logger.info("Dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing dependencies: {e}")
        raise


def create_app():
    """Create and configure the Dash app."""
    # Initialize the FastAPI app
    server = FastAPI(title="Continuous Improvement Dashboard")
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname="/",
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    
    # Set layout
    app.layout = html.Div([
        dbc.Container([
            html.Div([
                html.H1("Continuous Improvement Dashboard", className="text-primary mt-4 mb-3"),
                html.P("Monitor and manage the automated improvement system for sentiment analysis", className="lead mb-4"),
            ]),
            
            # Main dashboard content
            create_layout()
        ], fluid=True)
    ])
    
    # Mount Dash app on FastAPI
    server.mount("/", WSGIMiddleware(app.server))
    
    return server


async def maintenance_task():
    """Background task for continuous improvement system maintenance."""
    while True:
        try:
            await continuous_improvement_manager.run_maintenance()
            logger.info("Continuous improvement maintenance completed")
        except Exception as e:
            logger.error(f"Error in continuous improvement maintenance: {e}")
        
        # Sleep for 1 hour
        await asyncio.sleep(3600)


async def start_maintenance_task():
    """Start the background maintenance task."""
    asyncio.create_task(maintenance_task())


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Continuous Improvement Dashboard")
    parser.add_argument(
        "--host",
        type=str,
        default=config.get("sentiment_analysis.continuous_improvement.dashboard.host", "0.0.0.0"),
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.get("sentiment_analysis.continuous_improvement.dashboard.port", 8053),
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reloading"
    )
    return parser.parse_args()


async def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Initialize dependencies
    await initialize_dependencies()
    
    # Start maintenance task
    await start_maintenance_task()
    
    # Create app
    app = create_app()
    
    # Import uvicorn here to avoid any startup conflicts
    import uvicorn
    
    # Show server info
    logger.info(f"Starting Continuous Improvement Dashboard on {args.host}:{args.port}")
    
    # Run the server
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())