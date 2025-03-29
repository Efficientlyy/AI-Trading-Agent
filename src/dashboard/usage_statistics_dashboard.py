"""Usage Statistics Dashboard.

This module provides a dashboard for visualizing LLM API usage statistics,
including cost, token usage, and performance metrics.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.wsgi import WSGIMiddleware

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.sentiment.usage_statistics import usage_tracker
from src.dashboard.components.usage_statistics_dashboard import create_usage_statistics_layout


# Initialize logging
logger = get_logger("dashboard", "usage_statistics_dashboard")

# Initialize FastAPI
app = FastAPI(title="LLM Usage Statistics Dashboard")

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Set app title
dash_app.title = "LLM Usage Statistics Dashboard"

# Create app layout
dash_app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LLM Usage Statistics Dashboard", className="mt-4"),
                html.Hr(),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                create_usage_statistics_layout()
            ])
        ])
    ], fluid=True)
])

# Mount Dash app
app.mount("/dashboard", WSGIMiddleware(dash_app.server))


@app.get("/", response_class=HTMLResponse)
async def get_usage_dashboard(request: Request):
    """Render the usage statistics dashboard.
    
    Args:
        request: The FastAPI request
        
    Returns:
        HTML response redirecting to the dashboard
    """
    # Redirect to the Dash app
    return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirecting...</title>
            <meta http-equiv="refresh" content="0;url=/dashboard/" />
        </head>
        <body>
            <p>Redirecting to dashboard...</p>
        </body>
        </html>
    """)


async def initialize():
    """Initialize the dashboard components."""
    logger.info("Initializing usage statistics dashboard")
    
    # Initialize the usage statistics tracker
    usage_tracker.initialize()
    
    logger.info("Usage statistics dashboard initialized")


def run_dashboard(host: str = "0.0.0.0", port: int = 8050):
    """Run the usage statistics dashboard.
    
    Args:
        host: Host address
        port: Port number
    """
    import uvicorn
    
    # Run initialization in event loop before starting
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize())
    
    logger.info(f"Starting usage statistics dashboard on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Read config
    host = config.get("dashboard.usage_statistics.host", "0.0.0.0")
    port = config.get("dashboard.usage_statistics.port", 8050)
    
    # Run dashboard
    run_dashboard(host=host, port=port)