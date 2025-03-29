#!/usr/bin/env python3
"""Run the provider health dashboard.

This script launches a dashboard for monitoring LLM provider health
and failover status.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.common.config import config
from src.common.logging import setup_logging, get_logger
from src.dashboard.components.provider_health_dashboard import create_layout


async def main(host, port):
    """Run the provider health dashboard.
    
    Args:
        host: Host to run the dashboard on
        port: Port to run the dashboard on
    """
    # Set up logging
    setup_logging()
    logger = get_logger("dashboard", "provider_health")
    
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="LLM Provider Health Dashboard"
    )
    
    # Set the layout
    app.layout = html.Div([
        dbc.Container([
            html.H1("LLM Provider Health Dashboard", className="mt-4 mb-4"),
            html.Hr(),
            create_layout(),
        ], fluid=True)
    ])
    
    # Run the app
    logger.info(f"Starting provider health dashboard on {host}:{port}")
    app.run_server(host=host, port=port, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the provider health dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the dashboard on")
    parser.add_argument("--port", type=int, default=8051, help="Port to run the dashboard on")
    
    args = parser.parse_args()
    
    # Run the dashboard
    asyncio.run(main(args.host, args.port))