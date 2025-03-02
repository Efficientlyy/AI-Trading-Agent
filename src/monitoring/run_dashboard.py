"""
Run the monitoring dashboard.

This script starts the monitoring API server and dashboard.
"""

import os
import argparse
import asyncio
import uvicorn
from pathlib import Path

from src.common.config import config
from src.common.logging import get_logger
from src.monitoring import get_monitoring_service


logger = get_logger("monitoring", "dashboard")


async def initialize_monitoring():
    """Initialize the monitoring service."""
    logger.info("Initializing monitoring service...")
    monitoring_service = get_monitoring_service()
    await monitoring_service.initialize()
    await monitoring_service.start()
    logger.info("Monitoring service started")


def main():
    """Run the monitoring dashboard server."""
    parser = argparse.ArgumentParser(description="Run the trading system monitoring dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development mode)")
    args = parser.parse_args()
    
    # Initialize monitoring service
    asyncio.run(initialize_monitoring())
    
    # Load configuration
    dashboard_host = config.get("monitoring.dashboard.host", args.host)
    dashboard_port = config.get("monitoring.dashboard.port", args.port)
    
    # Log startup information
    logger.info(f"Starting monitoring dashboard on http://{dashboard_host}:{dashboard_port}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Run the FastAPI server with uvicorn
    uvicorn.run(
        "src.monitoring.api:app", 
        host=dashboard_host,
        port=dashboard_port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main() 