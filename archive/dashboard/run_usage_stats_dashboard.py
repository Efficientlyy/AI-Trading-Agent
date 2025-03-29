"""Run Usage Statistics Dashboard.

This script launches the Usage Statistics Dashboard for monitoring LLM API usage.
"""

import os
import sys
import asyncio
import logging
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("run_usage_stats_dashboard")


def main():
    """Main entry point for the usage statistics dashboard."""
    parser = ArgumentParser(description="Run the Usage Statistics Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the dashboard on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Usage Statistics Dashboard")
    
    try:
        from src.dashboard.usage_statistics_dashboard import run_dashboard
        run_dashboard(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()