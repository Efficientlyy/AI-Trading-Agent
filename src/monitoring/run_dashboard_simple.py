"""
Run the monitoring dashboard.

This script starts a simple version of the monitoring API server and dashboard.
"""

import os
import argparse
import uvicorn
from pathlib import Path

# Parse command line arguments
def main():
    """Run the monitoring dashboard server."""
    parser = argparse.ArgumentParser(description="Run the trading system monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development mode)")
    args = parser.parse_args()
    
    # Log startup information
    print(f"Starting monitoring dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the FastAPI server with uvicorn
    uvicorn.run(
        "src.monitoring.api:app", 
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 