#!/usr/bin/env python
"""
Dashboard runner for the AI Trading Agent application.
This script launches the new integrated dashboard that provides a unified interface
for system monitoring, sentiment analysis, risk management, and log analysis.
"""

import os
import sys
import time
import socket
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard_runner")

# Apply Flask compatibility patches
try:
    from src.common.flask_compat import apply_flask_patches
    apply_flask_patches()
    print("✓ Successfully applied Flask compatibility patches")
except ImportError as e:
    print(f"Error applying Flask compatibility patches: {e}")
    sys.exit(1)

# Apply dateutil compatibility patches
try:
    from src.common.dateutil_compat import apply_dateutil_patches
    print("Applying dateutil.tz compatibility patches for Python 3.13...")
    apply_dateutil_patches()
    print("✓ Successfully applied dateutil.tz compatibility patches")
except ImportError as e:
    print(f"Error applying dateutil compatibility patches: {e}")
    sys.exit(1)

def is_port_in_use(host, port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def wait_for_port_release(host, port, timeout=30):
    """Wait for a port to be released."""
    start_time = time.time()
    while is_port_in_use(host, port):
        if time.time() - start_time > timeout:
            return False
        logger.info(f"Port {port} is in use. Waiting for it to be released...")
        time.sleep(1)
    return True

def main():
    """Run the integrated dashboard application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the AI Trading Agent integrated dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")
    args = parser.parse_args()
    
    # Configure logging based on arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Import the integrated dashboard application
        from integrated_dashboard import create_app
        
        # Create Flask application
        app = create_app()
        
        # Setup host and port
        host = os.environ.get('HOST', args.host)
        port = int(os.environ.get('PORT', args.port))
        
        # Check if port is already in use
        if is_port_in_use(host, port):
            logger.warning(f"Port {port} is already in use. Attempting to wait for it to be released...")
            if not wait_for_port_release(host, port):
                # If port isn't released after timeout, try different port
                for alt_port in range(port+1, port+10):
                    if not is_port_in_use(host, alt_port):
                        logger.info(f"Using alternative port {alt_port}")
                        port = alt_port
                        break
                else:
                    logger.error("Could not find available port. Please check for running instances.")
                    sys.exit(1)
        
        # Create template directory if it doesn't exist
        Path('templates').mkdir(exist_ok=True)
        
        logger.info(f"Starting integrated dashboard on http://{host}:{port}/")
        logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
        
        # Print dashboard features
        logger.info("Dashboard features:")
        logger.info("- System monitoring dashboard")
        logger.info("- Sentiment analysis dashboard")
        logger.info("- Risk management dashboard")
        logger.info("- Logs and monitoring dashboard")
        logger.info("- Market regime analysis")
        
        # Run the application
        app.run(host=host, port=port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        raise

if __name__ == "__main__":
    main()
