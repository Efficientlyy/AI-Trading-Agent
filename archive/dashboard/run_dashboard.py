#!/usr/bin/env python
"""
Dashboard runner for the AI Trading Agent application.
This script launches the dashboard based on the specified version.
- Modern: The redesigned dashboard with all the advanced features
- Legacy: The original integrated dashboard
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
    while time.time() - start_time < timeout:
        if not is_port_in_use(host, port):
            return True
        time.sleep(1)
    return False

def run_modern_dashboard(host, port, debug):
    """Run the modern dashboard with all the advanced features."""
    try:
        # Import necessary components for the modern dashboard
        from src.dashboard.modern_components import create_modern_app
        
        # Create Flask application with the modern components
        app = create_modern_app()
        
        logger.info(f"Starting MODERN dashboard on http://{host}:{port}/")
        logger.info("This is the redesigned dashboard with all advanced features:")
        logger.info("- DataService with caching mechanism")
        logger.info("- WebSocket for real-time updates")
        logger.info("- User authentication with roles")
        logger.info("- Dark/light theme toggle")
        logger.info("- Notifications center")
        logger.info("- Settings management")
        logger.info("- Lazy loading and performance optimizations")
        
        # Run the application
        app.run(host=host, port=port, debug=debug)
        
    except ImportError:
        logger.error("Could not import necessary components for modern dashboard.")
        logger.error("Falling back to enhanced integrated dashboard...")
        run_enhanced_dashboard(host, port, debug)

def run_enhanced_dashboard(host, port, debug):
    """Run the enhanced integrated dashboard."""
    try:
        # Import the integrated dashboard application with enhancements
        from integrated_dashboard import create_app
        
        # Create Flask application
        app = create_app(use_modern_features=True)
        
        logger.info(f"Starting ENHANCED dashboard on http://{host}:{port}/")
        logger.info("This dashboard includes the following features:")
        logger.info("- System monitoring dashboard")
        logger.info("- Sentiment analysis dashboard with enhanced visuals")
        logger.info("- Risk management dashboard with improved metrics")
        logger.info("- Logs and monitoring dashboard")
        logger.info("- Market regime analysis with pattern detection")
        
        # Run the application
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Error starting enhanced dashboard: {e}")
        raise

def run_legacy_dashboard(host, port, debug):
    """Run the original integrated dashboard."""
    try:
        # Import the original integrated dashboard application
        from integrated_dashboard import create_app
        
        # Create Flask application
        app = create_app(use_modern_features=False)
        
        logger.info(f"Starting LEGACY dashboard on http://{host}:{port}/")
        logger.info("This is the original dashboard with the following features:")
        logger.info("- System monitoring dashboard")
        logger.info("- Sentiment analysis dashboard")
        logger.info("- Risk management dashboard")
        logger.info("- Logs and monitoring dashboard")
        logger.info("- Market regime analysis")
        
        # Run the application
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Error starting legacy dashboard: {e}")
        raise

def main():
    """Run the dashboard application based on specified version."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the AI Trading Agent dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")
    parser.add_argument("--version", choices=["modern", "enhanced", "legacy"], default="modern",
                        help="Dashboard version to launch (modern, enhanced, or legacy)")
    args = parser.parse_args()
    
    # Configure logging based on arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
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
    
    # Launch the selected dashboard version
    if args.version == "modern":
        run_modern_dashboard(host, port, args.debug)
    elif args.version == "enhanced":
        run_enhanced_dashboard(host, port, args.debug)
    else:
        run_legacy_dashboard(host, port, args.debug)

if __name__ == "__main__":
    main()
