#!/usr/bin/env python3
"""
Modular Dashboard Runner

This script launches the modularized version of the modern dashboard.
It addresses the Single Responsibility Principle by separating
runner logic from dashboard implementation.
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

def is_port_in_use(host, port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def wait_for_port_release(host, port, timeout=30):
    """Wait for a port to be released."""
    start_time = time.time()
    while is_port_in_use(host, port):
        if time.time() - start_time > timeout:
            logger.warning(f"Timeout waiting for port {port} to be released")
            return False
        logger.info(f"Waiting for port {port} to be released...")
        time.sleep(1)
    return True

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    # If we get here, we didn't find an available port
    logger.warning(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")
    return start_port

def ensure_directories_exist():
    """Ensure needed directories exist for the dashboard"""
    # Check for template and static directories
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # Create directories if they don't exist
    for dir_path in [templates_dir, static_dir, logs_dir]:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Create empty __init__.py files if needed
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "dashboard")
    components_dir = os.path.join(dashboard_dir, "components")
    utils_dir = os.path.join(dashboard_dir, "utils")
    
    # Ensure dashboard module directories exist
    for dir_path in [dashboard_dir, components_dir, utils_dir]:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            
        # Create __init__.py if it doesn't exist
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""Dashboard module components."""\n')
    
    return templates_dir, static_dir

def run_dashboard(host, port, debug):
    """Run the modern dashboard with all advanced features"""
    # Import the modular dashboard implementation
    try:
        from src.dashboard.modern_dashboard_refactored import ModernDashboard
        logger.info("Using modular dashboard implementation")
    except ImportError:
        # Fall back to the original implementation if modular one not found
        try:
            from src.dashboard.modern_dashboard import ModernDashboard
            logger.info("Using original dashboard implementation (modular version not found)")
        except ImportError:
            logger.error("Failed to import ModernDashboard. Please ensure the module exists.")
            return 1
    
    # Ensure required directories exist
    templates_dir, static_dir = ensure_directories_exist()
    
    # Create and run the dashboard
    dashboard = ModernDashboard(
        template_folder=templates_dir,
        static_folder=static_dir
    )
    
    # Run with the specified host and port
    logger.info(f"Starting dashboard on http://{host}:{port}/")
    try:
        dashboard.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1
    
    return 0

def main():
    """Run the dashboard application based on specified options"""
    parser = argparse.ArgumentParser(description="Launch the AI Trading Agent Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the dashboard server")
    parser.add_argument("--port", type=int, default=0, help="Port to run the dashboard on (0 for auto)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Find an available port if not specified
    if args.port == 0:
        args.port = find_available_port()
    
    # Run the dashboard with the specified host, port, and debug setting
    return run_dashboard(args.host, args.port, args.debug)

if __name__ == "__main__":
    sys.exit(main())
