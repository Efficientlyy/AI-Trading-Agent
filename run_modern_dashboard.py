#!/usr/bin/env python3
"""
Fixed modern dashboard launcher with template error fixes
"""

import os
import sys
import argparse
import logging
from src.dashboard.modern_dashboard import ModernDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard_launcher")

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
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

def main():
    """Main entry point for the dashboard launcher"""
    parser = argparse.ArgumentParser(description="Launch the AI Trading Agent Modern Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the dashboard server")
    parser.add_argument("--port", type=int, default=0, help="Port to run the dashboard on (0 for auto)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Find an available port if not specified
    if args.port == 0:
        args.port = find_available_port()
    
    # Ensure template and static directories are correct
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    
    if not os.path.exists(templates_dir):
        logger.warning(f"Templates directory does not exist: {templates_dir}")
    
    if not os.path.exists(static_dir):
        logger.warning(f"Static directory does not exist: {static_dir}")
    
    # Fix recursive tooltip include before starting the dashboard
    tooltip_path = os.path.join(templates_dir, "tooltip.html")
    if os.path.exists(tooltip_path):
        with open(tooltip_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if this might cause recursion
        if "{% include 'tooltip.html' %}" in content:
            logger.warning("Found potential recursive include in tooltip.html, fixing it")
            content = content.replace(
                "{% include 'tooltip.html' %}", 
                "<!-- WARNING: Do not include this template recursively -->"
            )
            with open(tooltip_path, "w", encoding="utf-8") as f:
                f.write(content)
    
    try:
        # Set environment variables for the Flask app to use
        os.environ["FLASK_TEMPLATE_FOLDER"] = templates_dir
        os.environ["FLASK_STATIC_FOLDER"] = static_dir
        
        # Create and run the dashboard
        dashboard = ModernDashboard()
        print(f"\n🚀 AI Trading Agent Modern Dashboard is running at: http://{args.host}:{args.port}\n")
        print(f"Login Credentials:")
        print(f"  - Username: admin, Password: admin123")
        print(f"  - Username: operator, Password: operator123")
        print(f"  - Username: viewer, Password: viewer123")
        print(f"\nPress Ctrl+C to stop the server\n")
        
        dashboard.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())