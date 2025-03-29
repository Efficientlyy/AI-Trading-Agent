#!/usr/bin/env python
"""
Run script for the Market Regime Detection Web UI.

This script launches the Flask web application for the Market Regime Detection UI.
"""

import os
import argparse
from app import app

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Market Regime Detection Web UI')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind to (default: 0.0.0.0)')
    
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    parser.add_argument('--api-url', type=str, default=None,
                        help='Base URL for the API (default: http://localhost:8000)')
    
    return parser.parse_args()

def main():
    """Run the web application."""
    args = parse_args()
    
    # Set API URL if provided
    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url
    
    # Set Flask environment variables
    if args.debug:
        os.environ["FLASK_ENV"] = 'development'
        os.environ["FLASK_DEBUG"] = '1'
    else:
        os.environ["FLASK_ENV"] = 'production'
        os.environ["FLASK_DEBUG"] = '0'
    
    # Print info
    print(f"Starting Market Regime Detection Web UI at http://{args.host}:{args.port}")
    if args.api_url:
        print(f"Using API at: {args.api_url}")
    else:
        print(f"Using API at: {os.environ.get('API_BASE_URL', 'http://localhost:8000')}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    
    # Run the application
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 