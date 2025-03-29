#!/usr/bin/env python
"""
Run the sentiment analysis performance dashboard.

This script launches the dashboard for monitoring sentiment analysis 
performance metrics, model accuracy, and confidence calibration.
"""

import argparse
import asyncio
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Banner and styling constants
BANNER = """
╭───────────────────────────────────────────────────────╮
│                                                       │
│    SENTIMENT ANALYSIS PERFORMANCE DASHBOARD           │
│                                                       │
╰───────────────────────────────────────────────────────╯
"""

HEADER = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"


async def run_dashboard(port, debug=False, theme="light"):
    """Run the performance dashboard.
    
    Args:
        port: The port to run the dashboard on
        debug: Whether to run in debug mode
        theme: Dashboard theme (light or dark)
    """
    # Import here to allow argument parsing first
    from src.dashboard.performance_dashboard import PerformanceDashboard
    from src.analysis_agents.sentiment.performance_tracker import performance_tracker
    
    print(f"{BLUE}Initializing performance tracker...{ENDC}")
    await performance_tracker.initialize()
    
    print(f"{BLUE}Creating performance dashboard...{ENDC}")
    dashboard = PerformanceDashboard()
    
    print(f"{GREEN}Starting dashboard server on port {port}...{ENDC}")
    await dashboard.run_server(port=port, debug=debug)


def main():
    """Main function to parse arguments and run the dashboard."""
    print(BANNER)
    
    parser = argparse.ArgumentParser(description="Run the Sentiment Analysis Performance Dashboard")
    parser.add_argument("--port", type=int, default=8051, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--theme", type=str, default="light", choices=["light", "dark"], 
                        help="Dashboard theme (light or dark)")
    
    args = parser.parse_args()
    
    # Set environment variable for theme
    os.environ["DASHBOARD_THEME"] = args.theme
    
    print(f"{BOLD}Starting Sentiment Analysis Performance Dashboard{ENDC}")
    print(f"Port: {args.port}")
    print(f"Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Theme: {args.theme.capitalize()}")
    print()
    
    try:
        asyncio.run(run_dashboard(args.port, args.debug, args.theme))
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Dashboard terminated by user.{ENDC}")
    except Exception as e:
        print(f"\n{RED}Error running dashboard: {str(e)}{ENDC}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())