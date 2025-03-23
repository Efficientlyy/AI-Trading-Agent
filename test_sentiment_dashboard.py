#!/usr/bin/env python
"""
Sentiment Dashboard Test Script

This script tests the sentiment dashboard components to ensure they're working properly.
"""

import asyncio
import os
import sys
import logging
import json
import time
import webbrowser
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.sentiment_dashboard import SentimentDashboard
from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
from src.common.logging import setup_logging, get_logger


# Configure logging
setup_logging(level=logging.INFO)
logger = get_logger("test_script", "sentiment_dashboard_test")


class SentimentDashboardTester:
    """Tester for the sentiment dashboard."""
    
    def __init__(self):
        """Initialize the dashboard tester."""
        self.logger = get_logger("test_script", "sentiment_dashboard_test")
        self.manager = None
        self.dashboard = None
        
        # Create output directory
        os.makedirs("test_output", exist_ok=True)
    
    async def initialize(self):
        """Initialize the sentiment manager and dashboard."""
        self.logger.info("Initializing sentiment components")
        
        # Create and initialize sentiment manager
        self.manager = SentimentAnalysisManager()
        await self.manager.initialize()
        
        # Create and initialize dashboard
        self.dashboard = SentimentDashboard(self.manager)
        await self.dashboard.initialize()
        
        self.logger.info("Sentiment components initialized")
    
    async def start_components(self):
        """Start the sentiment manager and dashboard."""
        self.logger.info("Starting sentiment components")
        
        # Start sentiment manager
        await self.manager.start()
        
        # Start dashboard
        dashboard_url = await self.dashboard.start()
        
        self.logger.info(f"Dashboard started at: {dashboard_url}")
        return dashboard_url
    
    async def stop_components(self):
        """Stop the sentiment manager and dashboard."""
        self.logger.info("Stopping sentiment components")
        
        # Stop dashboard first
        if self.dashboard:
            await self.dashboard.stop()
        
        # Then stop manager
        if self.manager:
            await self.manager.stop()
        
        self.logger.info("Sentiment components stopped")
    
    async def test_dashboard(self, duration=60):
        """Test the sentiment dashboard.
        
        Args:
            duration: How long to run the test in seconds
        """
        self.logger.info(f"Testing sentiment dashboard for {duration} seconds")
        
        # Initialize components
        await self.initialize()
        
        # Start components
        dashboard_url = await self.start_components()
        
        # Try to open the dashboard in a browser
        try:
            self.logger.info(f"Attempting to open dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            self.logger.warning(f"Could not open browser: {str(e)}")
            print(f"\nPlease manually open the dashboard at: {dashboard_url}")
        
        # Wait for components to run
        print(f"\nSentiment dashboard running at: {dashboard_url}")
        print(f"Test will automatically end in {duration} seconds.")
        print("Press Ctrl+C to end the test early.")
        
        try:
            # Run for specified duration
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            self.logger.info("Test cancelled")
        
        # Stop components
        await self.stop_components()
        
        # Generate a simple test report
        report = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "dashboard_url": dashboard_url,
            "test_duration": duration,
            "status": "completed"
        }
        
        # Save report
        report_path = "test_output/dashboard_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Dashboard test report saved to {report_path}")
        
        return report


async def main():
    """Run the sentiment dashboard test."""
    # Display banner
    print("=" * 80)
    print("SENTIMENT DASHBOARD TEST".center(80))
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test the sentiment dashboard")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")
    args = parser.parse_args()
    
    # Create and run tester
    tester = SentimentDashboardTester()
    try:
        await tester.test_dashboard(duration=args.duration)
        return 0
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        await tester.stop_components()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)