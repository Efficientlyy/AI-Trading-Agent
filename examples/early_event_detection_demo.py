"""
Early Event Detection System Demo

This script demonstrates how to use the early event detection system
to detect market-moving events before they go viral.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add the project root to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import config
from src.common.logging import setup_logging, get_logger
from src.analysis_agents.early_detection.system import EarlyEventDetectionSystem


async def main():
    """Run the early event detection demo."""
    # Set up logging
    setup_logging(log_dir="logs", log_level="INFO")
    logger = get_logger("examples", "early_event_detection_demo")
    
    logger.info("Starting Early Event Detection Demo")
    
    try:
        # Initialize the Early Event Detection System
        system = EarlyEventDetectionSystem()
        
        # Initialize the system
        await system.initialize()
        
        # Start the system
        await system.start()
        
        logger.info("The system has been started. Press Ctrl+C to stop.")
        
        # Run a single detection cycle
        logger.info("Running a detection cycle manually...")
        await system._detection_cycle()
        
        # Get active events
        active_events = await system.get_active_events()
        logger.info(f"Found {len(active_events)} active events")
        
        # Print event details
        for i, event in enumerate(active_events[:5]):  # Show only the first 5 to avoid excessive output
            logger.info(f"Event {i+1}: {event.title}")
            logger.info(f"  Category: {event.category.value}")
            logger.info(f"  Confidence: {event.confidence.name}")
            logger.info(f"  Detected at: {event.detected_at}")
            logger.info(f"  Sources: {len(event.sources)}")
            
            # Print impact assessment if available
            if event.impact_assessment:
                assets = event.impact_assessment.get("assets", {})
                for asset, impact in assets.items():
                    logger.info(f"  Impact on {asset}: {impact.get('direction', 'unknown').value}, "
                                f"Score: {impact.get('score', 0):.2f}")
        
        # Get active signals
        active_signals = await system.get_active_signals()
        logger.info(f"Found {len(active_signals)} active signals")
        
        # Print signal details
        for i, signal in enumerate(active_signals[:5]):  # Show only the first 5
            logger.info(f"Signal {i+1}: {signal.title}")
            logger.info(f"  Confidence: {signal.confidence:.2f}")
            logger.info(f"  Created at: {signal.created_at}")
            logger.info(f"  Expires at: {signal.expires_at}")
            logger.info(f"  Priority: {signal.priority}")
            
            # Print asset recommendations
            for asset, details in signal.assets.items():
                logger.info(f"  Asset {asset}: {details}")
        
        # Generate a market impact report
        logger.info("Generating market impact report...")
        market_report = await system.generate_market_impact_report()
        
        logger.info(f"Market report summary: {market_report.get('summary')}")
        logger.info(f"Overall sentiment: {market_report.get('overall_sentiment', 0):.2f}")
        
        # Print asset impact reports
        for asset, impact in market_report.get("assets", {}).items():
            positive_count = len(impact.get("positive_events", []))
            negative_count = len(impact.get("negative_events", []))
            mixed_count = len(impact.get("mixed_events", []))
            
            logger.info(f"  {asset}: Impact score: {impact.get('impact_score', 0):.2f}")
            logger.info(f"    Positive events: {positive_count}, Negative events: {negative_count}, Mixed events: {mixed_count}")
        
        # Allow the system to run for a while
        logger.info("Demo completed. The system would normally continue running.")
        logger.info("In a production environment, the system would run continuously and publish events to a message bus.")
        
        # Stop the system
        await system.stop()
        
    except KeyboardInterrupt:
        logger.info("Stopping due to user interrupt")
        if 'system' in locals():
            await system.stop()
    except Exception as e:
        logger.error(f"Error in early event detection demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())