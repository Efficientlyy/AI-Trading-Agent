#!/usr/bin/env python
"""Simplified order routing debug example to identify issues."""

import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up basic logging to see all messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout  # Force output to stdout
)
logger = logging.getLogger("order_routing_debug")

def main():
    """Run a simplified check to debug issues."""
    logger.info("Starting debug check")
    
    # Check if data directory exists
    data_dir = Path("./data/fees")
    if not data_dir.exists():
        logger.info(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Data directory already exists: {data_dir}")
    
    # Import only after logging is initialized
    try:
        logger.info("Trying to import FeeManager")
        from src.fees.service import FeeManager
        logger.info("FeeManager imported successfully")
        
        # Try to create a FeeManager instance
        logger.info("Creating FeeManager instance")
        fee_manager = FeeManager(data_dir=data_dir)
        logger.info(f"FeeManager created: {fee_manager}")
        
        # Check fee schedules dict
        logger.info(f"Fee schedules: {getattr(fee_manager, 'fee_schedules', 'Not found')}")
        
        # Try to access the correct method
        logger.info("Checking available methods on FeeManager")
        methods = [m for m in dir(fee_manager) if not m.startswith('_')]
        logger.info(f"Available methods: {methods}")
        
    except Exception as e:
        logger.exception(f"Error importing or using FeeManager: {e}")
    
    # Try to import OrderRouter
    try:
        logger.info("Trying to import OrderRouter")
        from src.execution.routing import OrderRouter
        logger.info("OrderRouter imported successfully")
    except Exception as e:
        logger.exception(f"Error importing OrderRouter: {e}")
    
    logger.info("Debug check completed")

if __name__ == "__main__":
    main() 