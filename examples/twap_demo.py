#!/usr/bin/env python3
"""
TWAP Execution Algorithm Demo

This script demonstrates how to use the TWAP (Time-Weighted Average Price) 
execution algorithm to split a large order into smaller chunks that are 
executed over time to minimize market impact.

The demo uses the MockExchangeConnector for simulation purposes.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from src.execution.exchange.mock import MockExchangeConnector
from src.execution.interface import ExchangeInterface
from src.execution.algorithms.twap import TWAPExecutor
from src.models.order import OrderSide


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("twap_demo")


async def run_demo():
    """Run the TWAP demo."""
    logger.info("Starting TWAP execution algorithm demo")
    
    # Create a mock exchange connector
    mock_connector = MockExchangeConnector(
        exchange_id="binance",
        is_paper_trading=True,
        latency_ms=100,  # Simulate 100ms latency
        fill_probability=0.9,  # 90% chance of fill for limit orders
        price_volatility=0.002  # 0.2% price volatility
    )
    
    # Initialize the mock connector
    initialized = await mock_connector.initialize()
    if not initialized:
        logger.error("Failed to initialize mock exchange connector")
        return
    
    # Create exchange interface with the mock connector
    exchange_interface = ExchangeInterface({
        "binance": mock_connector
    })
    
    # Create TWAP executor
    twap_executor = TWAPExecutor(exchange_interface)
    
    # Set the parameters for the TWAP execution
    exchange_id = "binance"
    symbol = "BTC/USDT"
    side = OrderSide.BUY
    total_quantity = 1.0  # Buy 1 BTC
    duration_minutes = 10  # Execute over 10 minutes
    num_slices = 5  # Split into 5 equal slices
    
    logger.info(f"Starting TWAP execution to {side.value} {total_quantity} {symbol} over {duration_minutes} minutes in {num_slices} slices")
    
    # Start the TWAP execution
    success, job_id, error = await twap_executor.start_twap(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        use_limit_orders=True,  # Use limit orders
        limit_price_offset_percent=0.1,  # 0.1% price offset
        metadata={"demo": True}
    )
    
    if not success or job_id is None:
        logger.error(f"Failed to start TWAP execution: {error}")
        return
    
    logger.info(f"TWAP execution started with job ID: {job_id}")
    
    # Monitor the TWAP execution progress
    execution_complete = False
    while not execution_complete:
        # Get the current job status
        job_status = twap_executor.get_job_status(job_id)
        
        if job_status is None:
            logger.error(f"TWAP job not found: {job_id}")
            break
        
        # Check if the job is still active
        if not job_status["is_active"]:
            execution_complete = True
        
        # Log the current status
        logger.info(f"TWAP execution status: {job_status['percent_complete']:.2f}% complete")
        logger.info(f"  Slices executed: {job_status['slices_executed']}/{job_status['slices_total']}")
        logger.info(f"  Slices successful: {job_status['slices_successful']}")
        logger.info(f"  Remaining quantity: {job_status['remaining_quantity']}")
        
        if job_status["average_execution_price"] is not None:
            logger.info(f"  Average execution price: ${job_status['average_execution_price']:.2f}")
        
        # If there are errors, log them
        if job_status["errors"]:
            logger.warning(f"  Errors: {', '.join(job_status['errors'])}")
        
        # Wait for a moment before checking again
        await asyncio.sleep(min(10, job_status["interval_seconds"]))
    
    # Get the final status
    final_status = twap_executor.get_job_status(job_id)
    
    if final_status is None:
        logger.error(f"TWAP job not found: {job_id}")
        return
    
    logger.info("TWAP execution completed")
    logger.info(f"  Final status: {final_status['percent_complete']:.2f}% complete")
    logger.info(f"  Slices executed: {final_status['slices_executed']}/{final_status['slices_total']}")
    logger.info(f"  Slices successful: {final_status['slices_successful']}")
    logger.info(f"  Average execution price: ${final_status['average_execution_price'] or 0:.2f}")
    
    # Print details of individual orders
    logger.info("Order details:")
    for i, order_info in enumerate(final_status["orders"]):
        logger.info(f"  Order {i+1}:")
        logger.info(f"    Slice: {order_info['slice_num'] + 1}")
        logger.info(f"    Quantity: {order_info['quantity']}")
        logger.info(f"    Price: ${order_info['price']}")
        logger.info(f"    Status: {order_info['status']}")
        logger.info(f"    Filled: {order_info['filled_quantity']}")
        if order_info['average_price'] is not None:
            logger.info(f"    Average fill price: ${order_info['average_price']}")
    
    # Shutdown the mock connector
    await mock_connector.shutdown()
    logger.info("Demo completed")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 