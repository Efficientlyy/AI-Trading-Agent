#!/usr/bin/env python
"""Iceberg Order Execution Algorithm Demo

This script demonstrates the use of the Iceberg order execution algorithm
to execute large orders with minimal market impact by revealing only a small
portion of the total order quantity at a time.

The demo shows how to:
1. Initialize the Iceberg executor
2. Configure and start an Iceberg execution job
3. Monitor the execution progress
4. Analyze the execution results
"""

import asyncio
import logging
import sys
from datetime import datetime
import os
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.exchange.mock import MockExchangeConnector
from src.execution.interface import ExchangeInterface
from src.execution.algorithms.iceberg import IcebergExecutor
from src.models.order import OrderSide

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("examples.iceberg_demo")


async def run_demo():
    """Run the Iceberg order execution algorithm demo."""
    logger.info("Starting Iceberg Order Execution Algorithm Demo")
    
    # Initialize a mock exchange connector with simulated market data and fills
    mock_connector = MockExchangeConnector(
        exchange_id="mock",
        is_paper_trading=True,
        latency_ms=50,  # 50ms simulated latency
        fill_probability=0.7,  # 70% chance of order fill (more realistic)
        price_volatility=0.005  # 0.5% price volatility
    )
    
    # Initialize the exchange interface with the mock connector
    exchange_interface = ExchangeInterface({mock_connector.exchange_id: mock_connector})
    
    # Initialize the mock connector
    await mock_connector.initialize()
    
    try:
        # Create an Iceberg executor
        iceberg_executor = IcebergExecutor(exchange_interface)
        
        # Define parameters for the Iceberg order
        exchange_id = "mock"
        symbol = "BTC/USDT"
        side = OrderSide.BUY
        total_quantity = 10.0  # A large order of 10 BTC
        visible_quantity = 0.5  # Show only 0.5 BTC at a time (5% of total)
        
        # Get the current market price
        current_price = await exchange_interface.get_market_price(exchange_id, symbol)
        if current_price is None:
            logger.error("Failed to get current market price")
            return
        
        # Set limit price slightly above market for buy (would be below for sell)
        price = current_price * 1.001  # 0.1% above current price
        
        logger.info(f"Starting Iceberg execution to {side.value} {total_quantity} {symbol}")
        logger.info(f"Current market price: {current_price}, limit price: {price}")
        logger.info(f"Visible quantity: {visible_quantity} (showing {(visible_quantity/total_quantity)*100:.1f}% at a time)")
        
        # Start the Iceberg execution
        success, job_id, error = await iceberg_executor.start_iceberg(
            exchange_id=exchange_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            visible_quantity=visible_quantity,
            price=price,
            min_execution_interval_seconds=1.0,  # Wait 1 second between orders
            max_retry_attempts=5,  # Retry up to 5 times
            metadata={"demo": True}
        )
        
        if not success or job_id is None:
            logger.error(f"Failed to start Iceberg execution: {error}")
            return
        
        logger.info(f"Iceberg execution started with job ID: {job_id}")
        
        # Monitor the execution progress
        execution_complete = False
        while not execution_complete:
            # Get the current job status
            job_status = iceberg_executor.get_job_status(job_id)
            
            if job_status is None:
                logger.error(f"Iceberg job not found: {job_id}")
                break
            
            # Check if the job is still active
            if not job_status["is_active"]:
                execution_complete = True
            
            # Log the current status
            logger.info(f"Iceberg execution status: {job_status['percent_complete']:.2f}% complete")
            logger.info(f"  Remaining quantity: {job_status['remaining_quantity']} of {job_status['total_quantity']}")
            logger.info(f"  Slices executed: {job_status['slices_executed']}")
            logger.info(f"  Slices successful: {job_status['slices_successful']}")
            
            if job_status["average_execution_price"] is not None:
                logger.info(f"  Average execution price: ${job_status['average_execution_price']:.2f}")
            
            # If there are errors, log them
            if job_status["errors"]:
                logger.warning(f"  Errors: {', '.join(job_status['errors'])}")
            
            # Wait a moment before checking again
            await asyncio.sleep(2)
            
            # For demo purposes, we'll cancel the job after a few slices
            # to show the cancel functionality
            if job_status["slices_executed"] >= 5 and job_status["percent_complete"] < 50:
                logger.info("For demo purposes, cancelling the Iceberg execution halfway through")
                await iceberg_executor.cancel_iceberg(job_id)
                break
        
        # Get the final status
        final_status = iceberg_executor.get_job_status(job_id)
        
        if final_status is None:
            logger.error(f"Iceberg job not found: {job_id}")
            return
        
        logger.info("Iceberg execution completed or cancelled")
        logger.info(f"  Final status: {final_status['percent_complete']:.2f}% complete")
        logger.info(f"  Total quantity executed: {final_status['total_quantity'] - final_status['remaining_quantity']}")
        logger.info(f"  Slices executed: {final_status['slices_executed']}")
        logger.info(f"  Slices successful: {final_status['slices_successful']}")
        
        if final_status["average_execution_price"] is not None:
            logger.info(f"  Average execution price: ${final_status['average_execution_price']:.2f}")
            price_impact = ((final_status['average_execution_price'] - current_price) / current_price) * 100
            logger.info(f"  Price impact: {price_impact:.4f}%")
        
        # Print details of individual orders
        logger.info("Order details:")
        for i, order_info in enumerate(final_status["orders"]):
            logger.info(f"  Order {i+1}:")
            logger.info(f"    Quantity: {order_info['quantity']}")
            logger.info(f"    Price: ${order_info['price']}")
            logger.info(f"    Status: {order_info['status']}")
            logger.info(f"    Filled: {order_info['filled_quantity']}")
            if order_info['average_price'] is not None:
                logger.info(f"    Average fill price: ${order_info['average_price']}")
        
        # Demo the effect of different visible quantities
        logger.info("\nComparing different visible quantity sizes:")
        
        # Small visible quantity (1% of total)
        small_visible = 0.1  # 0.1 BTC (1% of total)
        logger.info(f"Small visible quantity: {small_visible} BTC (1% of total)")
        logger.info("  Advantages: Minimal market impact, better price")
        logger.info("  Disadvantages: Longer execution time, higher risk of partial fills")
        
        # Medium visible quantity (5% of total)
        medium_visible = 0.5  # 0.5 BTC (5% of total)
        logger.info(f"Medium visible quantity: {medium_visible} BTC (5% of total)")
        logger.info("  Advantages: Balance of speed and market impact")
        logger.info("  Disadvantages: Some market impact still possible")
        
        # Large visible quantity (20% of total)
        large_visible = 2.0  # 2.0 BTC (20% of total)
        logger.info(f"Large visible quantity: {large_visible} BTC (20% of total)")
        logger.info("  Advantages: Faster execution, higher fill probability")
        logger.info("  Disadvantages: Higher market impact, worse average price")
        
        logger.info("\nDemo completed successfully")
    
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
    
    finally:
        # Shutdown the mock connector
        await mock_connector.shutdown()


if __name__ == "__main__":
    asyncio.run(run_demo()) 