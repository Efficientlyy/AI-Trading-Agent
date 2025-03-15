#!/usr/bin/env python
"""Execution Algorithms Demo

This script demonstrates the TWAP, VWAP, and Iceberg execution algorithms using the
MockExchangeConnector. This is useful for understanding how these algorithms
work before using them with real exchange connections.

The script shows:
1. How to initialize the execution algorithms
2. How to start execution jobs for multiple algorithm types
3. How to monitor the progress of execution jobs
4. How to analyze the execution results

When run, the script executes a series of market trades that demonstrate
the different characteristics of each execution algorithm.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
import random
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.logging import get_logger
from src.execution.exchange.mock import MockExchangeConnector
from src.execution.interface import ExchangeInterface
from src.execution.algorithms.twap import TWAPExecutor
from src.execution.algorithms.vwap import VWAPExecutor
from src.execution.algorithms.iceberg import IcebergExecutor
from src.models.order import OrderSide

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("examples.execution_algorithms_demo")


async def run_twap_demo(exchange_interface):
    """Run a demonstration of the TWAP execution algorithm.
    
    Args:
        exchange_interface: Initialized ExchangeInterface
    """
    logger.info("-----------------------------------------------------------")
    logger.info("Starting TWAP Execution Demo")
    logger.info("-----------------------------------------------------------")
    
    # Initialize the TWAP executor
    twap_executor = TWAPExecutor(exchange_interface)
    
    # Set parameters for the TWAP execution
    exchange_id = "mock"
    symbol = "BTC/USDT"
    side = OrderSide.BUY
    total_quantity = 1.0  # BTC
    duration_minutes = 5  # Complete execution in 5 minutes
    num_slices = 5        # Split into 5 equal slices
    
    logger.info(f"Starting TWAP execution for {total_quantity} {symbol} over {duration_minutes} minutes in {num_slices} slices")
    
    # Start the TWAP execution
    success, job_id, error = await twap_executor.start_twap(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        use_limit_orders=False  # Use market orders for simplicity
    )
    
    if not success or job_id is None:
        logger.error(f"Failed to start TWAP execution: {error}")
        return
    
    logger.info(f"TWAP execution started with job ID: {job_id}")
    
    # Monitor the execution progress
    start_time = time.time()
    completed = False
    
    while not completed and (time.time() - start_time) < duration_minutes * 60 + 30:  # Add 30 seconds buffer
        # Get the current job status
        job_status = twap_executor.get_job_status(job_id)
        
        if job_status is None:
            logger.error(f"Failed to get TWAP job status for job ID: {job_id}")
            break
        
        # Log the current progress
        logger.info(f"TWAP Progress: {job_status['percent_complete']:.2f}% complete, "
                   f"{job_status['slices_executed']}/{job_status['slices_total']} slices executed")
        
        # Check if the job is completed
        completed = not job_status["is_active"]
        
        if completed:
            logger.info(f"TWAP execution completed")
            
            # Log final results
            avg_price = job_status["average_execution_price"]
            if avg_price is not None:
                logger.info(f"Average execution price: {avg_price:.2f} USDT")
            
            # Log details of individual orders
            logger.info(f"Order details:")
            for i, order_info in enumerate(job_status["orders"]):
                logger.info(f"  Slice {i+1}: {order_info['filled_quantity']} BTC at {order_info['average_price']:.2f} USDT")
        else:
            # Wait a bit before checking again
            await asyncio.sleep(5)
    
    if not completed:
        logger.warning(f"TWAP execution did not complete within the expected time")


async def run_vwap_demo(exchange_interface):
    """Run a demonstration of the VWAP execution algorithm.
    
    Args:
        exchange_interface: Initialized ExchangeInterface
    """
    logger.info("-----------------------------------------------------------")
    logger.info("Starting VWAP Execution Demo")
    logger.info("-----------------------------------------------------------")
    
    # Initialize the VWAP executor
    vwap_executor = VWAPExecutor(exchange_interface)
    
    # Set parameters for the VWAP execution
    exchange_id = "mock"
    symbol = "ETH/USDT"
    side = OrderSide.BUY
    total_quantity = 10.0  # ETH
    duration_minutes = 10  # Complete execution in 10 minutes
    num_slices = 5        # Split into 5 slices according to volume profile
    
    logger.info(f"Starting VWAP execution for {total_quantity} {symbol} over {duration_minutes} minutes in {num_slices} slices")
    
    # Start the VWAP execution
    success, job_id, error = await vwap_executor.start_vwap(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        use_limit_orders=False  # Use market orders for simplicity
    )
    
    if not success or job_id is None:
        logger.error(f"Failed to start VWAP execution: {error}")
        return
    
    logger.info(f"VWAP execution started with job ID: {job_id}")
    
    # Monitor the execution progress
    start_time = time.time()
    completed = False
    
    while not completed and (time.time() - start_time) < duration_minutes * 60 + 30:  # Add 30 seconds buffer
        # Get the current job status
        job_status = vwap_executor.get_job_status(job_id)
        
        if job_status is None:
            logger.error(f"Failed to get VWAP job status for job ID: {job_id}")
            break
        
        # Log the current progress
        logger.info(f"VWAP Progress: {job_status['percent_complete']:.2f}% complete, "
                   f"{job_status['slices_executed']}/{job_status['slices_total']} slices executed")
        
        # Check if the job is completed
        completed = not job_status["is_active"]
        
        if completed:
            logger.info(f"VWAP execution completed")
            
            # Log final results
            avg_price = job_status["average_execution_price"]
            if avg_price is not None:
                logger.info(f"Average execution price: {avg_price:.2f} USDT")
            
            # Log details of individual orders
            logger.info(f"Order details:")
            for i, order_info in enumerate(job_status["orders"]):
                logger.info(f"  Slice {i+1}: {order_info['filled_quantity']} ETH at {order_info['average_price']:.2f} USDT")
        else:
            # Wait a bit before checking again
            await asyncio.sleep(5)
    
    if not completed:
        logger.warning(f"VWAP execution did not complete within the expected time")


async def run_iceberg_demo(exchange_interface):
    """Run a demonstration of the Iceberg execution algorithm.
    
    Args:
        exchange_interface: Initialized ExchangeInterface
    """
    logger.info("-----------------------------------------------------------")
    logger.info("Starting Iceberg Execution Demo")
    logger.info("-----------------------------------------------------------")
    
    # Initialize the Iceberg executor
    iceberg_executor = IcebergExecutor(exchange_interface)
    
    # Set parameters for the Iceberg execution
    exchange_id = "mock"
    symbol = "BTC/USDT"
    side = OrderSide.BUY
    total_quantity = 5.0  # BTC
    visible_quantity = 0.5  # Show only 0.5 BTC at a time (10% of total)
    
    # Get current market price
    current_price = await exchange_interface.get_market_price(exchange_id, symbol)
    if current_price is None:
        logger.error("Failed to get current market price")
        return
    
    # Set limit price slightly above market for buy
    price = current_price * 1.001  # 0.1% above current price
    
    logger.info(f"Starting Iceberg execution for {total_quantity} {symbol} with {visible_quantity} visible (10%)")
    logger.info(f"Current market price: {current_price}, limit price: {price}")
    
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
    start_time = time.time()
    completed = False
    max_execution_time = 300  # 5 minutes max
    
    while not completed and (time.time() - start_time) < max_execution_time:
        # Get the current job status
        job_status = iceberg_executor.get_job_status(job_id)
        
        if job_status is None:
            logger.error(f"Failed to get Iceberg job status for job ID: {job_id}")
            break
        
        # Log the current progress
        logger.info(f"Iceberg Progress: {job_status['percent_complete']:.2f}% complete, "
                   f"executed {job_status['slices_successful']} of {job_status['slices_executed']} slices")
        
        # Check if the job is completed
        completed = not job_status["is_active"]
        
        if completed:
            logger.info(f"Iceberg execution completed")
            
            # Log final results
            avg_price = job_status["average_execution_price"]
            if avg_price is not None:
                logger.info(f"Average execution price: {avg_price:.2f} USDT")
                
                # Calculate price impact
                price_impact = ((avg_price - current_price) / current_price) * 100
                logger.info(f"Price impact: {price_impact:.4f}%")
            
            # Log details of individual orders
            logger.info(f"Order details:")
            for i, order_info in enumerate(job_status["orders"]):
                if order_info['filled_quantity'] > 0:
                    logger.info(f"  Order {i+1}: {order_info['filled_quantity']}/{order_info['quantity']} BTC " +
                              f"at {order_info['average_price']:.2f} USDT")
        else:
            # Wait a bit before checking again
            await asyncio.sleep(5)
    
    if not completed:
        logger.warning(f"Iceberg execution did not complete within the expected time")
        await iceberg_executor.cancel_iceberg(job_id)
        logger.info("Cancelled remaining Iceberg execution for demo purposes")


async def run_comparison_demo(exchange_interface):
    """Compare all execution algorithms for the same order.
    
    Args:
        exchange_interface: Initialized ExchangeInterface
    """
    logger.info("-----------------------------------------------------------")
    logger.info("Starting Execution Algorithms Comparison Demo")
    logger.info("-----------------------------------------------------------")
    
    # Initialize all executors
    twap_executor = TWAPExecutor(exchange_interface)
    vwap_executor = VWAPExecutor(exchange_interface)
    iceberg_executor = IcebergExecutor(exchange_interface)
    
    # Common parameters
    exchange_id = "mock"
    symbol = "BTC/USDT"
    side = OrderSide.BUY
    total_quantity = 2.0  # BTC
    duration_minutes = 10  # Complete execution in 10 minutes
    num_slices = 10        # Split into 10 slices for TWAP/VWAP
    visible_quantity = 0.2  # Show 0.2 BTC at a time for Iceberg (10% of total)
    
    # Get current market price for Iceberg
    current_price = await exchange_interface.get_market_price(exchange_id, symbol)
    if current_price is None:
        logger.error("Failed to get current market price")
        return
    
    # Set limit price for Iceberg
    price = current_price * 1.001  # 0.1% above current price
    
    logger.info(f"Starting comparison of execution algorithms for {total_quantity} {symbol}")
    logger.info(f"  TWAP: {num_slices} equal time slices over {duration_minutes} minutes")
    logger.info(f"  VWAP: {num_slices} volume-weighted slices over {duration_minutes} minutes")
    logger.info(f"  Iceberg: {visible_quantity} BTC visible at a time (10% of total)")
    
    # Start TWAP execution
    twap_success, twap_job_id, twap_error = await twap_executor.start_twap(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        use_limit_orders=False
    )
    
    if not twap_success or twap_job_id is None:
        logger.error(f"Failed to start TWAP execution: {twap_error}")
        return
    
    # Start VWAP execution
    vwap_success, vwap_job_id, vwap_error = await vwap_executor.start_vwap(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
        use_limit_orders=False
    )
    
    if not vwap_success or vwap_job_id is None:
        logger.error(f"Failed to start VWAP execution: {vwap_error}")
        # Cancel TWAP job if VWAP failed
        await twap_executor.cancel_twap(twap_job_id)
        return
    
    # Start Iceberg execution
    iceberg_success, iceberg_job_id, iceberg_error = await iceberg_executor.start_iceberg(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        total_quantity=total_quantity,
        visible_quantity=visible_quantity,
        price=price
    )
    
    if not iceberg_success or iceberg_job_id is None:
        logger.error(f"Failed to start Iceberg execution: {iceberg_error}")
        # Cancel other jobs if Iceberg failed
        await twap_executor.cancel_twap(twap_job_id)
        await vwap_executor.cancel_vwap(vwap_job_id)
        return
    
    logger.info(f"TWAP execution started with job ID: {twap_job_id}")
    logger.info(f"VWAP execution started with job ID: {vwap_job_id}")
    logger.info(f"Iceberg execution started with job ID: {iceberg_job_id}")
    
    # Monitor all executions
    start_time = time.time()
    twap_completed = False
    vwap_completed = False
    iceberg_completed = False
    max_execution_time = duration_minutes * 60 + 60  # Add 1 minute buffer
    
    while (not twap_completed or not vwap_completed or not iceberg_completed) and (time.time() - start_time) < max_execution_time:
        # Get current job statuses
        twap_status = twap_executor.get_job_status(twap_job_id)
        vwap_status = vwap_executor.get_job_status(vwap_job_id)
        iceberg_status = iceberg_executor.get_job_status(iceberg_job_id)
        
        if twap_status is None or vwap_status is None or iceberg_status is None:
            logger.error("Failed to get job status for one or more algorithms")
            break
        
        # Log progress
        logger.info(f"Progress - TWAP: {twap_status['percent_complete']:.2f}%, "
                   f"VWAP: {vwap_status['percent_complete']:.2f}%, "
                   f"Iceberg: {iceberg_status['percent_complete']:.2f}%")
        
        # Update completion status
        twap_completed = not twap_status["is_active"]
        vwap_completed = not vwap_status["is_active"]
        iceberg_completed = not iceberg_status["is_active"]
        
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Cancel any incomplete jobs for demo purposes
    if not twap_completed:
        await twap_executor.cancel_twap(twap_job_id)
    if not vwap_completed:
        await vwap_executor.cancel_vwap(vwap_job_id)
    if not iceberg_completed:
        await iceberg_executor.cancel_iceberg(iceberg_job_id)
    
    # Get final statuses
    twap_final = twap_executor.get_job_status(twap_job_id)
    vwap_final = vwap_executor.get_job_status(vwap_job_id)
    iceberg_final = iceberg_executor.get_job_status(iceberg_job_id)
    
    if twap_final is None or vwap_final is None or iceberg_final is None:
        logger.error("Failed to get final job status for one or more algorithms")
        return
    
    # Compare results
    logger.info("-----------------------------------------------------------")
    logger.info("Execution Algorithms Comparison Results")
    logger.info("-----------------------------------------------------------")
    
    twap_avg_price = twap_final["average_execution_price"]
    vwap_avg_price = vwap_final["average_execution_price"]
    iceberg_avg_price = iceberg_final["average_execution_price"]
    
    if twap_avg_price is not None:
        logger.info(f"TWAP Average Price: {twap_avg_price:.2f} USDT")
        logger.info(f"  Completed: {twap_final['percent_complete']:.2f}%")
        logger.info(f"  Slices executed: {twap_final['slices_executed']}/{twap_final['slices_total']}")
    
    if vwap_avg_price is not None:
        logger.info(f"VWAP Average Price: {vwap_avg_price:.2f} USDT")
        logger.info(f"  Completed: {vwap_final['percent_complete']:.2f}%")
        logger.info(f"  Slices executed: {vwap_final['slices_executed']}/{vwap_final['slices_total']}")
    
    if iceberg_avg_price is not None:
        logger.info(f"Iceberg Average Price: {iceberg_avg_price:.2f} USDT")
        logger.info(f"  Completed: {iceberg_final['percent_complete']:.2f}%")
        logger.info(f"  Orders executed: {iceberg_final['slices_executed']}")
        logger.info(f"  Orders successful: {iceberg_final['slices_successful']}")
    
    # Determine which performed best (for a buy order, lower price is better)
    if twap_avg_price is not None and vwap_avg_price is not None and iceberg_avg_price is not None:
        prices = [
            ("TWAP", twap_avg_price),
            ("VWAP", vwap_avg_price),
            ("Iceberg", iceberg_avg_price)
        ]
        
        if side == OrderSide.BUY:
            # For buys, lower price is better
            sorted_prices = sorted(prices, key=lambda x: x[1])
        else:
            # For sells, higher price is better
            sorted_prices = sorted(prices, key=lambda x: x[1], reverse=True)
        
        logger.info("\nAlgorithm Performance Ranking (best to worst):")
        for i, (algo_name, price) in enumerate(sorted_prices):
            logger.info(f"{i+1}. {algo_name}: {price:.2f} USDT")
        
        # Calculate price differences
        best_algo, best_price = sorted_prices[0]
        logger.info("\nPrice Differences Compared to Best Performer:")
        for algo_name, price in prices:
            if algo_name != best_algo:
                diff = price - best_price
                diff_percent = (diff / best_price) * 100
                logger.info(f"{algo_name} vs {best_algo}: {diff:.2f} USDT ({diff_percent:.2f}%)")
    
    logger.info("\nKey Characteristics:")
    logger.info("TWAP: Even time distribution, predictable execution pattern")
    logger.info("VWAP: Distribution based on volume profile, follows market patterns")
    logger.info("Iceberg: Conceals total order size, minimal market impact, opportunistic")
    
    logger.info("\nSuitable Use Cases:")
    logger.info("TWAP: Disciplined execution over fixed timeframe, benchmark performance")
    logger.info("VWAP: Best execution in line with market volume, benchmark performance")
    logger.info("Iceberg: Large orders, illiquid markets, minimal market impact needed")


async def run_demo():
    """Run the full execution algorithms demonstration."""
    logger.info("Starting Execution Algorithms Demo")
    
    # Create a mock exchange connector with simulated market data and fills
    mock_connector = MockExchangeConnector(
        exchange_id="mock",
        is_paper_trading=True,
        latency_ms=100,  # 100ms simulated latency
        fill_probability=0.95,  # 95% chance of order fill
        price_volatility=0.01  # 1% price volatility
    )
    
    # Initialize the exchange interface with the mock connector
    exchange_interface = ExchangeInterface({mock_connector.exchange_id: mock_connector})
    
    # Start the mock connector
    await mock_connector.initialize()
    
    try:
        # Run TWAP demo
        await run_twap_demo(exchange_interface)
        
        # Short pause between demos
        await asyncio.sleep(2)
        
        # Run VWAP demo
        await run_vwap_demo(exchange_interface)
        
        # Short pause between demos
        await asyncio.sleep(2)
        
        # Run Iceberg demo
        await run_iceberg_demo(exchange_interface)
        
        # Short pause between demos
        await asyncio.sleep(2)
        
        # Run comparison demo
        await run_comparison_demo(exchange_interface)
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
    
    finally:
        # Clean up
        await mock_connector.shutdown()
        logger.info("Execution Algorithms Demo completed")


if __name__ == "__main__":
    asyncio.run(run_demo()) 