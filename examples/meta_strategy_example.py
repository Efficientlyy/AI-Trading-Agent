#!/usr/bin/env python
"""
Example script for combining multiple strategies using the MetaStrategy.

This script demonstrates how to:
1. Create multiple strategies
2. Combine them using the MetaStrategy
3. Configure different weighting and combination methods
4. Process order book updates through the meta-strategy
"""

import asyncio
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.logging import get_logger
from src.models.signals import Signal, SignalType
from src.rust_bridge import create_order_book_processor
from src.strategy.market_imbalance import MarketImbalanceStrategy
from src.strategy.volume_absorption import VolumeAbsorptionStrategy
from src.strategy.meta_strategy import MetaStrategy, SignalCombinationMethod, StrategyWeighting


# Configure logging
logger = get_logger("examples", "meta_strategy")


def setup_termination_handler():
    """Set up a signal handler for graceful termination."""
    termination_requested = False
    
    def handle_termination(sig, frame):
        nonlocal termination_requested
        print(f"\nReceived signal {sig}, shutting down...")
        termination_requested = True
    
    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)
    
    return lambda: termination_requested


def generate_orderbook_update(symbol, side=None, large_order=False):
    """Generate a simulated order book update.
    
    Args:
        symbol: The trading pair symbol
        side: Optional side to focus the update on ("bid" or "ask")
        large_order: Whether to include a large order
        
    Returns:
        List of order book updates
    """
    base_price = 50000.0  # Base price for BTC
    updates = []
    
    # Choose a side if not specified
    if side is None:
        side = "bid" if random.random() < 0.5 else "ask"
    
    # Generate updates for the specified side
    if side == "bid":
        # Generate bid updates
        for i in range(10):
            price = base_price * (1 - 0.001 * i)
            size = random.uniform(0.1, 5.0)
            
            # Add large order if requested
            if large_order and i < 3:
                size *= random.uniform(5.0, 10.0)
            
            updates.append({
                "type": "bid",
                "price": price,
                "size": size
            })
    else:
        # Generate ask updates
        for i in range(10):
            price = base_price * (1 + 0.001 * i)
            size = random.uniform(0.1, 5.0)
            
            # Add large order if requested
            if large_order and i < 3:
                size *= random.uniform(5.0, 10.0)
            
            updates.append({
                "type": "ask",
                "price": price,
                "size": size
            })
    
    return updates


def generate_orderbook_imbalance(symbol, direction="bid", imbalance_factor=3.0):
    """Generate a simulated order book update with imbalance.
    
    Args:
        symbol: The trading pair symbol
        direction: Direction of the imbalance ("bid" or "ask")
        imbalance_factor: Factor by which to increase the volume on one side
        
    Returns:
        List of order book updates
    """
    base_price = 50000.0  # Base price for BTC
    updates = []
    
    # Generate bid updates
    for i in range(10):
        price = base_price * (1 - 0.001 * i)
        
        # Apply imbalance factor to bids if direction is bid
        if direction == "bid":
            size = random.uniform(0.5, 5.0) * imbalance_factor
        else:
            size = random.uniform(0.1, 3.0)
        
        updates.append({
            "type": "bid",
            "price": price,
            "size": size
        })
    
    # Generate ask updates
    for i in range(10):
        price = base_price * (1 + 0.001 * i)
        
        # Apply imbalance factor to asks if direction is ask
        if direction == "ask":
            size = random.uniform(0.5, 5.0) * imbalance_factor
        else:
            size = random.uniform(0.1, 3.0)
        
        updates.append({
            "type": "ask",
            "price": price,
            "size": size
        })
    
    return updates


async def simulate_large_order_absorption(processor, symbol, side="bid", absorption_pct=0.8):
    """Simulate the absorption of a large order.
    
    Args:
        processor: OrderBookProcessor instance
        symbol: Trading pair symbol
        side: Side of the large order ("bid" or "ask")
        absorption_pct: Percentage of the order to absorb
        
    Returns:
        None
    """
    # First, add a large order
    logger.info(f"Adding large {side} order for {symbol}")
    large_updates = generate_orderbook_update(symbol, side=side, large_order=True)
    processor.process_updates(large_updates)
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Now absorb the order
    logger.info(f"Absorbing {absorption_pct*100:.0f}% of the {side} order")
    
    # Generate absorption updates
    absorption_updates = []
    for update in large_updates:
        if update["type"] == side:
            # Reduce the size to simulate absorption
            remaining_size = update["size"] * (1 - absorption_pct)
            absorption_updates.append({
                "type": update["type"],
                "price": update["price"],
                "size": remaining_size
            })
    
    # Process the absorption
    processor.process_updates(absorption_updates)


async def simulate_market_imbalance(processor, symbol, direction="bid", imbalance_factor=3.0):
    """Simulate a market imbalance.
    
    Args:
        processor: OrderBookProcessor instance
        symbol: Trading pair symbol
        direction: Direction of the imbalance ("bid" or "ask")
        imbalance_factor: Strength of the imbalance
        
    Returns:
        None
    """
    logger.info(f"Creating {direction} imbalance for {symbol} (factor: {imbalance_factor})")
    imbalance_updates = generate_orderbook_imbalance(symbol, direction, imbalance_factor)
    processor.process_updates(imbalance_updates)


async def run_scenario(meta_strategy, symbol, processors):
    """Run a test scenario with various order book events.
    
    Args:
        meta_strategy: The meta-strategy to test
        symbol: Trading pair symbol
        processors: Dictionary of order book processors
        
    Returns:
        None
    """
    logger.info(f"Running test scenario for {symbol}")
    
    # Get the processor for this symbol
    processor = processors[symbol]
    
    # Scenario steps with delays between them
    
    # 1. Normal order book updates
    logger.info("Scenario 1: Normal order book updates")
    for _ in range(5):
        updates = generate_orderbook_update(symbol)
        processor.process_updates(updates)
        await asyncio.sleep(0.5)
    
    # 2. Market imbalance (buy pressure)
    logger.info("Scenario 2: Market imbalance (buy pressure)")
    await simulate_market_imbalance(processor, symbol, "bid", 4.0)
    await asyncio.sleep(2)
    
    # 3. Large order absorption
    logger.info("Scenario 3: Large order absorption")
    await simulate_large_order_absorption(processor, symbol, "ask", 0.9)
    await asyncio.sleep(2)
    
    # 4. Combined scenario: imbalance + absorption
    logger.info("Scenario 4: Combined imbalance and absorption")
    await simulate_market_imbalance(processor, symbol, "bid", 3.5)
    await asyncio.sleep(1)
    await simulate_large_order_absorption(processor, symbol, "ask", 0.85)
    await asyncio.sleep(2)
    
    # 5. Reset to normal state
    logger.info("Scenario 5: Resetting to normal state")
    normal_updates = generate_orderbook_update(symbol)
    processor.process_updates(normal_updates)
    
    # Log statistics
    stats = meta_strategy.get_statistics()
    logger.info(f"Meta-strategy statistics: {stats}")
    
    active_signals = meta_strategy.get_active_signals()
    for strat_id, signals in active_signals.items():
        if signals:
            logger.info(f"Active signals from {strat_id}: {len(signals)}")
            for signal in signals:
                logger.info(f"  - {signal.direction} {signal.signal_type.name} @ {signal.price} (conf: {signal.confidence:.2f})")


async def run_example():
    """Run the meta-strategy example."""
    # Set up termination handler
    should_terminate = setup_termination_handler()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Define symbols to use
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    # Create individual strategies
    logger.info("Creating strategies")
    market_imbalance = MarketImbalanceStrategy(
        strategy_id="market_imbalance",
        imbalance_threshold=1.5,
        depth_shallow=5,
        depth_medium=10,
        depth_deep=20,
        min_trade_size=0.01,
        max_trade_size=0.5
    )
    
    volume_absorption = VolumeAbsorptionStrategy(
        strategy_id="volume_absorption",
        min_order_size=5.0,
        absorption_threshold=0.8,
        price_impact_threshold=0.1,
        min_trade_size=0.05,
        max_trade_size=0.5
    )
    
    # Create meta-strategy
    logger.info("Creating meta-strategy")
    
    # Create strategy weightings
    strategy_weights = {
        "market_imbalance": StrategyWeighting(
            strategy_id="market_imbalance",
            weight=1.5,  # Higher weight for market imbalance
            min_confidence=0.6  # Higher confidence threshold
        ),
        "volume_absorption": StrategyWeighting(
            strategy_id="volume_absorption",
            weight=1.0,
            min_confidence=0.5
        )
    }
    
    # Create meta-strategy
    meta_strategy = MetaStrategy(
        strategy_id="combined_orderbook",
        sub_strategies=[market_imbalance, volume_absorption],
        strategy_weights=strategy_weights,
        combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE,
        min_consensus_pct=0.5,  # At least 50% of strategies must agree
        min_overall_confidence=0.6,  # Minimum combined confidence
        signal_window=60  # 1 minute window for concurrent signals
    )
    
    # Initialize strategies
    logger.info("Initializing strategies")
    await meta_strategy.initialize()
    
    # Start strategies
    logger.info("Starting strategies")
    await meta_strategy.start()
    
    try:
        # Create order book processors for each symbol
        logger.info("Creating order book processors")
        processors = {}
        for symbol in symbols:
            processors[symbol] = create_order_book_processor(symbol, "binance", max_depth=50)
        
        # Process initial updates to establish the order book
        for symbol, processor in processors.items():
            logger.info(f"Initializing order book for {symbol}")
            initial_updates = generate_orderbook_update(symbol)
            processor.process_updates(initial_updates)
        
        # Run scenarios until termination is requested
        logger.info("Running scenarios")
        while not should_terminate():
            # Run a scenario for each symbol
            for symbol in symbols:
                await run_scenario(meta_strategy, symbol, processors)
                
                # Check for termination request
                if should_terminate():
                    break
            
            # Wait between iterations
            if not should_terminate():
                logger.info("Waiting for next scenario run...")
                await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"Error during example: {e}")
        raise
    
    finally:
        # Clean up
        logger.info("Stopping strategies")
        await meta_strategy.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example()) 