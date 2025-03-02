#!/usr/bin/env python
"""
Order Book Processing Example

This example demonstrates how to use the high-performance OrderBookProcessor
to process real-time market data and analyze order book liquidity.
"""

import os
import sys
import time
import json
import logging
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rust_bridge import create_order_book_processor, is_rust_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_orderbook(base_price: float = 50000.0, depth: int = 20, spread_pct: float = 0.02) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate a sample order book with realistic price levels.
    
    Args:
        base_price: The base price around which to generate the order book
        depth: The number of price levels on each side
        spread_pct: The percentage spread between best bid and ask
        
    Returns:
        A tuple of (bid updates, ask updates) for initializing the order book
    """
    spread = base_price * spread_pct
    best_bid = base_price - spread / 2
    best_ask = base_price + spread / 2
    
    # Generate bids (sorted high to low)
    bids = []
    for i in range(depth):
        # Price decreases as we move away from the best bid
        # Volume increases to simulate larger resting orders further from the top of book
        price = best_bid * (1 - 0.001 * i)
        volume = 0.5 + (i * 0.15)  # Increasing volume
        
        # Add some randomness
        price_jitter = price * random.uniform(-0.0005, 0.0005)
        volume_jitter = volume * random.uniform(-0.1, 0.1)
        
        bids.append({
            "price": price + price_jitter,
            "side": "buy",
            "quantity": volume + volume_jitter,
            "timestamp": time.time(),
            "sequence": i + 1
        })
    
    # Generate asks (sorted low to high)
    asks = []
    for i in range(depth):
        # Price increases as we move away from the best ask
        # Volume increases to simulate larger resting orders further from the top of book
        price = best_ask * (1 + 0.001 * i)
        volume = 0.5 + (i * 0.15)  # Increasing volume
        
        # Add some randomness
        price_jitter = price * random.uniform(-0.0005, 0.0005)
        volume_jitter = volume * random.uniform(-0.1, 0.1)
        
        asks.append({
            "price": price + price_jitter,
            "side": "sell",
            "quantity": volume + volume_jitter,
            "timestamp": time.time(),
            "sequence": depth + i + 1
        })
    
    return bids, asks

def generate_order_book_update(processor, num_updates: int = 5, change_pct: float = 0.2) -> List[Dict[str, Any]]:
    """
    Generate a realistic order book update, simulating changes in the market.
    
    Args:
        processor: The OrderBookProcessor to use for context
        num_updates: Number of price level updates to generate
        change_pct: Percentage of price levels to modify/remove/add
        
    Returns:
        A list of order book updates
    """
    snapshot = processor.snapshot()
    bids = snapshot['bids']
    asks = snapshot['asks']
    
    updates = []
    sequence_start = int(time.time() * 1000)  # Millisecond timestamp as sequence
    
    # Decide what kind of update to generate
    for i in range(num_updates):
        update_type = random.choices(
            ["modify", "remove", "add"],
            weights=[0.7, 0.15, 0.15],
            k=1
        )[0]
        
        # 50/50 chance for bid or ask update
        is_bid = random.random() < 0.5
        side = "buy" if is_bid else "sell"
        levels = bids if is_bid else asks
        
        if not levels and update_type != "add":
            # If no levels on this side, we can only add
            update_type = "add"
        
        if update_type == "modify":
            # Modify an existing price level
            level_idx = random.randint(0, min(len(levels) - 1, 5))  # Bias towards top of book
            price = levels[level_idx][0]
            quantity = levels[level_idx][1] * random.uniform(0.8, 1.2)  # Adjust quantity
            updates.append({
                "price": price,
                "side": side,
                "quantity": quantity,
                "timestamp": time.time(),
                "sequence": sequence_start + i
            })
            
        elif update_type == "remove":
            # Remove a price level
            level_idx = random.randint(0, len(levels) - 1)
            price = levels[level_idx][0]
            updates.append({
                "price": price,
                "side": side,
                "quantity": 0,  # Zero quantity means remove
                "timestamp": time.time(),
                "sequence": sequence_start + i
            })
            
        else:  # "add"
            # Add a new price level
            if is_bid:
                # New bid somewhere between existing bids
                if len(levels) > 1:
                    idx1 = random.randint(0, len(levels) - 2)
                    idx2 = idx1 + 1
                    min_price = levels[idx2][0]  # Lower price (bids are high to low)
                    max_price = levels[idx1][0]  # Higher price
                    new_price = min_price + (max_price - min_price) * random.random()
                else:
                    # Only one level or no levels
                    if levels:
                        new_price = levels[0][0] * random.uniform(0.998, 0.999)
                    else:
                        # No levels at all, use best ask as reference or a default
                        if asks:
                            new_price = asks[0][0] * 0.99  # 1% below best ask
                        else:
                            new_price = 50000.0  # Default if no bids or asks
            else:
                # New ask somewhere between existing asks
                if len(levels) > 1:
                    idx1 = random.randint(0, len(levels) - 2)
                    idx2 = idx1 + 1
                    min_price = levels[idx1][0]  # Lower price (asks are low to high)
                    max_price = levels[idx2][0]  # Higher price
                    new_price = min_price + (max_price - min_price) * random.random()
                else:
                    # Only one level or no levels
                    if levels:
                        new_price = levels[0][0] * random.uniform(1.001, 1.002)
                    else:
                        # No levels at all, use best bid as reference or a default
                        if bids:
                            new_price = bids[0][0] * 1.01  # 1% above best bid
                        else:
                            new_price = 50000.0  # Default if no bids or asks
            
            # Generate a random quantity
            quantity = random.uniform(0.1, 2.0)
            
            updates.append({
                "price": new_price,
                "side": side,
                "quantity": quantity,
                "timestamp": time.time(),
                "sequence": sequence_start + i
            })
    
    return updates

def run_example():
    """Run the order book processing example."""
    # Check if Rust is available
    using_rust = is_rust_available()
    logger.info(f"Using Rust implementation: {using_rust}")
    
    # Create an order book processor
    symbol = "BTC/USD"
    exchange = "example_exchange"
    max_depth = 20
    
    processor = create_order_book_processor(symbol, exchange, max_depth)
    logger.info(f"Created OrderBookProcessor for {symbol} on {exchange}")
    
    # Generate initial order book data
    bids, asks = generate_sample_orderbook(base_price=50000.0, depth=max_depth)
    all_updates = bids + asks
    
    # Initialize the order book
    processing_time = processor.process_updates(all_updates)
    logger.info(f"Initialized order book with {len(bids)} bids and {len(asks)} asks in {processing_time:.3f}ms")
    
    # Display initial state
    snapshot = processor.snapshot()
    logger.info(f"Best bid: {processor.best_bid_price():.2f}")
    logger.info(f"Best ask: {processor.best_ask_price():.2f}")
    logger.info(f"Mid price: {processor.mid_price():.2f}")
    logger.info(f"Spread: {processor.spread():.2f} ({processor.spread_pct():.3f}%)")
    
    # Perform market impact analysis
    buy_sizes = [1.0, 5.0, 10.0, 25.0, 50.0]
    logger.info("\nMarket Impact Analysis (BUY orders):")
    logger.info(f"{'Size':>10} | {'Avg Price':>12} | {'Slippage %':>10} | {'Levels':>6} | {'Fillable':>8}")
    logger.info("-" * 60)
    
    for size in buy_sizes:
        impact = processor.calculate_market_impact("buy", size)
        logger.info(f"{size:10.2f} | {impact['avg_price']:12.2f} | {impact['slippage_pct']:10.3f} | {impact['levels_consumed']:6d} | {impact['fillable_quantity']:8.2f}")
    
    # Simulate real-time updates
    logger.info("\nSimulating real-time updates...")
    total_updates = 0
    cumulative_time = 0
    
    try:
        for i in range(10):
            # Generate and process a batch of updates
            updates = generate_order_book_update(processor, num_updates=random.randint(5, 15))
            start_time = time.time()
            processing_time = processor.process_updates(updates)
            end_time = time.time()
            
            total_updates += len(updates)
            cumulative_time += processing_time
            
            # Print statistics every few iterations
            if i % 2 == 0:
                imbalance = processor.book_imbalance(5)
                bid_liquidity = processor.liquidity_up_to("bid", 100.0)
                ask_liquidity = processor.liquidity_up_to("ask", 100.0)
                
                logger.info(f"\nUpdate {i+1}: Processed {len(updates)} updates in {processing_time:.3f}ms")
                logger.info(f"Best bid: {processor.best_bid_price():.2f}, Best ask: {processor.best_ask_price():.2f}")
                logger.info(f"Spread: {processor.spread():.2f} ({processor.spread_pct():.3f}%)")
                logger.info(f"Book imbalance (bid/ask ratio): {imbalance:.2f}")
                logger.info(f"Liquidity within $100: {bid_liquidity:.2f} BTC (bid), {ask_liquidity:.2f} BTC (ask)")
            
            # Simulate the passage of time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("\nExample interrupted by user.")
    
    # Print final statistics
    stats = processor.processing_stats()
    logger.info("\nFinal Statistics:")
    logger.info(f"Total updates processed: {stats['updates_processed']}")
    logger.info(f"Levels added: {stats['levels_added']}")
    logger.info(f"Levels modified: {stats['levels_modified']}")
    logger.info(f"Levels removed: {stats['levels_removed']}")
    logger.info(f"Average processing time: {stats['avg_processing_time_us'] / 1000:.3f}ms")
    logger.info(f"Max processing time: {stats['max_processing_time_us'] / 1000:.3f}ms")
    logger.info(f"Min processing time: {stats['min_processing_time_us'] / 1000:.3f}ms")
    
    # Calculate performance metrics
    avg_time_per_update = cumulative_time / total_updates if total_updates > 0 else 0
    updates_per_second = total_updates / (cumulative_time / 1000) if cumulative_time > 0 else 0
    
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"Average time per update: {avg_time_per_update:.3f}ms")
    logger.info(f"Updates per second: {updates_per_second:.1f}")
    logger.info(f"Implementation: {'Rust' if using_rust else 'Python'}")

if __name__ == "__main__":
    run_example() 