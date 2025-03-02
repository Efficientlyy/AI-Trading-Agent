#!/usr/bin/env python
"""
Example script for order book strategy backtesting.

This script demonstrates how to use the order book strategy backtesting framework
to test and evaluate the performance of order book strategies against historical
order book data.
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.common.logging import get_logger
from src.models.signals import SignalType
from src.rust_bridge import OrderBookProcessor, create_order_book_processor
from src.strategy.market_imbalance import MarketImbalanceStrategy
from src.strategy.volume_absorption import VolumeAbsorptionStrategy
from src.backtesting.orderbook_backtesting import (
    OrderBookSnapshot, 
    OrderBookDataset, 
    OrderBookBacktester
)


logger = get_logger("examples", "orderbook_backtest")


def generate_sample_dataset(symbol="BTC/USD", num_snapshots=1000, base_price=50000.0):
    """Generate a sample order book dataset for backtesting.
    
    Args:
        symbol: The trading pair symbol
        num_snapshots: Number of snapshots to generate
        base_price: The base price to use
        
    Returns:
        An OrderBookDataset with generated snapshots
    """
    logger.info(f"Generating sample dataset with {num_snapshots} snapshots")
    
    # Create a dataset
    dataset = OrderBookDataset(symbol=symbol)
    
    # Current timestamp and price
    timestamp = datetime.now() - timedelta(days=1)
    price = base_price
    
    # Price volatility
    volatility = 0.001  # 0.1% per step
    
    # Generate snapshots
    for i in range(num_snapshots):
        # Update price with some random walk
        price_change = random.normalvariate(0, 1) * volatility * price
        price += price_change
        
        # Generate order book levels
        bids = []
        asks = []
        
        # Generate 10 bid levels
        for j in range(10):
            bid_price = price * (1 - 0.001 * j)
            bid_size = random.uniform(0.1, 10.0) * (1 - j * 0.05)
            bids.append((bid_price, bid_size))
        
        # Generate 10 ask levels
        for j in range(10):
            ask_price = price * (1 + 0.001 * j)
            ask_size = random.uniform(0.1, 10.0) * (1 - j * 0.05)
            asks.append((ask_price, ask_size))
        
        # Create a snapshot
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            exchange="sample",
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )
        
        # Add to dataset
        dataset.add_snapshot(snapshot)
        
        # Increment timestamp
        timestamp += timedelta(seconds=10)
        
        # Occasionally create imbalance events for testing
        if random.random() < 0.05:  # 5% chance
            # Decide between bid or ask imbalance
            if random.random() < 0.5:
                # Bid imbalance (more buying pressure)
                for j in range(len(bids)):
                    bids[j] = (bids[j][0], bids[j][1] * random.uniform(2.0, 5.0))
            else:
                # Ask imbalance (more selling pressure)
                for j in range(len(asks)):
                    asks[j] = (asks[j][0], asks[j][1] * random.uniform(2.0, 5.0))
            
            # Create another snapshot with the imbalance
            imbalance_snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange="sample",
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )
            
            # Add to dataset
            dataset.add_snapshot(imbalance_snapshot)
            
            # Increment timestamp again
            timestamp += timedelta(seconds=10)
        
        # Occasionally create large order events for testing
        if random.random() < 0.05:  # 5% chance
            # Decide between bid or ask large order
            if random.random() < 0.5:
                # Large bid
                level = random.randint(0, min(3, len(bids)-1))
                bids[level] = (bids[level][0], bids[level][1] * random.uniform(5.0, 10.0))
            else:
                # Large ask
                level = random.randint(0, min(3, len(asks)-1))
                asks[level] = (asks[level][0], asks[level][1] * random.uniform(5.0, 10.0))
            
            # Create a snapshot with the large order
            large_order_snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange="sample",
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )
            
            # Add to dataset
            dataset.add_snapshot(large_order_snapshot)
            
            # Increment timestamp again
            timestamp += timedelta(seconds=10)
            
            # Create additional snapshots showing order absorption
            if random.random() < 0.7:  # 70% chance of absorption
                # Determine how much of the order gets absorbed
                absorption_pct = random.uniform(0.7, 0.95)
                
                if random.random() < 0.5:  # 50% of the time it's a good signal (low price impact)
                    price_impact = random.uniform(0.0, 0.05) / 100  # 0-0.05% price impact
                else:
                    price_impact = random.uniform(0.1, 0.5) / 100  # 0.1-0.5% price impact
                
                # Update price with impact
                if random.random() < 0.5:
                    price += price * price_impact
                else:
                    price -= price * price_impact
                
                # Update the large order
                if random.random() < 0.5:
                    # Bid absorption
                    level = random.randint(0, min(3, len(bids)-1))
                    bids[level] = (bids[level][0], bids[level][1] * (1 - absorption_pct))
                else:
                    # Ask absorption
                    level = random.randint(0, min(3, len(asks)-1))
                    asks[level] = (asks[level][0], asks[level][1] * (1 - absorption_pct))
                
                # Create a snapshot showing absorption
                absorption_snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    exchange="sample",
                    timestamp=timestamp + timedelta(seconds=5),
                    bids=bids,
                    asks=asks
                )
                
                # Add to dataset
                dataset.add_snapshot(absorption_snapshot)
                
                # Increment timestamp again
                timestamp += timedelta(seconds=5)
    
    # Sort the snapshots by timestamp
    dataset.sort_snapshots()
    logger.info(f"Generated dataset with {len(dataset)} snapshots")
    
    return dataset


async def run_backtest(strategy_type, dataset, output_dir):
    """Run a backtest with the specified strategy and dataset.
    
    Args:
        strategy_type: The strategy class to use
        dataset: The order book dataset to test against
        output_dir: Directory to save results
        
    Returns:
        The backtest results
    """
    # Create the strategy
    strategy_id = strategy_type.__name__.lower().replace("strategy", "")
    strategy = strategy_type(strategy_id=strategy_id)
    
    # Create the backtester
    backtester = OrderBookBacktester(
        strategy=strategy,
        dataset=dataset,
        log_level="INFO"
    )
    
    # Run the backtest
    logger.info(f"Running backtest for {strategy_id}...")
    results = await backtester.run()
    
    # Save the results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"{strategy_id}_results.json"
    results.save_to_file(results_path)
    logger.info(f"Saved results to {results_path}")
    
    # Plot results
    fig, _ = results.plot_summary()
    plot_path = output_dir / f"{strategy_id}_summary.png"
    fig.savefig(str(plot_path))
    plt.close(fig)
    logger.info(f"Saved plot to {plot_path}")
    
    return results


async def save_dataset_to_file(dataset, filepath):
    """Save the dataset to a file.
    
    Args:
        dataset: The dataset to save
        filepath: Path to save the dataset to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_file(filepath)
    logger.info(f"Saved dataset to {filepath}")


async def main():
    """Run the order book backtesting example."""
    parser = argparse.ArgumentParser(description="Order book strategy backtesting example")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["market_imbalance", "volume_absorption", "both"],
        default="both",
        help="Strategy to backtest"
    )
    parser.add_argument(
        "--snapshots", 
        type=int, 
        default=1000, 
        help="Number of snapshots to generate"
    )
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="BTC/USD", 
        help="Symbol to use for backtesting"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="backtest_results", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--dataset-file", 
        type=str,
        help="Path to load/save the dataset (if not provided, generates a new dataset)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Load or generate dataset
    if args.dataset_file and os.path.exists(args.dataset_file):
        logger.info(f"Loading dataset from {args.dataset_file}")
        dataset = OrderBookDataset(symbol=args.symbol)
        dataset.load_from_file(args.dataset_file)
    else:
        logger.info("Generating sample dataset")
        dataset = generate_sample_dataset(
            symbol=args.symbol,
            num_snapshots=args.snapshots
        )
        
        # Save dataset if path provided
        if args.dataset_file:
            await save_dataset_to_file(dataset, args.dataset_file)
    
    # Run backtest(s)
    if args.strategy == "market_imbalance" or args.strategy == "both":
        logger.info("Running MarketImbalanceStrategy backtest")
        await run_backtest(MarketImbalanceStrategy, dataset, args.output_dir)
    
    if args.strategy == "volume_absorption" or args.strategy == "both":
        logger.info("Running VolumeAbsorptionStrategy backtest")
        await run_backtest(VolumeAbsorptionStrategy, dataset, args.output_dir)
    
    logger.info("Backtesting completed")


if __name__ == "__main__":
    asyncio.run(main()) 