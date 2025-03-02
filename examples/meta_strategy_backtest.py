#!/usr/bin/env python
"""
Example script for backtesting a meta-strategy.

This script demonstrates how to:
1. Create multiple strategies
2. Combine them using the MetaStrategy
3. Backtest the meta-strategy against historical order book data
4. Compare performance of the meta-strategy vs. individual strategies
"""

import asyncio
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
from src.strategy.market_imbalance import MarketImbalanceStrategy
from src.strategy.volume_absorption import VolumeAbsorptionStrategy
from src.strategy.meta_strategy import MetaStrategy, SignalCombinationMethod, StrategyWeighting
from src.backtesting.orderbook_backtesting import (
    OrderBookSnapshot, 
    OrderBookDataset, 
    OrderBookBacktester
)


logger = get_logger("examples", "meta_strategy_backtest")


def generate_sample_dataset(
    symbol="BTC/USD", 
    num_snapshots=1000, 
    base_price=50000.0, 
    imbalance_rate=0.05,
    large_order_rate=0.05
):
    """Generate a sample order book dataset for backtesting.
    
    Args:
        symbol: The trading pair symbol
        num_snapshots: Number of snapshots to generate
        base_price: The base price to use
        imbalance_rate: How often to generate imbalance events
        large_order_rate: How often to generate large order events
        
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
        if random.random() < imbalance_rate:
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
        if random.random() < large_order_rate:
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


async def run_single_strategy_backtest(strategy, dataset, output_dir, strategy_id=None):
    """Run a backtest for a single strategy.
    
    Args:
        strategy: The strategy to backtest
        dataset: The order book dataset to test against
        output_dir: Directory to save results
        strategy_id: Optional strategy ID override
        
    Returns:
        The backtest results
    """
    # Get strategy ID
    strategy_id = strategy_id or strategy.strategy_id
    
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


async def run_meta_strategy_backtest(dataset, output_dir, combination_method=SignalCombinationMethod.WEIGHTED_AVERAGE):
    """Run a backtest with a meta-strategy combining multiple strategies.
    
    Args:
        dataset: The order book dataset to test against
        output_dir: Directory to save results
        combination_method: The method to use for combining signals
        
    Returns:
        The backtest results
    """
    # Create individual strategies
    logger.info("Creating individual strategies...")
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
    
    # Create meta-strategy with the specified combination method
    meta_strategy = MetaStrategy(
        strategy_id=f"meta_{combination_method}",
        sub_strategies=[market_imbalance, volume_absorption],
        strategy_weights=strategy_weights,
        combination_method=combination_method,
        min_consensus_pct=0.5,  # At least 50% of strategies must agree
        min_overall_confidence=0.6,  # Minimum combined confidence
        signal_window=60  # 1 minute window for concurrent signals
    )
    
    # Run the backtest
    logger.info(f"Running meta-strategy backtest with {combination_method} combination method...")
    results = await run_single_strategy_backtest(meta_strategy, dataset, output_dir)
    
    return results


async def compare_strategies(individual_results, meta_results, output_dir):
    """Compare the performance of individual strategies vs meta-strategy.
    
    Args:
        individual_results: Dict of strategy_id -> results for individual strategies
        meta_results: Dict of combination_method -> results for meta-strategies
        output_dir: Directory to save comparison plots
    """
    logger.info("Comparing strategy performance...")
    
    # Create a directory for comparison
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare signal counts
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = list(individual_results.keys()) + [f"meta_{m}" for m in meta_results.keys()]
    signal_counts = []
    
    for strategy_id in individual_results:
        results = individual_results[strategy_id]
        signal_counts.append(len(results.signals))
    
    for method in meta_results:
        results = meta_results[method]
        signal_counts.append(len(results.signals))
    
    ax.bar(strategies, signal_counts)
    ax.set_title("Signal Count Comparison")
    ax.set_ylabel("Number of Signals")
    ax.set_xlabel("Strategy")
    plt.xticks(rotation=45)
    
    fig.tight_layout()
    fig.savefig(str(output_dir / "signal_count_comparison.png"))
    plt.close(fig)
    
    # Compare signal confidence
    fig, ax = plt.subplots(figsize=(12, 6))
    
    confidence_means = []
    
    for strategy_id in individual_results:
        results = individual_results[strategy_id]
        entry_signals = [s for s in results.signals if s.signal_type == SignalType.ENTRY]
        if entry_signals:
            confidence_means.append(sum(s.confidence for s in entry_signals) / len(entry_signals))
        else:
            confidence_means.append(0)
    
    for method in meta_results:
        results = meta_results[method]
        entry_signals = [s for s in results.signals if s.signal_type == SignalType.ENTRY]
        if entry_signals:
            confidence_means.append(sum(s.confidence for s in entry_signals) / len(entry_signals))
        else:
            confidence_means.append(0)
    
    ax.bar(strategies, confidence_means)
    ax.set_title("Average Signal Confidence Comparison")
    ax.set_ylabel("Average Confidence")
    ax.set_xlabel("Strategy")
    plt.xticks(rotation=45)
    
    fig.tight_layout()
    fig.savefig(str(output_dir / "confidence_comparison.png"))
    plt.close(fig)
    
    # Create a summary table
    data = []
    
    for strategy_id in individual_results:
        results = individual_results[strategy_id]
        entry_signals = [s for s in results.signals if s.signal_type == SignalType.ENTRY]
        data.append({
            "Strategy": strategy_id,
            "Total Signals": len(results.signals),
            "Entry Signals": len(entry_signals),
            "Long Signals": len([s for s in entry_signals if s.direction == "long"]),
            "Short Signals": len([s for s in entry_signals if s.direction == "short"]),
            "Avg Confidence": sum(s.confidence for s in entry_signals) / len(entry_signals) if entry_signals else 0
        })
    
    for method in meta_results:
        results = meta_results[method]
        entry_signals = [s for s in results.signals if s.signal_type == SignalType.ENTRY]
        data.append({
            "Strategy": f"meta_{method}",
            "Total Signals": len(results.signals),
            "Entry Signals": len(entry_signals),
            "Long Signals": len([s for s in entry_signals if s.direction == "long"]),
            "Short Signals": len([s for s in entry_signals if s.direction == "short"]),
            "Avg Confidence": sum(s.confidence for s in entry_signals) / len(entry_signals) if entry_signals else 0
        })
    
    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / "strategy_comparison.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved comparison to {csv_path}")
    
    # Display summary
    logger.info("\nStrategy Comparison Summary:")
    for row in data:
        logger.info(f"{row['Strategy']}: {row['Entry Signals']} signals, {row['Avg Confidence']:.2f} avg confidence")


async def main():
    """Run the meta-strategy backtesting example."""
    parser = argparse.ArgumentParser(description="Meta-strategy backtesting example")
    parser.add_argument(
        "--snapshots", 
        type=int, 
        default=2000, 
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
        default="backtest_results/meta_strategy", 
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
            dataset.save_to_file(args.dataset_file)
            logger.info(f"Saved dataset to {args.dataset_file}")
    
    # Run individual strategy backtests
    logger.info("Running individual strategy backtests...")
    
    # Create the strategies
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
    
    # Run backtests
    individual_results = {}
    
    # Market Imbalance Strategy
    market_imbalance_results = await run_single_strategy_backtest(
        market_imbalance, 
        dataset, 
        args.output_dir
    )
    individual_results["market_imbalance"] = market_imbalance_results
    
    # Volume Absorption Strategy
    volume_absorption_results = await run_single_strategy_backtest(
        volume_absorption, 
        dataset, 
        args.output_dir
    )
    individual_results["volume_absorption"] = volume_absorption_results
    
    # Run meta-strategy backtests with different combination methods
    logger.info("Running meta-strategy backtests...")
    
    meta_results = {}
    
    # Weighted Average
    meta_weighted_results = await run_meta_strategy_backtest(
        dataset, 
        args.output_dir,
        SignalCombinationMethod.WEIGHTED_AVERAGE
    )
    meta_results[SignalCombinationMethod.WEIGHTED_AVERAGE] = meta_weighted_results
    
    # Majority Vote
    meta_majority_results = await run_meta_strategy_backtest(
        dataset, 
        args.output_dir,
        SignalCombinationMethod.MAJORITY_VOTE
    )
    meta_results[SignalCombinationMethod.MAJORITY_VOTE] = meta_majority_results
    
    # Compare strategies
    await compare_strategies(individual_results, meta_results, args.output_dir)
    
    logger.info("Backtesting completed")


if __name__ == "__main__":
    asyncio.run(main()) 