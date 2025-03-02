#!/usr/bin/env python
"""
Order Book Processor Benchmark

This script benchmarks the performance of the OrderBookProcessor,
comparing the Rust implementation with the Python fallback.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rust_bridge import create_order_book_processor, is_rust_available
try:
    from src.rust_bridge.market_data_py import OrderBookProcessor as PyOrderBookProcessor
    PYTHON_IMPL_AVAILABLE = True
except ImportError:
    PYTHON_IMPL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(num_levels: int = 100, base_price: float = 50000.0, 
                         num_batches: int = 100, updates_per_batch: int = 50) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Generate a large dataset for benchmarking.
    
    Args:
        num_levels: Number of price levels to initialize the book with
        base_price: Base price for the order book
        num_batches: Number of update batches to generate
        updates_per_batch: Number of updates per batch
        
    Returns:
        A tuple containing (initial_data, update_batches)
    """
    # Generate initial data
    spread = base_price * 0.02  # 0.02% spread
    best_bid = base_price - spread / 2
    best_ask = base_price + spread / 2
    
    initial_data = []
    
    # Generate bids
    for i in range(num_levels):
        price = best_bid * (1 - 0.0005 * i)
        volume = 0.5 + (i * 0.1)  # Increasing volume
        initial_data.append({
            "price": price,
            "side": "buy",
            "quantity": volume,
            "timestamp": time.time(),
            "sequence": i + 1
        })
    
    # Generate asks
    for i in range(num_levels):
        price = best_ask * (1 + 0.0005 * i)
        volume = 0.5 + (i * 0.1)  # Increasing volume
        initial_data.append({
            "price": price,
            "side": "sell",
            "quantity": volume,
            "timestamp": time.time(),
            "sequence": num_levels + i + 1
        })
    
    # Generate update batches
    update_batches = []
    sequence_start = num_levels * 2 + 1
    
    for batch in range(num_batches):
        batch_updates = []
        for i in range(updates_per_batch):
            update_type = random.choices(
                ["modify", "remove", "add"],
                weights=[0.7, 0.15, 0.15],
                k=1
            )[0]
            
            side = "buy" if random.random() < 0.5 else "sell"
            
            if update_type == "modify":
                # Modify an existing price level
                price = best_bid * (1 - 0.0005 * random.randint(0, num_levels-1)) if side == "buy" else \
                        best_ask * (1 + 0.0005 * random.randint(0, num_levels-1))
                quantity = random.uniform(0.1, 5.0)
                
            elif update_type == "remove":
                # Remove a price level
                price = best_bid * (1 - 0.0005 * random.randint(0, num_levels-1)) if side == "buy" else \
                        best_ask * (1 + 0.0005 * random.randint(0, num_levels-1))
                quantity = 0.0
                
            else:  # "add"
                # Add a new price level
                price = best_bid * (1 - 0.0005 * (num_levels + random.randint(0, 10))) if side == "buy" else \
                        best_ask * (1 + 0.0005 * (num_levels + random.randint(0, 10)))
                quantity = random.uniform(0.1, 5.0)
            
            batch_updates.append({
                "price": price,
                "side": side,
                "quantity": quantity,
                "timestamp": time.time(),
                "sequence": sequence_start + batch * updates_per_batch + i
            })
        
        update_batches.append(batch_updates)
    
    return initial_data, update_batches

def run_benchmark(args: argparse.Namespace):
    """Run the benchmark with specified parameters."""
    # Prepare benchmark parameters
    num_warmup_rounds = args.warmup
    num_benchmark_rounds = args.rounds
    book_depth = args.depth
    updates_per_batch = args.updates
    num_batches = args.batches
    batch_size = args.batch_size
    
    logger.info("Generating benchmark data...")
    initial_data, update_batches = generate_sample_data(
        num_levels=book_depth,
        base_price=50000.0,
        num_batches=num_batches,
        updates_per_batch=updates_per_batch
    )
    
    results = {}
    
    # Run benchmark for Rust implementation
    if is_rust_available():
        logger.info("Benchmarking Rust implementation...")
        rust_times = run_implementation_benchmark(
            "rust",
            initial_data,
            update_batches,
            batch_size,
            num_warmup_rounds,
            num_benchmark_rounds
        )
        results["rust"] = rust_times
    else:
        logger.warning("Rust implementation not available, skipping benchmark.")
    
    # Run benchmark for Python implementation
    if PYTHON_IMPL_AVAILABLE:
        logger.info("Benchmarking Python implementation...")
        python_times = run_implementation_benchmark(
            "python",
            initial_data,
            update_batches,
            batch_size,
            num_warmup_rounds,
            num_benchmark_rounds
        )
        results["python"] = python_times
    else:
        logger.warning("Python fallback implementation not available, skipping benchmark.")
    
    # Compare results if both implementations were benchmarked
    if "rust" in results and "python" in results:
        rust_median = np.median(results["rust"])
        python_median = np.median(results["python"])
        speedup = python_median / rust_median
        
        logger.info("\nPerformance Comparison:")
        logger.info(f"Rust median time: {rust_median:.6f} seconds")
        logger.info(f"Python median time: {python_median:.6f} seconds")
        logger.info(f"Speedup factor: {speedup:.2f}x")
        
        # Plot comparison chart
        if args.plot:
            plot_benchmark_results(results, batch_size * updates_per_batch)
    
    return results

def run_implementation_benchmark(implementation: str, initial_data: List[Dict[str, Any]], 
                               update_batches: List[List[Dict[str, Any]]], batch_size: int,
                               num_warmup_rounds: int, num_benchmark_rounds: int) -> List[float]:
    """Run benchmark for a specific implementation."""
    symbol = "BTC/USD"
    exchange = "benchmark"
    max_depth = 1000  # Large depth to prevent truncation
    
    # Create processor based on implementation
    if implementation == "rust":
        processor = create_order_book_processor(symbol, exchange, max_depth)
    else:  # Python
        processor = PyOrderBookProcessor(symbol, exchange, max_depth)
    
    # Initialize with initial data
    processor.process_updates(initial_data)
    
    # Warmup rounds
    logger.info(f"Running {num_warmup_rounds} warmup rounds...")
    for _ in range(num_warmup_rounds):
        batch_indices = random.sample(range(len(update_batches)), batch_size)
        updates = []
        for idx in batch_indices:
            updates.extend(update_batches[idx])
        processor.process_updates(updates)
    
    # Benchmark rounds
    logger.info(f"Running {num_benchmark_rounds} benchmark rounds...")
    times = []
    
    for i in range(num_benchmark_rounds):
        batch_indices = random.sample(range(len(update_batches)), batch_size)
        updates = []
        for idx in batch_indices:
            updates.extend(update_batches[idx])
        
        # Measure time
        start_time = time.time()
        processor.process_updates(updates)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        if (i + 1) % 10 == 0 or i == num_benchmark_rounds - 1:
            logger.info(f"Completed round {i+1}/{num_benchmark_rounds} - Time: {execution_time:.6f} seconds")
    
    # Calculate statistics
    times = np.array(times)
    min_time = np.min(times)
    max_time = np.max(times)
    mean_time = np.mean(times)
    median_time = np.median(times)
    
    logger.info(f"\n{implementation.capitalize()} Implementation Results:")
    logger.info(f"Min time: {min_time:.6f} seconds")
    logger.info(f"Max time: {max_time:.6f} seconds")
    logger.info(f"Mean time: {mean_time:.6f} seconds")
    logger.info(f"Median time: {median_time:.6f} seconds")
    
    # Calculate throughput
    updates_per_batch = len(update_batches[0]) if update_batches else 0
    total_updates = batch_size * updates_per_batch
    throughput = total_updates / median_time
    logger.info(f"Throughput: {throughput:.0f} updates/second")
    
    # Processor statistics
    stats = processor.processing_stats()
    logger.info(f"Processing stats from processor: {stats}")
    
    return times.tolist()

def plot_benchmark_results(results: Dict[str, List[float]], updates_per_round: int):
    """Plot benchmark results as a box plot."""
    plt.figure(figsize=(10, 6))
    
    # Box plot for processing times
    plt.subplot(1, 2, 1)
    plt.boxplot([results["rust"], results["python"]], labels=["Rust", "Python"])
    plt.title("Processing Time Comparison")
    plt.ylabel("Processing Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Bar chart for throughput
    plt.subplot(1, 2, 2)
    rust_throughput = updates_per_round / np.median(results["rust"])
    python_throughput = updates_per_round / np.median(results["python"])
    
    bar_width = 0.35
    index = np.array([1, 2])
    
    plt.bar(index, [rust_throughput, python_throughput], bar_width, 
            color=['#1f77b4', '#ff7f0e'],
            label=['Rust', 'Python'])
    
    plt.xlabel('Implementation')
    plt.ylabel('Throughput (updates/second)')
    plt.title('Throughput Comparison')
    plt.xticks(index, ['Rust', 'Python'])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add throughput values on top of bars
    for i, v in enumerate([rust_throughput, python_throughput]):
        plt.text(i + 1, v + 0.1, f"{int(v)}", ha='center')
    
    plt.tight_layout()
    plt.savefig("orderbook_benchmark_results.png")
    logger.info("Benchmark plot saved to 'orderbook_benchmark_results.png'")
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark OrderBookProcessor performance")
    parser.add_argument("--depth", type=int, default=100, help="Number of price levels")
    parser.add_argument("--updates", type=int, default=50, help="Updates per batch")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of batches to process per round")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup rounds")
    parser.add_argument("--rounds", type=int, default=20, help="Number of benchmark rounds")
    parser.add_argument("--plot", action="store_true", help="Generate performance comparison plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args) 