#!/usr/bin/env python
"""Transaction Cost Analysis (TCA) Demo

This script demonstrates the use of the Transaction Cost Analysis (TCA) module for
analyzing and comparing the performance of different execution algorithms.

The demo shows:
1. How to calculate implementation shortfall, market impact, and slippage
2. How to analyze an individual execution
3. How to compare the performance of different execution algorithms
4. How to use real-time metrics for monitoring execution quality

This is useful for evaluating the quality of executions and optimizing trading strategies.
"""

import asyncio
import logging
import random
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.logging import get_logger
from src.execution.tca import TransactionCostAnalyzer, RealTimeMetrics
from src.models.order import OrderSide
from src.execution.exchange.mock import MockExchangeConnector
from src.execution.interface import ExchangeInterface
from src.execution.algorithms.twap import TWAPExecutor
from src.execution.algorithms.vwap import VWAPExecutor
from src.execution.algorithms.iceberg import IcebergExecutor
from src.execution.algorithms.smart_order_routing import SmartOrderRouter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("examples.tca_demo")


def generate_simulated_order_data(num_orders: int = 100) -> List[Dict[str, Any]]:
    """Generate simulated order execution data for demonstration.
    
    Args:
        num_orders: Number of orders to generate
        
    Returns:
        List of dictionaries containing order execution details
    """
    now = datetime.now()
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    algorithms = ["TWAP", "VWAP", "Iceberg", "SOR"]
    
    orders = []
    for i in range(num_orders):
        symbol = random.choice(symbols)
        algo = random.choice(algorithms)
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        
        # Base price depends on the symbol
        if symbol == "BTC/USDT":
            base_price = 50000
        elif symbol == "ETH/USDT":
            base_price = 3000
        elif symbol == "SOL/USDT":
            base_price = 100
        else:  # BNB/USDT
            base_price = 400
        
        # Generate realistic price variations
        decision_price = base_price * (1 + random.uniform(-0.01, 0.01))
        arrival_price = decision_price * (1 + random.uniform(-0.001, 0.001))
        expected_price = arrival_price * (1 + random.uniform(-0.001, 0.001))
        
        # Execution price varies by algorithm
        algo_variation = {
            "TWAP": random.uniform(0.0005, 0.002),
            "VWAP": random.uniform(0.0003, 0.0015),
            "Iceberg": random.uniform(0.0002, 0.001),
            "SOR": random.uniform(0.0001, 0.0008)
        }
        
        # For buys, execution price is typically higher; for sells, it's typically lower
        if side == OrderSide.BUY:
            execution_price = expected_price * (1 + algo_variation[algo])
        else:
            execution_price = expected_price * (1 - algo_variation[algo])
        
        # Quantity varies by symbol
        if symbol == "BTC/USDT":
            quantity = random.uniform(0.1, 2.0)
        elif symbol == "ETH/USDT":
            quantity = random.uniform(1.0, 20.0)
        elif symbol == "SOL/USDT":
            quantity = random.uniform(10.0, 100.0)
        else:  # BNB/USDT
            quantity = random.uniform(5.0, 50.0)
        
        # Generate realistic fees (0.1% to 0.15% of executed value)
        fee_rate = random.uniform(0.001, 0.0015)
        fees = execution_price * quantity * fee_rate
        
        # Generate execution time (faster for SOR, slower for TWAP)
        execution_time_seconds = {
            "TWAP": random.uniform(120, 600),
            "VWAP": random.uniform(90, 450),
            "Iceberg": random.uniform(60, 300),
            "SOR": random.uniform(10, 60)
        }
        
        execution_time = execution_time_seconds[algo]
        execution_timestamp = now - timedelta(days=random.randint(0, 30))
        
        order = {
            "order_id": f"order-{i}",
            "symbol": symbol,
            "side": side,
            "algorithm": algo,
            "quantity": quantity,
            "decision_price": decision_price,
            "arrival_price": arrival_price,
            "expected_price": expected_price,
            "execution_price": execution_price,
            "fees": fees,
            "execution_time": execution_time,
            "timestamp": execution_timestamp
        }
        
        orders.append(order)
    
    return orders


def generate_simulated_market_data(orders: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Generate simulated market data for the orders.
    
    Args:
        orders: List of order execution details
        
    Returns:
        Dictionary mapping order IDs to market data
    """
    market_data = {}
    
    for order in orders:
        order_id = order["order_id"]
        symbol = order["symbol"]
        
        # Pre-trade price is typically close to arrival price
        pre_trade_price = order["arrival_price"] * (1 + random.uniform(-0.0005, 0.0005))
        
        # VWAP price varies by algorithm and is influenced by market conditions
        if order["algorithm"] == "VWAP":
            # VWAP should perform close to the actual VWAP
            vwap_price = order["execution_price"] * (1 + random.uniform(-0.0003, 0.0003))
        else:
            # Other algorithms might deviate more from VWAP
            vwap_price = order["arrival_price"] * (1 + random.uniform(-0.001, 0.001))
        
        # Market volume depends on symbol and varies randomly
        base_volume = {
            "BTC/USDT": 1000,
            "ETH/USDT": 10000,
            "SOL/USDT": 50000,
            "BNB/USDT": 5000
        }
        
        volume = base_volume[symbol] * (1 + random.uniform(-0.3, 0.3))
        
        market_data[order_id] = {
            "pre_trade_price": pre_trade_price,
            "vwap_price": vwap_price,
            "volume": volume,
            "symbol": symbol,
            "timestamp": order["timestamp"]
        }
    
    return market_data


def visualize_tca_results(algo_comparison: Dict[str, Any]) -> None:
    """Visualize TCA results with matplotlib.
    
    Args:
        algo_comparison: Algorithm comparison results from TCA
    """
    try:
        # Extract data for visualization
        algo_results = algo_comparison.get("detailed_results", {})
        metrics = ["avg_percentage_shortfall", "avg_percentage_impact", "avg_percentage_slippage"]
        
        # Prepare data for plotting
        algo_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for algo_name, result in algo_results.items():
            algo_names.append(algo_name)
            for metric in metrics:
                value = result.get("average_metrics", {}).get(metric, 0)
                metric_values[metric].append(value)
        
        # Check if we have data to plot
        if not algo_names:
            logger.warning("No data available for visualization")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
        fig.suptitle("Execution Algorithm Performance Comparison", fontsize=16)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(algo_names, metric_values[metric], color=['blue', 'green', 'orange', 'red'])
            
            # Add labels and formatting
            ax.set_title(f"{metric.replace('avg_', '').replace('_', ' ').title()}")
            ax.set_ylabel("Percentage (%)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f"{height:.3f}%", ha='center', va='bottom')
        
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig("tca_results.png")
        logger.info("TCA visualization saved to 'tca_results.png'")
        
    except Exception as e:
        logger.error(f"Error visualizing TCA results: {e}")


async def simulate_real_time_execution():
    """Simulate a real-time execution with metrics updates."""
    logger.info("Simulating real-time execution metrics tracking")
    
    # Setup real-time metrics calculator
    metrics = RealTimeMetrics(
        side=OrderSide.BUY,
        decision_price=50000.0,
        expected_quantity=1.0
    )
    
    # Simulate fills arriving over time
    fill_prices = [
        50005.0,  # Small slippage on first fill
        50010.0,  # Price moves against us
        50008.0,  # Price recovers slightly
        50015.0,  # More slippage
        50012.0   # Final fill
    ]
    
    fill_quantities = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal-sized fills for simplicity
    
    logger.info(f"Starting execution of 1.0 BTC at decision price: {metrics.decision_price}")
    
    # Process each fill with a delay to simulate real-time
    for i, (price, quantity) in enumerate(zip(fill_prices, fill_quantities)):
        # Update metrics with new fill
        metrics.update(price, quantity)
        
        # Get current metrics
        current_metrics = metrics.get_metrics()
        
        logger.info(f"Fill {i+1}: {quantity} BTC at {price}")
        logger.info(f"  Completed: {current_metrics['completion_percentage']:.2f}%")
        logger.info(f"  Average Price: {current_metrics['average_price']:.2f}")
        logger.info(f"  Shortfall: {current_metrics['percentage_shortfall']:.4f}%")
        logger.info(f"  Execution Rate: {current_metrics['execution_rate']:.4f} BTC/s")
        
        # Pause to simulate time passing
        await asyncio.sleep(1)
    
    # Final metrics
    final_metrics = metrics.get_metrics()
    logger.info("\nFinal Execution Metrics:")
    logger.info(f"  Quantity Executed: {final_metrics['executed_quantity']} BTC")
    logger.info(f"  Average Execution Price: {final_metrics['average_price']:.2f}")
    logger.info(f"  Implementation Shortfall: {final_metrics['percentage_shortfall']:.4f}%")
    logger.info(f"  Total Shortfall Cost: {final_metrics['total_shortfall_cost']:.2f} USD")
    logger.info(f"  Execution Time: {final_metrics['execution_time_seconds']:.2f} seconds")
    logger.info(f"  Price Dispersion: {final_metrics['price_dispersion']:.2f}")


async def run_demo():
    """Run the full TCA demo."""
    logger.info("Starting Transaction Cost Analysis Demo")
    
    # Part 1: Generate simulated data
    logger.info("Generating simulated execution data...")
    orders = generate_simulated_order_data(100)
    market_data = generate_simulated_market_data(orders)
    
    # Create the TCA analyzer
    tca = TransactionCostAnalyzer()
    
    # Part 2: Analyze a single order
    logger.info("\n=== Part 1: Analyzing a Single Order ===")
    sample_order = orders[0]
    sample_market_data = market_data[sample_order["order_id"]]
    
    logger.info(f"Order Details:")
    logger.info(f"  Symbol: {sample_order['symbol']}")
    logger.info(f"  Side: {sample_order['side']}")
    logger.info(f"  Algorithm: {sample_order['algorithm']}")
    logger.info(f"  Quantity: {sample_order['quantity']}")
    logger.info(f"  Decision Price: {sample_order['decision_price']:.2f}")
    logger.info(f"  Execution Price: {sample_order['execution_price']:.2f}")
    
    # Calculate individual metrics
    shortfall = tca.calculate_implementation_shortfall(
        side=sample_order["side"],
        decision_price=sample_order["decision_price"],
        execution_price=sample_order["execution_price"],
        quantity=sample_order["quantity"],
        fees=sample_order["fees"]
    )
    
    logger.info("\nImplementation Shortfall:")
    logger.info(f"  Price Shortfall: {shortfall['implementation_shortfall']:.2f}")
    logger.info(f"  Percentage Shortfall: {shortfall['percentage_shortfall']:.4f}%")
    logger.info(f"  Shortfall Cost: {shortfall['shortfall_cost']:.2f}")
    logger.info(f"  Fees: {shortfall['fees']:.2f}")
    logger.info(f"  Total Cost: {shortfall['total_cost']:.2f}")
    
    # Calculate full analysis
    analysis = tca.analyze_execution_quality(sample_order, sample_market_data)
    
    if "market_impact" in analysis:
        impact = analysis["market_impact"]
        logger.info("\nMarket Impact:")
        logger.info(f"  Price Impact: {impact['price_impact']:.2f}")
        logger.info(f"  Percentage Impact: {impact['percentage_impact']:.4f}%")
        logger.info(f"  Participation Rate: {impact['participation_rate']:.2f}%")
    
    if "slippage" in analysis:
        slippage = analysis["slippage"]
        logger.info("\nSlippage:")
        logger.info(f"  Slippage: {slippage['slippage']:.2f}")
        logger.info(f"  Percentage Slippage: {slippage['percentage_slippage']:.4f}%")
    
    if "timing_cost" in analysis:
        timing = analysis["timing_cost"]
        logger.info("\nTiming Cost:")
        logger.info(f"  Arrival Performance: {timing['percentage_arrival']:.4f}%")
        logger.info(f"  VWAP Performance: {timing['percentage_vwap']:.4f}%")
    
    # Part 3: Compare algorithm performance
    logger.info("\n=== Part 2: Comparing Execution Algorithms ===")
    
    # Group orders by algorithm
    orders_by_algo = {}
    for order in orders:
        algo = order["algorithm"]
        if algo not in orders_by_algo:
            orders_by_algo[algo] = []
        orders_by_algo[algo].append(order)
    
    # Analyze each algorithm's performance
    algo_results = {}
    for algo, algo_orders in orders_by_algo.items():
        result = tca.analyze_algo_performance(algo, algo_orders, market_data)
        algo_results[algo] = result
        
        logger.info(f"\n{algo} Performance:")
        logger.info(f"  Order Count: {result['order_count']}")
        logger.info(f"  Total Volume: {result['total_volume']:.2f}")
        
        metrics = result.get("average_metrics", {})
        if "avg_percentage_shortfall" in metrics:
            logger.info(f"  Avg Percentage Shortfall: {metrics['avg_percentage_shortfall']:.4f}%")
        if "avg_percentage_impact" in metrics:
            logger.info(f"  Avg Percentage Impact: {metrics['avg_percentage_impact']:.4f}%")
        if "avg_percentage_slippage" in metrics:
            logger.info(f"  Avg Percentage Slippage: {metrics['avg_percentage_slippage']:.4f}%")
    
    # Compare algorithms
    comparison = tca.compare_algos(algo_results)
    
    logger.info("\nAlgorithm Rankings:")
    for metric, rankings in comparison.get("metric_rankings", {}).items():
        logger.info(f"  {metric.replace('avg_', '').replace('_', ' ').title()}:")
        for i, (algo, value) in enumerate(rankings):
            logger.info(f"    {i+1}. {algo}: {value:.4f}%")
    
    logger.info("\nOverall Algorithm Ranking:")
    for i, (algo, rank) in enumerate(comparison.get("overall_ranking", [])):
        logger.info(f"  {i+1}. {algo} (Avg Rank: {rank:.2f})")
    
    best_algo = comparison.get("best_algorithm")
    if best_algo:
        logger.info(f"\nBest Performing Algorithm: {best_algo}")
    
    # Visualize results
    visualize_tca_results(comparison)
    
    # Part 4: Real-time metrics simulation
    logger.info("\n=== Part 3: Real-Time Execution Metrics ===")
    await simulate_real_time_execution()
    
    logger.info("\nTransaction Cost Analysis Demo completed")


if __name__ == "__main__":
    asyncio.run(run_demo()) 