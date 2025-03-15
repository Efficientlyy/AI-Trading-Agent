"""
Portfolio Rebalancing Demo

This script demonstrates the portfolio rebalancing functionality, including:
1. Creating allocation plans with different methods
2. Checking when rebalance is needed based on allocation drift
3. Generating and executing rebalance trades
4. Visualizing portfolio allocations before and after rebalancing
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Set, Optional
import random

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from tabulate import tabulate
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Visualization requires matplotlib, numpy, and tabulate. Install with:")
    print("pip install matplotlib numpy tabulate")
    VISUALIZATION_AVAILABLE = False

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("portfolio_rebalance_demo")

# Import the necessary modules, skipping the config-dependent logging module
from src.models.portfolio import (
    AllocationMethod,
    RebalanceTrigger,
    AssetAllocation,
    RebalanceConfig,
    PortfolioAllocation
)
from src.portfolio.portfolio_manager import PortfolioManager, Position, PositionType
from src.portfolio.rebalancing import PortfolioRebalancer


def setup_test_portfolio(portfolio_manager: PortfolioManager) -> None:
    """
    Set up a test portfolio with sample positions.
    
    Args:
        portfolio_manager: Portfolio manager instance
    """
    # Create some test positions manually
    positions = [
        # BTC with increased allocation (for demonstrating drift)
        Position(
            symbol="BTC/USD",
            position_type=PositionType.LONG,
            size=Decimal("2"),
            entry_price=Decimal("50000")
        ),
        # ETH position
        Position(
            symbol="ETH/USD",
            position_type=PositionType.LONG,
            size=Decimal("40"),
            entry_price=Decimal("3000")
        ),
        # SOL position
        Position(
            symbol="SOL/USD",
            position_type=PositionType.LONG,
            size=Decimal("500"),
            entry_price=Decimal("100")
        ),
        # AVAX position
        Position(
            symbol="AVAX/USD",
            position_type=PositionType.LONG,
            size=Decimal("100"),
            entry_price=Decimal("20")
        ),
    ]
    
    # Add positions to portfolio
    for position in positions:
        # Generate a position ID
        position.position_id = f"pos_{position.symbol.split('/')[0].lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        portfolio_manager.positions[position.position_id] = position
        portfolio_manager.open_positions.add(position.position_id)
        
        # Add to symbol tracking
        if position.symbol not in portfolio_manager.open_positions_by_symbol:
            portfolio_manager.open_positions_by_symbol[position.symbol] = set()
        portfolio_manager.open_positions_by_symbol[position.symbol].add(position.position_id)
    
    # Set current prices
    portfolio_manager.current_prices = {
        "BTC/USD": Decimal("55000"),  # Up 10% from entry
        "ETH/USD": Decimal("2900"),   # Down a bit
        "SOL/USD": Decimal("110"),    # Up a bit
        "AVAX/USD": Decimal("21"),    # Up a bit
    }
    
    # Update portfolio value
    portfolio_manager.current_balance = sum(
        position.size * portfolio_manager.current_prices.get(position.symbol, Decimal("0"))
        for position in portfolio_manager.get_open_positions()
    ) or Decimal("0")  # Ensure it's always a Decimal, even if sum is zero
    
    logger.info(f"Set up test portfolio with total value: {portfolio_manager.current_balance}")


async def run_equal_weight_demo(rebalancer: PortfolioRebalancer) -> None:
    """
    Demonstrate equal weight allocation and rebalancing.
    
    Args:
        rebalancer: Portfolio rebalancer instance
    """
    logger.info("=== Equal Weight Allocation Demo ===")
    
    # Create equal weight allocation plan
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    plan = rebalancer.create_allocation_plan(
        name="equal_weight_demo",
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        symbols=symbols,
        config=RebalanceConfig(
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            trigger_method=RebalanceTrigger.THRESHOLD_BASED,
            drift_threshold=Decimal("0.05")  # 5% drift threshold
        )
    )
    
    # Check allocation status before rebalance
    status_before = await rebalancer.get_allocation_status("equal_weight_demo")
    
    logger.info("Current portfolio allocation (before rebalance):")
    display_allocation_status(status_before)
    
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status_before, "Before Rebalance (Equal Weight)")
    
    # Check if rebalance is needed
    rebalance_needed = await rebalancer.check_rebalance_needed("equal_weight_demo")
    logger.info(f"Rebalance needed: {rebalance_needed}")
    
    if rebalance_needed:
        # Generate rebalance plan
        trades = await rebalancer.generate_rebalance_plan("equal_weight_demo")
        
        logger.info("Proposed trades:")
        for symbol, amount in trades.items():
            action = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {action} {abs(amount)} of {symbol}")
        
        # Execute rebalance
        success = await rebalancer.execute_rebalance("equal_weight_demo")
        logger.info(f"Rebalance execution {'succeeded' if success else 'failed'}")
        
        # Display status after rebalance
        status_after = await rebalancer.get_allocation_status("equal_weight_demo")
        
        logger.info("Simulated portfolio allocation (after rebalance):")
        display_allocation_status(status_after)
        
        if VISUALIZATION_AVAILABLE:
            plot_allocation(status_after, "After Rebalance (Equal Weight)")


async def run_volatility_weighted_demo(rebalancer: PortfolioRebalancer) -> None:
    """
    Demonstrate volatility weighted allocation and rebalancing.
    
    Args:
        rebalancer: Portfolio rebalancer instance
    """
    logger.info("\n=== Volatility Weighted Allocation Demo ===")
    
    # Set up market data with volatility information
    rebalancer.update_market_data("BTC/USD", {"volatility": 0.80})   # High volatility
    rebalancer.update_market_data("ETH/USD", {"volatility": 0.60})   # Medium-high volatility
    rebalancer.update_market_data("SOL/USD", {"volatility": 1.20})   # Very high volatility
    rebalancer.update_market_data("AVAX/USD", {"volatility": 0.40})  # Medium volatility
    
    # Create volatility weighted allocation plan
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    plan = rebalancer.create_allocation_plan(
        name="volatility_weighted_demo",
        allocation_method=AllocationMethod.VOLATILITY_WEIGHTED,
        symbols=symbols,
        config=RebalanceConfig(
            allocation_method=AllocationMethod.VOLATILITY_WEIGHTED,
            trigger_method=RebalanceTrigger.THRESHOLD_BASED,
            drift_threshold=Decimal("0.05")  # 5% drift threshold
        )
    )
    
    # Check allocation status before rebalance
    status_before = await rebalancer.get_allocation_status("volatility_weighted_demo")
    
    logger.info("Current portfolio allocation (before rebalance):")
    display_allocation_status(status_before)
    
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status_before, "Before Rebalance (Volatility Weighted)")
    
    # Check if rebalance is needed
    rebalance_needed = await rebalancer.check_rebalance_needed("volatility_weighted_demo")
    logger.info(f"Rebalance needed: {rebalance_needed}")
    
    if rebalance_needed:
        # Generate rebalance plan
        trades = await rebalancer.generate_rebalance_plan("volatility_weighted_demo")
        
        logger.info("Proposed trades:")
        for symbol, amount in trades.items():
            action = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {action} {abs(amount)} of {symbol}")
        
        # Execute rebalance
        success = await rebalancer.execute_rebalance("volatility_weighted_demo")
        logger.info(f"Rebalance execution {'succeeded' if success else 'failed'}")
        
        # Display status after rebalance
        status_after = await rebalancer.get_allocation_status("volatility_weighted_demo")
        
        logger.info("Simulated portfolio allocation (after rebalance):")
        display_allocation_status(status_after)
        
        if VISUALIZATION_AVAILABLE:
            plot_allocation(status_after, "After Rebalance (Volatility Weighted)")


async def run_custom_weight_demo(rebalancer: PortfolioRebalancer) -> None:
    """
    Demonstrate custom weight allocation and rebalancing.
    
    Args:
        rebalancer: Portfolio rebalancer instance
    """
    logger.info("\n=== Custom Weight Allocation Demo ===")
    
    # Create custom config with custom weights
    custom_config = RebalanceConfig(
        allocation_method=AllocationMethod.CUSTOM,
        trigger_method=RebalanceTrigger.THRESHOLD_BASED,
        drift_threshold=Decimal("0.05"),  # 5% drift threshold
        custom_weights={
            "BTC/USD": Decimal("0.40"),  # 40% Bitcoin
            "ETH/USD": Decimal("0.30"),  # 30% Ethereum
            "SOL/USD": Decimal("0.20"),  # 20% Solana
            "AVAX/USD": Decimal("0.10"),  # 10% Avalanche
        }
    )
    
    # Create custom weight allocation plan
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    plan = rebalancer.create_allocation_plan(
        name="custom_weight_demo",
        allocation_method=AllocationMethod.CUSTOM,
        symbols=symbols,
        config=custom_config
    )
    
    # Check allocation status before rebalance
    status_before = await rebalancer.get_allocation_status("custom_weight_demo")
    
    logger.info("Current portfolio allocation (before rebalance):")
    display_allocation_status(status_before)
    
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status_before, "Before Rebalance (Custom Weights)")
    
    # Check if rebalance is needed
    rebalance_needed = await rebalancer.check_rebalance_needed("custom_weight_demo")
    logger.info(f"Rebalance needed: {rebalance_needed}")
    
    if rebalance_needed:
        # Generate rebalance plan
        trades = await rebalancer.generate_rebalance_plan("custom_weight_demo")
        
        logger.info("Proposed trades:")
        for symbol, amount in trades.items():
            action = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {action} {abs(amount)} of {symbol}")
        
        # Execute rebalance
        success = await rebalancer.execute_rebalance("custom_weight_demo")
        logger.info(f"Rebalance execution {'succeeded' if success else 'failed'}")
        
        # Display status after rebalance
        status_after = await rebalancer.get_allocation_status("custom_weight_demo")
        
        logger.info("Simulated portfolio allocation (after rebalance):")
        display_allocation_status(status_after)
        
        if VISUALIZATION_AVAILABLE:
            plot_allocation(status_after, "After Rebalance (Custom Weights)")


def display_allocation_status(status: Dict[str, Any]) -> None:
    """
    Display allocation status in a readable format.
    
    Args:
        status: Allocation status dictionary
    """
    # Check if status has error
    if "error" in status:
        print(f"Error: {status['error']}")
        return
    
    # Format the data for display
    headers = ["Symbol", "Target %", "Current %", "Drift %"]
    rows = []
    
    for allocation in status["allocations"]:
        target_pct = f"{allocation['target_percentage'] * 100:.2f}%"
        current_pct = f"{allocation.get('current_percentage', 0) * 100:.2f}%" if allocation.get('current_percentage') is not None else "N/A"
        
        drift = allocation.get('drift')
        if drift is not None:
            drift_pct = f"{drift * 100:+.2f}%"  # Add + sign for positive drift
            # Add color indicators (in terminal)
            if drift > 0.05:
                drift_pct = f"\033[91m{drift_pct}\033[0m"  # Red for high positive drift
            elif drift < -0.05:
                drift_pct = f"\033[91m{drift_pct}\033[0m"  # Red for high negative drift
            elif abs(drift) > 0.02:
                drift_pct = f"\033[93m{drift_pct}\033[0m"  # Yellow for moderate drift
        else:
            drift_pct = "N/A"
        
        rows.append([allocation["symbol"], target_pct, current_pct, drift_pct])
    
    # Add summary row
    rows.append(["TOTAL", 
                f"{status.get('total_target_allocation', 0) * 100:.2f}%", 
                f"{status.get('total_current_allocation', 0) * 100:.2f}%",
                f"{status.get('max_drift', 0) * 100:.2f}% max"])
    
    if "tabulate" in sys.modules:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Fallback to simple format if tabulate is not available
        print("\t".join(headers))
        print("-" * 60)
        for row in rows:
            print("\t".join(str(cell) for cell in row))
    
    # Print rebalance status
    needs_rebalance = status.get("needs_rebalance", False)
    last_rebalance = status.get("last_rebalance", "Never")
    
    print(f"\nNeeds rebalance: {'Yes' if needs_rebalance else 'No'}")
    print(f"Last rebalance: {last_rebalance}")


def plot_allocation(status: Dict[str, Any], title: str) -> None:
    """
    Plot allocation as a pie chart.
    
    Args:
        status: Allocation status dictionary
        title: Plot title
    """
    if not VISUALIZATION_AVAILABLE:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Prepare data
    labels = []
    target_sizes = []
    current_sizes = []
    
    for allocation in status["allocations"]:
        labels.append(allocation["symbol"].split("/")[0])  # Remove /USD part
        target_sizes.append(allocation["target_percentage"])
        current_sizes.append(allocation.get("current_percentage", 0) or 0)
    
    # Target allocation pie chart
    ax1.pie(target_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Target Allocation')
    
    # Current allocation pie chart
    ax2.pie(current_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Current Allocation')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("data/plots", exist_ok=True)
    filename = f"data/plots/allocation_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    logger.info(f"Saved allocation chart to {filename}")
    
    # Show plot
    plt.show()


class SimplePortfolioManager(PortfolioManager):
    """
    A simplified PortfolioManager for the demo that doesn't rely on config.
    """
    
    def _record_balance(self, balance: Decimal) -> None:
        """Record balance in history."""
        # This is a simplified version that just logs the balance
        logger.info(f"Recording balance: {balance}")


async def main():
    """Main demo function."""
    logger.info("Starting Portfolio Rebalancing Demo")
    
    # Initialize portfolio manager with test data
    portfolio_manager = SimplePortfolioManager(
        initial_balance=Decimal("300000")  # $300k initial balance
    )
    
    # Set up test portfolio
    setup_test_portfolio(portfolio_manager)
    
    # Initialize portfolio rebalancer
    rebalancer = PortfolioRebalancer(portfolio_manager)
    
    # Run the demos
    await run_equal_weight_demo(rebalancer)
    await run_volatility_weighted_demo(rebalancer)
    await run_custom_weight_demo(rebalancer)
    
    logger.info("Portfolio Rebalancing Demo completed")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data/plots", exist_ok=True)
    
    # Run the demo
    asyncio.run(main()) 