#!/usr/bin/env python3
"""
Risk Budget Management Demo

This script demonstrates the risk budget management system, which allows for:
1. Hierarchical risk allocation across strategies, markets, and assets
2. Dynamic risk budget adjustments based on performance
3. Risk utilization tracking
4. Visualization of risk allocation

This is a focused demo that only uses the risk budget manager component.
"""

import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import risk budget manager
from src.risk.risk_budget_manager import (
    RiskBudgetManager, RiskBudget, RiskBudgetLevel, RiskAllocationMethod
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("risk_budget_demo")

# Create output directory
output_dir = Path("examples/output/risk_budget")
output_dir.mkdir(parents=True, exist_ok=True)


def setup_risk_budget_manager():
    """
    Set up the risk budget manager with a hierarchical structure.
    
    Returns:
        RiskBudgetManager instance
    """
    logger.info("Setting up risk budget manager")
    
    # Create Risk Budget Manager with 5% total risk
    risk_budget_manager = RiskBudgetManager(total_risk_percent=5.0)
    
    # Create strategy-level budgets
    trend_following = risk_budget_manager.create_strategy_budget(
        strategy_name="trend_following",
        max_risk_percent=2.5,
        allocation_method=RiskAllocationMethod.PERFORMANCE_BASED
    )
    
    mean_reversion = risk_budget_manager.create_strategy_budget(
        strategy_name="mean_reversion",
        max_risk_percent=1.5,
        allocation_method=RiskAllocationMethod.PERFORMANCE_BASED
    )
    
    breakout = risk_budget_manager.create_strategy_budget(
        strategy_name="breakout",
        max_risk_percent=1.0,
        allocation_method=RiskAllocationMethod.EQUAL
    )
    
    # Create market-level budgets for Trend Following strategy
    tf_crypto = risk_budget_manager.create_market_budget(
        strategy_name="trend_following",
        market_name="crypto",
        max_risk_percent=1.5,
        allocation_method=RiskAllocationMethod.VOLATILITY_ADJUSTED
    )
    
    tf_forex = risk_budget_manager.create_market_budget(
        strategy_name="trend_following",
        market_name="forex",
        max_risk_percent=1.0,
        allocation_method=RiskAllocationMethod.EQUAL
    )
    
    # Create market-level budgets for Mean Reversion strategy
    mr_crypto = risk_budget_manager.create_market_budget(
        strategy_name="mean_reversion",
        market_name="crypto",
        max_risk_percent=1.0,
        allocation_method=RiskAllocationMethod.VOLATILITY_ADJUSTED
    )
    
    mr_commodities = risk_budget_manager.create_market_budget(
        strategy_name="mean_reversion",
        market_name="commodities",
        max_risk_percent=0.5,
        allocation_method=RiskAllocationMethod.EQUAL
    )
    
    # Create market-level budgets for Breakout strategy
    bo_crypto = risk_budget_manager.create_market_budget(
        strategy_name="breakout",
        market_name="crypto",
        max_risk_percent=1.0,
        allocation_method=RiskAllocationMethod.VOLATILITY_ADJUSTED
    )
    
    # Create asset-level budgets
    risk_budget_manager.create_asset_budget(
        strategy_name="trend_following",
        market_name="crypto",
        asset_name="BTC",
        max_risk_percent=0.6
    )
    
    risk_budget_manager.create_asset_budget(
        strategy_name="trend_following",
        market_name="crypto",
        asset_name="ETH",
        max_risk_percent=0.5
    )
    
    risk_budget_manager.create_asset_budget(
        strategy_name="mean_reversion",
        market_name="crypto",
        asset_name="BTC",
        max_risk_percent=0.4
    )
    
    risk_budget_manager.create_asset_budget(
        strategy_name="breakout",
        market_name="crypto",
        asset_name="BTC",
        max_risk_percent=0.5
    )
    
    logger.info("Risk budget hierarchy created")
    return risk_budget_manager


def update_risk_utilization(risk_budget_manager):
    """
    Simulate updating risk utilization for various strategies and assets.
    
    Args:
        risk_budget_manager: RiskBudgetManager instance
    """
    logger.info("Updating risk utilization")
    
    # Update risk utilization for some assets
    risk_budget_manager.update_risk_utilization(
        path=["trend_following", "crypto", "BTC"],
        risk_value=0.3
    )
    
    risk_budget_manager.update_risk_utilization(
        path=["trend_following", "crypto", "ETH"],
        risk_value=0.2
    )
    
    risk_budget_manager.update_risk_utilization(
        path=["mean_reversion", "crypto", "BTC"],
        risk_value=0.15
    )
    
    # Print current risk utilization
    risk_report = risk_budget_manager.risk_report()
    
    logger.info("Current risk utilization:")
    logger.info(f"Total risk: {risk_report['risk_utilization_percent']:.2f}%")
    
    for strategy_name, strategy_data in risk_report["strategies"].items():
        logger.info(f"Strategy '{strategy_name}':")
        logger.info(f"  Max risk: {strategy_data['max_risk']:.2f}%")
        logger.info(f"  Current risk: {strategy_data['current_risk']:.2f}%")
        logger.info(f"  Utilization: {strategy_data['risk_utilization_percent']:.2f}%")
        
        for market_name, market_data in strategy_data["markets"].items():
            logger.info(f"  Market '{market_name}':")
            logger.info(f"    Max risk: {market_data['max_risk']:.2f}%")
            logger.info(f"    Current risk: {market_data['current_risk']:.2f}%")
            logger.info(f"    Utilization: {market_data['risk_utilization_percent']:.2f}%")
            
            for asset_name, asset_data in market_data["assets"].items():
                logger.info(f"    Asset '{asset_name}':")
                logger.info(f"      Max risk: {asset_data['max_risk']:.2f}%")
                logger.info(f"      Current risk: {asset_data['current_risk']:.2f}%")
                logger.info(f"      Utilization: {asset_data['risk_utilization_percent']:.2f}%")


def optimize_risk_allocation(risk_budget_manager):
    """
    Demonstrate risk optimization based on performance metrics.
    
    Args:
        risk_budget_manager: RiskBudgetManager instance
    """
    logger.info("Optimizing risk allocation based on performance")
    
    # Simulate performance metrics (e.g., Sharpe ratios)
    performance_data = {
        "trend_following": 1.8,
        "mean_reversion": 2.2,
        "breakout": 1.5,
        "trend_following/crypto": 1.9,
        "trend_following/forex": 1.6,
        "mean_reversion/crypto": 2.3,
        "mean_reversion/commodities": 1.9,
        "breakout/crypto": 1.5
    }
    
    # Optimize allocations based on performance
    risk_budget_manager.optimize_allocations(performance_data)
    
    # Print updated allocations
    risk_report = risk_budget_manager.risk_report()
    
    logger.info("Optimized risk allocation:")
    for strategy_name, strategy_data in risk_report["strategies"].items():
        logger.info(f"Strategy '{strategy_name}':")
        logger.info(f"  Max risk: {strategy_data['max_risk']:.2f}%")
        
        for market_name, market_data in strategy_data["markets"].items():
            logger.info(f"  Market '{market_name}':")
            logger.info(f"    Max risk: {market_data['max_risk']:.2f}%")


def visualize_risk_allocation(risk_budget_manager):
    """
    Create visualizations of risk allocation.
    
    Args:
        risk_budget_manager: RiskBudgetManager instance
    """
    logger.info("Creating risk allocation visualizations")
    
    # Generate treemap visualization
    treemap_path = output_dir / "risk_allocation_treemap.png"
    risk_budget_manager.visualize_risk_allocation(save_path=str(treemap_path))
    
    # Get risk report
    risk_report = risk_budget_manager.risk_report()
    
    # Create bar chart of strategy risk utilization
    plt.figure(figsize=(12, 6))
    strategies = list(risk_report["strategies"].keys())
    utilization = [risk_report["strategies"][s]["risk_utilization_percent"] for s in strategies]
    max_risk = [risk_report["strategies"][s]["max_risk"] for s in strategies]
    current_risk = [risk_report["strategies"][s]["current_risk"] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(x - width/2, max_risk, width, label='Maximum Risk', color='#3498db', alpha=0.7)
    plt.bar(x + width/2, current_risk, width, label='Current Risk', color='#e74c3c', alpha=0.7)
    
    plt.xlabel('Strategy')
    plt.ylabel('Risk (%)')
    plt.title('Risk Allocation by Strategy')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(utilization):
        plt.annotate(f"{v:.1f}%", xy=(i, current_risk[i] + 0.1), 
                    ha='center', va='bottom', fontsize=9, color='#555')
    
    bar_chart_path = output_dir / "strategy_risk_allocation.png"
    plt.savefig(bar_chart_path)
    plt.close()
    
    logger.info(f"Risk visualizations saved to {output_dir}")


def run_demo():
    """Main demo function to demonstrate risk budget management."""
    logger.info("Starting Risk Budget Management Demo")
    
    try:
        # Setup risk budget manager
        logger.info("Setting up risk budget manager")
        risk_budget_manager = setup_risk_budget_manager()
        print("Risk budget manager created successfully")
        
        # Update risk utilization
        logger.info("Updating risk utilization")
        update_risk_utilization(risk_budget_manager)
        print("Risk utilization updated successfully")
        
        # Optimize risk allocation
        logger.info("Optimizing risk allocation")
        optimize_risk_allocation(risk_budget_manager)
        print("Risk allocation optimized successfully")
        
        # Create visualizations
        logger.info("Creating visualizations")
        visualize_risk_allocation(risk_budget_manager)
        print("Visualizations created successfully")
        
        # Generate final report
        logger.info("Generating final report")
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "risk_budget": risk_budget_manager.risk_report()
        }
        
        # Save report to file
        report_path = output_dir / "risk_budget_report.json"
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Risk budget report saved to {report_path}")
        logger.info("Risk Budget Management Demo completed")
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    run_demo()
