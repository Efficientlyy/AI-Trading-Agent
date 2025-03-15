#!/usr/bin/env python3
"""
Simple Risk Management Demo

This script demonstrates the core risk management components:
1. Risk Budget Management
2. Position Risk Analysis
3. Dynamic Risk Limits

This is a simplified version that doesn't rely on the Config class.
"""

import sys
import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
from decimal import Decimal

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import risk management components
from src.risk.risk_budget_manager import (
    RiskBudgetManager, RiskBudget, RiskBudgetLevel, RiskAllocationMethod
)
from src.risk.position_risk_analyzer import (
    PositionRiskAnalyzer, Position as RiskPosition, VaRMethod, ConfidenceLevel
)
from src.risk.dynamic_risk_limits import DynamicRiskLimits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_risk_demo")

# Create output directory
output_dir = Path("examples/output/simple_risk")
output_dir.mkdir(parents=True, exist_ok=True)


def setup_risk_management_system():
    """
    Set up core risk management components.
    
    Returns:
        Tuple containing risk management components
    """
    logger.info("Setting up risk management system")
    
    # 1. Create Risk Budget Manager
    risk_budget_manager = RiskBudgetManager(total_risk_percent=5.0)
    
    # 2. Create Position Risk Analyzer
    position_risk_analyzer = PositionRiskAnalyzer()
    
    # 3. Create Dynamic Risk Limits
    dynamic_risk_limits = DynamicRiskLimits()
    
    # Set up risk budget hierarchy
    
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
    
    # Add dynamic risk limits for BTC
    dynamic_risk_limits.add_volatility_limit(
        symbol="BTC",
        threshold=0.05,
        lookback_period=30,
        volatility_scale=1.2
    )
    
    dynamic_risk_limits.add_drawdown_limit(
        symbol="BTC",
        max_drawdown=0.15,
        recovery_threshold=0.5
    )
    
    logger.info("Risk management system setup complete")
    return (risk_budget_manager, position_risk_analyzer, dynamic_risk_limits)


def generate_sample_price_data(symbols, days=60, seed=42):
    """
    Generate sample price data for demonstration.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days of data
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataFrames with OHLCV data by symbol
    """
    np.random.seed(seed)
    
    # Symbol-specific parameters for more realistic price movements
    symbol_params = {
        "BTC": {"volatility": 0.03, "trend": 0.0005, "start_price": 35000.0},
        "ETH": {"volatility": 0.035, "trend": 0.0003, "start_price": 2200.0},
    }
    
    # Default parameters for symbols not in the list
    default_params = {"volatility": 0.025, "trend": 0.0, "start_price": 100.0}
    
    price_data = {}
    
    for symbol in symbols:
        # Get parameters for this symbol
        params = symbol_params.get(symbol, default_params)
        
        # Generate timestamps (hourly data)
        now = datetime.now()
        hours = days * 24
        timestamps = [now - timedelta(hours=i) for i in range(hours)]
        timestamps.reverse()  # Oldest first
        
        # Generate returns with random noise and trend
        returns = np.random.normal(params["trend"], params["volatility"], hours)
        
        # Calculate prices from returns
        prices = [params["start_price"]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove the seed price
        
        # Create OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            high = price * (1 + random.uniform(0, params["volatility"]/2))
            low = price * (1 - random.uniform(0, params["volatility"]/2))
            open_price = price * (1 + random.uniform(-params["volatility"]/4, params["volatility"]/4))
            volume = random.uniform(10000, 100000)
            
            data.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume
            })
        
        # Create DataFrame
        price_data[symbol] = pd.DataFrame(data)
    
    return price_data


def analyze_position_risks(price_data, position_risk_analyzer):
    """
    Analyze risk metrics for sample positions.
    
    Args:
        price_data: Dictionary of price DataFrames by symbol
        position_risk_analyzer: PositionRiskAnalyzer instance
        
    Returns:
        Dictionary of risk metrics by position
    """
    logger.info("Analyzing position risks")
    
    # Create sample positions
    positions = [
        RiskPosition(
            symbol="BTC",
            quantity=2.5,
            entry_price=35000.0,
            current_price=price_data["BTC"].iloc[-1]["close"],
            exchange="Binance",
            timestamp=datetime.now()
        ),
        RiskPosition(
            symbol="ETH",
            quantity=15.0,
            entry_price=2200.0,
            current_price=price_data["ETH"].iloc[-1]["close"],
            exchange="Coinbase",
            timestamp=datetime.now()
        )
    ]
    
    # Analyze each position
    risk_metrics = {}
    
    for position in positions:
        # Extract historical price data for this symbol
        symbol_data = price_data[position.symbol]
        historical_prices = symbol_data["close"].values
        
        # Analyze risk
        metrics = position_risk_analyzer.analyze_position_risk(
            position=position,
            historical_prices=historical_prices,
            confidence_level=0.95,
            var_method=VaRMethod.HISTORICAL
        )
        
        # Store metrics
        risk_metrics[position.symbol] = metrics
        
        # Print a summary
        logger.info(f"Risk metrics for {position.symbol} position:")
        logger.info(f"  Value at Risk (95%): ${metrics.var_1d_95:.2f}")
        logger.info(f"  Expected Shortfall: ${metrics.expected_shortfall:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        logger.info(f"  Volatility: {metrics.volatility*100:.2f}%")
    
    # Create portfolio for combined analysis
    portfolio_data = {pos.symbol: price_data[pos.symbol]["close"].values for pos in positions}
    portfolio_metrics = position_risk_analyzer.analyze_portfolio_risk(
        positions=positions,
        historical_data=portfolio_data,
        confidence_level=0.95
    )
    
    # Print portfolio metrics
    logger.info(f"Portfolio risk metrics:")
    logger.info(f"  Portfolio VaR (95%): ${portfolio_metrics.var_1d_95:.2f}")
    logger.info(f"  Portfolio Expected Shortfall: ${portfolio_metrics.expected_shortfall:.2f}")
    logger.info(f"  Portfolio Volatility: {portfolio_metrics.volatility*100:.2f}%")
    
    # Visualize VaR for BTC position
    var_chart_path = output_dir / "btc_var_distribution.png"
    position_risk_analyzer.visualize_var(
        position=positions[0],
        returns=np.diff(np.log(historical_prices)),
        confidence_level=0.95,
        save_path=str(var_chart_path)
    )
    
    return risk_metrics


def test_dynamic_risk_limits(price_data, dynamic_risk_limits):
    """
    Test the dynamic risk limits functionality.
    
    Args:
        price_data: Dictionary of price DataFrames by symbol
        dynamic_risk_limits: DynamicRiskLimits instance
    """
    logger.info("Testing dynamic risk limits")
    
    # Update market data for the dynamic risk limits
    for symbol, data in price_data.items():
        dynamic_risk_limits.update_market_data(symbol, data)
    
    # Test position size calculation for BTC
    btc_multiplier = dynamic_risk_limits.get_position_size_multiplier("BTC")
    logger.info(f"Position size multiplier for BTC: {btc_multiplier:.2f}")
    
    # Test drawdown protection
    # Simulate a portfolio drawdown
    dynamic_risk_limits.update_portfolio_value(1000000.0)  # Initial value
    dynamic_risk_limits.update_portfolio_value(850000.0)   # 15% drawdown
    
    breached, current_drawdown = dynamic_risk_limits.check_drawdown_limits()
    logger.info(f"Current drawdown: {current_drawdown*100:.2f}%")
    
    if breached:
        logger.info("Drawdown limit breached - risk reduction required")
    else:
        logger.info("Drawdown limit not breached")
    
    # Get overall risk summary
    risk_summary = dynamic_risk_limits.get_risk_summary()
    logger.info("Dynamic risk limits summary:")
    
    # Print portfolio information
    logger.info(f"  Portfolio current value: ${risk_summary['portfolio']['current_value']:.2f}")
    logger.info(f"  Portfolio peak value: ${risk_summary['portfolio']['peak_value']:.2f}")
    logger.info(f"  Portfolio drawdown: {risk_summary['portfolio']['drawdown']*100:.2f}%")
    
    # Print symbol-specific information
    for symbol, symbol_data in risk_summary['symbols'].items():
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"    Volatility: {symbol_data['volatility']*100:.2f}%")
        logger.info(f"    Position size multiplier: {symbol_data['position_size_multiplier']:.2f}")
        
        for limit in symbol_data['limits']:
            logger.info(f"    {limit['type']} limit: {limit['status']}")


def visualize_risk_budget_utilization(risk_budget_manager):
    """
    Create visualizations of risk budget utilization.
    
    Args:
        risk_budget_manager: RiskBudgetManager instance
    """
    logger.info("Creating risk budget visualizations")
    
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
    """Main demo function to run all risk management components."""
    logger.info("Starting Simple Risk Management Demo")
    
    # Setup all risk management components
    risk_budget_manager, position_risk_analyzer, dynamic_risk_limits = setup_risk_management_system()
    
    # Generate sample price data
    symbols = ["BTC", "ETH"]
    price_data = generate_sample_price_data(symbols)
    
    # Analyze position risks
    risk_metrics = analyze_position_risks(price_data, position_risk_analyzer)
    
    # Test dynamic risk limits
    test_dynamic_risk_limits(price_data, dynamic_risk_limits)
    
    # Create risk budget visualizations
    visualize_risk_budget_utilization(risk_budget_manager)
    
    # Perform risk optimization based on performance
    # In a real system, this would use actual strategy performance metrics
    performance_data = {
        "trend_following": 1.8,  # Sharpe ratio
        "mean_reversion": 2.2,   # Sharpe ratio
        "breakout": 1.5,         # Sharpe ratio
    }
    
    risk_budget_manager.optimize_allocations(performance_data)
    
    # Generate final report
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "risk_budget": risk_budget_manager.risk_report(),
        "position_risk": {
            symbol: {
                "var_95": metrics.var_1d_95,
                "expected_shortfall": metrics.expected_shortfall,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility
            } for symbol, metrics in risk_metrics.items()
        },
        "dynamic_limits": dynamic_risk_limits.get_risk_summary()
    }
    
    # Save report to file
    report_path = output_dir / "simple_risk_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Risk report saved to {report_path}")
    logger.info("Simple Risk Management Demo completed")


if __name__ == "__main__":
    # Run the demo
    run_demo() 