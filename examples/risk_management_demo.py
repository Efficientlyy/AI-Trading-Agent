#!/usr/bin/env python
"""
Risk Management Demo for the AI Trading System.

This script demonstrates the use of the Position Risk Analyzer to evaluate
risk metrics for individual positions and portfolios, perform stress testing,
and visualize risk metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
from pathlib import Path
import json
import random

# Add the parent directory to sys.path to allow importing from src
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import risk management components
from src.risk.position_risk_analyzer import (
    PositionRiskAnalyzer, Position, RiskMetrics, 
    VaRMethod, ConfidenceLevel, create_example_stress_scenarios
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("risk_management_demo")

# Create data directory if it doesn't exist
data_dir = current_dir / "data" / "risk"
data_dir.mkdir(parents=True, exist_ok=True)


def generate_sample_price_data(mean_return, volatility, days=252, seed=42):
    """Generate sample price data for demonstration."""
    np.random.seed(seed)
    
    # Generate daily returns from a normal distribution
    daily_returns = np.random.normal(mean_return / 252, volatility / np.sqrt(252), days)
    
    # Convert returns to price series starting from 100
    price_series = 100 * np.cumprod(1 + daily_returns)
    
    return price_series


def create_sample_positions():
    """Create sample positions for demonstration."""
    # Current timestamp
    now = datetime.datetime.now()
    
    positions = [
        Position(
            symbol="BTC/USD",
            quantity=2.5,
            entry_price=35000.0,
            current_price=36500.0,
            exchange="Binance",
            timestamp=now,
            metadata={"strategy": "trend_following", "timeframe": "1h"}
        ),
        Position(
            symbol="ETH/USD",
            quantity=15.0,
            entry_price=2300.0,
            current_price=2250.0,
            exchange="Coinbase",
            timestamp=now,
            metadata={"strategy": "mean_reversion", "timeframe": "4h"}
        ),
        Position(
            symbol="SOL/USD",
            quantity=150.0,
            entry_price=85.0,
            current_price=92.0,
            exchange="Kraken",
            timestamp=now,
            metadata={"strategy": "breakout", "timeframe": "1d"}
        ),
        Position(
            symbol="BNB/USD",
            quantity=25.0,
            entry_price=320.0,
            current_price=315.0,
            exchange="Binance",
            timestamp=now,
            metadata={"strategy": "oscillator", "timeframe": "1h"}
        ),
    ]
    
    logger.info(f"Created {len(positions)} sample positions")
    return positions


def generate_historical_data(positions):
    """Generate historical price data for the sample positions."""
    historical_data = {}
    
    # Generate different price paths for each symbol
    historical_data["BTC/USD"] = generate_sample_price_data(
        mean_return=0.8, volatility=0.65, days=365, seed=42
    ) * (36500 / 100)  # Scale to current price level
    
    historical_data["ETH/USD"] = generate_sample_price_data(
        mean_return=1.0, volatility=0.85, days=365, seed=43
    ) * (2250 / 100)
    
    historical_data["SOL/USD"] = generate_sample_price_data(
        mean_return=1.2, volatility=1.2, days=365, seed=44
    ) * (92 / 100)
    
    historical_data["BNB/USD"] = generate_sample_price_data(
        mean_return=0.7, volatility=0.75, days=365, seed=45
    ) * (315 / 100)
    
    logger.info(f"Generated historical data for {len(historical_data)} symbols")
    
    return historical_data


def analyze_single_position_risk(risk_analyzer, position, historical_data):
    """Analyze risk for a single position and display results."""
    logger.info(f"\n==== Risk Analysis for {position.symbol} ====")
    logger.info(f"Position: {position.quantity} @ ${position.current_price:.2f} = ${position.position_value:.2f}")
    
    # Analyze position risk using different VaR methods
    for method in [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]:
        logger.info(f"\nVaR Method: {method.value}")
        
        risk_metrics = risk_analyzer.analyze_position_risk(
            position=position,
            historical_prices=historical_data[position.symbol],
            var_method=method
        )
        
        # Display risk metrics
        logger.info(f"1-Day VaR (95%): ${risk_metrics.var_1d_95:.2f} ({risk_metrics.var_1d_95 / position.position_value:.2%})")
        logger.info(f"1-Day VaR (99%): ${risk_metrics.var_1d_99:.2f} ({risk_metrics.var_1d_99 / position.position_value:.2%})")
        logger.info(f"10-Day VaR (99%): ${risk_metrics.var_10d_99:.2f} ({risk_metrics.var_10d_99 / position.position_value:.2%})")
        logger.info(f"Expected Shortfall: ${risk_metrics.expected_shortfall:.2f} ({risk_metrics.expected_shortfall / position.position_value:.2%})")
        logger.info(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        logger.info(f"Annualized Volatility: {risk_metrics.volatility:.2%}")
        
        # Create visualization for historical method
        if method == VaRMethod.HISTORICAL:
            # Calculate returns
            prices = historical_data[position.symbol]
            returns = np.diff(prices) / prices[:-1]
            
            # Visualize VaR
            save_path = data_dir / f"{position.symbol.replace('/', '_')}_var.png"
            risk_analyzer.visualize_var(
                position=position,
                returns=returns,
                method=method,
                save_path=str(save_path)
            )
            logger.info(f"Saved VaR visualization to {save_path}")
    
    return risk_metrics


def analyze_portfolio_risk(risk_analyzer, positions, historical_data):
    """Analyze risk for a portfolio of positions and display results."""
    logger.info("\n==== Portfolio Risk Analysis ====")
    
    # Calculate total portfolio value
    portfolio_value = sum(pos.position_value for pos in positions)
    logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
    
    # Position information
    for pos in positions:
        logger.info(f"{pos.symbol}: {pos.quantity} @ ${pos.current_price:.2f} = ${pos.position_value:.2f} ({pos.position_value / portfolio_value:.2%})")
    
    # Analyze portfolio risk using different VaR methods
    for method in [VaRMethod.PARAMETRIC, VaRMethod.HISTORICAL, VaRMethod.MONTE_CARLO]:
        logger.info(f"\nVaR Method: {method.value}")
        
        risk_metrics = risk_analyzer.analyze_portfolio_risk(
            positions=positions,
            historical_data=historical_data,
            var_method=method
        )
        
        # Display risk metrics
        logger.info(f"1-Day VaR (95%): ${risk_metrics.var_1d_95:.2f} ({risk_metrics.var_1d_95 / portfolio_value:.2%})")
        logger.info(f"1-Day VaR (99%): ${risk_metrics.var_1d_99:.2f} ({risk_metrics.var_1d_99 / portfolio_value:.2%})")
        logger.info(f"10-Day VaR (99%): ${risk_metrics.var_10d_99:.2f} ({risk_metrics.var_10d_99 / portfolio_value:.2%})")
        logger.info(f"Expected Shortfall: ${risk_metrics.expected_shortfall:.2f} ({risk_metrics.expected_shortfall / portfolio_value:.2%})")
        logger.info(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        logger.info(f"Annualized Volatility: {risk_metrics.volatility:.2%}")
        
        # Visualize correlation matrix
        if method == VaRMethod.PARAMETRIC:
            save_path = data_dir / "correlation_heatmap.png"
            risk_analyzer.visualize_correlation_heatmap(
                correlation_matrix=risk_metrics.correlation_matrix,
                save_path=str(save_path)
            )
            logger.info(f"Saved correlation heatmap to {save_path}")
    
    return risk_metrics


def perform_stress_testing(risk_analyzer, positions):
    """Perform stress testing on the portfolio and display results."""
    logger.info("\n==== Portfolio Stress Testing ====")
    
    # Get stress scenarios
    scenarios = create_example_stress_scenarios()
    logger.info(f"Testing {len(scenarios)} stress scenarios")
    
    # Run stress tests
    stress_test_results = risk_analyzer.perform_stress_test(positions, scenarios)
    
    # Display results
    for scenario, impact in stress_test_results.items():
        # A negative result means a gain, positive means a loss
        if impact < 0:
            logger.info(f"{scenario}: Gain of {abs(impact):.2f}%")
        else:
            logger.info(f"{scenario}: Loss of {impact:.2f}%")
    
    # Visualize stress test results
    plt.figure(figsize=(12, 8))
    
    # Sort scenarios by impact
    sorted_scenarios = sorted(stress_test_results.items(), key=lambda x: x[1])
    scenarios = [item[0] for item in sorted_scenarios]
    impacts = [item[1] for item in sorted_scenarios]
    
    # Plot horizontal bar chart
    colors = ['green' if impact < 0 else 'red' for impact in impacts]
    bars = plt.barh(scenarios, impacts, color=colors)
    
    # Add labels
    plt.xlabel('Portfolio Impact (%)')
    plt.title('Stress Test Scenarios Impact on Portfolio')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        label = f"{abs(width):.1f}%"
        if width < 0:
            plt.text(width - 1.0, bar.get_y() + bar.get_height()/2, label,
                   ha='right', va='center')
        else:
            plt.text(width + 1.0, bar.get_y() + bar.get_height()/2, label,
                   ha='left', va='center')
    
    # Save the figure
    save_path = data_dir / "stress_test_results.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved stress test visualization to {save_path}")
    
    return stress_test_results


def main():
    """Run the risk management demo."""
    logger.info("Starting Risk Management Demo")
    
    # Create risk analyzer
    risk_analyzer = PositionRiskAnalyzer(risk_free_rate=0.04)  # 4% risk-free rate
    
    # Create sample positions
    positions = create_sample_positions()
    
    # Generate historical data
    historical_data = generate_historical_data(positions)
    
    # Analyze individual position risk
    for position in positions:
        analyze_single_position_risk(risk_analyzer, position, historical_data)
    
    # Analyze portfolio risk
    portfolio_risk = analyze_portfolio_risk(risk_analyzer, positions, historical_data)
    
    # Perform stress testing
    stress_results = perform_stress_testing(risk_analyzer, positions)
    
    # Save portfolio risk metrics to file
    risk_metrics_dict = portfolio_risk.to_dict()
    risk_metrics_dict["stress_test_results"] = stress_results
    
    with open(data_dir / "portfolio_risk_metrics.json", "w") as f:
        json.dump(risk_metrics_dict, f, indent=2, default=str)
    
    logger.info("Risk Management Demo completed successfully.")
    logger.info(f"Results saved to {data_dir}")


if __name__ == "__main__":
    main() 