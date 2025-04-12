"""
Agent Architecture Example

This script demonstrates how to use the modular agent architecture to create and run a backtest.
It shows how the different components (DataManager, Strategy, PortfolioManager, RiskManager, 
ExecutionHandler, and Orchestrator) work together.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the agent factory
from ai_trading_agent.agent.factory import (
    create_agent_from_config,
    is_rust_available
)

# Import the trading engine models
from ai_trading_agent.trading_engine.models import (
    Order,
    OrderSide,
    OrderType,
    Portfolio
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration for the agent.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Define the test period
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # Define the symbols to trade
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Create the configuration
    config = {
        "data_manager": {
            "type": "SimpleDataManager",
            "config": {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": "1d",
                "data_dir": "data/",  # Add data directory
                "use_mock_data": True,  # Use mock data for this example
                "mock_data_params": {
                    "volatility": 0.02,
                    "drift": 0.001,
                    "gap_probability": 0.05,
                    "random_seed": 42
                }
            }
        },
        "strategy": {
            "type": "SentimentStrategy",
            "config": {
                "sentiment_threshold_buy": 0.3,
                "sentiment_threshold_sell": -0.3,
                "use_mock_sentiment": True,  # Use mock sentiment for this example
                "mock_sentiment_params": {
                    "mean": 0.1,
                    "std": 0.5,
                    "random_seed": 42
                }
            }
        },
        "portfolio_manager": {
            "type": "PortfolioManager",
            "config": {
                "initial_cash": 100000.0,
                "position_size_pct": 0.1,  # Use 10% of portfolio per position
                "max_positions": 5,
                "fractional_shares": True
            }
        },
        "risk_manager": {
            "type": "SimpleRiskManager",
            "config": {
                "max_drawdown_pct": 0.1,  # 10% max drawdown
                "max_position_pct": 0.2,   # 20% max position size
                "stop_loss_pct": 0.05,     # 5% stop loss
                "take_profit_pct": 0.1     # 10% take profit
            }
        },
        "execution_handler": {
            "type": "SimulatedExecutionHandler",
            "config": {
                "slippage_model": "normal",
                "slippage_std": 0.001,     # 0.1% standard deviation for slippage
                "commission_model": "percentage",
                "commission_pct": 0.001    # 0.1% commission
            }
        },
        "backtest": {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "timeframe": "1d",
            "initial_cash": 100000.0,
            "verbose": True
        }
    }
    
    return config


def run_example_backtest():
    """
    Run an example backtest using the agent architecture.
    """
    logger.info("Creating agent configuration...")
    config = create_example_config()
    
    # Check if Rust components are available
    use_rust = is_rust_available()
    if use_rust:
        logger.info("Using Rust-accelerated components")
    else:
        logger.info("Using Python components (Rust not available)")
    
    # Create the agent (orchestrator with all components)
    logger.info("Creating agent from configuration...")
    agent = create_agent_from_config(config, use_rust=use_rust)
    
    # Run the backtest
    logger.info("Running backtest...")
    results = agent.run()
    
    # Display results
    if results:
        logger.info("Backtest completed successfully")
        
        # Display performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info("Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Display final portfolio state
        if 'final_state' in results:
            final_state = results['final_state']
            logger.info("Final Portfolio State:")
            logger.info(f"  Cash: ${final_state.get('cash', 0):.2f}")
            
            positions = final_state.get('positions', {})
            if positions:
                logger.info("  Positions:")
                for symbol, position in positions.items():
                    qty = position.get('quantity', 0)
                    avg_price = position.get('average_price', 0)
                    current_price = position.get('current_price', 0)
                    market_value = qty * current_price
                    pnl = (current_price - avg_price) * qty
                    pnl_pct = (current_price / avg_price - 1) * 100 if avg_price > 0 else 0
                    
                    logger.info(f"    {symbol}: {qty} shares @ ${avg_price:.2f} (Current: ${current_price:.2f}, "
                               f"Value: ${market_value:.2f}, P&L: ${pnl:.2f} / {pnl_pct:.2f}%)")
            else:
                logger.info("  No positions")
        
        # Display trade statistics
        if 'trade_stats' in results:
            trade_stats = results['trade_stats']
            logger.info("Trade Statistics:")
            for key, value in trade_stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    else:
        logger.error("Backtest failed to complete")


if __name__ == "__main__":
    run_example_backtest()
