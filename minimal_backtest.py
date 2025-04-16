#!/usr/bin/env python
# minimal_backtest.py

"""
A minimal working backtest script to test the core functionality.
This version uses the factory system to create agent components.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("minimal_backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Added project root to sys.path: {project_root}")

# Import required modules
try:
    # Import trading engine components
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
    logger.info("Successfully imported enums")
    
    from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
    logger.info("Successfully imported models")
    
    # Import backtesting components
    from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
    logger.info("Successfully imported performance_metrics")
    
    # Import agent components
    from ai_trading_agent.agent.data_manager import SimpleDataManager
    logger.info("Successfully imported data_manager")
    
    from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
    logger.info("Successfully imported strategy")
    
    from ai_trading_agent.agent.risk_manager import SimpleRiskManager
    logger.info("Successfully imported risk_manager")
    
    from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
    logger.info("Successfully imported execution_handler")
    
    from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
    logger.info("Successfully imported orchestrator")
    
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    logger.info("Successfully imported portfolio_manager")
    
    # Import the new factory system
    from ai_trading_agent.agent.factory import (
        create_data_manager,
        create_strategy,
        create_strategy_manager,
        create_risk_manager,
        create_portfolio_manager,
        create_execution_handler,
        create_orchestrator,
        create_agent_from_config
    )
    logger.info("Successfully imported factory system")
    
    # Import config validation
    from ai_trading_agent.common.config_validator import validate_agent_config, check_config_compatibility
    logger.info("Successfully imported config validator")
    
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def generate_sample_data(symbols, start_date, end_date):
    """
    Generate sample price data for backtesting.
    """
    logger.info(f"Generating sample data for {len(symbols)} symbols")
    
    data = {}
    for symbol in symbols:
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Generate price data (simple random walk)
        close = 100 + np.random.normal(0, 1, n).cumsum()
        close = np.maximum(close, 1)  # Ensure prices are positive
        
        # Generate OHLCV data
        high = close * (1 + np.random.uniform(0, 0.02, n))
        low = close * (1 - np.random.uniform(0, 0.02, n))
        open_price = low + np.random.uniform(0, 1, n) * (high - low)
        volume = np.random.uniform(100000, 1000000, n)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        data[symbol] = df
    
    logger.info(f"Generated sample data with {len(data[symbols[0]])} bars per symbol")
    return data

def generate_sample_sentiment(symbols, start_date, end_date):
    """
    Generate sample sentiment data for backtesting.
    """
    logger.info(f"Generating sample sentiment data for {len(symbols)} symbols")
    
    sentiment_data = {}
    for symbol in symbols:
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Generate sentiment scores (random values between -1 and 1)
        sentiment = np.random.normal(0, 0.5, n)
        sentiment = np.clip(sentiment, -1, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'sentiment': sentiment
        }, index=dates)
        
        sentiment_data[symbol] = df
    
    logger.info(f"Generated sample sentiment data with {len(sentiment_data[symbols[0]])} bars per symbol")
    return sentiment_data

class MinimalDataManager(SimpleDataManager):
    """
    A custom data manager that works with in-memory data for the minimal backtest.
    """
    
    def __init__(self, price_data, sentiment_data, symbols, start_date, end_date):
        """
        Initialize with in-memory data.
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            sentiment_data: Dictionary of sentiment DataFrames by symbol
            symbols: List of symbols
            start_date: Start date
            end_date: End date
        """
        # Create a temporary directory for data
        self.temp_dir = os.path.join(project_root, 'temp_data')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Initialize with basic config
        config = {
            'data_dir': self.temp_dir,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': '1d',
            'data_types': ['ohlcv', 'sentiment']
        }
        
        # Initialize parent class
        super().__init__(config)
        
        # Override data with in-memory data
        self.data = price_data
        
        # Create combined sentiment data
        self.sentiment_data = pd.DataFrame()
        for symbol, df in sentiment_data.items():
            self.sentiment_data[f"{symbol}_sentiment_score"] = df['sentiment']
        
        # Prepare combined index from all data
        self._prepare_combined_index()
        
        logger.info(f"MinimalDataManager initialized with {len(symbols)} symbols")

def run_backtest_with_factory():
    """
    Run a simple backtest using the factory system.
    """
    logger.info("Starting minimal backtest with factory system")
    
    # Define backtest parameters
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    symbols = ["AAPL", "GOOG", "MSFT"]
    initial_capital = 100000.0
    
    # Generate sample data
    price_data = generate_sample_data(symbols, start_date, end_date)
    sentiment_data = generate_sample_sentiment(symbols, start_date, end_date)
    
    # Create agent configuration
    agent_config = {
        "data_manager": {
            "type": "SimpleDataManager",
            "config": {
                "data_dir": os.path.join(project_root, 'temp_data'),
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": "1d",
                "data_types": ["ohlcv", "sentiment"]
            }
        },
        "strategy": {
            "type": "SentimentStrategy",
            "config": {
                "name": "SentimentStrategy",
                "symbols": symbols,
                "sentiment_threshold": 0.3,
                "position_size_pct": 0.1
            }
        },
        "risk_manager": {
            "type": "SimpleRiskManager",
            "config": {
                "max_position_size": None,
                "max_portfolio_risk_pct": 0.05,
                "stop_loss_pct": 0.05
            }
        },
        "portfolio_manager": {
            "type": "PortfolioManager",
            "config": {
                "initial_capital": initial_capital,
                "risk_per_trade": 0.02,
                "max_position_size": 0.2
            }
        },
        "execution_handler": {
            "type": "SimulatedExecutionHandler",
            "config": {
                "commission_rate": 0.001,
                "slippage_pct": 0.001
            }
        },
        "backtest": {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "initial_capital": initial_capital
        }
    }
    
    # Validate the configuration
    is_valid, error_message = validate_agent_config(agent_config)
    if not is_valid:
        logger.error(f"Invalid agent configuration: {error_message}")
        return None
    
    # Check for compatibility issues
    warnings = check_config_compatibility(agent_config)
    for warning in warnings:
        logger.warning(warning)
    
    # Create a custom data manager with in-memory data
    # (We'll create this separately since it needs our generated data)
    data_manager = MinimalDataManager(
        price_data=price_data,
        sentiment_data=sentiment_data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Create the rest of the components using the factory system
    strategy = create_strategy(agent_config["strategy"])
    strategy_manager = create_strategy_manager(strategy)
    risk_manager = create_risk_manager(agent_config["risk_manager"])
    portfolio_manager = create_portfolio_manager(agent_config["portfolio_manager"])
    execution_handler = create_execution_handler(
        agent_config["execution_handler"],
        portfolio_manager
    )
    
    # Create the orchestrator
    orchestrator = create_orchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=agent_config["backtest"]
    )
    
    # Run backtest using the orchestrator
    logger.info("Running backtest...")
    results = orchestrator.run()
    
    # Print results
    if results:
        logger.info(f"Backtest completed successfully")
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info(f"Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    else:
        logger.warning("Backtest did not return results")
    
    return results

def run_backtest_with_full_factory():
    """
    Run a simple backtest using the complete factory system.
    This demonstrates how to create an agent from a configuration file.
    """
    logger.info("Starting minimal backtest with full factory system")
    
    # Define backtest parameters
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    symbols = ["AAPL", "GOOG", "MSFT"]
    initial_capital = 100000.0
    
    # Generate sample data (we'll still need this for our custom data manager)
    price_data = generate_sample_data(symbols, start_date, end_date)
    sentiment_data = generate_sample_sentiment(symbols, start_date, end_date)
    
    # Create agent configuration
    agent_config = {
        "data_manager": {
            "type": "SimpleDataManager",
            "config": {
                "data_dir": os.path.join(project_root, 'temp_data'),
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": "1d",
                "data_types": ["ohlcv", "sentiment"]
            }
        },
        "strategy": {
            "type": "SentimentStrategy",
            "config": {
                "name": "SentimentStrategy",
                "symbols": symbols,
                "sentiment_threshold": 0.3,
                "position_size_pct": 0.1
            }
        },
        "risk_manager": {
            "type": "SimpleRiskManager",
            "config": {
                "max_position_size": None,
                "max_portfolio_risk_pct": 0.05,
                "stop_loss_pct": 0.05
            }
        },
        "portfolio_manager": {
            "type": "PortfolioManager",
            "config": {
                "initial_capital": initial_capital,
                "risk_per_trade": 0.02,
                "max_position_size": 0.2
            }
        },
        "execution_handler": {
            "type": "SimulatedExecutionHandler",
            "config": {
                "commission_rate": 0.001,
                "slippage_pct": 0.001
            }
        },
        "backtest": {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "initial_capital": initial_capital
        }
    }
    
    # Save the configuration to a temporary file
    config_path = os.path.join(project_root, 'temp_data', 'agent_config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(agent_config, f, default_flow_style=False)
    
    logger.info(f"Saved agent configuration to {config_path}")
    
    # For this example, we'll still use our custom data manager
    # In a real scenario, you'd register the MinimalDataManager in the factory
    
    # Create a custom data manager with in-memory data
    data_manager = MinimalDataManager(
        price_data=price_data,
        sentiment_data=sentiment_data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Create the agent using the factory system
    # In a real scenario, you'd use create_agent_from_config(agent_config)
    # But for this example, we'll create each component separately
    
    strategy = create_strategy(agent_config["strategy"])
    strategy_manager = create_strategy_manager(strategy)
    risk_manager = create_risk_manager(agent_config["risk_manager"])
    portfolio_manager = create_portfolio_manager(agent_config["portfolio_manager"])
    execution_handler = create_execution_handler(
        agent_config["execution_handler"],
        portfolio_manager
    )
    
    # Create the orchestrator
    orchestrator = create_orchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=agent_config["backtest"]
    )
    
    # Run backtest using the orchestrator
    logger.info("Running backtest...")
    results = orchestrator.run()
    
    # Print results
    if results:
        logger.info(f"Backtest completed successfully")
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info(f"Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    else:
        logger.warning("Backtest did not return results")
    
    return results

if __name__ == "__main__":
    try:
        # Choose which version to run
        use_factory = True
        
        if use_factory:
            results = run_backtest_with_factory()
            logger.info("Minimal backtest with factory completed successfully")
        else:
            # Legacy method for comparison
            from run_backtest import run_backtest
            results = run_backtest()
            logger.info("Minimal backtest (legacy) completed successfully")
    except Exception as e:
        logger.error(f"Error running minimal backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
