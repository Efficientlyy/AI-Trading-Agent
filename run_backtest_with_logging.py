#!/usr/bin/env python
# run_backtest_with_logging.py

"""
A script to run the backtest with proper error handling and logging.
"""

import os
import sys
import traceback
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backtest_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_backtest():
    """
    Run the backtest with proper error handling.
    """
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to sys.path: {project_root}")
        
        # Import required modules
        logger.info("Importing required modules...")
        
        # Import trading engine components
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        logger.info("Successfully imported enums")
        
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        logger.info("Successfully imported models")
        
        # Import backtesting components
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        logger.info("Successfully imported performance_metrics")
        
        from ai_trading_agent.backtesting.backtester import Backtester
        logger.info("Successfully imported Backtester")
        
        # Try to import RustBacktester (optional)
        try:
            from ai_trading_agent.backtesting.rust_backtester import RustBacktester
            logger.info("Successfully imported RustBacktester")
            rust_available = True
        except ImportError as e:
            logger.warning(f"RustBacktester not available: {e}")
            logger.info("This is expected if Rust extensions are not installed")
            rust_available = False
        
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
        
        # Import the run_backtest script
        logger.info("Importing run_backtest...")
        from scripts.run_backtest import run_backtest as run_backtest_func
        logger.info("Successfully imported run_backtest")
        
        # Run the backtest
        logger.info("Running backtest...")
        run_backtest_func()
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting backtest run...")
    success = run_backtest()
    
    if success:
        logger.info("Backtest completed successfully")
    else:
        logger.error("Backtest failed")
        sys.exit(1)
