#!/usr/bin/env python
# minimal_run_backtest.py

"""
A minimal version of run_backtest.py to identify the exact import issue.
"""

import sys
import os
import traceback

# Create a log file for errors
error_log = open("minimal_backtest_error.log", "w")

try:
    # Ensure project root is in path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    error_log.write(f"Added project root to sys.path: {project_root}\n")
    
    # Import components one by one with detailed error logging
    
    # First, try importing enums directly
    error_log.write("Attempting to import enums...\n")
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
    error_log.write("Successfully imported enums\n")
    
    # Then try importing models
    error_log.write("Attempting to import models...\n")
    from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio
    error_log.write("Successfully imported models\n")
    
    # Then try importing portfolio_manager
    error_log.write("Attempting to import portfolio_manager...\n")
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    error_log.write("Successfully imported portfolio_manager\n")
    
    # Try importing backtesting components
    error_log.write("Attempting to import backtesting components...\n")
    from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
    error_log.write("Successfully imported performance_metrics\n")
    
    error_log.write("Attempting to import Backtester...\n")
    from ai_trading_agent.backtesting.backtester import Backtester
    error_log.write("Successfully imported Backtester\n")
    
    # Try importing agent components
    error_log.write("Attempting to import agent components...\n")
    from ai_trading_agent.agent.data_manager import SimpleDataManager
    error_log.write("Successfully imported data_manager\n")
    
    from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
    error_log.write("Successfully imported strategy\n")
    
    from ai_trading_agent.agent.risk_manager import SimpleRiskManager
    error_log.write("Successfully imported risk_manager\n")
    
    from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
    error_log.write("Successfully imported execution_handler\n")
    
    from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
    error_log.write("Successfully imported orchestrator\n")
    
    error_log.write("All imports successful!\n")
    print("All imports successful! See minimal_backtest_error.log for details.")

except Exception as e:
    error_log.write(f"ERROR: {e}\n")
    error_log.write("Traceback:\n")
    traceback.print_exc(file=error_log)
    print(f"Error occurred. See minimal_backtest_error.log for details.")

finally:
    error_log.close()
