#!/usr/bin/env python
# debug_backtest_to_file.py

"""
This script runs the backtest with detailed logging and writes the output to a file.
"""

import os
import sys
import traceback
import logging
from datetime import datetime

# Configure logging to write to both console and file
log_file = "backtest_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_backtest")

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Added project root to sys.path: {project_root}")
logger.info(f"Full sys.path: {sys.path}")

def import_with_logging(module_name, items=None):
    """Import a module with detailed logging"""
    logger.info(f"Attempting to import {module_name}...")
    try:
        if items:
            # Import specific items from the module
            module_parts = module_name.split('.')
            import_stmt = f"from {module_name} import {', '.join(items)}"
            logger.info(f"Executing: {import_stmt}")
            exec(import_stmt, globals())
            logger.info(f"Successfully imported {', '.join(items)} from {module_name}")
        else:
            # Import the entire module
            import_stmt = f"import {module_name}"
            logger.info(f"Executing: {import_stmt}")
            exec(import_stmt, globals())
            logger.info(f"Successfully imported {module_name}")
        return True
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        logger.error("Traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        return False

# Import modules one by one with detailed logging
imports_successful = True

# First, import the logging config
if not import_with_logging("ai_trading_agent.common.logging_config", ["setup_logging"]):
    imports_successful = False
else:
    # Setup custom logging if import was successful
    try:
        setup_logging()
        logger.info("Custom logging setup complete")
    except Exception as e:
        logger.error(f"Error setting up custom logging: {e}")
        imports_successful = False

# Import trading engine components in a specific order
if imports_successful:
    # First import enums
    if not import_with_logging("ai_trading_agent.trading_engine.enums", ["OrderSide", "OrderType", "OrderStatus"]):
        imports_successful = False

if imports_successful:
    # Then import models
    if not import_with_logging("ai_trading_agent.trading_engine.models", ["Order", "Trade"]):
        imports_successful = False

if imports_successful:
    # Then import portfolio_manager
    if not import_with_logging("ai_trading_agent.trading_engine.portfolio_manager", ["PortfolioManager"]):
        imports_successful = False

# Import agent components
if imports_successful:
    if not import_with_logging("ai_trading_agent.agent.data_manager", ["SimpleDataManager"]):
        imports_successful = False

if imports_successful:
    if not import_with_logging("ai_trading_agent.agent.strategy", ["SimpleStrategyManager", "SentimentStrategy"]):
        imports_successful = False

if imports_successful:
    if not import_with_logging("ai_trading_agent.agent.risk_manager", ["SimpleRiskManager"]):
        imports_successful = False

if imports_successful:
    if not import_with_logging("ai_trading_agent.agent.execution_handler", ["SimulatedExecutionHandler"]):
        imports_successful = False

if imports_successful:
    if not import_with_logging("ai_trading_agent.agent.orchestrator", ["BacktestOrchestrator"]):
        imports_successful = False

# Check if all imports were successful
if imports_successful:
    logger.info("All imports successful! Proceeding with backtest...")
    
    # Set up a simple backtest configuration
    config = {
        'start_date': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=30),
        'end_date': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        'symbols': ['BTC/USD'],
        'data_types': ['ohlcv'],
        'timeframe': '1d',
        'initial_capital': 10000.0,
        'risk_per_trade': 0.02
    }
    
    logger.info(f"Backtest configuration: {config}")
    
    try:
        # Create the components
        data_manager = SimpleDataManager(config['symbols'], config['start_date'], config['end_date'])
        strategy_manager = SimpleStrategyManager()
        portfolio_manager = PortfolioManager(initial_capital=config['initial_capital'])
        risk_manager = SimpleRiskManager(risk_per_trade=config['risk_per_trade'])
        execution_handler = SimulatedExecutionHandler()
        
        logger.info("Components created successfully")
        
        # Create the orchestrator
        orchestrator = BacktestOrchestrator(
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            config=config
        )
        
        logger.info("Orchestrator created successfully")
        
        # Run the backtest
        logger.info("Running backtest...")
        results = orchestrator.run()
        
        # Print results
        if results:
            logger.info("Backtest completed successfully")
            if 'performance_metrics' in results:
                logger.info(f"Performance metrics: {results['performance_metrics']}")
        else:
            logger.warning("Backtest did not return results")
            
    except Exception as e:
        logger.error(f"Error during backtest execution: {e}")
        logger.error("Traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
else:
    logger.error("Some imports failed. Cannot proceed with backtest.")

logger.info(f"Debug completed. Full log written to {log_file}")
print(f"\nDebug completed. Check {log_file} for detailed output.")
