#!/usr/bin/env python
# simple_backtest.py

"""
A simplified version of the backtest script to isolate import issues.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

logger.info("Starting simplified backtest")

try:
    # Import core components
    from ai_trading_agent.agent.data_manager import SimpleDataManager
    logger.info("Imported SimpleDataManager")
    
    from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
    logger.info("Imported strategy components")
    
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    logger.info("Imported PortfolioManager")
    
    from ai_trading_agent.agent.risk_manager import SimpleRiskManager
    logger.info("Imported SimpleRiskManager")
    
    from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
    logger.info("Imported SimulatedExecutionHandler")
    
    from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
    logger.info("Imported BacktestOrchestrator")
    
    # Set up a simple backtest configuration
    config = {
        'start_date': datetime.now() - timedelta(days=30),
        'end_date': datetime.now(),
        'symbols': ['BTC/USD'],
        'data_types': ['ohlcv'],
        'timeframe': '1d',
        'initial_capital': 10000.0,
        'risk_per_trade': 0.02
    }
    
    logger.info("Configuration set up")
    
    # Create the components
    data_manager = SimpleDataManager(config={
        'symbols': config['symbols'],
        'start_date': config['start_date'],
        'end_date': config['end_date'],
        'data_dir': os.path.join(project_root, 'data'),
        'timeframe': config['timeframe'],
        'data_types': config['data_types']
    })
    strategy_manager = SimpleStrategyManager()
    portfolio_manager = PortfolioManager(initial_capital=config['initial_capital'])
    risk_manager = SimpleRiskManager(config={
        'max_portfolio_risk_pct': config['risk_per_trade'],
        'stop_loss_pct': 0.05
    })
    execution_handler = SimulatedExecutionHandler(
        portfolio_manager=portfolio_manager,
        config={
            'commission_rate': 0.001,
            'slippage_pct': 0.001
        }
    )
    
    logger.info("Components created")
    
    # Create the orchestrator
    orchestrator = BacktestOrchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=config
    )
    
    logger.info("Orchestrator created")
    
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
    logger.error(f"Error during backtest: {e}", exc_info=True)

logger.info("Simplified backtest script completed")
