"""
Tests for the agent factory system.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

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
from ai_trading_agent.agent.data_manager import SimpleDataManager
from ai_trading_agent.agent.strategy import SentimentStrategy, SimpleStrategyManager
from ai_trading_agent.agent.risk_manager import SimpleRiskManager
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
from ai_trading_agent.agent.orchestrator import BacktestOrchestrator

class TestFactory(unittest.TestCase):
    """
    Test the factory system for creating agent components.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.symbols = ["AAPL", "GOOG", "MSFT"]
        self.start_date = "2020-01-01"
        self.end_date = "2020-12-31"
        self.initial_capital = 100000.0
        
        # Create a basic configuration for testing
        self.config = {
            "data_manager": {
                "type": "SimpleDataManager",
                "config": {
                    "data_dir": os.path.join(project_root, 'temp_data'),
                    "symbols": self.symbols,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "timeframe": "1d",
                    "data_types": ["ohlcv", "sentiment"]
                }
            },
            "strategy": {
                "type": "SentimentStrategy",
                "config": {
                    "name": "SentimentStrategy",
                    "symbols": self.symbols,
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
                    "initial_capital": self.initial_capital,
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
                "start_date": self.start_date,
                "end_date": self.end_date,
                "symbols": self.symbols,
                "initial_capital": self.initial_capital
            }
        }
    
    def test_create_data_manager(self):
        """
        Test creating a data manager.
        """
        data_manager = create_data_manager(self.config["data_manager"])
        self.assertIsInstance(data_manager, SimpleDataManager)
        self.assertEqual(data_manager.symbols, self.symbols)
    
    def test_create_strategy(self):
        """
        Test creating a strategy.
        """
        strategy = create_strategy(self.config["strategy"])
        self.assertIsInstance(strategy, SentimentStrategy)
        self.assertEqual(strategy.name, "SentimentStrategy")
        self.assertEqual(strategy.symbols, self.symbols)
    
    def test_create_strategy_manager(self):
        """
        Test creating a strategy manager.
        """
        strategy = create_strategy(self.config["strategy"])
        strategy_manager = create_strategy_manager(strategy)
        self.assertIsInstance(strategy_manager, SimpleStrategyManager)
        self.assertEqual(strategy_manager.strategy, strategy)
    
    def test_create_risk_manager(self):
        """
        Test creating a risk manager.
        """
        risk_manager = create_risk_manager(self.config["risk_manager"])
        self.assertIsInstance(risk_manager, SimpleRiskManager)
        self.assertEqual(risk_manager.config["max_portfolio_risk_pct"], 0.05)
    
    def test_create_portfolio_manager(self):
        """
        Test creating a portfolio manager.
        """
        portfolio_manager = create_portfolio_manager(self.config["portfolio_manager"])
        self.assertIsInstance(portfolio_manager, PortfolioManager)
        self.assertEqual(portfolio_manager.initial_capital, self.initial_capital)
    
    def test_create_execution_handler(self):
        """
        Test creating an execution handler.
        """
        portfolio_manager = create_portfolio_manager(self.config["portfolio_manager"])
        execution_handler = create_execution_handler(
            self.config["execution_handler"],
            portfolio_manager
        )
        self.assertIsInstance(execution_handler, SimulatedExecutionHandler)
        self.assertEqual(execution_handler.config["commission_rate"], 0.001)
    
    def test_create_orchestrator(self):
        """
        Test creating an orchestrator.
        """
        data_manager = create_data_manager(self.config["data_manager"])
        strategy = create_strategy(self.config["strategy"])
        strategy_manager = create_strategy_manager(strategy)
        risk_manager = create_risk_manager(self.config["risk_manager"])
        portfolio_manager = create_portfolio_manager(self.config["portfolio_manager"])
        execution_handler = create_execution_handler(
            self.config["execution_handler"],
            portfolio_manager
        )
        
        orchestrator = create_orchestrator(
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            config=self.config["backtest"]
        )
        
        self.assertIsInstance(orchestrator, BacktestOrchestrator)
        self.assertEqual(orchestrator.config["initial_capital"], self.initial_capital)
    
    def test_invalid_component_type(self):
        """
        Test handling of invalid component types.
        """
        invalid_config = {
            "type": "NonExistentDataManager",
            "config": {}
        }
        
        with self.assertRaises(ValueError):
            create_data_manager(invalid_config)

if __name__ == "__main__":
    unittest.main()
