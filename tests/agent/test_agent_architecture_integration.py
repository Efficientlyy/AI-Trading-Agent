"""
Integration tests for the agent architecture components.

This test suite verifies that all components of the agent architecture
work together correctly in an end-to-end scenario.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading_agent.agent.data_manager import SimpleDataManager
from ai_trading_agent.agent.strategy import SentimentStrategy, BaseStrategyManager
from ai_trading_agent.agent.risk_manager import SimpleRiskManager
from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
from ai_trading_agent.agent.factory import create_agent_from_config


class TestAgentArchitectureIntegration(unittest.TestCase):
    """Test the integration of all agent architecture components."""

    def setUp(self):
        """Set up test fixtures."""
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 31)
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.initial_cash = 100000.0

    def create_test_config(self) -> Dict[str, Any]:
        """Create a test configuration for the agent."""
        config = {
            "data_manager": {
                "type": "SimpleDataManager",
                "config": {
                    "symbols": self.symbols,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "timeframe": "1d",
                    "data_dir": "data/",
                    "use_mock_data": True,
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
                    "use_mock_sentiment": True,
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
                    "initial_cash": self.initial_cash,
                    "position_size_pct": 0.1,
                    "max_positions": 5,
                    "fractional_shares": True
                }
            },
            "risk_manager": {
                "type": "SimpleRiskManager",
                "config": {
                    "max_drawdown_pct": 0.1,
                    "max_position_pct": 0.2,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.1
                }
            },
            "execution_handler": {
                "type": "SimulatedExecutionHandler",
                "config": {
                    "slippage_model": "normal",
                    "slippage_std": 0.001,
                    "commission_model": "percentage",
                    "commission_pct": 0.001
                }
            },
            "backtest": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "symbols": self.symbols,
                "timeframe": "1d",
                "initial_cash": self.initial_cash,
                "verbose": True
            }
        }
        return config

    def test_factory_creates_valid_agent(self):
        """Test that the factory creates a valid agent from configuration."""
        config = self.create_test_config()
        agent = create_agent_from_config(config)
        
        # Verify agent is an instance of BacktestOrchestrator
        self.assertIsInstance(agent, BacktestOrchestrator)
        
        # Verify agent has all required components
        self.assertIsInstance(agent.data_manager, SimpleDataManager)
        self.assertIsInstance(agent.strategy_manager, BaseStrategyManager)
        self.assertIsInstance(agent.portfolio_manager, PortfolioManager)
        self.assertIsInstance(agent.risk_manager, SimpleRiskManager)
        self.assertIsInstance(agent.execution_handler, SimulatedExecutionHandler)

    def test_agent_runs_backtest(self):
        """Test that the agent can run a backtest and produce valid results."""
        config = self.create_test_config()
        agent = create_agent_from_config(config)
        
        # Run the backtest
        results = agent.run()
        
        # Verify results exist
        self.assertIsNotNone(results)
        
        # Verify results contain expected keys
        self.assertIn('portfolio_history', results)
        self.assertIn('trade_history', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('final_state', results)
        
        # Verify portfolio history is a list
        self.assertIsInstance(results['portfolio_history'], list)
        
        # Verify trade history is a list
        self.assertIsInstance(results['trade_history'], list)
        
        # Verify performance metrics is a dictionary
        self.assertIsInstance(results['performance_metrics'], dict)
        
        # Verify final state is a dictionary
        self.assertIsInstance(results['final_state'], dict)
        
        # Verify final cash is present in final state
        self.assertIn('cash', results['final_state'])
        
        # Verify positions are present in final state
        self.assertIn('positions', results['final_state'])

    def test_agent_components_interaction(self):
        """Test that the agent components interact correctly."""
        # Create components manually
        data_manager = SimpleDataManager(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe="1d",
            data_dir="data/",
            use_mock_data=True,
            mock_data_params={
                "volatility": 0.02,
                "drift": 0.001,
                "gap_probability": 0.05,
                "random_seed": 42
            }
        )
        
        strategy = SentimentStrategy(
            sentiment_threshold_buy=0.3,
            sentiment_threshold_sell=-0.3,
            use_mock_sentiment=True,
            mock_sentiment_params={
                "mean": 0.1,
                "std": 0.5,
                "random_seed": 42
            }
        )
        
        strategy_manager = BaseStrategyManager(strategy)
        
        portfolio_manager = PortfolioManager(
            initial_cash=self.initial_cash,
            position_size_pct=0.1,
            max_positions=5,
            fractional_shares=True
        )
        
        risk_manager = SimpleRiskManager(
            max_drawdown_pct=0.1,
            max_position_pct=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.1
        )
        
        execution_handler = SimulatedExecutionHandler(
            portfolio_manager=portfolio_manager,
            slippage_model="normal",
            slippage_std=0.001,
            commission_model="percentage",
            commission_pct=0.001
        )
        
        # Create orchestrator
        orchestrator = BacktestOrchestrator(
            data_manager=data_manager,
            strategy_manager=strategy_manager,
            portfolio_manager=portfolio_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            config={
                "start_date": self.start_date,
                "end_date": self.end_date,
                "symbols": self.symbols,
                "timeframe": "1d",
                "initial_cash": self.initial_cash,
                "verbose": True
            }
        )
        
        # Run the backtest
        results = orchestrator.run()
        
        # Verify results exist
        self.assertIsNotNone(results)
        
        # Verify portfolio history contains entries
        self.assertTrue(len(results['portfolio_history']) > 0)
        
        # Verify trade history exists (may be empty if no trades were made)
        self.assertIsInstance(results['trade_history'], list)


if __name__ == '__main__':
    unittest.main()
