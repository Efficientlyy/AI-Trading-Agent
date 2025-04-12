"""
Integration tests for the agent architecture.

These tests verify that the entire agent pipeline works correctly,
from data management through strategy execution to performance calculation.
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

from ai_trading_agent.agent.factory import create_agent_from_config
from ai_trading_agent.common.config_validator import validate_agent_config

# Custom data manager for testing
class TestDataManager:
    """
    A custom data manager for testing that provides synthetic data.
    """
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = self._generate_price_data()
        self.sentiment_data = self._generate_sentiment_data()
        
    def _generate_price_data(self):
        """Generate synthetic price data for testing."""
        data = {}
        for symbol in self.symbols:
            # Generate dates
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
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
        
        return data
    
    def _generate_sentiment_data(self):
        """Generate synthetic sentiment data for testing."""
        sentiment_data = pd.DataFrame()
        
        for symbol in self.symbols:
            # Generate dates
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            n = len(dates)
            
            # Generate sentiment scores (random values between -1 and 1)
            sentiment = np.random.normal(0, 0.5, n)
            sentiment = np.clip(sentiment, -1, 1)
            
            # Add to DataFrame
            sentiment_data[f"{symbol}_sentiment_score"] = pd.Series(sentiment, index=dates)
        
        return sentiment_data
    
    def get_latest_data(self, symbol, n=1):
        """Get the latest n bars of price data for a symbol."""
        return self.price_data[symbol].iloc[-n:]
    
    def get_latest_sentiment(self, symbol):
        """Get the latest sentiment score for a symbol."""
        return self.sentiment_data[f"{symbol}_sentiment_score"].iloc[-1]
    
    def update(self):
        """Update the data (no-op for testing)."""
        pass
    
    def get_symbols(self):
        """Get the list of symbols."""
        return self.symbols
    
    def get_current_datetime(self):
        """Get the current datetime in the data."""
        return self.price_data[self.symbols[0]].index[-1]

class TestAgentIntegration(unittest.TestCase):
    """
    Integration tests for the agent architecture.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.symbols = ["AAPL", "GOOG", "MSFT"]
        self.start_date = "2020-01-01"
        self.end_date = "2020-12-31"
        self.initial_capital = 100000.0
        
        # Create a valid configuration for testing
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
        
        # Create a test data manager
        self.test_data_manager = TestDataManager(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Create temp directory if it doesn't exist
        os.makedirs(os.path.join(project_root, 'temp_data'), exist_ok=True)
    
    def test_end_to_end_backtest(self):
        """
        Test an end-to-end backtest using the agent architecture.
        
        This test verifies that:
        1. The agent can be created from configuration
        2. The backtest runs without errors
        3. The backtest produces valid performance metrics
        """
        try:
            # Import required components
            from ai_trading_agent.agent.data_manager import SimpleDataManager
            from ai_trading_agent.agent.strategy import SentimentStrategy, SimpleStrategyManager
            from ai_trading_agent.agent.risk_manager import SimpleRiskManager
            from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
            from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
            from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
            
            # Validate the configuration
            is_valid, error_message = validate_agent_config(self.config)
            self.assertTrue(is_valid, f"Configuration is invalid: {error_message}")
            
            # Create the components manually for more control
            data_manager = SimpleDataManager(config=self.config["data_manager"]["config"])
            
            # Override the data manager's data with our test data
            data_manager.data = self.test_data_manager.price_data
            data_manager.sentiment_data = self.test_data_manager.sentiment_data
            
            # Create the strategy
            strategy = SentimentStrategy(
                name=self.config["strategy"]["config"]["name"],
                config=self.config["strategy"]["config"]
            )
            strategy_manager = SimpleStrategyManager(strategy)
            
            # Create the risk manager
            risk_manager = SimpleRiskManager(config=self.config["risk_manager"]["config"])
            
            # Create the portfolio manager
            portfolio_manager = PortfolioManager(
                initial_capital=self.config["portfolio_manager"]["config"]["initial_capital"]
            )
            
            # Create the execution handler
            execution_handler = SimulatedExecutionHandler(
                portfolio_manager=portfolio_manager,
                config=self.config["execution_handler"]["config"]
            )
            
            # Create the orchestrator
            orchestrator = BacktestOrchestrator(
                data_manager=data_manager,
                strategy_manager=strategy_manager,
                portfolio_manager=portfolio_manager,
                risk_manager=risk_manager,
                execution_handler=execution_handler,
                config=self.config["backtest"]
            )
            
            # Run the backtest
            results = orchestrator.run()
            
            # Verify the results
            self.assertIsNotNone(results, "Backtest did not return results")
            self.assertIn('performance_metrics', results, "Results do not contain performance metrics")
            
            metrics = results['performance_metrics']
            self.assertIn('total_return', metrics, "Metrics do not contain total_return")
            self.assertIn('sharpe_ratio', metrics, "Metrics do not contain sharpe_ratio")
            self.assertIn('max_drawdown', metrics, "Metrics do not contain max_drawdown")
            
            # Verify that the metrics are reasonable
            self.assertIsInstance(metrics['total_return'], float, "Total return is not a float")
            self.assertIsInstance(metrics['sharpe_ratio'], float, "Sharpe ratio is not a float")
            self.assertIsInstance(metrics['max_drawdown'], float, "Max drawdown is not a float")
            
            # Verify that the portfolio value history is present
            self.assertIn('portfolio_history', results, "Results do not contain portfolio history")
            self.assertIsInstance(results['portfolio_history'], pd.DataFrame, "Portfolio history is not a DataFrame")
            self.assertGreater(len(results['portfolio_history']), 0, "Portfolio history is empty")
            
        except Exception as e:
            self.fail(f"End-to-end backtest test failed with error: {e}")
    
    def test_factory_end_to_end(self):
        """
        Test an end-to-end backtest using the factory system.
        
        This test verifies that:
        1. The agent can be created from configuration using the factory
        2. The backtest runs without errors
        3. The backtest produces valid performance metrics
        """
        try:
            # Import the factory
            from ai_trading_agent.agent.factory import (
                create_data_manager,
                create_strategy,
                create_strategy_manager,
                create_risk_manager,
                create_portfolio_manager,
                create_execution_handler,
                create_orchestrator
            )
            
            # Create the components using the factory
            data_manager = create_data_manager(self.config["data_manager"])
            
            # Override the data manager's data with our test data
            data_manager.data = self.test_data_manager.price_data
            data_manager.sentiment_data = self.test_data_manager.sentiment_data
            
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
            
            # Run the backtest
            results = orchestrator.run()
            
            # Verify the results
            self.assertIsNotNone(results, "Backtest did not return results")
            self.assertIn('performance_metrics', results, "Results do not contain performance metrics")
            
            metrics = results['performance_metrics']
            self.assertIn('total_return', metrics, "Metrics do not contain total_return")
            self.assertIn('sharpe_ratio', metrics, "Metrics do not contain sharpe_ratio")
            self.assertIn('max_drawdown', metrics, "Metrics do not contain max_drawdown")
            
        except Exception as e:
            self.fail(f"Factory end-to-end test failed with error: {e}")
    
    def test_error_handling(self):
        """
        Test error handling in the agent architecture.
        
        This test verifies that:
        1. The agent handles invalid configurations gracefully
        2. The agent handles runtime errors gracefully
        """
        # Test with invalid configuration
        invalid_config = self.config.copy()
        del invalid_config["data_manager"]
        
        is_valid, error_message = validate_agent_config(invalid_config)
        self.assertFalse(is_valid, "Invalid configuration was not detected")
        self.assertIsNotNone(error_message, "Error message is None for invalid configuration")
        
        # Test with invalid component configuration
        invalid_component_config = self.config.copy()
        invalid_component_config["data_manager"]["type"] = "NonExistentDataManager"
        
        try:
            from ai_trading_agent.agent.factory import create_data_manager
            with self.assertRaises(ValueError):
                create_data_manager(invalid_component_config["data_manager"])
        except Exception as e:
            self.fail(f"Error handling test failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
