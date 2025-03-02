"""
Basic test script for the modular backtesting framework.

This script runs a simple backtest to verify the functionality
of the key components of the modular backtesting system.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
import tempfile

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.modular_backtester.models import TimeFrame
from examples.modular_backtester.strategies import (
    MovingAverageCrossoverStrategy,
    RSIStrategy,
    MultiStrategySystem
)
from examples.modular_backtester.backtester import StrategyBacktester
from examples.modular_backtester.data_utils import generate_sample_data


class TestSimpleBacktest(unittest.TestCase):
    """Test cases for backtesting functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Generate sample data
        self.symbol = "BTC-USD"
        self.start_time = datetime.now() - timedelta(days=30)
        self.end_time = datetime.now()
        
        self.candles = generate_sample_data(
            symbol=self.symbol,
            timeframe=TimeFrame.HOUR_1,
            start_time=self.start_time,
            end_time=self.end_time,
            base_price=10000.0,
            volatility=0.015,
            trend_strength=0.0001,
            with_cycles=True,
            seed=42  # For reproducible tests
        )
        
        # Create a temporary directory for reports
        self.temp_dir = tempfile.mkdtemp()
    
    def test_ma_crossover_strategy(self):
        """Test the Moving Average Crossover strategy."""
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            fast_ma_period=8,
            slow_ma_period=21,
            fast_ma_type="EMA",
            slow_ma_type="EMA",
            min_confidence=0.4
        )
        
        # Set up backtester
        backtester = StrategyBacktester(
            initial_capital=10000.0,
            position_size=0.1,
            use_stop_loss=True,
            stop_loss_pct=0.05,
            commission_pct=0.001
        )
        
        # Set strategy and add data
        backtester.set_strategy(strategy)
        backtester.add_historical_data(self.symbol, self.candles)
        
        # Run backtest
        metrics = backtester.run_backtest()
        
        # Verify metrics exist
        self.assertIsNotNone(metrics)
        self.assertTrue(hasattr(metrics, 'total_trades'))
        self.assertTrue(hasattr(metrics, 'total_pnl'))
        
        # Generate report
        report_path = backtester.generate_report(self.temp_dir)
        
        # Verify report was created
        self.assertTrue(os.path.exists(report_path))
    
    def test_rsi_strategy(self):
        """Test the RSI strategy."""
        # Create strategy
        strategy = RSIStrategy(
            period=14,
            overbought_threshold=70.0,
            oversold_threshold=30.0,
            min_confidence=0.5
        )
        
        # Set up backtester
        backtester = StrategyBacktester(
            initial_capital=10000.0,
            position_size=0.1
        )
        
        # Set strategy and add data
        backtester.set_strategy(strategy)
        backtester.add_historical_data(self.symbol, self.candles)
        
        # Run backtest
        metrics = backtester.run_backtest()
        
        # Verify metrics
        self.assertIsNotNone(metrics)
        
        # Check for closed positions
        self.assertIsNotNone(backtester.closed_positions)
    
    def test_multi_strategy(self):
        """Test the Multi-Strategy system."""
        # Create strategies
        ma_strategy = MovingAverageCrossoverStrategy(
            fast_ma_period=8,
            slow_ma_period=21
        )
        
        rsi_strategy = RSIStrategy(
            period=14,
            overbought_threshold=70.0,
            oversold_threshold=30.0
        )
        
        # Create multi-strategy
        multi_strategy = MultiStrategySystem(
            strategies=[
                (ma_strategy, 0.6),
                (rsi_strategy, 0.4)
            ],
            min_consensus=0.5
        )
        
        # Set up backtester
        backtester = StrategyBacktester(
            initial_capital=10000.0,
            position_size=0.1,
            use_stop_loss=True,
            stop_loss_pct=0.05,
            use_take_profit=True,
            take_profit_pct=0.1,
            enable_trailing_stop=True,
            trailing_stop_pct=0.03
        )
        
        # Set strategy and add data
        backtester.set_strategy(multi_strategy)
        backtester.add_historical_data(self.symbol, self.candles)
        
        # Run backtest
        metrics = backtester.run_backtest()
        
        # Verify metrics
        self.assertIsNotNone(metrics)
        metrics_dict = metrics.to_dict()
        
        print(f"Multi-Strategy Results:")
        print(f"Total P&L: ${metrics_dict['total_pnl']:.2f}")
        print(f"Total Trades: {metrics_dict['total_trades']}")
        print(f"Win Rate: {metrics_dict['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics_dict['profit_factor']:.2f}")
        
        # Generate report
        report_path = backtester.generate_report(self.temp_dir)
        
        # Verify report was created
        self.assertTrue(os.path.exists(report_path))
    
    def test_parameter_optimization(self):
        """Test the parameter optimization functionality."""
        # Create strategy
        strategy = MovingAverageCrossoverStrategy()
        
        # Set up backtester
        backtester = StrategyBacktester(
            initial_capital=10000.0,
            position_size=0.1
        )
        
        # Set strategy and add data
        backtester.set_strategy(strategy)
        backtester.add_historical_data(self.symbol, self.candles)
        
        # Define parameter ranges for optimization
        parameter_ranges = {
            'fast_ma_period': [5, 8, 13],
            'slow_ma_period': [21, 34],
            'min_confidence': [0.4, 0.5]
        }
        
        # Run optimization
        results = backtester.optimize_parameters(
            parameter_ranges=parameter_ranges,
            optimization_metric='sharpe_ratio',
            max_iterations=5
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('parameters', results)
        self.assertIn('metrics', results)
        
        print(f"Optimization Results:")
        print(f"Best Parameters: {results['parameters']}")
        print(f"Best Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main() 