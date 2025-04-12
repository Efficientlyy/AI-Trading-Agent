"""
Tests for strategy optimization and performance benchmarking.
"""

import unittest
import pandas as pd
import numpy as np
import time
import os
import tempfile
from datetime import datetime, timedelta

from ai_trading_agent.optimization.strategy_comparison import (
    StrategyComparison,
    optimize_strategy_parameters
)
from ai_trading_agent.optimization.performance_benchmark import (
    PerformanceBenchmark,
    benchmark_strategy_execution
)
from ai_trading_agent.performance.data_optimization import measure_performance

class TestStrategyOptimization(unittest.TestCase):
    """Test strategy optimization and performance benchmarking."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Create base configuration
        self.base_config = {
            "backtest": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_capital": 10000,
                "commission": 0.001
            },
            "data": {
                "source": "test",
                "resolution": "1d"
            }
        }
        
        # Create test strategies
        self.strategies = {
            "MovingAverageCrossover": {
                "type": "MovingAverageCrossover",
                "config": {
                    "fast_period": 10,
                    "slow_period": 30,
                    "risk_per_trade": 0.02
                }
            },
            "MeanReversion": {
                "type": "MeanReversion",
                "config": {
                    "lookback_period": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "risk_per_trade": 0.02
                }
            }
        }
        
        # Create parameter grid for optimization
        self.parameter_grid = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50]
        }
        
        # Create market conditions
        self.market_conditions = {
            "bull_market": {
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
                "description": "Bull market period"
            },
            "bear_market": {
                "start_date": "2023-07-01",
                "end_date": "2023-12-31",
                "description": "Bear market period"
            }
        }
        
        # Create temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        os.rmdir(self.temp_dir)
    
    def test_strategy_comparison_initialization(self):
        """Test initialization of strategy comparison."""
        # Create strategy comparison
        comparison = StrategyComparison(
            base_config=self.base_config,
            strategies=self.strategies,
            market_conditions=self.market_conditions,
            output_dir=self.temp_dir
        )
        
        # Verify initialization
        self.assertEqual(comparison.base_config, self.base_config)
        self.assertEqual(comparison.strategies, self.strategies)
        self.assertEqual(comparison.market_conditions, self.market_conditions)
        self.assertEqual(comparison.output_dir, self.temp_dir)
    
    def test_parallel_optimization(self):
        """Test parallel optimization of strategy parameters."""
        # Skip test if running in CI environment
        if os.environ.get("CI", "false").lower() == "true":
            self.skipTest("Skipping parallel optimization test in CI environment")
        
        # Create a simpler parameter grid for testing
        test_parameter_grid = {
            "fast_period": [5, 10],
            "slow_period": [20, 30]
        }
        
        # Create strategy comparison with a single strategy
        comparison = StrategyComparison(
            base_config=self.base_config,
            strategies={"MovingAverageCrossover": self.strategies["MovingAverageCrossover"]},
            output_dir=self.temp_dir
        )
        
        # Mock the _run_optimization_task method to avoid actual backtest execution
        def mock_run_optimization_task(strategy_name, strategy_config, params, base_config, use_rust):
            # Simulate backtest execution
            time.sleep(0.1)
            
            # Return mock results
            return {
                "strategy_name": strategy_name,
                "parameters": params,
                "metrics": {
                    "total_return": np.random.uniform(0.05, 0.2),
                    "sharpe_ratio": np.random.uniform(0.8, 2.0),
                    "max_drawdown": np.random.uniform(0.05, 0.15),
                    "win_rate": np.random.uniform(0.4, 0.6)
                }
            }
        
        # Replace the method with our mock
        comparison._run_optimization_task = mock_run_optimization_task
        
        # Run parallel optimization
        with measure_performance("Parallel optimization"):
            results_df = comparison.run_parallel_optimization(
                parameter_grid=test_parameter_grid,
                n_jobs=2,
                use_rust=False
            )
        
        # Verify results
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 4)  # 2x2 parameter combinations
        
        # Check columns
        self.assertIn("strategy", results_df.columns)
        self.assertIn("param_fast_period", results_df.columns)
        self.assertIn("param_slow_period", results_df.columns)
        self.assertIn("metric_total_return", results_df.columns)
        self.assertIn("metric_sharpe_ratio", results_df.columns)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation for strategy performance."""
        # Create a strategy comparison
        comparison = StrategyComparison(
            base_config=self.base_config,
            strategies=self.strategies,
            output_dir=self.temp_dir
        )
        
        # Create mock results
        comparison.results = {
            "full_period": {
                "MovingAverageCrossover": self._create_mock_strategy_result(
                    strategy_name="MovingAverageCrossover",
                    n_trades=50
                ),
                "MeanReversion": self._create_mock_strategy_result(
                    strategy_name="MeanReversion",
                    n_trades=40
                )
            }
        }
        
        # Run Monte Carlo simulation with fewer iterations for testing
        simulation_results = comparison.run_monte_carlo_simulation(
            n_simulations=100,
            confidence_level=0.95
        )
        
        # Verify results
        self.assertIsInstance(simulation_results, dict)
        self.assertIn("full_period", simulation_results)
        self.assertIn("MovingAverageCrossover", simulation_results["full_period"])
        self.assertIn("MeanReversion", simulation_results["full_period"])
        
        # Check simulation results structure
        for condition, strategies in simulation_results.items():
            for strategy, result in strategies.items():
                self.assertIn("simulations", result)
                self.assertIn("final_values", result)
                self.assertIn("mean", result)
                self.assertIn("median", result)
                self.assertIn("confidence_interval", result)
    
    def test_statistical_significance(self):
        """Test statistical significance testing."""
        # Create a strategy comparison
        comparison = StrategyComparison(
            base_config=self.base_config,
            strategies=self.strategies,
            output_dir=self.temp_dir
        )
        
        # Create mock results
        comparison.results = {
            "bull_market": {
                "MovingAverageCrossover": self._create_mock_strategy_result(
                    strategy_name="MovingAverageCrossover",
                    n_days=180,
                    mean_return=0.001,
                    std_return=0.01
                ),
                "MeanReversion": self._create_mock_strategy_result(
                    strategy_name="MeanReversion",
                    n_days=180,
                    mean_return=0.0005,
                    std_return=0.008
                )
            },
            "bear_market": {
                "MovingAverageCrossover": self._create_mock_strategy_result(
                    strategy_name="MovingAverageCrossover",
                    n_days=180,
                    mean_return=-0.0005,
                    std_return=0.012
                ),
                "MeanReversion": self._create_mock_strategy_result(
                    strategy_name="MeanReversion",
                    n_days=180,
                    mean_return=0.0002,
                    std_return=0.009
                )
            }
        }
        
        # Run statistical significance test
        results_df = comparison.statistical_significance_test(
            metric="daily_returns",
            alpha=0.05
        )
        
        # Verify results
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 0)
        
        # Check columns
        self.assertIn("strategy1", results_df.columns)
        self.assertIn("condition1", results_df.columns)
        self.assertIn("strategy2", results_df.columns)
        self.assertIn("condition2", results_df.columns)
        self.assertIn("p_value", results_df.columns)
        self.assertIn("is_significant", results_df.columns)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        # Create a performance benchmark
        benchmark = PerformanceBenchmark(
            output_dir=self.temp_dir
        )
        
        # Mock the benchmark_strategy method to avoid actual backtest execution
        def mock_benchmark_strategy(strategy_name, config, dataset_size, 
                                   optimization_level, use_rust, n_runs):
            # Simulate backtest execution
            time.sleep(0.1)
            
            # Return mock results based on optimization level
            execution_time = 10.0
            memory_usage = 100.0
            
            if optimization_level == "basic":
                execution_time *= 0.7
                memory_usage *= 0.8
            elif optimization_level == "advanced":
                execution_time *= 0.4
                memory_usage *= 0.6
            
            # Scale based on dataset size
            if dataset_size == "medium":
                execution_time *= 2
                memory_usage *= 2
            elif dataset_size == "large":
                execution_time *= 5
                memory_usage *= 5
            
            return {
                "strategy_name": strategy_name,
                "dataset_size": dataset_size,
                "optimization_level": optimization_level,
                "use_rust": use_rust,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "cpu_usage": 50.0,
                "performance_metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.1,
                    "win_rate": 0.55
                }
            }
        
        # Replace the method with our mock
        benchmark.benchmark_strategy = mock_benchmark_strategy
        
        # Run benchmark for a single strategy
        result = benchmark.benchmark_strategy(
            strategy_name="MovingAverageCrossover",
            config=self.base_config,
            dataset_size="small",
            optimization_level="none",
            use_rust=False,
            n_runs=1
        )
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["strategy_name"], "MovingAverageCrossover")
        self.assertEqual(result["dataset_size"], "small")
        self.assertEqual(result["optimization_level"], "none")
        
        # Generate report
        benchmark.generate_report()
        
        # Verify report files
        report_dirs = [d for d in os.listdir(self.temp_dir) if d.startswith("report_")]
        self.assertGreater(len(report_dirs), 0)
    
    def _create_mock_strategy_result(self, strategy_name, n_days=252, n_trades=None, 
                                    mean_return=0.0005, std_return=0.01):
        """Create mock strategy result for testing."""
        from ai_trading_agent.optimization.strategy_comparison import StrategyResult
        
        # Create date range
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(mean_return, std_return, n_days)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Create equity curve
        equity_curve = pd.DataFrame({
            "timestamp": dates,
            "portfolio_value": 10000 * cumulative_returns,
            "daily_returns": daily_returns,
            "drawdown": np.random.uniform(0, 0.1, n_days)
        })
        equity_curve.set_index("timestamp", inplace=True)
        
        # Create trades
        if n_trades is None:
            n_trades = n_days // 5  # Approximately one trade per week
        
        trade_dates = np.random.choice(dates, n_trades, replace=False)
        trade_dates.sort()
        
        trades = pd.DataFrame({
            "timestamp": trade_dates,
            "symbol": "TEST",
            "side": np.random.choice(["BUY", "SELL"], n_trades),
            "quantity": np.random.uniform(1, 10, n_trades),
            "price": np.random.uniform(90, 110, n_trades),
            "return": np.random.normal(mean_return * 5, std_return * 5, n_trades)
        })
        
        # Create drawdowns
        drawdowns = pd.DataFrame({
            "start_date": [dates[0], dates[n_days // 3], dates[2 * n_days // 3]],
            "end_date": [dates[n_days // 6], dates[n_days // 2], dates[n_days - 1]],
            "depth": [0.05, 0.08, 0.06],
            "length": [n_days // 6, n_days // 6, n_days // 3]
        })
        
        # Create and return strategy result
        return StrategyResult(
            strategy_name=strategy_name,
            parameters={"param1": 10, "param2": 20},
            metrics={
                "total_return": cumulative_returns[-1] - 1,
                "sharpe_ratio": mean_return / std_return * np.sqrt(252),
                "max_drawdown": np.random.uniform(0.05, 0.15),
                "win_rate": np.random.uniform(0.4, 0.6)
            },
            equity_curve=equity_curve,
            trades=trades,
            drawdowns=drawdowns
        )


if __name__ == "__main__":
    unittest.main()
