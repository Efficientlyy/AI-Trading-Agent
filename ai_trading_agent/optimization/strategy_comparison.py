"""
Strategy Comparison Framework for AI Trading Agent.

This module provides tools for comparing different trading strategies and their
optimized parameters across various market conditions and performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
import os
import json
from dataclasses import dataclass
from datetime import datetime
import time
import concurrent.futures
from functools import partial
import scipy.stats as stats

from ..agent.factory import create_agent_from_config
from ..backtesting.performance_metrics import PerformanceMetrics
from ..common.logging_config import logger
from ..performance.data_optimization import measure_performance

@dataclass
class StrategyResult:
    """Results from a strategy backtest."""
    strategy_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    drawdowns: pd.DataFrame
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            # Convert DataFrames to dict for JSON serialization
            "equity_curve": self.equity_curve.to_dict() if self.equity_curve is not None else None,
            "trades": self.trades.to_dict() if self.trades is not None else None,
            "drawdowns": self.drawdowns.to_dict() if self.drawdowns is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyResult':
        """Create from dictionary."""
        return cls(
            strategy_name=data["strategy_name"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            equity_curve=pd.DataFrame(data["equity_curve"]) if data["equity_curve"] else None,
            trades=pd.DataFrame(data["trades"]) if data["trades"] else None,
            drawdowns=pd.DataFrame(data["drawdowns"]) if data["drawdowns"] else None
        )

class StrategyComparison:
    """
    Framework for comparing trading strategies across different market conditions.
    
    Features:
    - Run multiple strategies with different parameters
    - Compare performance across various metrics
    - Analyze strategy behavior in different market conditions
    - Generate detailed comparison reports and visualizations
    - Parallel strategy optimization for faster processing
    - Performance profiling for strategy execution
    - Statistical significance testing of strategy performance
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        strategies: Dict[str, Dict[str, Any]],
        market_conditions: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
        output_dir: str = "strategy_comparison_results",
    ):
        """
        Initialize the strategy comparison framework.
        
        Args:
            base_config: Base configuration for the agent
            strategies: Dictionary of strategies to compare
                Format: {
                    "strategy_name": {
                        "type": "SentimentStrategy",
                        "config": {...}
                    }
                }
            market_conditions: Dictionary of market conditions to test
                Format: {
                    "condition_name": {
                        "start_date": "2020-01-01",
                        "end_date": "2020-12-31",
                        "description": "Bull market"
                    }
                }
            metrics: List of metrics to compare (default: all available)
            output_dir: Directory for saving results
        """
        self.base_config = base_config
        self.strategies = strategies
        self.market_conditions = market_conditions or {
            "full_period": {
                "start_date": base_config.get("backtest", {}).get("start_date"),
                "end_date": base_config.get("backtest", {}).get("end_date"),
                "description": "Full backtest period"
            }
        }
        self.metrics = metrics or [
            "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "win_rate", "profit_factor", "calmar_ratio", "omega_ratio"
        ]
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results: Dict[str, Dict[str, StrategyResult]] = {}
    
    def run_comparison(self, use_rust: bool = False, n_jobs: int = -1):
        """
        Run the strategy comparison.
        
        Args:
            use_rust: Whether to use Rust-accelerated components if available
            n_jobs: Number of parallel jobs to run (-1 for all available cores)
            
        Returns:
            DataFrame with comparison results
        """
        self.results = {}
        start_time = time.time()
        
        # Determine number of workers
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Run strategies in parallel for each market condition
        for condition_name, condition_config in self.market_conditions.items():
            logger.info(f"Running comparison for market condition: {condition_name}")
            self.results[condition_name] = {}
            
            # Create tasks for parallel execution
            tasks = []
            for strategy_name, strategy_config in self.strategies.items():
                # Create a copy of the base config
                config = self.base_config.copy()
                
                # Update with strategy config
                config["strategy"] = strategy_config
                
                # Update with market condition config
                if "backtest" not in config:
                    config["backtest"] = {}
                config["backtest"]["start_date"] = condition_config["start_date"]
                config["backtest"]["end_date"] = condition_config["end_date"]
                
                tasks.append((strategy_name, config, use_rust))
            
            # Execute tasks in parallel
            if n_jobs > 1 and len(tasks) > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(self._run_strategy, *task) for task in tasks]
                    
                    for future, (strategy_name, _, _) in zip(futures, tasks):
                        try:
                            result = future.result()
                            self.results[condition_name][strategy_name] = result
                        except Exception as e:
                            logger.error(f"Error running strategy {strategy_name}: {e}")
            else:
                # Run sequentially
                for strategy_name, config, use_rust in tasks:
                    try:
                        result = self._run_strategy(strategy_name, config, use_rust)
                        self.results[condition_name][strategy_name] = result
                    except Exception as e:
                        logger.error(f"Error running strategy {strategy_name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = self.create_comparison_dataframe()
        
        end_time = time.time()
        logger.info(f"Strategy comparison completed in {end_time - start_time:.2f} seconds")
        
        return comparison_df
    
    def _run_strategy(self, strategy_name: str, config: Dict[str, Any], use_rust: bool) -> StrategyResult:
        """Run a single strategy and return the results."""
        logger.info(f"Running strategy: {strategy_name}")
        
        # Create agent from config
        agent = create_agent_from_config(config, use_rust=use_rust)
        
        # Measure performance of strategy execution
        with measure_performance(f"Strategy execution: {strategy_name}"):
            # Run backtest
            agent.run()
        
        # Get results
        metrics = agent.get_performance_metrics()
        equity_curve = agent.get_equity_curve()
        trades = agent.get_trades()
        drawdowns = agent.get_drawdowns() if hasattr(agent, "get_drawdowns") else None
        
        # Extract parameters from config
        parameters = config["strategy"].get("config", {})
        
        # Create and return result
        return StrategyResult(
            strategy_name=strategy_name,
            parameters=parameters,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            drawdowns=drawdowns
        )
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with comparison results.
        
        Returns:
            DataFrame with comparison results
        """
        rows = []
        
        for condition_name, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                row = {
                    "market_condition": condition_name,
                    "strategy": strategy_name
                }
                
                # Add metrics
                for metric, value in result.metrics.items():
                    if metric in self.metrics:
                        row[metric] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self):
        """Save comparison results to files."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(self.output_dir, f"comparison_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save comparison DataFrame
        comparison_df = self.create_comparison_dataframe()
        comparison_df.to_csv(os.path.join(results_dir, "comparison_results.csv"), index=False)
        
        # Save detailed results for each strategy and condition
        for condition_name, strategies in self.results.items():
            condition_dir = os.path.join(results_dir, condition_name)
            os.makedirs(condition_dir, exist_ok=True)
            
            for strategy_name, result in strategies.items():
                # Save metrics
                with open(os.path.join(condition_dir, f"{strategy_name}_metrics.json"), "w") as f:
                    json.dump(result.metrics, f, indent=2)
                
                # Save equity curve
                if result.equity_curve is not None:
                    result.equity_curve.to_csv(
                        os.path.join(condition_dir, f"{strategy_name}_equity.csv"), 
                        index=True
                    )
                
                # Save trades
                if result.trades is not None:
                    result.trades.to_csv(
                        os.path.join(condition_dir, f"{strategy_name}_trades.csv"), 
                        index=False
                    )
        
        # Save full results as JSON
        results_json = {}
        for condition_name, strategies in self.results.items():
            results_json[condition_name] = {
                strategy_name: result.to_dict()
                for strategy_name, result in strategies.items()
            }
        
        with open(os.path.join(results_dir, "full_results.json"), "w") as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
    
    def generate_report(self):
        """Generate a detailed comparison report with visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_df = self.create_comparison_dataframe()
        
        # Generate heatmap of metrics
        self._generate_heatmap(comparison_df, report_dir)
        
        # Generate equity curve comparison
        self._generate_equity_curves(report_dir)
        
        # Generate drawdown comparison
        self._generate_drawdown_comparison(report_dir)
        
        # Generate metric comparison bar charts
        self._generate_metric_comparison(comparison_df, report_dir)
        
        logger.info(f"Report generated in {report_dir}")
    
    def _generate_heatmap(self, comparison_df: pd.DataFrame, report_dir: str):
        """Generate heatmap of metrics."""
        # Pivot the DataFrame for the heatmap
        for metric in self.metrics:
            if metric in comparison_df.columns:
                plt.figure(figsize=(12, 8))
                
                # Create pivot table
                pivot = comparison_df.pivot(
                    index="strategy", 
                    columns="market_condition", 
                    values=metric
                )
                
                # Create heatmap
                sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".4f")
                
                plt.title(f"{metric} Comparison")
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(report_dir, f"heatmap_{metric}.png"))
                plt.close()
    
    def _generate_equity_curves(self, report_dir: str):
        """Generate equity curve comparison plots."""
        # Generate equity curve for each market condition
        for condition_name, strategies in self.results.items():
            plt.figure(figsize=(12, 8))
            
            for strategy_name, result in strategies.items():
                if result.equity_curve is not None and "portfolio_value" in result.equity_curve.columns:
                    # Normalize to starting value of 1.0 for comparison
                    equity = result.equity_curve["portfolio_value"]
                    normalized_equity = equity / equity.iloc[0]
                    
                    plt.plot(normalized_equity.index, normalized_equity, label=strategy_name)
            
            plt.title(f"Equity Curve Comparison - {condition_name}")
            plt.xlabel("Date")
            plt.ylabel("Normalized Portfolio Value")
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(report_dir, f"equity_curves_{condition_name}.png"))
            plt.close()
    
    def _generate_drawdown_comparison(self, report_dir: str):
        """Generate drawdown comparison plots."""
        # Generate drawdown plot for each market condition
        for condition_name, strategies in self.results.items():
            plt.figure(figsize=(12, 8))
            
            for strategy_name, result in strategies.items():
                if result.equity_curve is not None and "drawdown" in result.equity_curve.columns:
                    drawdown = result.equity_curve["drawdown"]
                    plt.plot(drawdown.index, drawdown, label=strategy_name)
            
            plt.title(f"Drawdown Comparison - {condition_name}")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(report_dir, f"drawdowns_{condition_name}.png"))
            plt.close()
    
    def _generate_metric_comparison(self, comparison_df: pd.DataFrame, report_dir: str):
        """Generate metric comparison bar charts."""
        # Generate bar chart for each metric
        for metric in self.metrics:
            if metric in comparison_df.columns:
                plt.figure(figsize=(12, 8))
                
                # Create grouped bar chart
                sns.barplot(x="strategy", y=metric, hue="market_condition", data=comparison_df)
                
                plt.title(f"{metric} Comparison")
                plt.xlabel("Strategy")
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.legend(title="Market Condition")
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(report_dir, f"barplot_{metric}.png"))
                plt.close()
    
    def run_parallel_optimization(self, parameter_grid: Dict[str, List[Any]], n_jobs: int = -1, use_rust: bool = True):
        """
        Run parallel optimization of strategy parameters.
        
        Args:
            parameter_grid: Dictionary of parameter names and values to test
            n_jobs: Number of parallel jobs to run (-1 for all available cores)
            use_rust: Whether to use Rust-accelerated components
            
        Returns:
            DataFrame with optimization results
        """
        logger.info(f"Running parallel optimization with {len(self.strategies)} strategies")
        
        # Determine number of workers
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        optimization_results = {}
        
        # For each strategy, run optimization
        for strategy_name, strategy_config in self.strategies.items():
            logger.info(f"Optimizing strategy: {strategy_name}")
            
            # Create parameter combinations
            from itertools import product
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            param_combinations = list(product(*param_values))
            
            # Create tasks for parallel execution
            tasks = []
            for params in param_combinations:
                # Create parameter dictionary
                param_dict = {name: value for name, value in zip(param_names, params)}
                
                # Create a copy of the strategy config
                config = strategy_config.copy()
                
                # Update with parameters
                if "config" not in config:
                    config["config"] = {}
                config["config"].update(param_dict)
                
                # Create task
                tasks.append((strategy_name, config, param_dict))
            
            # Execute tasks in parallel
            results = []
            if n_jobs > 1 and len(tasks) > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # Create a partial function with fixed base_config and use_rust
                    run_optimization_task = partial(
                        self._run_optimization_task, 
                        base_config=self.base_config,
                        use_rust=use_rust
                    )
                    
                    # Submit tasks
                    futures = [executor.submit(run_optimization_task, *task) for task in tasks]
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error in optimization task: {e}")
            else:
                # Run sequentially
                for task in tasks:
                    try:
                        result = self._run_optimization_task(*task, base_config=self.base_config, use_rust=use_rust)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in optimization task: {e}")
            
            # Store results
            optimization_results[strategy_name] = results
        
        # Convert results to DataFrame
        return self._create_optimization_dataframe(optimization_results)
    
    @staticmethod
    def _run_optimization_task(strategy_name: str, strategy_config: Dict[str, Any], 
                              params: Dict[str, Any], base_config: Dict[str, Any], 
                              use_rust: bool) -> Dict[str, Any]:
        """Run a single optimization task and return the results."""
        # Create a copy of the base config
        config = base_config.copy()
        
        # Update with strategy config
        config["strategy"] = strategy_config
        
        # Create agent from config
        agent = create_agent_from_config(config, use_rust=use_rust)
        
        # Run backtest
        agent.run()
        
        # Get metrics
        metrics = agent.get_performance_metrics()
        
        # Return results
        return {
            "strategy_name": strategy_name,
            "parameters": params,
            "metrics": metrics
        }
    
    def _create_optimization_dataframe(self, optimization_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Create a DataFrame from optimization results."""
        rows = []
        
        for strategy_name, results in optimization_results.items():
            for result in results:
                # Create a row with strategy name and parameters
                row = {
                    "strategy": strategy_name,
                    **{f"param_{k}": v for k, v in result["parameters"].items()}
                }
                
                # Add metrics
                row.update({f"metric_{k}": v for k, v in result["metrics"].items()})
                
                rows.append(row)
        
        return pd.DataFrame(rows)

    def statistical_significance_test(self, metric: str = "daily_returns", alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform statistical significance tests to compare strategies.
        
        Args:
            metric: The metric to compare (default: daily_returns)
            alpha: Significance level for hypothesis tests
            
        Returns:
            DataFrame with p-values for each strategy pair
        """
        logger.info(f"Performing statistical significance tests for metric: {metric}")
        
        # Extract time series data for each strategy
        strategy_data = {}
        
        for condition_name, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                if result.equity_curve is not None:
                    # Extract the specified metric
                    if metric == "daily_returns":
                        # Calculate daily returns if not already present
                        if "daily_returns" not in result.equity_curve.columns:
                            result.equity_curve["daily_returns"] = result.equity_curve["portfolio_value"].pct_change()
                        
                        # Store the time series
                        key = f"{strategy_name}_{condition_name}"
                        strategy_data[key] = result.equity_curve["daily_returns"].dropna()
        
        # Create a matrix of p-values
        strategy_keys = list(strategy_data.keys())
        n_strategies = len(strategy_keys)
        
        # Initialize results DataFrame
        p_values = pd.DataFrame(
            np.ones((n_strategies, n_strategies)),
            index=strategy_keys,
            columns=strategy_keys
        )
        
        # Perform statistical tests for each pair of strategies
        for i, strategy1 in enumerate(strategy_keys):
            for j, strategy2 in enumerate(strategy_keys):
                if i != j:  # Skip self-comparison
                    # Get data for both strategies
                    data1 = strategy_data[strategy1]
                    data2 = strategy_data[strategy2]
                    
                    # Perform statistical test
                    try:
                        # Perform t-test for difference in means
                        t_stat, p_value = stats.ttest_ind(
                            data1, 
                            data2, 
                            equal_var=False,  # Welch's t-test (don't assume equal variance)
                            nan_policy='omit'
                        )
                        
                        # Store p-value
                        p_values.loc[strategy1, strategy2] = p_value
                    except Exception as e:
                        logger.error(f"Error performing statistical test for {strategy1} vs {strategy2}: {e}")
        
        # Create a more readable result DataFrame
        result_df = self._format_significance_results(p_values, alpha)
        
        return result_df
    
    def _format_significance_results(self, p_values: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Format significance test results for better readability."""
        # Create a new DataFrame for formatted results
        results = []
        
        for idx, row in p_values.iterrows():
            for col in p_values.columns:
                if idx != col:  # Skip self-comparison
                    strategy1 = idx.split('_')[0]
                    condition1 = '_'.join(idx.split('_')[1:])
                    strategy2 = col.split('_')[0]
                    condition2 = '_'.join(col.split('_')[1:])
                    
                    p_value = row[col]
                    is_significant = p_value < alpha
                    
                    results.append({
                        'strategy1': strategy1,
                        'condition1': condition1,
                        'strategy2': strategy2,
                        'condition2': condition2,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'better_strategy': strategy1 if is_significant else 'No significant difference'
                    })
        
        return pd.DataFrame(results)
    
    def generate_statistical_report(self, metric: str = "daily_returns", alpha: float = 0.05) -> None:
        """
        Generate a statistical significance report with visualizations.
        
        Args:
            metric: The metric to compare (default: daily_returns)
            alpha: Significance level for hypothesis tests
        """
        # Perform statistical tests
        results_df = self.statistical_significance_test(metric, alpha)
        
        # Create output directory
        report_dir = os.path.join(self.output_dir, "statistical_analysis")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save results to CSV
        results_df.to_csv(os.path.join(report_dir, "significance_tests.csv"), index=False)
        
        # Create heatmap of p-values
        self._generate_pvalue_heatmap(results_df, report_dir)
        
        # Generate distribution comparison plots
        self._generate_distribution_plots(metric, report_dir)
        
        logger.info(f"Statistical report generated in {report_dir}")
    
    def _generate_pvalue_heatmap(self, results_df: pd.DataFrame, report_dir: str) -> None:
        """Generate heatmap of p-values."""
        # Create a pivot table of p-values
        pivot_df = results_df.pivot_table(
            index=['strategy1', 'condition1'],
            columns=['strategy2', 'condition2'],
            values='p_value'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            cmap='coolwarm_r',  # Reverse colormap so lower p-values (more significant) are darker
            vmin=0, 
            vmax=0.1,  # Cap at 0.1 for better visualization
            linewidths=0.5
        )
        plt.title('P-values for Strategy Comparisons')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(report_dir, "pvalue_heatmap.png"))
        plt.close()
    
    def _generate_distribution_plots(self, metric: str, report_dir: str) -> None:
        """Generate distribution plots for the specified metric."""
        # Extract data for each strategy
        for condition_name, strategies in self.results.items():
            plt.figure(figsize=(12, 8))
            
            for strategy_name, result in strategies.items():
                if result.equity_curve is not None:
                    # Extract the specified metric
                    if metric == "daily_returns":
                        # Calculate daily returns if not already present
                        if "daily_returns" not in result.equity_curve.columns:
                            result.equity_curve["daily_returns"] = result.equity_curve["portfolio_value"].pct_change()
                        
                        # Plot distribution
                        sns.kdeplot(
                            result.equity_curve["daily_returns"].dropna(),
                            label=strategy_name
                        )
            
            plt.title(f"{metric.replace('_', ' ').title()} Distribution - {condition_name}")
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(report_dir, f"distribution_{metric}_{condition_name}.png"))
            plt.close()
    
    def run_monte_carlo_simulation(self, n_simulations: int = 1000, confidence_level: float = 0.95) -> Dict[str, Dict[str, Any]]:
        """
        Run Monte Carlo simulations to estimate confidence intervals for strategy performance.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% confidence)
            
        Returns:
            Dictionary with simulation results for each strategy
        """
        logger.info(f"Running Monte Carlo simulations with {n_simulations} iterations")
        
        simulation_results = {}
        
        # For each strategy in each market condition
        for condition_name, strategies in self.results.items():
            simulation_results[condition_name] = {}
            
            for strategy_name, result in strategies.items():
                if result.trades is not None and len(result.trades) > 0:
                    # Extract trade returns
                    trade_returns = result.trades.get("return", result.trades.get("pnl_pct", None))
                    
                    if trade_returns is not None:
                        # Run Monte Carlo simulation
                        sim_result = self._run_single_monte_carlo(
                            trade_returns=trade_returns.values,
                            n_simulations=n_simulations,
                            confidence_level=confidence_level
                        )
                        
                        # Store results
                        simulation_results[condition_name][strategy_name] = sim_result
                    else:
                        logger.warning(f"No trade returns found for strategy {strategy_name} in {condition_name}")
                else:
                    logger.warning(f"No trades found for strategy {strategy_name} in {condition_name}")
        
        # Generate Monte Carlo report
        self._generate_monte_carlo_report(simulation_results)
        
        return simulation_results
    
    def _run_single_monte_carlo(self, trade_returns: np.ndarray, n_simulations: int, confidence_level: float) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a single strategy."""
        # Number of trades
        n_trades = len(trade_returns)
        
        # Initialize array for simulation results
        cumulative_returns = np.zeros((n_simulations, n_trades + 1))
        cumulative_returns[:, 0] = 1.0  # Start with $1
        
        # Run simulations
        for i in range(n_simulations):
            # Resample trade returns with replacement
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
            
            # Calculate cumulative returns
            for j in range(n_trades):
                cumulative_returns[i, j+1] = cumulative_returns[i, j] * (1 + sampled_returns[j])
        
        # Calculate statistics
        final_values = cumulative_returns[:, -1]
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        confidence_interval = np.percentile(final_values, [lower_percentile, upper_percentile])
        
        # Return results
        return {
            "simulations": cumulative_returns,
            "final_values": final_values,
            "mean": np.mean(final_values),
            "median": np.median(final_values),
            "std": np.std(final_values),
            "confidence_interval": confidence_interval,
            "confidence_level": confidence_level,
            "worst_case": np.min(final_values),
            "best_case": np.max(final_values)
        }
    
    def _generate_monte_carlo_report(self, simulation_results: Dict[str, Dict[str, Any]]) -> None:
        """Generate Monte Carlo simulation report."""
        # Create output directory
        report_dir = os.path.join(self.output_dir, "monte_carlo")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate summary table
        summary_rows = []
        
        for condition_name, strategies in simulation_results.items():
            for strategy_name, result in strategies.items():
                summary_rows.append({
                    "market_condition": condition_name,
                    "strategy": strategy_name,
                    "mean_return": result["mean"] - 1,
                    "median_return": result["median"] - 1,
                    "std_dev": result["std"],
                    "lower_bound": result["confidence_interval"][0] - 1,
                    "upper_bound": result["confidence_interval"][1] - 1,
                    "worst_case": result["worst_case"] - 1,
                    "best_case": result["best_case"] - 1
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # Save to CSV
        summary_df.to_csv(os.path.join(report_dir, "monte_carlo_summary.csv"), index=False)
        
        # Generate plots
        for condition_name, strategies in simulation_results.items():
            for strategy_name, result in strategies.items():
                # Plot simulation paths
                plt.figure(figsize=(12, 8))
                
                # Plot a subset of paths for clarity
                n_paths_to_plot = min(100, result["simulations"].shape[0])
                for i in range(n_paths_to_plot):
                    plt.plot(result["simulations"][i], alpha=0.1, color='blue')
                
                # Plot mean path
                mean_path = np.mean(result["simulations"], axis=0)
                plt.plot(mean_path, linewidth=2, color='red', label='Mean')
                
                # Plot confidence interval
                lower_bound = np.percentile(result["simulations"], (1 - result["confidence_level"]) / 2 * 100, axis=0)
                upper_bound = np.percentile(result["simulations"], (1 + result["confidence_level"]) / 2 * 100, axis=0)
                
                plt.fill_between(
                    range(len(mean_path)),
                    lower_bound,
                    upper_bound,
                    alpha=0.2,
                    color='red',
                    label=f"{int(result['confidence_level'] * 100)}% Confidence Interval"
                )
                
                plt.title(f"Monte Carlo Simulation - {strategy_name} in {condition_name}")
                plt.xlabel("Trade Number")
                plt.ylabel("Portfolio Value (Starting at $1)")
                plt.legend()
                plt.grid(True)
                
                # Save figure
                plt.savefig(os.path.join(report_dir, f"monte_carlo_{strategy_name}_{condition_name}.png"))
                plt.close()
                
                # Plot distribution of final values
                plt.figure(figsize=(12, 8))
                
                sns.histplot(result["final_values"], kde=True)
                
                # Add vertical lines for mean, median, and confidence interval
                plt.axvline(result["mean"], color='red', linestyle='--', label=f"Mean: {result['mean']:.4f}")
                plt.axvline(result["median"], color='green', linestyle='--', label=f"Median: {result['median']:.4f}")
                plt.axvline(result["confidence_interval"][0], color='orange', linestyle='--', 
                           label=f"Lower CI: {result['confidence_interval'][0]:.4f}")
                plt.axvline(result["confidence_interval"][1], color='orange', linestyle='--',
                           label=f"Upper CI: {result['confidence_interval'][1]:.4f}")
                
                plt.title(f"Distribution of Final Portfolio Values - {strategy_name} in {condition_name}")
                plt.xlabel("Final Portfolio Value (Starting at $1)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True)
                
                # Save figure
                plt.savefig(os.path.join(report_dir, f"distribution_{strategy_name}_{condition_name}.png"))
                plt.close()
        
        logger.info(f"Monte Carlo report generated in {report_dir}")

def optimize_strategy_parameters(
    base_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    parameter_grid: Dict[str, List[Any]],
    optimization_metric: str = "sharpe_ratio",
    n_jobs: int = -1,
    use_rust: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        base_config: Base configuration for the agent
        strategy_config: Strategy configuration
        parameter_grid: Dictionary of parameter names and values to test
        optimization_metric: Metric to optimize (default: sharpe_ratio)
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
        use_rust: Whether to use Rust-accelerated components
        
    Returns:
        Tuple of (best_parameters, results_dataframe)
    """
    # Create a strategy comparison with a single strategy
    comparison = StrategyComparison(
        base_config=base_config,
        strategies={"strategy_to_optimize": strategy_config}
    )
    
    # Run optimization
    results_df = comparison.run_parallel_optimization(
        parameter_grid=parameter_grid,
        n_jobs=n_jobs,
        use_rust=use_rust
    )
    
    # Find best parameters
    metric_col = f"metric_{optimization_metric}"
    if metric_col not in results_df.columns:
        raise ValueError(f"Optimization metric '{optimization_metric}' not found in results")
    
    # Get row with best metric
    best_row = results_df.loc[results_df[metric_col].idxmax()]
    
    # Extract parameters
    best_params = {
        k.replace("param_", ""): v 
        for k, v in best_row.items() 
        if k.startswith("param_")
    }
    
    return best_params, results_df

def compare_strategies(
    base_config: Dict[str, Any],
    strategies: Dict[str, Dict[str, Any]],
    market_conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics: Optional[List[str]] = None,
    output_dir: str = "strategy_comparison_results",
    use_rust: bool = False
) -> pd.DataFrame:
    """
    Compare multiple trading strategies across different market conditions.
    
    Args:
        base_config: Base configuration for the agent
        strategies: Dictionary of strategies to compare
        market_conditions: Dictionary of market conditions to test
        metrics: List of metrics to compare
        output_dir: Directory for saving results
        use_rust: Whether to use Rust-accelerated components if available
        
    Returns:
        DataFrame with comparison results
    """
    comparison = StrategyComparison(
        base_config=base_config,
        strategies=strategies,
        market_conditions=market_conditions,
        metrics=metrics,
        output_dir=output_dir
    )
    
    return comparison.run_comparison(use_rust=use_rust)
