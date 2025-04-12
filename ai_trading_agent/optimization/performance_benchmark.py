"""
Performance Benchmarking Tool for AI Trading Agent.

This module provides tools for benchmarking the performance of the trading agent
with different datasets, strategies, and optimization techniques.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
import os
import json
from datetime import datetime
import psutil
import tracemalloc
from functools import wraps
import logging

from ..common.logging_config import logger
from ..performance.data_optimization import measure_performance
from ..agent.factory import create_agent_from_config
from ..backtesting.performance_metrics import PerformanceMetrics

class PerformanceBenchmark:
    """
    Tool for benchmarking the performance of the trading agent.
    
    Features:
    - Measure execution time
    - Track memory usage
    - Compare different optimization techniques
    - Generate performance reports
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        log_level: str = "INFO"
    ):
        """
        Initialize the performance benchmark tool.
        
        Args:
            output_dir: Directory for saving benchmark results
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_dir = output_dir
        self.log_level = log_level
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(numeric_level)
        
        # Initialize results storage
        self.results = []
    
    def benchmark_strategy(
        self,
        strategy_name: str,
        config: Dict[str, Any],
        dataset_size: str,
        optimization_level: str = "none",
        use_rust: bool = False,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark a trading strategy with specified configuration.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration for the agent
            dataset_size: Size of the dataset (small, medium, large)
            optimization_level: Level of optimization to apply (none, basic, advanced)
            use_rust: Whether to use Rust-accelerated components
            n_runs: Number of runs to average results
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking strategy: {strategy_name} with {dataset_size} dataset")
        
        # Apply optimization configuration
        config = self._apply_optimization_config(config, optimization_level)
        
        # Initialize metrics
        execution_times = []
        memory_usages = []
        cpu_usages = []
        
        # Run multiple times to get average performance
        for i in range(n_runs):
            logger.info(f"Run {i+1}/{n_runs}")
            
            # Start memory tracking
            tracemalloc.start()
            
            # Record start time and resources
            start_time = time.time()
            start_memory = tracemalloc.get_traced_memory()[0] / (1024 * 1024)  # MB
            start_cpu = psutil.cpu_percent(interval=None)
            
            # Create agent from config
            agent = create_agent_from_config(config, use_rust=use_rust)
            
            # Run backtest
            agent.run()
            
            # Record end time and resources
            end_time = time.time()
            end_memory = tracemalloc.get_traced_memory()[0] / (1024 * 1024)  # MB
            end_cpu = psutil.cpu_percent(interval=None)
            
            # Stop memory tracking
            tracemalloc.stop()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = end_cpu - start_cpu if end_cpu > start_cpu else end_cpu
            
            # Store metrics
            execution_times.append(execution_time)
            memory_usages.append(memory_usage)
            cpu_usages.append(cpu_usage)
            
            logger.info(f"Run {i+1} completed in {execution_time:.2f} seconds")
        
        # Calculate average metrics
        avg_execution_time = np.mean(execution_times)
        avg_memory_usage = np.mean(memory_usages)
        avg_cpu_usage = np.mean(cpu_usages)
        
        # Get performance metrics
        performance_metrics = agent.get_performance_metrics()
        
        # Create result
        result = {
            "strategy_name": strategy_name,
            "dataset_size": dataset_size,
            "optimization_level": optimization_level,
            "use_rust": use_rust,
            "execution_time": avg_execution_time,
            "memory_usage": avg_memory_usage,
            "cpu_usage": avg_cpu_usage,
            "performance_metrics": performance_metrics
        }
        
        # Store result
        self.results.append(result)
        
        return result
    
    def _apply_optimization_config(
        self,
        config: Dict[str, Any],
        optimization_level: str
    ) -> Dict[str, Any]:
        """Apply optimization configuration based on the level."""
        # Create a copy of the config
        optimized_config = config.copy()
        
        # Apply optimization based on level
        if optimization_level == "none":
            # No optimization
            pass
        
        elif optimization_level == "basic":
            # Basic optimization
            if "optimization" not in optimized_config:
                optimized_config["optimization"] = {}
            
            optimized_config["optimization"].update({
                "use_caching": True,
                "chunk_size": 10000,
                "parallel_processing": False,
                "use_numpy": True
            })
        
        elif optimization_level == "advanced":
            # Advanced optimization
            if "optimization" not in optimized_config:
                optimized_config["optimization"] = {}
            
            optimized_config["optimization"].update({
                "use_caching": True,
                "chunk_size": 10000,
                "parallel_processing": True,
                "use_numpy": True,
                "use_pyarrow": True,
                "memory_efficient": True,
                "n_jobs": -1
            })
        
        else:
            raise ValueError(f"Invalid optimization level: {optimization_level}")
        
        return optimized_config
    
    def generate_report(self) -> None:
        """Generate a benchmark report with visualizations."""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results to CSV
        results_df.to_csv(os.path.join(report_dir, "benchmark_results.csv"), index=False)
        
        # Generate visualizations
        self._generate_execution_time_plot(results_df, report_dir)
        self._generate_memory_usage_plot(results_df, report_dir)
        self._generate_optimization_comparison_plot(results_df, report_dir)
        
        logger.info(f"Benchmark report generated in {report_dir}")
    
    def _generate_execution_time_plot(self, results_df: pd.DataFrame, report_dir: str) -> None:
        """Generate execution time comparison plot."""
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        sns.barplot(
            x="strategy_name",
            y="execution_time",
            hue="optimization_level",
            data=results_df
        )
        
        plt.title("Execution Time Comparison")
        plt.xlabel("Strategy")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(rotation=45)
        plt.legend(title="Optimization Level")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(report_dir, "execution_time.png"))
        plt.close()
    
    def _generate_memory_usage_plot(self, results_df: pd.DataFrame, report_dir: str) -> None:
        """Generate memory usage comparison plot."""
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        sns.barplot(
            x="strategy_name",
            y="memory_usage",
            hue="optimization_level",
            data=results_df
        )
        
        plt.title("Memory Usage Comparison")
        plt.xlabel("Strategy")
        plt.ylabel("Memory Usage (MB)")
        plt.xticks(rotation=45)
        plt.legend(title="Optimization Level")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(report_dir, "memory_usage.png"))
        plt.close()
    
    def _generate_optimization_comparison_plot(self, results_df: pd.DataFrame, report_dir: str) -> None:
        """Generate optimization comparison plot."""
        # Create a pivot table for optimization comparison
        pivot_df = results_df.pivot_table(
            index=["strategy_name", "dataset_size"],
            columns="optimization_level",
            values="execution_time"
        )
        
        # Calculate speedup relative to no optimization
        if "none" in pivot_df.columns:
            for col in pivot_df.columns:
                if col != "none":
                    pivot_df[f"{col}_speedup"] = pivot_df["none"] / pivot_df[col]
        
        # Reset index for plotting
        pivot_df = pivot_df.reset_index()
        
        # Plot speedup comparison
        if any(col.endswith("_speedup") for col in pivot_df.columns):
            plt.figure(figsize=(12, 8))
            
            # Melt DataFrame for easier plotting
            speedup_cols = [col for col in pivot_df.columns if col.endswith("_speedup")]
            id_vars = ["strategy_name", "dataset_size"]
            
            melted_df = pd.melt(
                pivot_df,
                id_vars=id_vars,
                value_vars=speedup_cols,
                var_name="optimization_level",
                value_name="speedup"
            )
            
            # Clean up optimization level names
            melted_df["optimization_level"] = melted_df["optimization_level"].str.replace("_speedup", "")
            
            # Create grouped bar chart
            sns.barplot(
                x="strategy_name",
                y="speedup",
                hue="optimization_level",
                data=melted_df
            )
            
            plt.title("Optimization Speedup Comparison")
            plt.xlabel("Strategy")
            plt.ylabel("Speedup Factor (relative to no optimization)")
            plt.xticks(rotation=45)
            plt.legend(title="Optimization Level")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(report_dir, "optimization_speedup.png"))
            plt.close()

def benchmark_strategy_execution(
    strategy_name: str,
    config: Dict[str, Any],
    dataset_sizes: List[str] = ["small", "medium", "large"],
    optimization_levels: List[str] = ["none", "basic", "advanced"],
    use_rust: bool = False,
    n_runs: int = 3,
    output_dir: str = "benchmark_results"
) -> pd.DataFrame:
    """
    Benchmark a trading strategy with different dataset sizes and optimization levels.
    
    Args:
        strategy_name: Name of the strategy
        config: Configuration for the agent
        dataset_sizes: List of dataset sizes to test
        optimization_levels: List of optimization levels to test
        use_rust: Whether to use Rust-accelerated components
        n_runs: Number of runs to average results
        output_dir: Directory for saving benchmark results
        
    Returns:
        DataFrame with benchmark results
    """
    # Create benchmark tool
    benchmark = PerformanceBenchmark(output_dir=output_dir)
    
    # Run benchmarks for each combination
    for dataset_size in dataset_sizes:
        for optimization_level in optimization_levels:
            benchmark.benchmark_strategy(
                strategy_name=strategy_name,
                config=config,
                dataset_size=dataset_size,
                optimization_level=optimization_level,
                use_rust=use_rust,
                n_runs=n_runs
            )
    
    # Generate report
    benchmark.generate_report()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(benchmark.results)
    
    return results_df
