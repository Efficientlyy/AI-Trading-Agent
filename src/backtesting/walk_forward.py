"""
Walk-forward analysis for trading strategy backtesting.

This module provides a robust framework for performing walk-forward analysis,
which helps verify strategy performance on out-of-sample data and prevents overfitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Union, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

class WalkForwardAnalysis:
    """
    Implements walk-forward analysis for trading strategy validation.
    
    Walk-forward analysis trains a strategy on in-sample data and tests on out-of-sample data,
    simulating real-world strategy deployment with periodic reoptimization.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str,
        optimization_func: Callable,
        backtest_func: Callable,
        train_size: int = 90,    # Training window in days
        test_size: int = 30,     # Testing window in days
        step_size: int = 30,     # Step size in days
        param_bounds: Dict = None,
        anchor_date: Optional[datetime] = None,
        verbose: bool = True
    ):
        """
        Initialize the walk-forward analysis.
        
        Args:
            data: DataFrame containing historical price and sentiment data
            date_column: Name of the column containing timestamps
            optimization_func: Function to optimize strategy parameters
            backtest_func: Function to backtest the strategy
            train_size: Size of the training window in days
            test_size: Size of the testing window in days
            step_size: Number of days to move forward for each iteration
            param_bounds: Dictionary of parameter bounds for optimization
            anchor_date: Starting date for analysis (defaults to earliest date in data)
            verbose: If True, print detailed progress information
        """
        self.data = data.copy()
        self.date_column = date_column
        self.optimization_func = optimization_func
        self.backtest_func = backtest_func
        self.train_size = timedelta(days=train_size)
        self.test_size = timedelta(days=test_size)
        self.step_size = timedelta(days=step_size)
        self.param_bounds = param_bounds or {}
        self.verbose = verbose
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            
        # Set anchor date
        self.min_date = self.data[date_column].min()
        self.max_date = self.data[date_column].max()
        self.anchor_date = anchor_date or self.min_date
        
        # Results storage
        self.results = {
            'windows': [],
            'train_periods': [],
            'test_periods': [],
            'optimized_params': [],
            'train_metrics': [],
            'test_metrics': [],
            'equity_curves': []
        }
        
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the walk-forward analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        if self.verbose:
            print("Starting walk-forward analysis...")
            
        # Generate time windows
        current_date = self.anchor_date
        window_count = 0
        
        while True:
            # Define training period
            train_start = current_date
            train_end = train_start + self.train_size
            
            # Define testing period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + self.test_size
            
            # Stop if we've reached the end of the data
            if test_end > self.max_date:
                break
                
            window_count += 1
            if self.verbose:
                print(f"\nWindow {window_count}:")
                print(f"  Training: {train_start.date()} to {train_end.date()}")
                print(f"  Testing: {test_start.date()} to {test_end.date()}")
            
            # Filter data for training and testing
            train_data = self.data[
                (self.data[self.date_column] >= train_start) & 
                (self.data[self.date_column] <= train_end)
            ]
            
            test_data = self.data[
                (self.data[self.date_column] >= test_start) & 
                (self.data[self.date_column] <= test_end)
            ]
            
            # Skip if insufficient data
            if len(train_data) < 10 or len(test_data) < 5:
                if self.verbose:
                    print("  Insufficient data, skipping window")
                current_date += self.step_size
                continue
                
            # Optimize parameters on training data
            if self.verbose:
                print("  Optimizing parameters...")
            
            optimization_results = self.optimization_func(
                train_data, self.param_bounds
            )
            
            best_params = optimization_results['best_params']
            train_metrics = optimization_results['metrics']
            
            if self.verbose:
                print(f"  Best parameters: {best_params}")
                print(f"  Training performance: {train_metrics['total_return']:.2%}")
            
            # Test on out-of-sample data
            if self.verbose:
                print("  Testing on out-of-sample data...")
                
            test_results = self.backtest_func(test_data, best_params)
            test_metrics = test_results['metrics']
            
            if self.verbose:
                print(f"  Testing performance: {test_metrics['total_return']:.2%}")
            
            # Store results
            self.results['windows'].append(window_count)
            self.results['train_periods'].append((train_start, train_end))
            self.results['test_periods'].append((test_start, test_end))
            self.results['optimized_params'].append(best_params)
            self.results['train_metrics'].append(train_metrics)
            self.results['test_metrics'].append(test_metrics)
            self.results['equity_curves'].append(test_results.get('equity_curve'))
            
            # Move to next window
            current_date += self.step_size
            
        # Calculate summary statistics
        if window_count > 0:
            train_returns = [m['total_return'] for m in self.results['train_metrics']]
            test_returns = [m['total_return'] for m in self.results['test_metrics']]
            
            self.results["summary"] = {
                'windows': window_count,
                'avg_train_return': np.mean(train_returns),
                'avg_test_return': np.mean(test_returns),
                'profitable_windows': sum(r > 0 for r in test_returns),
                'robustness_ratio': np.mean(test_returns) / np.mean(train_returns) if np.mean(train_returns) != 0 else 0,
                'execution_time': time.time() - start_time
            }
            
            if self.verbose:
                print("\nWalk-forward analysis complete:")
                print(f"  Windows analyzed: {window_count}")
                print(f"  Average training return: {self.results['summary']['avg_train_return']:.2%}")
                print(f"  Average testing return: {self.results['summary']['avg_test_return']:.2%}")
                print(f"  Profitable test windows: {self.results['summary']['profitable_windows']}/{window_count} "
                      f"({self.results['summary']['profitable_windows']/window_count*100:.1f}%)")
                print(f"  Robustness ratio: {self.results['summary']['robustness_ratio']:.2f}")
                print(f"  Total time: {self.results['summary']['execution_time']:.2f} seconds")
        else:
            if self.verbose:
                print("No valid windows found for analysis")
                
        return self.results
    
    def plot_window_performance(self, figsize=(12, 6)):
        """
        Plot performance comparison between training and testing periods.
        
        Args:
            figsize: Figure dimensions
        """
        if not self.results['windows']:
            print("No results to plot. Run analysis first.")
            return
            
        plt.figure(figsize=figsize)
        
        windows = self.results['windows']
        train_returns = [m['total_return'] * 100 for m in self.results['train_metrics']]
        test_returns = [m['total_return'] * 100 for m in self.results['test_metrics']]
        
        bar_width = 0.35
        opacity = 0.8
        
        plt.bar(
            [w - bar_width/2 for w in windows], 
            train_returns, 
            bar_width,
            alpha=opacity,
            color='b',
            label='Training (In-Sample)'
        )
        
        plt.bar(
            [w + bar_width/2 for w in windows], 
            test_returns, 
            bar_width,
            alpha=opacity,
            color='g',
            label='Testing (Out-of-Sample)'
        )
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Window')
        plt.ylabel('Return (%)')
        plt.title('Walk-Forward Analysis: Training vs Testing Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_parameter_evolution(self, param_name=None, figsize=(12, 6)):
        """
        Plot how a specific parameter evolves across windows.
        
        Args:
            param_name: Name of parameter to plot (if None, plots first parameter)
            figsize: Figure dimensions
        """
        if not self.results['windows']:
            print("No results to plot. Run analysis first.")
            return
            
        if not param_name:
            # Get the first parameter name from the first window
            param_name = next(iter(self.results['optimized_params'][0]))
            
        # Check if parameter exists in all windows
        if not all(param_name in params for params in self.results['optimized_params']):
            print(f"Parameter '{param_name}' not found in all windows")
            return
            
        param_values = [params[param_name] for params in self.results['optimized_params']]
        windows = self.results['windows']
        
        plt.figure(figsize=figsize)
        plt.plot(windows, param_values, 'o-', markersize=8)
        plt.xlabel('Window')
        plt.ylabel(f'Parameter Value: {param_name}')
        plt.title(f'Parameter Evolution Across Walk-Forward Windows: {param_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def get_robust_parameters(self) -> Dict[str, Any]:
        """
        Identify the most robust parameter set across all windows.
        
        Returns:
            Dictionary containing the robust parameter set and metrics
        """
        if not self.results['windows']:
            print("No results to analyze. Run analysis first.")
            return {}
            
        # Calculate average parameters across all windows
        all_params = self.results['optimized_params']
        param_keys = set()
        for params in all_params:
            param_keys.update(params.keys())
            
        robust_params = {}
        for key in param_keys:
            # Only average numerical parameters
            values = [params.get(key) for params in all_params if key in params]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if numeric_values:
                robust_params[key] = np.mean(numeric_values)
            else:
                # For non-numeric, use most common value
                from collections import Counter
                counter = Counter(values)
                robust_params[key] = counter.most_common(1)[0][0]
                
        # Calculate window performance stats
        train_returns = [m['total_return'] for m in self.results['train_metrics']]
        test_returns = [m['total_return'] for m in self.results['test_metrics']]
        
        return {
            'robust_params': robust_params,
            'train_performance': {
                'mean': np.mean(train_returns),
                'median': np.median(train_returns),
                'std': np.std(train_returns),
                'min': np.min(train_returns),
                'max': np.max(train_returns)
            },
            'test_performance': {
                'mean': np.mean(test_returns),
                'median': np.median(test_returns),
                'std': np.std(test_returns),
                'min': np.min(test_returns),
                'max': np.max(test_returns)
            },
            'robustness_ratio': np.mean(test_returns) / np.mean(train_returns) if np.mean(train_returns) != 0 else 0
        }
