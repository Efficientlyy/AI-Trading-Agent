"""
Walk-forward analysis framework for trading strategy optimization.

This module provides tools for performing walk-forward analysis (WFA) to test how well
a trading strategy adapts to changing market conditions and avoid overfitting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Tuple, Optional, Union
import time
from datetime import datetime, timedelta
from .genetic_optimizer import GeneticOptimizer


class WalkForwardAnalyzer:
    """
    Walk-forward analysis framework for trading strategy optimization.
    
    Walk-forward analysis divides historical data into multiple training and testing
    segments, optimizing strategy parameters on each training segment and validating
    on the corresponding test segment. This helps assess how well a strategy performs
    on unseen data and reduces the risk of overfitting.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str,
        training_window: int = 90,  # Days for training window
        testing_window: int = 30,   # Days for testing window
        step_size: int = 30,        # Days to move forward for each iteration
        optimization_function: Callable = None,
        backtest_function: Callable = None,
        param_bounds: Dict[str, Union[Tuple[float, float], List[Any]]] = None,
        optimization_args: Dict[str, Any] = None,
        anchor_date: Optional[datetime] = None,
        verbose: bool = True
    ):
        """
        Initialize the walk-forward analyzer.
        
        Args:
            data: DataFrame containing historical data
            date_column: Name of the column containing dates/timestamps
            training_window: Number of days for the training window
            testing_window: Number of days for the testing window
            step_size: Number of days to move forward for each iteration
            optimization_function: Function to optimize strategy parameters
            backtest_function: Function to evaluate the strategy on testing data
            param_bounds: Dictionary of parameter bounds for optimization
            optimization_args: Additional arguments for the optimization function
            anchor_date: Optional start date for the first training window
            verbose: If True, print progress information
        """
        self.data = data.copy()
        
        # Ensure the date column is a datetime
        if pd.api.types.is_datetime64_any_dtype(data[date_column]):
            self.data[date_column] = pd.to_datetime(data[date_column])
        
        self.date_column = date_column
        self.training_window = timedelta(days=training_window)
        self.testing_window = timedelta(days=testing_window)
        self.step_size = timedelta(days=step_size)
        self.optimization_function = optimization_function
        self.backtest_function = backtest_function
        self.param_bounds = param_bounds or {}
        self.optimization_args = optimization_args or {}
        self.verbose = verbose
        
        # Determine first and last dates in the dataset
        self.first_date = self.data[date_column].min()
        self.last_date = self.data[date_column].max()
        
        # Set the anchor date
        if anchor_date:
            self.anchor_date = anchor_date
        else:
            self.anchor_date = self.first_date
            
        # Prepare results storage
        self.results = {
            'training_periods': [],
            'testing_periods': [],
            'optimized_params': [],
            'training_performance': [],
            'testing_performance': [],
            'metrics': []
        }
        
    def generate_periods(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate the training and testing periods for walk-forward analysis.
        
        Returns:
            List of tuples containing (train_start, train_end, test_start, test_end) dates
        """
        periods = []
        current_start = self.anchor_date
        
        while True:
            train_start = current_start
            train_end = train_start + self.training_window
            
            test_start = train_end + timedelta(days=1)
            test_end = test_start + self.testing_window
            
            # Stop if we've reached the end of the data
            if test_end > self.last_date:
                break
                
            periods.append((train_start, train_end, test_start, test_end))
            
            # Move forward by the step size
            current_start += self.step_size
            
        return periods
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run the walk-forward analysis.
        
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Generate time periods
        periods = self.generate_periods()
        
        if self.verbose:
            print(f"Running walk-forward analysis with {len(periods)} periods")
            
        # For each period, run the analysis
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            period_start_time = time.time()
            
            if self.verbose:
                print(f"\nPeriod {i+1}/{len(periods)}:")
                print(f"  Training: {train_start.date()} to {train_end.date()}")
                print(f"  Testing: {test_start.date()} to {test_end.date()}")
            
            # Filter data for this period
            train_data = self.data[
                (self.data[self.date_column] >= train_start) & 
                (self.data[self.date_column] <= train_end)
            ]
            
            test_data = self.data[
                (self.data[self.date_column] >= test_start) & 
                (self.data[self.date_column] <= test_end)
            ]
            
            # Skip if not enough data
            if len(train_data) < 10 or len(test_data) < 5:
                if self.verbose:
                    print(f"  Skipping period due to insufficient data")
                continue
                
            # Optimize parameters on training data
            if self.verbose:
                print(f"  Optimizing parameters on training data...")
                
            # Create a fitness function that uses the training data
            def fitness_function(params):
                return self.optimization_function(train_data, params, **self.optimization_args)
            
            # Run the optimization
            optimizer = GeneticOptimizer(
                param_bounds=self.param_bounds,
                fitness_function=fitness_function,
                **self.optimization_args
            )
            
            optimization_results = optimizer.optimize(verbose=self.verbose)
            best_params = optimization_results['best_params']
            training_performance = optimization_results['best_fitness']
            
            if self.verbose:
                print(f"  Testing optimized parameters on out-of-sample data...")
                
            # Run a backtest on the test data with the optimized parameters
            testing_performance = self.backtest_function(test_data, best_params)
            
            # Store the results
            self.results['training_periods'].append((train_start, train_end))
            self.results['testing_periods'].append((test_start, test_end))
            self.results['optimized_params'].append(best_params)
            self.results['training_performance'].append(training_performance)
            self.results['testing_performance'].append(testing_performance)
            
            # Calculate any additional metrics
            metrics = {
                'params': best_params,
                'training_score': training_performance,
                'testing_score': testing_performance,
                'score_difference': training_performance - testing_performance,
                'is_profitable': testing_performance > 0
            }
            self.results['metrics'].append(metrics)
            
            period_time = time.time() - period_start_time
            if self.verbose:
                print(f"  Period completed in {period_time:.2f}s")
                print(f"  Training performance: {training_performance:.4f}")
                print(f"  Testing performance: {testing_performance:.4f}")
                
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nWalk-forward analysis completed in {total_time:.2f}s")
            print(f"Total periods analyzed: {len(self.results['training_periods'])}")
            
            # Summary statistics
            if self.results['testing_performance']:
                avg_test_performance = np.mean(self.results['testing_performance'])
                profitable_periods = sum(tp > 0 for tp in self.results['testing_performance'])
                
                print(f"Average out-of-sample performance: {avg_test_performance:.4f}")
                print(f"Profitable periods: {profitable_periods}/{len(self.results['testing_performance'])} "
                      f"({profitable_periods/len(self.results['testing_performance'])*100:.1f}%)")
            
        return self.results
    
    def plot_performance(self, figsize=(12, 8)):
        """
        Plot the results of the walk-forward analysis.
        
        Args:
            figsize: Figure size as a tuple (width, height)
        """
        if not self.results['training_performance']:
            raise ValueError("No analysis results available. Run analyze() first.")
            
        plt.figure(figsize=figsize)
        
        # Convert periods to labels
        period_labels = [f"{i+1}" for i in range(len(self.results['training_performance']))]
        
        # Plot training and testing performance
        width = 0.35
        x = np.arange(len(period_labels))
        
        plt.bar(x - width/2, self.results['training_performance'], width, label='Training (In-Sample)')
        plt.bar(x + width/2, self.results['testing_performance'], width, label='Testing (Out-of-Sample)')
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Period')
        plt.ylabel('Performance')
        plt.title('Walk-Forward Analysis: Training vs Testing Performance')
        plt.xticks(x, period_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_parameters_over_time(self, figsize=(12, 10)):
        """
        Plot how optimized parameters change over time.
        
        Args:
            figsize: Figure size as a tuple (width, height)
        """
        if not self.results['optimized_params']:
            raise ValueError("No analysis results available. Run analyze() first.")
            
        # Extract all parameter names
        all_params = set()
        for params in self.results['optimized_params']:
            all_params.update(params.keys())
            
        # Filter out non-numeric parameters
        numeric_params = []
        for param in all_params:
            is_numeric = all(isinstance(params.get(param, 0), (int, float)) 
                            for params in self.results['optimized_params'])
            if is_numeric:
                numeric_params.append(param)
                
        # Skip if no numeric parameters
        if not numeric_params:
            print("No numeric parameters found to plot.")
            return
            
        # Calculate number of subplots needed
        n_params = len(numeric_params)
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes for easy iteration
        if n_params > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # Period labels
        period_labels = [f"{i+1}" for i in range(len(self.results['training_performance']))]
        
        # Plot each parameter
        for i, param in enumerate(numeric_params):
            ax = axes[i]
            
            # Extract parameter values for each period
            values = [params.get(param, None) for params in self.results['optimized_params']]
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [values[i] for i in valid_indices]
            valid_labels = [period_labels[i] for i in valid_indices]
            
            # Plot
            ax.plot(valid_labels, valid_values, 'o-', markersize=6)
            ax.set_title(f'Parameter: {param}')
            ax.set_xlabel('Period')
            ax.set_xticks(valid_labels)
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best overall parameters based on out-of-sample performance.
        
        Returns:
            Dictionary with the best parameters
        """
        if not self.results['testing_performance']:
            raise ValueError("No analysis results available. Run analyze() first.")
            
        # Find the period with the best out-of-sample performance
        best_idx = np.argmax(self.results['testing_performance'])
        best_params = self.results['optimized_params'][best_idx]
        
        # Get the period information
        train_period = self.results['training_periods'][best_idx]
        test_period = self.results['testing_periods'][best_idx]
        
        return {
            'parameters': best_params,
            'training_period': train_period,
            'testing_period': test_period,
            'training_performance': self.results['training_performance'][best_idx],
            'testing_performance': self.results['testing_performance'][best_idx],
        }
    
    def get_period_statistics(self) -> pd.DataFrame:
        """
        Get detailed statistics for each period.
        
        Returns:
            DataFrame with period statistics
        """
        if not self.results['training_performance']:
            raise ValueError("No analysis results available. Run analyze() first.")
            
        periods = []
        
        for i in range(len(self.results['training_periods'])):
            train_start, train_end = self.results['training_periods'][i]
            test_start, test_end = self.results['testing_periods'][i]
            
            period_data = {
                'period': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'training_performance': self.results['training_performance'][i],
                'testing_performance': self.results['testing_performance'][i],
                'is_profitable': self.results['testing_performance'][i] > 0,
                'overfitting_ratio': self.results['training_performance'][i] / 
                                      max(0.0001, abs(self.results['testing_performance'][i]))
            }
            
            # Add parameters
            for param, value in self.results['optimized_params'][i].items():
                period_data[f'param_{param}'] = value
                
            periods.append(period_data)
            
        return pd.DataFrame(periods)
