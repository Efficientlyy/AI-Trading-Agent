"""
Basic Backtest Visualization Module

This module provides basic visualization tools for backtesting results,
including equity curves, drawdowns, and performance metrics.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime, timedelta
import os
import math

class BasicVisualizer:
    """
    Basic visualization tools for backtest results.
    """
    
    def __init__(self, output_dir: str = None, style: str = "darkgrid", 
                figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the basic visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Seaborn style for plots
            figsize: Default figure size
        """
        self.output_dir = output_dir or "."
        self.style = style
        self.figsize = figsize
        
        # Set style
        sns.set(style=style)
        
        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def plot_equity_curve(self, equity_data: pd.DataFrame, benchmark_data: pd.DataFrame = None,
                        title: str = "Equity Curve", filename: str = "equity_curve.png"):
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            equity_data: DataFrame with equity data (index=dates, columns=['equity'])
            benchmark_data: Optional benchmark DataFrame (index=dates, columns=['equity'])
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=self.figsize)
        
        # Plot equity curve
        plt.plot(equity_data.index, equity_data['equity'], 
                label="Strategy", linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            # Align benchmark to equity curve dates
            aligned_benchmark = benchmark_data.reindex(
                equity_data.index, method='ffill'
            )
            
            # Normalize benchmark to same starting value
            if not aligned_benchmark.empty and not equity_data.empty:
                benchmark_scale = equity_data['equity'].iloc[0] / aligned_benchmark['equity'].iloc[0]
                normalized_benchmark = aligned_benchmark['equity'] * benchmark_scale
                
                plt.plot(equity_data.index, normalized_benchmark, 
                        label="Benchmark", linewidth=2, linestyle='--', alpha=0.7)
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Format y-axis with dollar amounts
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_drawdowns(self, drawdown_data: pd.DataFrame, 
                     title: str = "Portfolio Drawdowns", 
                     filename: str = "drawdowns.png"):
        """
        Plot drawdowns over time.
        
        Args:
            drawdown_data: DataFrame with drawdown data (index=dates, columns=['drawdown_pct'])
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=self.figsize)
        
        # Plot drawdowns as negative values
        plt.fill_between(drawdown_data.index, 0, -drawdown_data['drawdown_pct'], 
                       color='red', alpha=0.3)
        plt.plot(drawdown_data.index, -drawdown_data['drawdown_pct'], 
                color='darkred', linewidth=1)
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Format y-axis with percentage
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'{-x:.1f}%')
        )
        
        # Add max drawdown annotation
        if not drawdown_data.empty:
            max_dd = drawdown_data['drawdown_pct'].max()
            max_dd_date = drawdown_data['drawdown_pct'].idxmax()
            
            plt.annotate(f'Max DD: {max_dd:.2f}%',
                       xy=(max_dd_date, -max_dd),
                       xytext=(max_dd_date, -max_dd/2 if max_dd > 5 else -max_dd-5),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                       fontsize=12,
                       ha='center')
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_monthly_returns(self, returns_data: pd.DataFrame,
                           title: str = "Monthly Returns", 
                           filename: str = "monthly_returns.png"):
        """
        Plot monthly returns heatmap.
        
        Args:
            returns_data: DataFrame with return data (index=dates, columns=['return'])
            title: Plot title
            filename: Output filename
        """
        # Convert daily returns to monthly
        if returns_data.empty:
            return
            
        # Ensure datetime index
        returns_data.index = pd.to_datetime(returns_data.index)
        
        # Calculate monthly returns
        monthly_returns = returns_data['return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create a pivot table with years as rows and months as columns
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        monthly_pivot = monthly_pivot.pivot('Year', 'Month', 'Return')
        
        # Plot heatmap
        plt.figure(figsize=self.figsize)
        
        # Create a custom diverging colormap centered at zero
        cmap = sns.diverging_palette(10, 240, as_cmap=True)
        
        # Find the maximum absolute return for symmetric color scaling
        vmax = max(abs(monthly_pivot.min().min()), abs(monthly_pivot.max().max()))
        vmin = -vmax
        
        # Plot heatmap
        ax = sns.heatmap(monthly_pivot, cmap=cmap, vmin=vmin, vmax=vmax,
                      annot=True, fmt='.1%', linewidths=0.5, center=0,
                      cbar_kws={'label': 'Monthly Return'})
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.ylabel('Year', fontsize=12)
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_performance_metrics(self, metrics: Dict[str, Any], 
                               title: str = "Performance Metrics", 
                               filename: str = "performance_metrics.png"):
        """
        Create a visual summary of performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Plot title
            filename: Output filename
        """
        # Select metrics to display
        display_metrics = {
            'Total Return': f"{metrics.get('total_return_pct', 0):.2f}%",
            'Annualized Return': f"{metrics.get('annualized_return_pct', 0):.2f}%",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            'Win Rate': f"{metrics.get('win_rate', 0):.2f}%",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
            'Total Trades': f"{metrics.get('total_trades', 0)}",
        }
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create a horizontal bar chart
        metrics_names = list(display_metrics.keys())
        metrics_values = [float(v.strip('%').strip('$')) for v in display_metrics.values()]
        
        # Create colors based on values (green for positive, red for negative)
        colors = ['green' if (not pd.isna(v) and v > 0) else 'red' for v in metrics_values]
        
        # Plot bars with different y positions
        y_pos = range(len(metrics_names))
        plt.barh(y_pos, metrics_values, align='center', color=colors, alpha=0.7)
        
        # Customize the plot
        plt.yticks(y_pos, metrics_names)
        plt.title(title, fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
        
        # Add values as text
        for i, v in enumerate(display_metrics.values()):
            plt.text(metrics_values[i] * 0.5, i, v, 
                   verticalalignment='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_trade_distribution(self, trades: List[Dict], 
                              title: str = "Trade P&L Distribution", 
                              filename: str = "trade_distribution.png"):
        """
        Plot distribution of trade P&Ls.
        
        Args:
            trades: List of trade dictionaries with 'realized_pnl' key
            title: Plot title
            filename: Output filename
        """
        if not trades:
            return
            
        # Extract P&L values
        pnl_values = [trade.get('realized_pnl', 0) for trade in trades]
        
        plt.figure(figsize=self.figsize)
        
        # Create histogram with KDE
        sns.histplot(pnl_values, kde=True, bins=20, color='skyblue')
        
        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Calculate statistics
        mean_pnl = np.mean(pnl_values)
        median_pnl = np.median(pnl_values)
        
        # Add annotations for mean and median
        plt.axvline(x=mean_pnl, color='green', linestyle='-', alpha=0.7, 
                  label=f'Mean: ${mean_pnl:.2f}')
        plt.axvline(x=median_pnl, color='purple', linestyle='-', alpha=0.7, 
                  label=f'Median: ${median_pnl:.2f}')
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.xlabel('Trade P&L ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis with dollar amounts
        plt.gca().xaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def create_performance_dashboard(self, equity_data: pd.DataFrame, 
                                  drawdown_data: pd.DataFrame,
                                  returns_data: pd.DataFrame,
                                  metrics: Dict[str, Any],
                                  trades: List[Dict] = None,
                                  benchmark_data: pd.DataFrame = None,
                                  title: str = "Performance Dashboard",
                                  filename: str = "performance_dashboard.png"):
        """
        Create a comprehensive performance dashboard with multiple plots.
        
        Args:
            equity_data: DataFrame with equity data (index=dates, columns=['equity'])
            drawdown_data: DataFrame with drawdown data (index=dates, columns=['drawdown_pct'])
            returns_data: DataFrame with return data (index=dates, columns=['return'])
            metrics: Dictionary of performance metrics
            trades: Optional list of trade dictionaries
            benchmark_data: Optional benchmark DataFrame (index=dates, columns=['equity'])
            title: Dashboard title
            filename: Output filename
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        
        # 1. Equity Curve with Benchmark
        ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        equity_data['equity'].plot(ax=ax1, linewidth=2, label='Strategy')
        
        if benchmark_data is not None:
            # Align benchmark to equity curve dates and normalize
            aligned_benchmark = benchmark_data.reindex(
                equity_data.index, method='ffill'
            )
            
            if not aligned_benchmark.empty and not equity_data.empty:
                benchmark_scale = equity_data['equity'].iloc[0] / aligned_benchmark['equity'].iloc[0]
                normalized_benchmark = aligned_benchmark['equity'] * benchmark_scale
                normalized_benchmark.plot(ax=ax1, linewidth=2, linestyle='--', 
                                       label='Benchmark', alpha=0.7)
        
        ax1.set_title('Equity Curve', fontsize=14)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdowns
        ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
        ax2.fill_between(drawdown_data.index, 0, -drawdown_data['drawdown_pct'], 
                       color='red', alpha=0.3)
        ax2.plot(drawdown_data.index, -drawdown_data['drawdown_pct'], 
                color='darkred', linewidth=1)
        
        ax2.set_title('Drawdowns', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{-x:.1f}%'))
        
        # 3. Monthly Returns Heatmap
        ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
        
        # Calculate monthly returns for heatmap
        if not returns_data.empty:
            returns_data.index = pd.to_datetime(returns_data.index)
            monthly_returns = returns_data['return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            monthly_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            monthly_pivot = monthly_pivot.pivot('Year', 'Month', 'Return')
            
            # Create a custom diverging colormap centered at zero
            cmap = sns.diverging_palette(10, 240, as_cmap=True)
            
            # Find the maximum absolute return for symmetric color scaling
            vmax = max(abs(monthly_pivot.min().min()), abs(monthly_pivot.max().max()))
            vmin = -vmax
            
            # Plot heatmap
            sns.heatmap(monthly_pivot, cmap=cmap, vmin=vmin, vmax=vmax,
                      annot=True, fmt='.1%', linewidths=0.5, center=0,
                      cbar_kws={'label': 'Monthly Return'}, ax=ax3)
            
            # Set month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax3.set_xticklabels(month_names)
            
        ax3.set_title('Monthly Returns', fontsize=14)
        ax3.set_ylabel('Year', fontsize=12)
        
        # 4. Performance Metrics
        ax4 = plt.subplot2grid((4, 2), (3, 0))
        
        # Select metrics to display
        display_metrics = {
            'Total Return': f"{metrics.get('total_return_pct', 0):.2f}%",
            'Annualized Return': f"{metrics.get('annualized_return_pct', 0):.2f}%",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            'Win Rate': f"{metrics.get('win_rate', 0):.2f}%",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
            'Total Trades': f"{metrics.get('total_trades', 0)}",
        }
        
        # Create a horizontal bar chart
        metrics_names = list(display_metrics.keys())
        metrics_values = [float(v.strip('%').strip('$')) for v in display_metrics.values()]
        
        # Create colors based on values (green for positive, red for negative)
        colors = ['green' if (not pd.isna(v) and v > 0) else 'red' for v in metrics_values]
        
        # Plot bars with different y positions
        y_pos = range(len(metrics_names))
        ax4.barh(y_pos, metrics_values, align='center', color=colors, alpha=0.7)
        
        # Customize the plot
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(metrics_names)
        ax4.set_title('Performance Metrics', fontsize=14)
        ax4.grid(True, axis='x', alpha=0.3)
        
        # Add values as text
        for i, v in enumerate(display_metrics.values()):
            ax4.text(metrics_values[i] * 0.5, i, v, 
                   verticalalignment='center', fontsize=10)
        
        # 5. Trade P&L Distribution
        ax5 = plt.subplot2grid((4, 2), (3, 1))
        
        if trades:
            # Extract P&L values
            pnl_values = [trade.get('realized_pnl', 0) for trade in trades]
            
            # Create histogram with KDE
            sns.histplot(pnl_values, kde=True, bins=20, color='skyblue', ax=ax5)
            
            # Add vertical line at zero
            ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Calculate statistics
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)
            
            # Add annotations for mean and median
            ax5.axvline(x=mean_pnl, color='green', linestyle='-', alpha=0.7, 
                      label=f'Mean: ${mean_pnl:.2f}')
            ax5.axvline(x=median_pnl, color='purple', linestyle='-', alpha=0.7, 
                      label=f'Median: ${median_pnl:.2f}')
            
            ax5.legend()
            ax5.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
        ax5.set_title('Trade P&L Distribution', fontsize=14)
        ax5.set_xlabel('Trade P&L ($)', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Overall dashboard title
        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
