"""Visualization utilities for backtesting results.

This module provides functions for creating visualizations of backtesting
results, including equity curves, drawdowns, regime performance charts,
and trade analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import seaborn as sns
from ..detection.base_detector import plot_regimes
import io
import os
import base64
import copy


def plot_equity_curve(
    dates: List[datetime],
    equity: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    regime_colors: Optional[Dict[int, str]] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    xlabel: str = "Date",
    ylabel: str = "Equity ($)",
    legend_loc: str = "upper left",
    save_path: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the equity curve with optional benchmark and regime background.
    
    Args:
        dates: List of dates
        equity: Array of equity values
        benchmark: Optional array of benchmark equity values
        regimes: Optional array of regime labels
        regime_colors: Optional dictionary mapping regime labels to colors
        title: Plot title
        figsize: Figure size (width, height) in inches
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        legend_loc: Location of the legend
        save_path: Optional path to save the figure
        show: Whether to show the figure
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert dates for plotting
    if not isinstance(dates[0], datetime):
        dates = [datetime.fromisoformat(d) if isinstance(d, str) else 
                datetime.fromtimestamp(d) if isinstance(d, (int, float)) else d
                for d in dates]
    
    # Plot regimes as background if provided
    if regimes is not None:
        plot_regimes(dates, regimes, ax=ax, colors=regime_colors, alpha=0.15)
    
    # Plot equity curve
    ax.plot(dates, equity, label="Strategy", linewidth=2, color="blue")
    
    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(dates, benchmark, label="Benchmark", linewidth=2, color="gray", alpha=0.7)
    
    # Configure axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    
    # Add legend
    if benchmark is not None or regimes is not None:
        ax.legend(loc=legend_loc)
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def plot_drawdown(
    dates: List[datetime],
    drawdowns: np.ndarray,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (12, 4),
    xlabel: str = "Date",
    ylabel: str = "Drawdown (%)",
    color: str = "red",
    fill: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot the drawdown curve.
    
    Args:
        dates: List of dates
        drawdowns: Array of drawdown values (negative percentages)
        title: Plot title
        figsize: Figure size (width, height) in inches
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        color: Color of the drawdown line/fill
        fill: Whether to fill the area under the curve
        save_path: Optional path to save the figure
        show: Whether to show the figure
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert dates for plotting if needed
    if not isinstance(dates[0], datetime):
        dates = [datetime.fromisoformat(d) if isinstance(d, str) else 
                datetime.fromtimestamp(d) if isinstance(d, (int, float)) else d
                for d in dates]
    
    # Plot drawdowns
    if fill:
        ax.fill_between(dates, 0, drawdowns * 100, color=color, alpha=0.3)
    ax.plot(dates, drawdowns * 100, color=color, linewidth=1.5)
    
    # Configure axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def plot_regime_performance(
    regime_metrics: Dict[int, Dict[str, float]],
    regime_names: Optional[Dict[int, str]] = None,
    metrics_to_plot: List[str] = ['total_return', 'sharpe', 'max_drawdown', 'duration_pct'],
    title: str = "Performance by Regime",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot performance metrics by regime.
    
    Args:
        regime_metrics: Dictionary of performance metrics for each regime
        regime_names: Optional dictionary mapping regime IDs to names
        metrics_to_plot: List of metrics to plot
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    # Create a DataFrame from regime metrics
    data = []
    for regime_id, metrics in regime_metrics.items():
        regime_name = regime_names.get(regime_id, f"Regime {regime_id}") if regime_names else f"Regime {regime_id}"
        row = {"Regime": regime_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    df.set_index("Regime", inplace=True)
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    if n_metrics <= 2:
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
                axes[i].set_title(f"{metric.replace('_', ' ').title()}")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f"Metric '{metric}' not found", 
                            horizontalalignment='center', verticalalignment='center')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def plot_trade_analysis(
    trades: List[Dict[str, Any]],
    title: str = "Trade Analysis",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot trade analysis visualizations.
    
    Args:
        trades: List of trade dictionaries
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    if not trades:
        # Return empty figure if no trades
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No trades to analyze", 
                horizontalalignment='center', verticalalignment='center')
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig
    
    # Create dataframe from trades
    df = pd.DataFrame(trades)
    
    # Convert entry and exit times to datetime if they are strings
    for col in ['entry_time', 'exit_time']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col])
    
    # Calculate holding period in days if not already present
    if 'holding_period' not in df.columns and 'entry_time' in df.columns and 'exit_time' in df.columns:
        df['holding_period'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / (60 * 60 * 24)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    
    # 1. P&L Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'pnl' in df.columns:
        sns.histplot(df['pnl'], kde=True, ax=ax1)
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_title("P&L Distribution")
        ax1.set_xlabel("Profit/Loss")
    else:
        ax1.text(0.5, 0.5, "P&L data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # 2. Win/Loss Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    if 'pnl' in df.columns:
        wins = (df['pnl'] > 0).sum()
        losses = (df['pnl'] < 0).sum()
        ax2.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90)
        ax2.set_title(f"Win/Loss Ratio: {wins}/{losses}")
    else:
        ax2.text(0.5, 0.5, "P&L data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # 3. P&L Over Time
    ax3 = fig.add_subplot(gs[1, :])
    if 'exit_time' in df.columns and 'pnl' in df.columns:
        df_sorted = df.sort_values('exit_time')
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        ax3.plot(df_sorted['exit_time'], df_sorted['cumulative_pnl'], marker='o', markersize=3)
        ax3.set_title("Cumulative P&L Over Time")
        ax3.set_xlabel("Exit Time")
        ax3.set_ylabel("Cumulative P&L")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Time series P&L data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # 4. Holding Period vs. P&L
    ax4 = fig.add_subplot(gs[2, 0])
    if 'holding_period' in df.columns and 'pnl' in df.columns:
        ax4.scatter(df['holding_period'], df['pnl'], alpha=0.6)
        ax4.set_title("Holding Period vs. P&L")
        ax4.set_xlabel("Holding Period (days)")
        ax4.set_ylabel("P&L")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Holding period or P&L data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # 5. Average P&L by Trade Type
    ax5 = fig.add_subplot(gs[2, 1])
    if 'type' in df.columns and 'pnl' in df.columns:
        average_pnl = df.groupby('type')['pnl'].mean()
        average_pnl.plot(kind='bar', ax=ax5, color=['blue', 'orange'])
        ax5.set_title("Average P&L by Trade Type")
        ax5.set_xlabel("Trade Type")
        ax5.set_ylabel("Average P&L")
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Trade type or P&L data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def plot_returns_heatmap(
    returns: np.ndarray,
    dates: List[datetime],
    title: str = "Returns Heatmap",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn",
    annot: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a calendar heatmap of returns.
    
    Args:
        returns: Array of daily returns
        dates: List of dates corresponding to returns
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        annot: Whether to annotate heatmap cells with values
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to pandas Series
    series = pd.Series(returns, index=pd.DatetimeIndex(dates))
    
    # Create DataFrame with months as columns and days as rows
    df = series.resample('D').mean()
    
    # Create a pivot table: year and month as index, day as columns
    df_pivot = pd.DataFrame({
        'year': df.index.year,
        'month': df.index.month,
        'day': df.index.day,
        'returns': df.values
    })
    
    monthly_returns = df_pivot.pivot_table(
        index=['year', 'month'], 
        columns='day', 
        values='returns'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(monthly_returns, cmap=cmap, ax=ax, 
                center=0, annot=annot, fmt=".1%")
    
    # Configure axis
    ax.set_title(title)
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Year-Month")
    
    # Format y-axis labels
    y_labels = [f"{year}-{month:02d}" for year, month in monthly_returns.index]
    ax.set_yticklabels(y_labels)
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def create_comprehensive_report(
    dates: List[datetime],
    equity: np.ndarray,
    returns: np.ndarray,
    trades: List[Dict[str, Any]],
    performance_metrics: Dict[str, Any],
    benchmark: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    regime_metrics: Optional[Dict[int, Dict[str, float]]] = None,
    regime_colors: Optional[Dict[int, str]] = None,
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Backtest Results",
    figsize: Tuple[int, int] = (15, 20),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive visual report of backtest results.
    
    Args:
        dates: List of dates
        equity: Array of equity values
        returns: Array of returns
        trades: List of trade dictionaries
        performance_metrics: Dictionary of performance metrics
        benchmark: Optional array of benchmark equity values
        regimes: Optional array of regime labels
        regime_metrics: Optional dictionary of performance metrics for each regime
        regime_colors: Optional dictionary mapping regime labels to colors
        regime_names: Optional dictionary mapping regime IDs to names
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    # Calculate drawdowns if not provided in metrics
    if 'drawdowns' in performance_metrics:
        drawdowns = performance_metrics['drawdowns']
    else:
        from .performance_metrics import calculate_drawdowns
        drawdowns, _, _ = calculate_drawdowns(returns)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 3, 3, 2])
    
    # 1. Equity Curve
    ax_equity = fig.add_subplot(gs[0, :])
    plot_equity_curve(
        dates, equity, benchmark, regimes, regime_colors,
        title="Equity Curve", figsize=None, show=False, ax=ax_equity
    )
    
    # 2. Drawdown
    ax_dd = fig.add_subplot(gs[1, :])
    plot_drawdown(
        dates, drawdowns, title="Drawdown", figsize=None, show=False, ax=ax_dd
    )
    
    # 3. Trade Analysis
    if trades:
        # Use the trade analysis function but don't show it
        trade_fig = plot_trade_analysis(trades, show=False)
        
        # Copy the relevant subplots to our main figure
        for i, ax in enumerate(trade_fig.axes[:4]):  # Take the first 4 subplots
            if i < 2:
                new_ax = fig.add_subplot(gs[2, i])
            else:
                new_ax = fig.add_subplot(gs[3, i-2])
            
            # Copy the contents of the subplot
            for item in ax.get_children():
                try:
                    new_item = copy.copy(item)
                    new_ax.add_artist(new_item)
                except:
                    pass
            
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
        
        plt.close(trade_fig)
    else:
        # If no trades, show a message
        ax_no_trades = fig.add_subplot(gs[2:4, :])
        ax_no_trades.text(0.5, 0.5, "No trades to analyze", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14)
        ax_no_trades.axis('off')
    
    # 4. Summary Metrics
    ax_metrics = fig.add_subplot(gs[4, :])
    ax_metrics.axis('off')
    
    # Create a formatted string of metrics
    metrics_text = "Performance Metrics:\n"
    important_metrics = [
        ('total_return', 'Total Return', '{:.2%}'),
        ('annual_return', 'Annual Return', '{:.2%}'),
        ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
        ('sortino_ratio', 'Sortino Ratio', '{:.2f}'),
        ('max_drawdown', 'Max Drawdown', '{:.2%}'),
        ('win_rate', 'Win Rate', '{:.2%}'),
        ('profit_factor', 'Profit Factor', '{:.2f}'),
        ('num_trades', 'Number of Trades', '{:.0f}')
    ]
    
    for key, label, fmt in important_metrics:
        if key in performance_metrics:
            value = performance_metrics[key]
            metrics_text += f"{label}: {fmt.format(value)}    "
    
    # Add text to the axis
    ax_metrics.text(0.5, 0.5, metrics_text, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, fontweight='bold')
    
    # Set overall title
    fig.suptitle(title, fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def save_html_report(
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
    figures: Dict[str, plt.Figure],
    output_path: str
) -> str:
    """
    Generate and save an HTML backtest report.
    
    Args:
        metrics: Dictionary of performance metrics
        trades: List of trade dictionaries
        figures: Dictionary of matplotlib figures
        output_path: Path to save the HTML report
        
    Returns:
        Path to the saved HTML report
    """
    # Convert figures to base64-encoded PNGs
    figure_imgs = {}
    for name, fig in figures.items():
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        figure_imgs[name] = f'data:image/png;base64,{img_data}'
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .metric-card {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
            .chart {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Results</h1>
            
            <div class="metrics">
                <div class="metric-card">
                    <div>Total Return</div>
                    <div class="metric-value">{metrics.get('total_return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div>Annual Return</div>
                    <div class="metric-value">{metrics.get('annual_return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div>Sharpe Ratio</div>
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div>Max Drawdown</div>
                    <div class="metric-value">{metrics.get('max_drawdown', 0):.2%}</div>
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div>Win Rate</div>
                    <div class="metric-value">{metrics.get('win_rate', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div>Profit Factor</div>
                    <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div>Number of Trades</div>
                    <div class="metric-value">{metrics.get('num_trades', 0)}</div>
                </div>
                <div class="metric-card">
                    <div>Expectancy</div>
                    <div class="metric-value">${metrics.get('expectancy', 0):.2f}</div>
                </div>
            </div>
            
    """
    
    # Add figures
    for name, img_data in figure_imgs.items():
        html_content += f"""
            <div class="chart">
                <h2>{name}</h2>
                <img src="{img_data}" style="max-width: 100%;" />
            </div>
        """
    
    # Add trade table (limit to 50 most recent trades)
    html_content += """
            <h2>Recent Trades</h2>
            <table>
                <tr>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Type</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Return</th>
                </tr>
    """
    
    for trade in trades[-50:]:
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        trade_type = trade.get('type', '')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('pnl', 0)
        return_pct = trade.get('return_pct', 0)
        
        html_content += f"""
                <tr>
                    <td>{entry_time}</td>
                    <td>{exit_time}</td>
                    <td>{trade_type}</td>
                    <td>${entry_price:.2f}</td>
                    <td>${exit_price:.2f}</td>
                    <td>${pnl:.2f}</td>
                    <td>{return_pct:.2%}</td>
                </tr>
        """
    
    # Close HTML
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path 