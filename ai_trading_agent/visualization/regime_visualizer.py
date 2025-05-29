"""
Regime Visualization Module

This module provides visualization tools specifically for market regime transitions
and parameter adaptations during backtesting.
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

class RegimeVisualizer:
    """
    Visualization tools for market regime transitions and parameter adaptations.
    """
    
    def __init__(self, output_dir: str = None, style: str = "darkgrid", 
                figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the regime visualizer.
        
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
            
    def plot_regime_transitions(self, regime_data: pd.DataFrame, price_data: pd.DataFrame = None,
                              title: str = "Market Regime Transitions", 
                              filename: str = "regime_transitions.png"):
        """
        Visualize market regime transitions with price overlay.
        
        Args:
            regime_data: DataFrame with regime data (index=dates, columns=['regime'])
            price_data: Optional price DataFrame for overlay (index=dates, columns=['close'])
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=self.figsize)
        
        # Set up axes
        ax1 = plt.gca()
        
        # Plot price data if provided
        if price_data is not None:
            ax1.plot(price_data.index, price_data['close'], color='gray', alpha=0.5, linewidth=1)
            ax1.set_ylabel('Price', fontsize=12)
            
            # Create second y-axis for regime
            ax2 = ax1.twinx()
        else:
            ax2 = ax1
            
        # Plot regime transitions
        unique_regimes = regime_data['regime'].unique()
        regime_colors = {
            'bull': 'green',
            'bear': 'red',
            'neutral': 'blue',
            'volatile': 'orange',
            'trending': 'purple',
            'ranging': 'cyan',
            'high_volatility': 'magenta',
            'low_volatility': 'teal'
        }
        
        # Ensure all regimes have a color
        for regime in unique_regimes:
            if regime.lower() not in regime_colors:
                regime_colors[regime.lower()] = 'gray'
        
        # Create numeric mapping for regimes
        regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
        regime_data['regime_numeric'] = regime_data['regime'].map(regime_map)
        
        # Plot each regime segment
        prev_date = None
        prev_regime = None
        
        for date, row in regime_data.iterrows():
            if prev_date is not None and prev_regime is not None:
                color = regime_colors.get(prev_regime.lower(), 'gray')
                ax2.axvspan(prev_date, date, alpha=0.2, color=color)
                
            prev_date = date
            prev_regime = row['regime']
            
        # Plot regime line
        ax2.plot(regime_data.index, regime_data['regime_numeric'], 'k-', linewidth=2)
        
        # Add regime labels
        ax2.set_yticks(range(len(unique_regimes)))
        ax2.set_yticklabels(unique_regimes)
        ax2.set_ylabel('Market Regime', fontsize=12)
        
        # Add regime transitions as vertical lines
        regime_changes = regime_data['regime'].ne(regime_data['regime'].shift()).fillna(False)
        transition_dates = regime_data.index[regime_changes]
        
        for date in transition_dates:
            ax1.axvline(x=date, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Annotate regime change
            if date in regime_data.index:
                regime = regime_data.loc[date, 'regime']
                ax1.annotate(f"{regime}", 
                           xy=(date, ax1.get_ylim()[1]),
                           xytext=(date, ax1.get_ylim()[1] * 1.05),
                           rotation=90,
                           ha='center',
                           fontsize=8)
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_parameter_adaptations(self, parameter_data: pd.DataFrame, price_data: pd.DataFrame = None,
                                 title: str = "Strategy Parameter Adaptations", 
                                 filename: str = "parameter_adaptations.png"):
        """
        Visualize strategy parameter adaptations over time.
        
        Args:
            parameter_data: DataFrame with parameter data (index=dates, columns=parameters)
            price_data: Optional price DataFrame for overlay (index=dates, columns=['close'])
            title: Plot title
            filename: Output filename
        """
        # Skip if no data
        if parameter_data.empty:
            return
            
        # Determine how many parameters to plot
        parameters = parameter_data.columns.tolist()
        num_params = len(parameters)
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_params + (1 if price_data is not None else 0), 1, 
                               figsize=(self.figsize[0], self.figsize[1] * 1.5), 
                               sharex=True)
        
        # Plot price data if provided
        if price_data is not None:
            ax_price = axes[0]
            ax_price.plot(price_data.index, price_data['close'], color='black', linewidth=1.5)
            ax_price.set_ylabel('Price', fontsize=12)
            ax_price.grid(True, alpha=0.3)
            ax_price.set_title('Price Chart', fontsize=12)
            param_axes = axes[1:]
        else:
            param_axes = axes
            
        # Ensure param_axes is iterable even with single parameter
        if num_params == 1:
            param_axes = [param_axes]
            
        # Plot each parameter
        for i, param in enumerate(parameters):
            ax = param_axes[i]
            ax.plot(parameter_data.index, parameter_data[param], linewidth=2)
            ax.set_ylabel(param, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line for default/baseline value if available
            # This assumes the first value is the baseline
            if not parameter_data[param].empty:
                baseline = parameter_data[param].iloc[0]
                ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5)
                ax.annotate(f"Baseline: {baseline:.4f}", 
                          xy=(parameter_data.index[0], baseline),
                          xytext=(parameter_data.index[0], baseline * 1.1),
                          fontsize=8)
                
            # Highlight significant changes
            param_changes = parameter_data[param].pct_change().abs()
            significant_changes = param_changes > 0.1  # 10% change threshold
            
            for date in parameter_data.index[significant_changes]:
                value = parameter_data.loc[date, param]
                ax.scatter(date, value, color='red', s=50, zorder=5)
                
                # Annotate significant change
                pct_change = param_changes.loc[date] * 100
                ax.annotate(f"{pct_change:.1f}%", 
                          xy=(date, value),
                          xytext=(date, value * 1.1),
                          fontsize=8,
                          ha='center')
        
        # Format plot
        plt.suptitle(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_risk_adjustments(self, risk_data: pd.DataFrame, equity_data: pd.DataFrame = None,
                           drawdown_data: pd.DataFrame = None,
                           title: str = "Risk Management Adjustments", 
                           filename: str = "risk_adjustments.png"):
        """
        Visualize risk management adjustments with equity and drawdown overlay.
        
        Args:
            risk_data: DataFrame with risk adjustment data
            equity_data: Optional equity curve DataFrame
            drawdown_data: Optional drawdown DataFrame
            title: Plot title
            filename: Output filename
        """
        # Skip if no data
        if risk_data.empty:
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), 
                               sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Plot equity curve if provided
        ax1 = axes[0]
        if equity_data is not None:
            ax1.plot(equity_data.index, equity_data['equity'], color='blue', linewidth=1.5)
            ax1.set_ylabel('Equity ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Equity Curve', fontsize=12)
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
        # 2. Plot drawdowns if provided
        ax2 = axes[1]
        if drawdown_data is not None:
            ax2.fill_between(drawdown_data.index, 0, -drawdown_data['drawdown_pct'], 
                           color='red', alpha=0.3)
            ax2.plot(drawdown_data.index, -drawdown_data['drawdown_pct'], 
                    color='darkred', linewidth=1)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Drawdowns', fontsize=12)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{-x:.1f}%'))
            
        # 3. Plot risk adjustments
        ax3 = axes[2]
        
        # Plot total risk adjustment
        if 'total_adjustment' in risk_data.columns:
            ax3.plot(risk_data.index, risk_data['total_adjustment'], 
                   color='purple', linewidth=2, label='Total Adjustment')
            
        # Plot individual components if available
        if 'drawdown_reduction' in risk_data.columns:
            ax3.plot(risk_data.index, risk_data['drawdown_reduction'], 
                   color='red', linewidth=1.5, linestyle='--', 
                   label='Drawdown Reduction', alpha=0.7)
            
        if 'volatility_adjustment' in risk_data.columns:
            # Convert to percentage adjustment from factor
            vol_adjustment_pct = (risk_data['volatility_adjustment'] - 1) * 100
            ax3.plot(risk_data.index, vol_adjustment_pct, 
                   color='orange', linewidth=1.5, linestyle='--', 
                   label='Volatility Adjustment', alpha=0.7)
            
        if 'streak_adjustment' in risk_data.columns:
            ax3.plot(risk_data.index, risk_data['streak_adjustment'], 
                   color='green', linewidth=1.5, linestyle='--', 
                   label='Streak Adjustment', alpha=0.7)
            
        ax3.set_ylabel('Adjustment (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Risk Adjustments', fontsize=12)
        ax3.legend(loc='best')
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add overall title
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def plot_position_scaling(self, position_data: pd.DataFrame, price_data: pd.DataFrame,
                           title: str = "Position Scaling Analysis", 
                           filename: str = "position_scaling.png"):
        """
        Visualize position scaling (entries and exits) with price overlay.
        
        Args:
            position_data: DataFrame with position data (entries and exits)
            price_data: Price DataFrame for overlay
            title: Plot title
            filename: Output filename
        """
        # Skip if no data
        if position_data.empty or price_data.empty:
            return
            
        plt.figure(figsize=self.figsize)
        
        # Plot price
        plt.plot(price_data.index, price_data['close'], color='black', linewidth=1.5, alpha=0.7)
        
        # Extract entries and exits
        entries = position_data[position_data['type'] == 'entry']
        exits = position_data[position_data['type'] == 'exit']
        
        # Plot entries as green triangles
        if not entries.empty:
            plt.scatter(entries.index, entries['price'], 
                      marker='^', color='green', s=100, label='Entries', zorder=5)
            
            # Annotate entry sizes
            for date, row in entries.iterrows():
                plt.annotate(f"{row['quantity']:.2f}", 
                           xy=(date, row['price']),
                           xytext=(date, row['price'] * 1.02),
                           fontsize=8,
                           ha='center')
        
        # Plot exits as red triangles
        if not exits.empty:
            plt.scatter(exits.index, exits['price'], 
                      marker='v', color='red', s=100, label='Exits', zorder=5)
            
            # Annotate exit sizes
            for date, row in exits.iterrows():
                plt.annotate(f"{row['quantity']:.2f}", 
                           xy=(date, row['price']),
                           xytext=(date, row['price'] * 0.98),
                           fontsize=8,
                           ha='center')
        
        # Format plot
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
        
    def create_regime_dashboard(self, price_data: pd.DataFrame, regime_data: pd.DataFrame,
                             parameter_data: pd.DataFrame, risk_data: pd.DataFrame,
                             position_data: pd.DataFrame = None,
                             title: str = "Regime and Adaptation Dashboard",
                             filename: str = "regime_dashboard.png"):
        """
        Create a comprehensive dashboard for regime transitions and adaptations.
        
        Args:
            price_data: Price DataFrame
            regime_data: Regime transition DataFrame
            parameter_data: Parameter adaptation DataFrame
            risk_data: Risk adjustment DataFrame
            position_data: Optional position scaling DataFrame
            title: Dashboard title
            filename: Output filename
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        
        # 1. Price and Regime
        ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        
        # Plot price
        ax1.plot(price_data.index, price_data['close'], color='black', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Price', fontsize=12)
        
        # Create second y-axis for regime
        ax1b = ax1.twinx()
        
        # Plot regime transitions if data available
        if not regime_data.empty:
            unique_regimes = regime_data['regime'].unique()
            regime_colors = {
                'bull': 'green',
                'bear': 'red',
                'neutral': 'blue',
                'volatile': 'orange',
                'trending': 'purple',
                'ranging': 'cyan',
                'high_volatility': 'magenta',
                'low_volatility': 'teal'
            }
            
            # Create numeric mapping for regimes
            regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
            regime_data['regime_numeric'] = regime_data['regime'].map(regime_map)
            
            # Plot regime line
            ax1b.plot(regime_data.index, regime_data['regime_numeric'], 'k-', linewidth=2)
            
            # Add regime labels
            ax1b.set_yticks(range(len(unique_regimes)))
            ax1b.set_yticklabels(unique_regimes)
            ax1b.set_ylabel('Market Regime', fontsize=12)
            
            # Add regime transitions as vertical lines
            regime_changes = regime_data['regime'].ne(regime_data['regime'].shift()).fillna(False)
            transition_dates = regime_data.index[regime_changes]
            
            for date in transition_dates:
                ax1.axvline(x=date, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_title('Price and Market Regime', fontsize=14)
        
        # 2. Parameter Adaptations
        ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
        
        # Plot parameter adaptations if data available
        if not parameter_data.empty:
            # Plot up to 3 most important parameters
            important_params = parameter_data.columns[:min(3, len(parameter_data.columns))]
            
            for param in important_params:
                ax2.plot(parameter_data.index, parameter_data[param], linewidth=2, label=param)
                
            ax2.legend(loc='best')
            
        ax2.set_title('Strategy Parameter Adaptations', fontsize=14)
        ax2.set_ylabel('Parameter Value', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk Adjustments
        ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
        
        # Plot risk adjustments if data available
        if not risk_data.empty:
            # Plot total risk adjustment
            if 'total_adjustment' in risk_data.columns:
                ax3.plot(risk_data.index, risk_data['total_adjustment'], 
                       color='purple', linewidth=2, label='Total Adjustment')
                
            # Plot individual components if available
            if 'drawdown_reduction' in risk_data.columns:
                ax3.plot(risk_data.index, risk_data['drawdown_reduction'], 
                       color='red', linewidth=1.5, linestyle='--', 
                       label='Drawdown Reduction', alpha=0.7)
                
            if 'volatility_adjustment' in risk_data.columns:
                # Convert to percentage adjustment from factor
                vol_adjustment_pct = (risk_data['volatility_adjustment'] - 1) * 100
                ax3.plot(risk_data.index, vol_adjustment_pct, 
                       color='orange', linewidth=1.5, linestyle='--', 
                       label='Volatility Adjustment', alpha=0.7)
                
            if 'streak_adjustment' in risk_data.columns:
                ax3.plot(risk_data.index, risk_data['streak_adjustment'], 
                       color='green', linewidth=1.5, linestyle='--', 
                       label='Streak Adjustment', alpha=0.7)
                
            ax3.legend(loc='best')
            
        ax3.set_title('Risk Management Adjustments', fontsize=14)
        ax3.set_ylabel('Adjustment (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Position Scaling (Left)
        ax4 = plt.subplot2grid((4, 2), (3, 0))
        
        # Plot position scaling if data available
        if position_data is not None and not position_data.empty:
            # Plot price
            ax4.plot(price_data.index, price_data['close'], color='black', linewidth=1.5, alpha=0.7)
            
            # Extract entries and exits
            entries = position_data[position_data['type'] == 'entry']
            exits = position_data[position_data['type'] == 'exit']
            
            # Plot entries as green triangles
            if not entries.empty:
                ax4.scatter(entries.index, entries['price'], 
                          marker='^', color='green', s=100, label='Entries', zorder=5)
                
            # Plot exits as red triangles
            if not exits.empty:
                ax4.scatter(exits.index, exits['price'], 
                          marker='v', color='red', s=100, label='Exits', zorder=5)
                
            ax4.legend(loc='best')
            
        ax4.set_title('Position Scaling', fontsize=14)
        ax4.set_ylabel('Price', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. Position Size vs. Risk (Right)
        ax5 = plt.subplot2grid((4, 2), (3, 1))
        
        # Plot position size vs. risk if data available
        if position_data is not None and not position_data.empty and 'risk_pct' in position_data.columns:
            entries = position_data[position_data['type'] == 'entry']
            
            if not entries.empty:
                ax5.scatter(entries['risk_pct'], entries['quantity'], 
                          marker='o', color='blue', s=80, alpha=0.7)
                
                # Add trend line
                if len(entries) > 1:
                    z = np.polyfit(entries['risk_pct'], entries['quantity'], 1)
                    p = np.poly1d(z)
                    ax5.plot(entries['risk_pct'], p(entries['risk_pct']), 
                           'r--', linewidth=1.5, alpha=0.7)
                    
                # Add annotations
                for _, row in entries.iterrows():
                    ax5.annotate(row.get('date', '').strftime('%Y-%m-%d') if isinstance(row.get('date'), datetime) else '',
                               xy=(row['risk_pct'], row['quantity']),
                               xytext=(row['risk_pct'], row['quantity'] * 1.05),
                               fontsize=8,
                               ha='center')
                    
        ax5.set_title('Position Size vs. Risk', fontsize=14)
        ax5.set_xlabel('Risk Percentage (%)', fontsize=12)
        ax5.set_ylabel('Position Size', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Format x-axis dates for all date-based subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.tick_params(axis='x', rotation=45)
        
        # Overall dashboard title
        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save if output directory specified
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            
        plt.close()
