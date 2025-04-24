"""
Visualization module for backtest results.

This module provides functions for visualizing backtest results, including
portfolio performance, sentiment data, and trade history.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import json
import logging

logger = logging.getLogger(__name__)

def format_value(value, pos):
    """Format values for y-axis."""
    if value >= 1_000_000:
        return f'${value/1_000_000:.1f}M'
    elif value >= 1_000:
        return f'${value/1_000:.1f}K'
    else:
        return f'${value:.0f}'

def visualize_backtest_results(results_file='backtest_results.json', output_file='backtest_results.png'):
    """
    Visualize backtest results.
    
    Args:
        results_file: Path to the JSON file containing backtest results
        output_file: Path to save the visualization
    """
    try:
        logger.info(f"Visualizing backtest results from {results_file}")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract portfolio history
        portfolio_history = results['portfolio_history']
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        ax1.plot(df.index, df['total_value'], linewidth=2, color='#1f77b4')
        ax1.set_title('Portfolio Performance', fontsize=16)
        ax1.set_ylabel('Portfolio Value', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.yaxis.set_major_formatter(FuncFormatter(format_value))
        
        # Fill the area under the curve
        ax1.fill_between(df.index, df['total_value'], alpha=0.2, color='#1f77b4')
        
        # Calculate drawdowns
        rolling_max = df['total_value'].cummax()
        drawdown = 100 * (df['total_value'] - rolling_max) / rolling_max
        
        # Plot drawdowns
        ax2.fill_between(df.index, drawdown, 0, alpha=0.5, color='r')
        ax2.set_title('Drawdown (%)', fontsize=14)
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add performance metrics as text
        textstr = '\n'.join([
            f"Total Return: {results['total_return']*100:.2f}%",
            f"Annualized Return: {results['annualized_return']*100:.2f}%",
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}",
            f"Max Drawdown: {results['max_drawdown']*100:.2f}%",
            f"Win Rate: {results['win_rate']*100:.2f}%",
            f"Profit Factor: {results['profit_factor']:.2f}",
            f"Total Trades: {results['trade_summary']['total_trades']}",
            f"Winning Trades: {results['trade_summary']['winning_trades']}",
            f"Losing Trades: {results['trade_summary']['losing_trades']}"
        ])
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        fig.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing backtest results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def visualize_sentiment_vs_price(sentiment_data, price_data, symbol, 
                                output_file='sentiment_vs_price.png'):
    """
    Visualize sentiment data against price data for a given symbol.
    
    Args:
        sentiment_data: Dictionary mapping symbols to DataFrames with sentiment data
        price_data: Dictionary mapping symbols to DataFrames with OHLCV data
        symbol: Symbol to visualize
        output_file: Path to save the visualization
    """
    try:
        logger.info(f"Visualizing sentiment vs price for {symbol}")
        
        if symbol not in sentiment_data or symbol not in price_data:
            logger.error(f"Symbol {symbol} not found in data")
            return False
        
        sent_df = sentiment_data[symbol]
        price_df = price_data[symbol]
        
        # Ensure indices are datetime
        if not isinstance(sent_df.index, pd.DatetimeIndex):
            sent_df.index = pd.to_datetime(sent_df.index)
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(price_df.index, price_df['close'], linewidth=2, color='#1f77b4')
        ax1.set_title(f'{symbol} Price vs Sentiment', fontsize=16)
        ax1.set_ylabel('Price', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot sentiment
        ax2.plot(sent_df.index, sent_df['sentiment'], linewidth=2, color='#ff7f0e')
        ax2.axhline(y=0, linestyle='--', color='gray', alpha=0.5)  # Zero line
        ax2.set_title('Sentiment', fontsize=14)
        ax2.set_ylabel('Sentiment Score', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Color the sentiment background based on positive/negative
        ax2.fill_between(sent_df.index, sent_df['sentiment'], 0, 
                        where=(sent_df['sentiment'] >= 0),
                        alpha=0.3, color='green')
        ax2.fill_between(sent_df.index, sent_df['sentiment'], 0,
                        where=(sent_df['sentiment'] < 0),
                        alpha=0.3, color='red')
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        fig.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing sentiment vs price: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def visualize_all_sentiment(sentiment_data, output_file='all_sentiment.png'):
    """
    Visualize sentiment data for all symbols.
    
    Args:
        sentiment_data: Dictionary mapping symbols to DataFrames with sentiment data
        output_file: Path to save the visualization
    """
    try:
        logger.info(f"Visualizing sentiment for all symbols")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot sentiment for each symbol
        for symbol, sent_df in sentiment_data.items():
            # Ensure index is datetime
            if not isinstance(sent_df.index, pd.DatetimeIndex):
                sent_df.index = pd.to_datetime(sent_df.index)
            
            ax.plot(sent_df.index, sent_df['sentiment'], linewidth=1.5, label=symbol)
        
        # Add zero line
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        
        # Add labels and title
        ax.set_title('Sentiment Comparison Across Symbols', fontsize=16)
        ax.set_ylabel('Sentiment Score', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        fig.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing all sentiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False