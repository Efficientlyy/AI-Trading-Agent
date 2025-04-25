"""
Sentiment Visualization Module

This module provides functions for visualizing sentiment data from Alpha Vantage
in a format compatible with the existing dashboard.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta
import os

from ..sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector

logger = logging.getLogger(__name__)

def visualize_sentiment_trends(sentiment_data: pd.DataFrame, symbol: str, output_file: str) -> None:
    """
    Visualize sentiment trends for a specific symbol.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        symbol: Symbol to visualize
        output_file: Output file path for the visualization
    """
    if sentiment_data.empty:
        logger.warning(f"No sentiment data to visualize for {symbol}")
        # Create a placeholder image with a message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"No sentiment data available for {symbol}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    # Determine sentiment column
    if 'ticker_sentiment_score' in sentiment_data.columns:
        sentiment_column = 'ticker_sentiment_score'
    else:
        sentiment_column = 'overall_sentiment_score'
    
    # Plot sentiment trends
    plt.figure(figsize=(12, 8))
    
    # Plot raw sentiment
    plt.plot(sentiment_data.index, sentiment_data[sentiment_column], 'o-', alpha=0.5, label='Raw Sentiment')
    
    # Plot rolling average if we have enough data
    if len(sentiment_data) >= 3:
        rolling_avg = sentiment_data[sentiment_column].rolling(window=3).mean()
        plt.plot(sentiment_data.index, rolling_avg, 'r-', linewidth=2, label='3-period Rolling Average')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=0.2, color='g', linestyle='--', alpha=0.7, label='Buy Threshold')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7, label='Sell Threshold')
    
    # Formatting
    plt.title(f"Sentiment Trends for {symbol}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to ensure -1 to 1 range is visible
    y_min, y_max = plt.ylim()
    plt.ylim(min(y_min, -1.0), max(y_max, 1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    logger.info(f"Sentiment trends visualization saved to {output_file}")

def visualize_multi_asset_sentiment(symbols: List[str], days_back: int = 30, 
                                   output_file: str = "multi_asset_sentiment.png") -> None:
    """
    Visualize sentiment for multiple assets together.
    
    Args:
        symbols: List of symbols to visualize
        days_back: Number of days to look back
        output_file: Output file path for the visualization
    """
    connector = AlphaVantageSentimentConnector()
    
    plt.figure(figsize=(14, 8))
    
    for symbol in symbols:
        # Get sentiment data
        df = connector.get_sentiment_for_symbol(symbol, days_back)
        
        if df.empty:
            logger.warning(f"No sentiment data for {symbol}, skipping")
            continue
        
        # Determine sentiment column
        if 'ticker_sentiment_score' in df.columns:
            sentiment_column = 'ticker_sentiment_score'
        else:
            sentiment_column = 'overall_sentiment_score'
        
        # Calculate rolling average to smooth the visualization
        if len(df) >= 3:
            sentiment_values = df[sentiment_column].rolling(window=3).mean()
            # Fill NaN values at the beginning
            sentiment_values = sentiment_values.fillna(df[sentiment_column])
        else:
            sentiment_values = df[sentiment_column]
        
        # Plot sentiment values
        plt.plot(df.index, sentiment_values, 'o-', label=symbol)
    
    # Add horizontal lines for thresholds
    plt.axhline(y=0.2, color='g', linestyle='--', alpha=0.7, label='Buy Threshold')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7, label='Sell Threshold')
    
    # Formatting
    plt.title("Sentiment Comparison Across Assets", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sentiment Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to ensure -1 to 1 range is visible
    y_min, y_max = plt.ylim()
    plt.ylim(min(y_min, -1.0), max(y_max, 1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    logger.info(f"Multi-asset sentiment visualization saved to {output_file}")

def visualize_sentiment_vs_price(sentiment_data: pd.DataFrame, price_data: pd.DataFrame, 
                               symbol: str, output_file: str) -> None:
    """
    Visualize sentiment vs price for a specific symbol.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        price_data: DataFrame with price data
        symbol: Symbol to visualize
        output_file: Output file path for the visualization
    """
    if sentiment_data.empty or price_data.empty:
        logger.warning(f"Missing data to visualize sentiment vs price for {symbol}")
        # Create a placeholder image with a message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Insufficient data available for {symbol}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    # Determine sentiment column
    if 'ticker_sentiment_score' in sentiment_data.columns:
        sentiment_column = 'ticker_sentiment_score'
    else:
        sentiment_column = 'overall_sentiment_score'
    
    # Combine data
    # First resample sentiment to daily to match price data
    sentiment_daily = sentiment_data[sentiment_column].resample('D').mean()
    
    # Ensure price data has a datetime index
    if not isinstance(price_data.index, pd.DatetimeIndex):
        try:
            price_data.index = pd.to_datetime(price_data.index)
        except Exception as e:
            logger.error(f"Failed to convert price data index to datetime: {e}")
            return
    
    # Determine price column
    price_column = 'close' if 'close' in price_data.columns else price_data.columns[0]
    
    # Prepare the plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot price data on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(price_data.index, price_data[price_column], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary y-axis for sentiment
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sentiment Score', color=color)
    
    # Plot resampled sentiment data
    ax2.plot(sentiment_daily.index, sentiment_daily.values, color=color, marker='o', linestyle='-', alpha=0.7)
    
    # Add horizontal lines for sentiment thresholds
    ax2.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='Buy Threshold')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.axhline(y=-0.2, color='r', linestyle='--', alpha=0.5, label='Sell Threshold')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, [f"{symbol} Price"] + labels2, loc='upper left')
    
    # Set title
    plt.title(f"Price vs Sentiment for {symbol}", fontsize=16)
    
    # Adjust y-axis range for sentiment
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(min(y_min, -1.0), max(y_max, 1.0))
    
    # Save figure
    fig.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    logger.info(f"Sentiment vs price visualization saved to {output_file}")

def export_sentiment_data_for_dashboard(symbols: List[str], days_back: int = 30, 
                                      output_file: str = "dashboard_sentiment_data.json") -> None:
    """
    Export sentiment data in a format ready for the dashboard.
    
    Args:
        symbols: List of symbols to export data for
        days_back: Number of days to look back
        output_file: Output JSON file path
    """
    connector = AlphaVantageSentimentConnector()
    sentiment_summary = connector.get_sentiment_summary(symbols, days_back)
    
    # Add historical data for each symbol
    sentiment_data = sentiment_summary['sentimentData']
    historical_data = {}
    
    for symbol in symbols:
        try:
            historical = connector.get_historical_sentiment(symbol, timeframe='1M')
            historical_data[symbol] = historical
        except Exception as e:
            logger.error(f"Error getting historical sentiment for {symbol}: {e}")
            historical_data[symbol] = []
    
    # Prepare final data structure
    dashboard_data = {
        'sentimentSummary': sentiment_summary,
        'historicalSentiment': historical_data,
        'generatedAt': datetime.now().isoformat()
    }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    logger.info(f"Dashboard sentiment data exported to {output_file}")
    
    return dashboard_data