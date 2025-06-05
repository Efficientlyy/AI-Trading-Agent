#!/usr/bin/env python
"""
Chart Manager

This module provides the ChartManager class for managing chart instances.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import io

logger = logging.getLogger("system_overseer.plugins.visualization.chart_manager")

class ChartManager:
    """Chart Manager for visualization plugin."""
    
    def __init__(self, data_dir: str = "./data/charts"):
        """Initialize Chart Manager.
        
        Args:
            data_dir: Directory for storing chart data and images
        """
        self.data_dir = data_dir
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Chart style settings
        plt.style.use('dark_background')
        
        logger.info("ChartManager initialized")
    
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str, interval: str, 
                                indicators: List[str] = None) -> Optional[bytes]:
        """Create a candlestick chart from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            interval: Time interval
            indicators: List of indicators to include
            
        Returns:
            Optional[bytes]: PNG image data or None if creation failed
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Plot candlesticks
            width = 0.6
            width2 = 0.05
            
            # Bull candles (close > open)
            bull = df[df.close > df.open]
            ax.bar(bull.index, bull.close - bull.open, width, bottom=bull.open, color='green')
            ax.bar(bull.index, bull.high - bull.close, width2, bottom=bull.close, color='green')
            ax.bar(bull.index, bull.low - bull.open, width2, bottom=bull.open, color='green')
            
            # Bear candles (close <= open)
            bear = df[df.close <= df.open]
            ax.bar(bear.index, bear.close - bear.open, width, bottom=bear.open, color='red')
            ax.bar(bear.index, bear.high - bear.open, width2, bottom=bear.open, color='red')
            ax.bar(bear.index, bear.low - bear.close, width2, bottom=bear.close, color='red')
            
            # Add indicators if requested
            if indicators:
                for indicator in indicators:
                    if indicator == 'sma':
                        sma = df['close'].rolling(window=20).mean()
                        ax.plot(df.index, sma, label='SMA (20)', color='blue', linewidth=1)
                    elif indicator == 'ema':
                        ema = df['close'].ewm(span=20, adjust=False).mean()
                        ax.plot(df.index, ema, label='EMA (20)', color='orange', linewidth=1)
            
            # Add title and labels
            plt.title(f'{symbol} {interval} Candlestick Chart')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            
            # Add legend if indicators were added
            if indicators:
                plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Close figure to free memory
            plt.close(fig)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return None
    
    def create_line_chart(self, df: pd.DataFrame, symbol: str, interval: str,
                         indicators: List[str] = None) -> Optional[bytes]:
        """Create a line chart from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            interval: Time interval
            indicators: List of indicators to include
            
        Returns:
            Optional[bytes]: PNG image data or None if creation failed
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Plot close price
            ax.plot(df.index, df.close, label='Close Price', color='cyan', linewidth=2)
            
            # Add indicators if requested
            if indicators:
                for indicator in indicators:
                    if indicator == 'sma':
                        sma = df['close'].rolling(window=20).mean()
                        ax.plot(df.index, sma, label='SMA (20)', color='blue', linewidth=1)
                    elif indicator == 'ema':
                        ema = df['close'].ewm(span=20, adjust=False).mean()
                        ax.plot(df.index, ema, label='EMA (20)', color='orange', linewidth=1)
            
            # Add title and labels
            plt.title(f'{symbol} {interval} Line Chart')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Close figure to free memory
            plt.close(fig)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            return None
    
    def create_volume_chart(self, df: pd.DataFrame, symbol: str, interval: str) -> Optional[bytes]:
        """Create a volume chart from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Optional[bytes]: PNG image data or None if creation failed
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Plot volume bars
            # Color based on price movement
            colors = ['green' if close > open else 'red' for close, open in zip(df.close, df.open)]
            ax.bar(df.index, df.volume, color=colors, alpha=0.8)
            
            # Add title and labels
            plt.title(f'{symbol} {interval} Volume Chart')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Close figure to free memory
            plt.close(fig)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating volume chart: {e}")
            return None
    
    def save_chart_image(self, image_data: bytes, symbol: str, chart_type: str, 
                        interval: str) -> Optional[str]:
        """Save chart image to file.
        
        Args:
            image_data: PNG image data
            symbol: Trading pair symbol
            chart_type: Type of chart
            interval: Time interval
            
        Returns:
            Optional[str]: Path to saved image or None if save failed
        """
        try:
            # Create filename
            filename = f"{symbol}_{chart_type}_{interval}_{int(time.time())}.png"
            filepath = os.path.join(self.data_dir, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Chart image saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving chart image: {e}")
            return None
