"""
Base class for sentiment data providers.

This module defines the abstract base class for all sentiment data providers
in the AI Trading Agent system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd


class BaseSentimentProvider(ABC):
    """
    Abstract base class for sentiment data providers.
    
    All sentiment data providers must implement the methods defined in this class.
    """
    
    @abstractmethod
    def fetch_sentiment_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for the specified symbols and date range.
        
        Args:
            symbols: List of ticker symbols to fetch sentiment for
            start_date: Start date for sentiment data
            end_date: End date for sentiment data
            **kwargs: Additional parameters for the data fetch
            
        Returns:
            DataFrame with sentiment data with at least the following columns:
            - timestamp: Datetime of the sentiment data
            - content: Text content of the sentiment data
            - symbol: Ticker symbol the sentiment is related to
            - source: Source of the sentiment data (e.g., 'twitter', 'reddit')
        """
        pass
    
    @abstractmethod
    def stream_sentiment_data(
        self,
        symbols: List[str],
        callback: callable,
        **kwargs
    ) -> None:
        """
        Stream sentiment data for the specified symbols.
        
        Args:
            symbols: List of ticker symbols to stream sentiment for
            callback: Callback function to process streamed data
            **kwargs: Additional parameters for the data stream
        """
        pass
