"""
Data Manager for AI Trading Agent.

This module provides the DataManagerABC abstract base class and DataManager implementation
for managing market data and other data sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class DataManagerABC(ABC):
    """
    Abstract base class for data management.
    
    Defines the interface for data acquisition, processing, and storage.
    """
    
    @abstractmethod
    def get_historical_data(self, symbols: Optional[List[str]] = None, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           interval: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical data for the specified symbols and time range.
        
        Args:
            symbols: List of symbols to get data for. If None, get data for all configured symbols.
            start_date: Start date for historical data in YYYY-MM-DD format. If None, use default lookback.
            end_date: End date for historical data in YYYY-MM-DD format. If None, use current date.
            interval: Data interval (e.g., '1d', '1h', '15m'). If None, use default interval.
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data.
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Retrieve the latest data point for the specified symbols.
        
        Args:
            symbols: List of symbols to get data for. If None, get data for all configured symbols.
            
        Returns:
            Dictionary mapping symbols to Series with the latest data.
        """
        pass
    
    @abstractmethod
    def update_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Update the data for the specified symbols.
        
        Args:
            symbols: List of symbols to update data for. If None, update all configured symbols.
            
        Returns:
            Dictionary mapping symbols to DataFrames with the updated data.
        """
        pass


class DataManager(DataManagerABC):
    """
    Data Manager implementation.
    
    Manages market data and other data sources for the trading agent.
    """
    
    def __init__(self, config: Dict[str, Any], data_service=None):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration dictionary.
            data_service: Data service instance for fetching data. If None, create a new one.
        """
        self.config = config
        self.data_service = data_service
        
        # Initialize cache
        self.cache = {}
        self.cache_enabled = config.get('cache_data', True)
        self.cache_dir = config.get('cache_dir', 'data/cache')
        self.data_lookback = config.get('data_lookback', 200)
        
        # Initialize data
        self.historical_data = {}
        self.latest_data = {}
        
        # Initialize symbols
        self.symbols = config.get('symbols', [])
        
        logger.info(f"DataManager initialized with {len(self.symbols)} symbols")
    
    def get_historical_data(self, symbols: Optional[List[str]] = None, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           interval: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical data for the specified symbols and time range.
        
        Args:
            symbols: List of symbols to get data for. If None, get data for all configured symbols.
            start_date: Start date for historical data in YYYY-MM-DD format. If None, use default lookback.
            end_date: End date for historical data in YYYY-MM-DD format. If None, use current date.
            interval: Data interval (e.g., '1d', '1h', '15m'). If None, use default interval.
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data.
        """
        # Use configured symbols if none provided
        if symbols is None:
            symbols = self.symbols
        
        # Use cached data if available and no specific date range requested
        if not start_date and not end_date and self.historical_data:
            return {symbol: self.historical_data[symbol] for symbol in symbols if symbol in self.historical_data}
        
        # If we have a data service, fetch the data
        if self.data_service:
            try:
                data = self.data_service.fetch_historical_data(
                    symbols, start_date, end_date, interval
                )
                
                # Update cache
                if self.cache_enabled:
                    for symbol, df in data.items():
                        self.historical_data[symbol] = df
                
                return data
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                # Return cached data if available
                if self.historical_data:
                    return {symbol: self.historical_data[symbol] for symbol in symbols if symbol in self.historical_data}
                raise
        else:
            logger.warning("No data service available, returning cached data")
            return {symbol: self.historical_data[symbol] for symbol in symbols if symbol in self.historical_data}
    
    def get_latest_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.Series]:
        """
        Retrieve the latest data point for the specified symbols.
        
        Args:
            symbols: List of symbols to get data for. If None, get data for all configured symbols.
            
        Returns:
            Dictionary mapping symbols to Series with the latest data.
        """
        # Use configured symbols if none provided
        if symbols is None:
            symbols = self.symbols
        
        # Get latest data from historical data
        result = {}
        for symbol in symbols:
            if symbol in self.historical_data and not self.historical_data[symbol].empty:
                result[symbol] = self.historical_data[symbol].iloc[-1]
        
        # If we have a data service and missing symbols, fetch the latest data
        missing_symbols = [symbol for symbol in symbols if symbol not in result]
        if missing_symbols and self.data_service:
            try:
                latest_data = self.data_service.fetch_latest_data(missing_symbols)
                result.update(latest_data)
            except Exception as e:
                logger.error(f"Error fetching latest data: {e}")
        
        return result
    
    def update_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Update the data for the specified symbols.
        
        Args:
            symbols: List of symbols to update data for. If None, update all configured symbols.
            
        Returns:
            Dictionary mapping symbols to DataFrames with the updated data.
        """
        # Use configured symbols if none provided
        if symbols is None:
            symbols = self.symbols
        
        # If we have a data service, fetch the latest data
        if self.data_service:
            try:
                # Get the latest timestamp for each symbol
                latest_timestamps = {}
                for symbol in symbols:
                    if symbol in self.historical_data and not self.historical_data[symbol].empty:
                        latest_timestamps[symbol] = self.historical_data[symbol].index[-1]
                
                # Fetch data since the latest timestamp
                updated_data = self.data_service.fetch_data_since(symbols, latest_timestamps)
                
                # Update historical data
                for symbol, df in updated_data.items():
                    if symbol in self.historical_data:
                        # Append new data
                        self.historical_data[symbol] = pd.concat([
                            self.historical_data[symbol],
                            df[~df.index.isin(self.historical_data[symbol].index)]
                        ]).sort_index()
                    else:
                        self.historical_data[symbol] = df
                
                return self.historical_data
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                return self.historical_data
        else:
            logger.warning("No data service available, returning cached data")
            return self.historical_data
