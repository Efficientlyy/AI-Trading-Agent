#!/usr/bin/env python
"""
Data Provider Interface

This module defines the interface for data providers used by the Visualization Plugin.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd

class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the data provider with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get klines (candlestick data) for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            interval: Time interval (e.g., '1m', '5m', '1h')
            limit: Maximum number of klines to retrieve
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns [timestamp, open, high, low, close, volume]
                                   or None if data retrieval failed
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with ticker data or None if retrieval failed
        """
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            limit: Maximum number of orders to retrieve for each side
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with order book data or None if retrieval failed
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pair symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        pass
    
    @abstractmethod
    def get_available_intervals(self) -> List[str]:
        """Get list of available time intervals.
        
        Returns:
            List[str]: List of available intervals
        """
        pass
