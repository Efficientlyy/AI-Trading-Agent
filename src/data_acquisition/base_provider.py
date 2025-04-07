"""
Base class definition for all data providers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional

class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.
    Defines the common interface for fetching historical and real-time data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data provider.

        Args:
            config (Optional[Dict[str, Any]]): Configuration specific to the provider.
        """
        self.config = config or {}
        self.provider_name = self.__class__.__name__

    @abstractmethod
    async def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical market data (OHLCV) for given symbols and timeframe.

        Args:
            symbols (List[str]): List of market symbols (e.g., ['BTC/USD', 'ETH/USD']).
            timeframe (str): Timeframe for the data (e.g., '1h', '1d').
            start_date (pd.Timestamp): Start date for the historical data.
            end_date (pd.Timestamp): End date for the historical data.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are symbols and values are
                                     pandas DataFrames containing OHLCV data.
                                     DataFrame should have a DatetimeIndex and columns
                                     ['open', 'high', 'low', 'close', 'volume'].
        """
        pass

    async def connect_realtime(self):
        """
        Establish connection for real-time data streaming (optional).
        Should be overridden by providers that support real-time data.
        """
        # Default implementation does nothing
        pass

    async def disconnect_realtime(self):
        """
        Disconnect from real-time data stream (optional).
        """
        # Default implementation does nothing
        pass

    async def subscribe_to_symbols(self, symbols: List[str]):
        """
        Subscribe to real-time updates for specific symbols (optional).

        Args:
            symbols (List[str]): List of symbols to subscribe to.
        """
        # Default implementation does nothing
        pass

    async def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest real-time data update (e.g., tick data, kline update) (optional).
        This method might be called periodically or use a callback mechanism depending
        on the provider's implementation.

        Returns:
            Optional[Dict[str, Any]]: Latest data update, format depends on provider.
                                       Could be a tick, a completed kline, etc.
                                       Returns None if no new data is available.
        """
        # Default implementation returns None
        return None

    def get_supported_timeframes(self) -> List[str]:
        """
        Get a list of timeframes supported by this provider.

        Returns:
            List[str]: List of supported timeframes (e.g., ['1m', '5m', '1h', '1d']).
                       Returns empty list if not applicable or unknown.
        """
        # Default implementation returns empty list
        return []

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the data provider.

        Returns:
            Dict[str, Any]: Dictionary containing provider details.
        """
        return {
            "provider_name": self.provider_name,
            "config": self.config,
            "supported_timeframes": self.get_supported_timeframes()
        }
