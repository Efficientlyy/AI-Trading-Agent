"""
Data Service Module

Acts as a central hub for accessing market data, delegating requests
to the configured data provider.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio
import inspect

from ..common import get_config, get_config_value, logger
from .base_provider import BaseDataProvider
from .mock_provider import MockDataProvider
# Import other providers as they are created
from .ccxt_provider import CcxtProvider
# from .yfinance_provider import YFinanceProvider

class DataService:
    """
    Manages data providers and provides a unified interface for data access.
    """

    def __init__(self):
        """
        Initialize the DataService by loading configuration and instantiating
        the appropriate data provider.
        """
        self.config = get_config()
        self.data_sources_config = get_config_value('data_sources', {})
        self.active_provider_name = self.data_sources_config.get('active_provider', 'mock')
        self.provider_config = self.data_sources_config.get(self.active_provider_name, {})

        logger.info(f"Initializing DataService with active provider: {self.active_provider_name}")

        self.provider: BaseDataProvider = self._create_provider()

    def _create_provider(self) -> BaseDataProvider:
        """
        Factory method to create the configured data provider instance.
        """
        provider_name = self.active_provider_name.lower()

        if provider_name == 'mock':
            logger.info(f"Creating MockDataProvider with config: {self.provider_config}")
            return MockDataProvider(config=self.provider_config)
        elif provider_name == 'ccxt':
             logger.info(f"Creating CcxtProvider with config: {self.provider_config}")
             # Ensure necessary config keys are present if needed later (e.g., API keys)
             return CcxtProvider(config=self.provider_config)
        # elif provider_name == 'yfinance':
        #     logger.info(f"Creating YFinanceProvider with config: {self.provider_config}")
        #     return YFinanceProvider(config=self.provider_config)
        else:
            logger.error(f"Unsupported data provider specified: {self.active_provider_name}")
            raise ValueError(f"Unsupported data provider: {self.active_provider_name}")

    async def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data using the active provider."""
        logger.debug(f"DataService forwarding fetch_historical_data to {self.provider.provider_name}")
        
        # Check if the provider's fetch_historical_data method accepts a params parameter
        provider_method = self.provider.fetch_historical_data
        sig = inspect.signature(provider_method)
        
        if 'params' in sig.parameters:
            # Provider accepts params parameter
            return await self.provider.fetch_historical_data(symbols, timeframe, start_date, end_date, params=params)
        else:
            # Provider doesn't accept params parameter
            return await self.provider.fetch_historical_data(symbols, timeframe, start_date, end_date)

    async def connect_realtime(self):
        """Connect to real-time data stream using the active provider."""
        logger.debug(f"DataService forwarding connect_realtime to {self.provider.provider_name}")
        await self.provider.connect_realtime()

    async def disconnect_realtime(self):
        """Disconnect from real-time data stream using the active provider."""
        logger.debug(f"DataService forwarding disconnect_realtime to {self.provider.provider_name}")
        await self.provider.disconnect_realtime()

    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time symbols using the active provider."""
        logger.debug(f"DataService forwarding subscribe_to_symbols to {self.provider.provider_name}")
        await self.provider.subscribe_to_symbols(symbols)

    async def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest real-time data update from the active provider."""
        # Debug log might be too noisy here, depends on call frequency
        # logger.debug(f"DataService forwarding get_realtime_data to {self.provider.provider_name}")
        return await self.provider.get_realtime_data()

    def get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes from the active provider."""
        logger.debug(f"DataService forwarding get_supported_timeframes to {self.provider.provider_name}")
        return self.provider.get_supported_timeframes()

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider."""
        logger.debug(f"DataService forwarding get_info to {self.provider.provider_name}")
        return self.provider.get_info()

    def get_active_provider_name(self) -> str:
        """Return the name of the active provider."""
        return self.active_provider_name

    async def close(self):
        """Cleanly close the underlying provider if it has a close method."""
        if hasattr(self.provider, 'close') and asyncio.iscoroutinefunction(self.provider.close):
            logger.info(f"Closing provider: {self.active_provider_name}")
            await self.provider.close()
        else:
            logger.debug(f"Provider {self.active_provider_name} does not have an async close method.")

# Optional: Create a singleton instance for easy access throughout the application
# data_service = DataService()
