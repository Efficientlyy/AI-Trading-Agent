"""
Base classes for alternative data sources integration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data sources."""
    api_key: str
    api_secret: Optional[str] = None
    endpoint: Optional[str] = None
    cache_duration: int = 3600  # Default cache duration in seconds
    max_retries: int = 3
    timeout: int = 30  # Request timeout in seconds
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    rate_limit: Optional[int] = None  # Requests per minute


class AlternativeDataSource(ABC):
    """
    Abstract base class for all alternative data sources.
    
    This class defines the common interface that all alternative data
    source implementations must adhere to. It provides methods for 
    initializing the data source, fetching data, and processing results.
    """
    
    def __init__(self, config: AlternativeDataConfig):
        """
        Initialize the alternative data source with configuration.
        
        Args:
            config: Configuration for the data source
        """
        self.config = config
        self.last_updated = None
        self._cache = {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the data source connection.
        
        This method should establish any necessary connections or authentication
        required by the data source. It will be called during initialization.
        """
        pass
    
    @abstractmethod
    async def fetch_data(self, query: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        Fetch data from the alternative data source.
        
        Args:
            query: Parameters defining what data to retrieve
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame containing the retrieved data
        """
        pass
    
    @abstractmethod
    def process_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the raw data into actionable trading signals.
        
        Args:
            data: Raw data retrieved from the data source
            
        Returns:
            Dictionary containing processed signals and metadata
        """
        pass
    
    def clear_cache(self) -> None:
        """Clear the internal data cache."""
        self._cache = {}
        
    def get_last_updated(self) -> Optional[datetime]:
        """Return the timestamp of the last data update."""
        return self.last_updated
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the data source connection.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "source_type": self.__class__.__name__,
            "connected": self._is_connected(),
            "last_updated": self.last_updated,
            "cache_size": len(self._cache),
            "config": {
                "endpoint": self.config.endpoint,
                "cache_duration": self.config.cache_duration,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "use_proxy": self.config.use_proxy,
                "rate_limit": self.config.rate_limit
            }
        }
    
    @abstractmethod
    def _is_connected(self) -> bool:
        """
        Check if the data source is currently connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass
