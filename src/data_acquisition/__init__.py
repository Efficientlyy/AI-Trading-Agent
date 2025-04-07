"""
Data Acquisition Module

Handles fetching historical and real-time market data from various sources.
"""

from .base_provider import BaseDataProvider
from .mock_provider import MockDataProvider
from .data_service import DataService
from .ccxt_provider import CcxtProvider
# Import other providers as needed

__all__ = [
    'BaseDataProvider',
    'MockDataProvider',
    'CcxtProvider',
    'DataService',
    # Add other provider classes here
]
