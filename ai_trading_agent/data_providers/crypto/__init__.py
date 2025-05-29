"""
Cryptocurrency market data provider package.

This package provides classes and utilities for fetching real-time
and historical cryptocurrency market data from various providers.
"""

from .provider import MarketDataProvider
from .binance_client import BinanceClient
from .coingecko_client import CoinGeckoClient
from .crypto_compare_client import CryptoCompareClient

__all__ = [
    'MarketDataProvider',
    'BinanceClient',
    'CoinGeckoClient',
    'CryptoCompareClient',
]
