"""Exchange connectors for cryptocurrency exchanges.

This module provides a standardized interface for interacting with various
cryptocurrency exchanges, allowing the trading system to execute orders,
retrieve market data, and manage API keys.

Available connectors:
- Base Connector: Abstract base class that defines the exchange connector interface
- Binance Connector: Implementation for the Binance exchange
- Mock Connector: Simulated exchange connector for testing
- Coinbase Connector: Connector for Coinbase exchange
"""

from src.execution.exchange.base import BaseExchangeConnector
from src.execution.exchange.mock import MockExchangeConnector
from src.execution.exchange.binance import BinanceExchangeConnector
from src.execution.exchange.coinbase import CoinbaseExchangeConnector

__all__ = [
    "BaseExchangeConnector",
    "MockExchangeConnector",
    "BinanceExchangeConnector",
    "CoinbaseExchangeConnector",
] 