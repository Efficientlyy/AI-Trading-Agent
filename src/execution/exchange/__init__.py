"""Exchange connectors for cryptocurrency exchanges.

This module provides a standardized interface for interacting with various
cryptocurrency exchanges, allowing the trading system to execute orders,
retrieve market data, and manage API keys.

Available connectors:
- Base Connector: Abstract base class that defines the exchange connector interface
- Binance Connector: Implementation for the Binance exchange
- Mock Connector: Simulated exchange connector for testing
"""

from src.execution.exchange.base import BaseExchangeConnector
from src.execution.exchange.mock import MockExchangeConnector

__all__ = [
    "BaseExchangeConnector",
    "MockExchangeConnector",
] 