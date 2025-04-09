"""
Custom exceptions for the AI Trading Agent trading engine.
"""

class TradingEngineError(Exception):
    """Base exception for trading engine errors."""
    pass

class OrderValidationError(TradingEngineError):
    """Raised when an order fails validation."""
    pass

class ExecutionError(TradingEngineError):
    """Raised when an order execution fails."""
    pass

class PortfolioUpdateError(TradingEngineError):
    """Raised when portfolio update fails."""
    pass

class DataProviderError(TradingEngineError):
    """Raised when data acquisition encounters an error."""
    pass