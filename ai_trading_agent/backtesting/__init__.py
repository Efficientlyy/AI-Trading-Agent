"""
Backtesting module for AI Trading Agent.

This module provides tools for backtesting trading strategies.
"""

# Define __all__ without any imports initially
__all__ = ["Backtester", "calculate_metrics", "PerformanceMetrics", "RUST_AVAILABLE"]

# Set RUST_AVAILABLE to False by default
RUST_AVAILABLE = False

# Import performance metrics first as it has no dependencies on other modules
try:
    from .performance_metrics import calculate_metrics, PerformanceMetrics
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing performance_metrics: {e}")
    # Create placeholder functions/classes if imports fail
    def calculate_metrics(*args, **kwargs):
        raise ImportError("calculate_metrics could not be imported")
    
    class PerformanceMetrics:
        def __init__(self, *args, **kwargs):
            raise ImportError("PerformanceMetrics could not be imported")

# Import the core backtester
try:
    from .backtester import Backtester
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing Backtester: {e}")
    # Create placeholder class if import fails
    class Backtester:
        def __init__(self, *args, **kwargs):
            raise ImportError("Backtester could not be imported")

# Try to import RustBacktester, but don't fail if it's not available
try:
    # Import Rust backtester - handle gracefully if not available
    from .rust_backtester import RustBacktester
    RUST_AVAILABLE = True
    __all__.append("RustBacktester")
except ImportError as e:
    # Log the import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"RustBacktester not available: {e}")
    # Create placeholder class for RustBacktester
    class RustBacktester:
        def __init__(self, *args, **kwargs):
            raise ImportError("RustBacktester is not available. Rust extensions may not be installed.")
