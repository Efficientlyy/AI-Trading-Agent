"""
Wrapper module that provides the same interface as the Rust extension.
This allows for seamless switching between the Python and Rust implementations.
"""
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from .models import (
    OrderSide, OrderType, OrderStatus, Order, Fill, Trade, Position,
    Portfolio, PortfolioSnapshot, OHLCVBar, BacktestConfig, PerformanceMetrics
)
from .backtester import run_backtest


def run_backtest_py(data: Dict[str, List[OHLCVBar]], orders: List[Order], config: BacktestConfig) -> Dict[str, Any]:
    """
    Python implementation of run_backtest_rs from the Rust extension.
    This function has the same interface as the Rust function to allow for seamless switching.
    """
    return run_backtest(data, orders, config)


def convert_numpy_arrays_to_lists(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert any NumPy arrays in the result dictionary to Python lists for JSON serialization.
    """
    for key, value in result_dict.items():
        if isinstance(value, np.ndarray):
            result_dict[key] = value.tolist()
        elif isinstance(value, dict):
            result_dict[key] = convert_numpy_arrays_to_lists(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    value[i] = convert_numpy_arrays_to_lists(item)
                elif isinstance(item, np.ndarray):
                    value[i] = item.tolist()
    
    return result_dict
