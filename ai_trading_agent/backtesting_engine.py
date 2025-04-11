"""
Backtesting engine integration module.
This module tries to use the Rust extension for performance, but falls back to the Python implementation if necessary.
"""
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import json
import traceback

# Setup logging
logger = logging.getLogger(__name__)

# Try to import the CFFI loader for Rust extensions
try:
    from ai_trading_agent.cffi_loader import run_backtest as run_backtest_rs
    logger.info("Successfully imported Rust backtesting function via CFFI")
    USE_RUST = True
except (ImportError, AttributeError) as e:
    logger.warning(f"Failed to import Rust backtesting function: {e}")
    USE_RUST = False

# If CFFI loader is not available, import the Python implementation
if not USE_RUST:
    from ai_trading_agent.python_backtesting import run_backtest_py as run_backtest_rs
    logger.info("Using Python implementation for backtesting")


def run_backtest(data: Dict[str, List[Any]], orders: List[Any], config: Any) -> Dict[str, Any]:
    """
    Run a backtest with the given data and configuration.
    This function will use the Rust implementation if available, otherwise it will use the Python implementation.
    
    Args:
        data: Dictionary mapping symbols to lists of OHLCV bars
        orders: List of orders to process
        config: Backtest configuration
        
    Returns:
        Dictionary containing backtest results
    """
    try:
        # Convert data and orders to the format expected by the backtesting function
        # This ensures compatibility with both Rust and Python implementations
        formatted_data = []
        for symbol, bars in data.items():
            formatted_bars = []
            for bar in bars:
                formatted_bar = {
                    'timestamp': bar.timestamp if hasattr(bar, 'timestamp') else bar['timestamp'],
                    'open': bar.open if hasattr(bar, 'open') else bar['open'],
                    'high': bar.high if hasattr(bar, 'high') else bar['high'],
                    'low': bar.low if hasattr(bar, 'low') else bar['low'],
                    'close': bar.close if hasattr(bar, 'close') else bar['close'],
                    'volume': bar.volume if hasattr(bar, 'volume') else bar['volume'],
                    'symbol': symbol
                }
                formatted_bars.append(formatted_bar)
            formatted_data.extend(formatted_bars)
        
        formatted_orders = []
        for order in orders:
            formatted_order = {
                'id': order.id if hasattr(order, 'id') else order['id'],
                'timestamp': order.timestamp if hasattr(order, 'timestamp') else order['timestamp'],
                'symbol': order.symbol if hasattr(order, 'symbol') else order['symbol'],
                'side': order.side if hasattr(order, 'side') else order['side'],
                'quantity': order.quantity if hasattr(order, 'quantity') else order['quantity'],
                'price': order.price if hasattr(order, 'price') else order.get('price', 0.0),
                'type': order.type if hasattr(order, 'type') else order['type']
            }
            formatted_orders.append(formatted_order)
        
        formatted_config = {
            'initial_capital': config.initial_capital if hasattr(config, 'initial_capital') else config['initial_capital'],
            'commission_rate': config.commission_rate if hasattr(config, 'commission_rate') else config['commission_rate'],
            'slippage': config.slippage if hasattr(config, 'slippage') else config['slippage']
        }
        
        # Log the formatted data for debugging
        logger.debug(f"Running backtest with {len(formatted_data)} bars, {len(formatted_orders)} orders")
        
        # Run the backtest
        result = run_backtest_rs(formatted_data, formatted_config)
        
        # Log the result for debugging
        logger.debug(f"Backtest result: {json.dumps(result, indent=4)}")
        
        # Process the result
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
        # Fallback to a minimal result in case of error
        return {
            'final_capital': config.initial_capital if hasattr(config, 'initial_capital') else config['initial_capital'],
            'metrics': {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            },
            'error': str(e)
        }


# Export the model classes from the Python implementation for convenience
from ai_trading_agent.python_backtesting import (
    OrderSide, OrderType, OrderStatus, Order, Fill, Trade, Position,
    Portfolio, PortfolioSnapshot, OHLCVBar, BacktestConfig, PerformanceMetrics, BacktestResult
)
