"""
Rust-accelerated backtesting module for AI Trading Agent.

This module provides a wrapper around the Rust backtesting implementation
for improved performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from datetime import datetime
import time
from dataclasses import dataclass, field

from src.trading_engine.models import Order, Trade, Position, Portfolio, OrderSide, OrderType, OrderStatus
from src.common import logger

# Import Rust extension
try:
    from rust_extensions import run_backtest_rs
    RUST_AVAILABLE = True
except ImportError:
    logger.warning("Rust extensions not available. Using Python implementation.")
    RUST_AVAILABLE = False


class RustBacktester:
    """
    Rust-accelerated backtester class for simulating trading strategies on historical data.
    
    This class provides a high-performance implementation of the backtesting loop
    by delegating the core simulation to Rust.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0,
        enable_fractional: bool = True,
    ):
        """
        Initialize the Rust backtester.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as a decimal (e.g., 0.001 = 0.1%)
            enable_fractional: Whether to allow fractional position sizes
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust extensions not available. Cannot use RustBacktester.")
        
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.enable_fractional = enable_fractional
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized Rust backtester with {len(self.symbols)} symbols and {initial_capital} initial capital")
    
    def _validate_data(self):
        """Validate input data format and alignment."""
        if not self.data:
            raise ValueError("No data provided for backtesting")
        
        # Check that all dataframes have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for symbol, df in self.data.items():
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns {missing_columns} for symbol {symbol}")
        
        # Check that all dataframes have the same index (timestamps)
        # This is important for multi-asset backtesting
        reference_index = next(iter(self.data.values())).index
        for symbol, df in self.data.items():
            if not df.index.equals(reference_index):
                raise ValueError(f"Index mismatch for symbol {symbol}. All data must have the same timestamps.")
        
        logger.info(f"Data validation passed. Timeframe: {reference_index[1] - reference_index[0]}")
    
    def _convert_data_to_rust_format(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert pandas DataFrames to a format that can be passed to Rust.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Data in Rust-compatible format
        """
        rust_data = {}
        
        for symbol, df in self.data.items():
            bars = []
            for idx, row in df.iterrows():
                timestamp = int(idx.timestamp())
                bar = {
                    "timestamp": timestamp,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                }
                bars.append(bar)
            
            rust_data[symbol] = bars
        
        return rust_data
    
    def _convert_orders_to_rust_format(self, orders: List[Order]) -> List[Dict[str, Any]]:
        """
        Convert Python Order objects to a format that can be passed to Rust.
        
        Args:
            orders: List of Order objects
            
        Returns:
            List[Dict[str, Any]]: Orders in Rust-compatible format
        """
        rust_orders = []
        
        for order in orders:
            # Convert OrderSide enum to Rust-compatible format
            side = "Buy" if order.side == OrderSide.BUY else "Sell"
            
            # Convert OrderType enum to Rust-compatible format
            order_type_map = {
                OrderType.MARKET: "Market",
                OrderType.LIMIT: "Limit",
                OrderType.STOP: "Stop",
                OrderType.STOP_LIMIT: "StopLimit",
            }
            order_type = order_type_map.get(order.order_type, "Market")
            
            # Convert OrderStatus enum to Rust-compatible format
            status_map = {
                OrderStatus.CREATED: "Created",
                OrderStatus.SUBMITTED: "Submitted",
                OrderStatus.PARTIAL: "Partial",
                OrderStatus.FILLED: "Filled",
                OrderStatus.CANCELED: "Canceled",
                OrderStatus.REJECTED: "Rejected",
            }
            status = status_map.get(order.status, "Created")
            
            # Convert fills to Rust-compatible format
            fills = []
            for fill in order.fills:
                fills.append({
                    "quantity": float(fill.quantity),
                    "price": float(fill.price),
                    "timestamp": int(fill.timestamp.timestamp()) if hasattr(fill.timestamp, 'timestamp') else int(fill.timestamp),
                })
            
            # Create Rust-compatible order
            rust_order = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": side,
                "order_type": order_type,
                "quantity": float(order.quantity),
                "limit_price": float(order.limit_price) if order.limit_price is not None else None,
                "stop_price": float(order.stop_price) if order.stop_price is not None else None,
                "status": status,
                "fills": fills,
                "created_at": int(order.created_at.timestamp()) if hasattr(order.created_at, 'timestamp') else int(order.created_at),
            }
            
            rust_orders.append(rust_order)
        
        return rust_orders
    
    def _convert_rust_results_to_python(self, rust_results: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Trade], List[Order], Dict[str, Any]]:
        """
        Convert Rust results back to Python objects.
        
        Args:
            rust_results: Results from Rust backtesting
            
        Returns:
            Tuple containing:
            - portfolio_history: List of portfolio snapshots
            - trade_history: List of Trade objects
            - order_history: List of Order objects
            - metrics: Dictionary of performance metrics
        """
        # Convert portfolio history
        portfolio_history = []
        for snapshot in rust_results.get('portfolio_history', []):
            timestamp = pd.Timestamp(snapshot['timestamp'], unit='s')
            
            # Convert positions
            positions = {}
            for symbol, pos_data in snapshot.get('positions', {}).items():
                position = Position(
                    symbol=symbol,
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                )
                position.market_price = pos_data['market_price']
                position.unrealized_pnl = pos_data['unrealized_pnl']
                position.realized_pnl = pos_data['realized_pnl']
                
                positions[symbol] = position
            
            portfolio_snapshot = {
                'timestamp': timestamp,
                'cash': snapshot['cash'],
                'total_value': snapshot['total_value'],
                'positions': positions,
            }
            
            portfolio_history.append(portfolio_snapshot)
        
        # Convert trade history
        trade_history = []
        for trade_data in rust_results.get('trade_history', []):
            # Convert side string back to enum
            side = OrderSide.BUY if trade_data['side'] == 'Buy' else OrderSide.SELL
            
            trade = Trade(
                symbol=trade_data['symbol'],
                order_id=trade_data['order_id'],
                side=side,
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                timestamp=pd.Timestamp(trade_data['timestamp'], unit='s'),
            )
            
            trade_history.append(trade)
        
        # Convert order history
        order_history = []
        for order_data in rust_results.get('order_history', []):
            # Convert side string back to enum
            side = OrderSide.BUY if order_data['side'] == 'Buy' else OrderSide.SELL
            
            # Convert order type string back to enum
            order_type_map = {
                'Market': OrderType.MARKET,
                'Limit': OrderType.LIMIT,
                'Stop': OrderType.STOP,
                'StopLimit': OrderType.STOP_LIMIT,
            }
            order_type = order_type_map.get(order_data['order_type'], OrderType.MARKET)
            
            # Convert status string back to enum
            status_map = {
                'Created': OrderStatus.CREATED,
                'Submitted': OrderStatus.SUBMITTED,
                'Partial': OrderStatus.PARTIAL,
                'Filled': OrderStatus.FILLED,
                'Canceled': OrderStatus.CANCELED,
                'Rejected': OrderStatus.REJECTED,
            }
            status = status_map.get(order_data['status'], OrderStatus.CREATED)
            
            # Create order
            order = Order(
                symbol=order_data['symbol'],
                side=side,
                order_type=order_type,
                quantity=order_data['quantity'],
                limit_price=order_data.get('limit_price'),
                stop_price=order_data.get('stop_price'),
                created_at=pd.Timestamp(order_data['created_at'], unit='s'),
            )
            
            # Set order ID and status
            order.order_id = order_data['order_id']
            order.status = status
            
            # Add fills
            for fill_data in order_data.get('fills', []):
                order.add_fill(
                    quantity=fill_data['quantity'],
                    price=fill_data['price'],
                    timestamp=pd.Timestamp(fill_data['timestamp'], unit='s'),
                )
            
            order_history.append(order)
        
        # Extract metrics
        metrics = rust_results.get('metrics', {})
        
        return portfolio_history, trade_history, order_history, metrics
    
    def run(self, strategy_fn: Callable[[Dict[str, pd.DataFrame], Portfolio, int], List[Order]]) -> Dict[str, Any]:
        """
        Run the backtest using the Rust implementation.
        
        Args:
            strategy_fn: A function that takes (data, portfolio, current_idx) and returns a list of orders
        
        Returns:
            Dict[str, Any]: Dictionary containing backtest results
        """
        logger.info(f"Starting Rust-accelerated backtest with {len(self.data[self.symbols[0]])} bars")
        start_time = time.time()
        
        # Initialize portfolio for strategy function
        portfolio = Portfolio(initial_capital=self.initial_capital)
        
        # Get reference dataframe for iteration
        reference_df = self.data[self.symbols[0]]
        
        # Collect all orders from strategy
        all_orders = []
        
        # Run strategy to generate orders
        for i in range(len(reference_df)):
            current_timestamp = reference_df.index[i]
            
            # Get current data slice (up to current timestamp)
            current_data = {
                symbol: df.iloc[:i+1] for symbol, df in self.data.items()
            }
            
            # Execute strategy to get new orders
            new_orders = strategy_fn(current_data, portfolio, i)
            
            # Add timestamp to orders
            for order in new_orders:
                if not hasattr(order, 'created_at') or order.created_at is None:
                    order.created_at = current_timestamp
            
            all_orders.extend(new_orders)
        
        # Convert data and orders to Rust format
        rust_data = self._convert_data_to_rust_format()
        rust_orders = self._convert_orders_to_rust_format(all_orders)
        
        # Create config for Rust
        rust_config = {
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "slippage": self.slippage,
            "enable_fractional": self.enable_fractional,
        }
        
        # Run Rust backtester
        rust_results = run_backtest_rs(rust_data, rust_orders, rust_config)
        
        # Convert results back to Python
        portfolio_history, trade_history, order_history, metrics = self._convert_rust_results_to_python(rust_results)
        
        execution_time = time.time() - start_time
        logger.info(f"Rust backtest completed in {execution_time:.2f} seconds")
        logger.info(f"Final portfolio value: {portfolio_history[-1]['total_value']:.2f}")
        
        # Return results
        return {
            'portfolio_history': portfolio_history,
            'trade_history': trade_history,
            'order_history': order_history,
            'metrics': metrics,
            'execution_time': execution_time,
        }
