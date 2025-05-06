"""
Backtester module for AI Trading Agent.

This module provides the core backtesting functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import time
from dataclasses import dataclass, field

from ..trading_engine.models import Order, Trade, Position, Portfolio
from ..trading_engine.order_manager import OrderManager
from ..common import logger
from .performance_metrics import calculate_metrics, PerformanceMetrics
from ..trading_engine.enums import OrderType, OrderSide


class Backtester:
    """
    Backtester class for simulating trading strategies on historical data.
    
    This class handles the core backtesting loop, including:
    - Processing historical data
    - Executing trading strategy signals
    - Simulating order execution
    - Tracking portfolio performance
    - Calculating performance metrics
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        fixed_commission: float = 0.0,
        slippage: float = 0.0,
        bid_ask_spread: float = 0.0,
        slippage_model: str = "fixed",  # "fixed", "random", "volatility", "size"
        slippage_volatility: float = 0.0,  # For volatility-based slippage
        slippage_size_factor: float = 0.0,  # For size-based slippage
        price_impact_factor: float = 0.0,
        enable_fractional: bool = True,
        execution_delay: int = 0,
    ):
        """
        Initialize the backtester.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as a decimal (e.g., 0.001 = 0.1%)
            enable_fractional: Whether to allow fractional position sizes
        """
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.fixed_commission = fixed_commission
        self.slippage = slippage
        self.enable_fractional = enable_fractional
        self.bid_ask_spread = bid_ask_spread
        self.execution_delay = execution_delay
        self._delayed_orders = {}  # bar_idx -> list of orders to execute at that bar
        self.slippage_model = slippage_model
        self.slippage_volatility = slippage_volatility
        self.slippage_size_factor = slippage_size_factor
        self.price_impact_factor = price_impact_factor
        
        # Validate data
        self._validate_data()
        
        # Initialize trading components
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.order_manager = OrderManager(portfolio=self.portfolio)
        
        # Initialize results storage
        self.portfolio_history = []
        self.trade_history = []
        self.order_history = []
        
        logger.info(f"Initialized backtester with {len(self.symbols)} symbols and {initial_capital} initial capital")
    
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
    
    def run(self, strategy_fn: Callable[[Dict[str, pd.DataFrame], Portfolio, int], List[Order]]) -> PerformanceMetrics:
        """
        Run the backtest.
        
        Args:
            strategy_fn: A function that takes (data, portfolio, current_idx) and returns a list of orders
        
        Returns:
            PerformanceMetrics: Performance metrics for the backtest
        """
        logger.info(f"Starting backtest with {len(self.data[self.symbols[0]])} bars")
        start_time = time.time()
        
        # Get reference dataframe for iteration
        reference_df = self.data[self.symbols[0]]
        
        # Main backtesting loop
        for i in range(len(reference_df)):
            current_timestamp = reference_df.index[i]
            
            # Get current data slice (up to current timestamp)
            current_data = {
                symbol: df.iloc[:i+1] for symbol, df in self.data.items()
            }
            
            # Execute strategy to get new orders
            new_orders = strategy_fn(current_data, self.portfolio, i)
            
            # If execution_delay > 0, schedule orders for future bar
            if self.execution_delay > 0:
                execute_at = i + self.execution_delay
                if execute_at not in self._delayed_orders:
                    self._delayed_orders[execute_at] = []
                self._delayed_orders[execute_at].extend(new_orders)
                new_orders = []
            
            # Add any delayed orders scheduled for this bar
            if i in self._delayed_orders:
                new_orders.extend(self._delayed_orders.pop(i))
            # Process new orders
            for order in new_orders:
                self.order_manager.place_order(order)
                self.order_history.append(order)
            
            # Process pending orders with current bar data
            self._process_orders(current_timestamp, i)
            
            # Update portfolio state and record history
            self._update_portfolio_state(current_timestamp, i)
        
        # Calculate performance metrics
        metrics = calculate_metrics(
            self.portfolio_history,
            self.trade_history,
            self.initial_capital
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        logger.info(f"Final portfolio value: {self.portfolio.total_value:.2f}")
        
        return metrics
    
    def _process_orders(self, timestamp: pd.Timestamp, bar_idx: int):
        """
        Process pending orders using current bar data.
        
        Args:
            timestamp: Current timestamp
            bar_idx: Current bar index
        """
        # Get pending orders
        pending_orders = self.order_manager.get_open_orders()
        
        if not pending_orders:
            return
        
        logger.info(f"Processing {len(pending_orders)} pending orders at {timestamp}")
        
        # Iterate over the values of the pending_orders dictionary
        for order_id, order in pending_orders.items():
            symbol = order.symbol
            
            # Skip if we don't have data for this symbol
            if symbol not in self.data:
                logger.warning(f"No data for symbol {symbol}, skipping order {order.order_id}")
                continue
                
            # Get current bar data
            current_bar = self.data[symbol].iloc[bar_idx]
            
            # Simulate execution
            executed_price = self._calculate_execution_price(order, current_bar)
            
            # Apply commission and slippage
            effective_price = self._apply_transaction_costs(order, executed_price)
            
            # Create fill
            fill_qty = order.quantity
            if not self.enable_fractional:
                fill_qty = int(fill_qty)  # Round down to nearest integer
            
            # Add fill to order
            order.add_fill(
                quantity=fill_qty,
                price=effective_price,
                timestamp=timestamp
            )
            
            # Create trade record
            trade = Trade(
                symbol=order.symbol,
                order_id=order.order_id,
                side=order.side,
                quantity=fill_qty,
                price=effective_price,
                timestamp=timestamp
            )
            
            # Update portfolio with the trade
            current_prices = self._get_current_prices(bar_idx)
            self.portfolio.update_from_trade(trade, current_prices)
            
            # Record trade
            self.trade_history.append(trade)
            
            # Finalize order
            self.order_manager.finalize_order(order.order_id)
    
    def _get_current_prices(self, bar_idx: int) -> Dict[str, float]:
        """
        Get current prices for all symbols at the given bar index.
        
        Args:
            bar_idx: Current bar index
            
        Returns:
            Dict[str, float]: Dictionary mapping symbols to their current prices
        """
        current_prices = {}
        for symbol, df in self.data.items():
            if bar_idx < len(df):
                current_prices[symbol] = df.iloc[bar_idx]['close']
        return current_prices
    
    def _calculate_execution_price(self, order: Order, bar: pd.Series) -> float:
        """
        Calculate execution price based on order type and current bar.
        
        Args:
            order: The order to execute
            bar: Current price bar
            
        Returns:
            float: Execution price
        """
        # For simplicity, we'll use the following rules:
        # - Market orders: Execute at the next bar's open
        # - Limit buy: Execute at limit price if low <= limit price
        # - Limit sell: Execute at limit price if high >= limit price
        # - Stop buy: Execute at stop price if high >= stop price
        # - Stop sell: Execute at stop price if low <= stop price
        
        if order.type == OrderType.MARKET:
            # Simulate bid/ask spread for market orders
            price = bar['open']
            if self.bid_ask_spread > 0:
                if order.side == OrderSide.BUY:
                    price = price * (1 + self.bid_ask_spread / 2)
                else:  # SELL
                    price = price * (1 - self.bid_ask_spread / 2)
            # Simulate price impact for large orders
            if self.price_impact_factor > 0:
                # Use volume if available, else assume 1
                bar_volume = bar.get('volume', 1)
                impact = self.price_impact_factor * (order.quantity / max(bar_volume, 1e-8))
                if order.side == OrderSide.BUY:
                    price = price * (1 + impact)
                else:
                    price = price * (1 - impact)
            return price
        
        elif order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and bar['low'] <= order.price:
                return min(bar['open'], order.price)
            elif order.side == OrderSide.SELL and bar['high'] >= order.price:
                return max(bar['open'], order.price)
        
        # Default case: order not executed
        return None
    
    def _apply_transaction_costs(self, order: Order, executed_price: float) -> float:
        """
        Apply commission and slippage to execution price.
        
        Args:
            order: The order being executed
            executed_price: The raw execution price
            
        Returns:
            float: Effective price after transaction costs
        """
        if executed_price is None:
            return None
        
        # Apply slippage
        effective_price = executed_price
        if self.slippage_model == "fixed":
            if order.side == OrderSide.BUY:
                effective_price = executed_price * (1 + self.slippage)
            else:
                effective_price = executed_price * (1 - self.slippage)
        elif self.slippage_model == "random":
            import random
            slip = random.uniform(-self.slippage, self.slippage)
            if order.side == OrderSide.BUY:
                effective_price = executed_price * (1 + abs(slip))
            else:
                effective_price = executed_price * (1 - abs(slip))
        elif self.slippage_model == "volatility":
            slip = self.slippage_volatility
            if order.side == OrderSide.BUY:
                effective_price = executed_price * (1 + slip)
            else:
                effective_price = executed_price * (1 - slip)
        elif self.slippage_model == "size":
            # Slippage increases with order size relative to bar volume
            bar_volume = getattr(order, "bar_volume", 1)
            slip = self.slippage_size_factor * (order.quantity / max(bar_volume, 1e-8))
            if order.side == OrderSide.BUY:
                effective_price = executed_price * (1 + slip)
            else:
                effective_price = executed_price * (1 - slip)
        
        # Apply commission (percentage-based and fixed)
        trade_value = abs(order.quantity * effective_price)
        commission = trade_value * self.commission_rate + self.fixed_commission
        # For buys, increase effective price; for sells, decrease
        if order.side == OrderSide.BUY:
            effective_price += commission / max(order.quantity, 1e-8)
        else:
            effective_price -= commission / max(order.quantity, 1e-8)
        
        return effective_price
    
    def _update_portfolio_state(self, timestamp: pd.Timestamp, bar_idx: int):
        """
        Update portfolio state and record history.
        
        Args:
            timestamp: Current timestamp
            bar_idx: Current bar index
        """
        # Update portfolio total value
        current_prices = self._get_current_prices(bar_idx)
        self.portfolio.update_total_value(current_prices)
        
        # Record portfolio state
        portfolio_snapshot = {
            'timestamp': timestamp,
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'positions': {symbol: {'quantity': pos.quantity, 'value': pos.get_position_value(current_prices.get(symbol, pos.entry_price))} 
                         for symbol, pos in self.portfolio.positions.items()}
        }
        self.portfolio_history.append(portfolio_snapshot)
