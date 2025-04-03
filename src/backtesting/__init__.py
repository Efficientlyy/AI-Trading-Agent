"""
Backtesting module for the AI Crypto Trading System.

This module provides backtesting capabilities using a high-performance Rust implementation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import os
import uuid
import json
from enum import Enum
import matplotlib.pyplot as plt

# Import the Rust bridge
from ..rust_bridge import is_rust_available
if is_rust_available():
    from ..rust_bridge import get_backtest_engine, BacktestStats
else:
    logging.warning("Rust implementation not available. Falling back to pure Python implementation.")
    # We'll define equivalent pure Python classes

# Define common enums
class TimeFrame(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class BacktestMode(str, Enum):
    CANDLES = "candles"
    TRADES = "trades"
    ORDERBOOK = "orderbook"

# Pure Python implementation for when Rust is not available
class PyBacktestStats:
    """Statistics from a completed backtest."""
    
    def __init__(self, 
                 start_time: int, 
                 end_time: int,
                 initial_balance: float,
                 final_balance: float,
                 total_trades: int = 0,
                 winning_trades: int = 0,
                 losing_trades: int = 0,
                 total_profit: float = 0.0,
                 total_loss: float = 0.0,
                 max_drawdown: float = 0.0,
                 max_drawdown_pct: float = 0.0,
                 sharpe_ratio: Optional[float] = None,
                 profit_factor: Optional[float] = None,
                 win_rate: float = 0.0,
                 avg_win: Optional[float] = None,
                 avg_loss: Optional[float] = None,
                 largest_win: Optional[float] = None,
                 largest_loss: Optional[float] = None):
        
        self.start_time = start_time
        self.end_time = end_time
        self.initial_balance = initial_balance
        self.final_balance = final_balance
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.total_profit = total_profit
        self.total_loss = total_loss
        self.max_drawdown = max_drawdown
        self.max_drawdown_pct = max_drawdown_pct
        self.sharpe_ratio = sharpe_ratio
        self.profit_factor = profit_factor
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.largest_win = largest_win
        self.largest_loss = largest_loss
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stats to a dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
        }
    
    def __str__(self) -> str:
        """Return a string representation of the stats."""
        profit_pct = ((self.final_balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        return f"BacktestStats(profit: {profit_pct:.2f}%, trades: {self.total_trades}, win_rate: {self.win_rate:.2f}%)"
    
    def __repr__(self) -> str:
        return self.__str__()

class PyBacktestEngine:
    """Pure Python implementation of the backtesting engine."""
    
    def __init__(self,
                 initial_balance: float,
                 symbols: List[str],
                 start_time: int,
                 end_time: int,
                 mode: str = "candles",
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005,
                 enable_fractional_sizing: bool = True):
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.peak_equity = initial_balance
        self.symbols = symbols
        self.start_time = start_time
        self.end_time = end_time
        self.mode = mode
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.enable_fractional_sizing = enable_fractional_sizing
        
        self.positions = {}  # Symbol -> Position
        self.open_orders = []  # List of orders
        self.filled_orders = []  # List of filled orders
        self.equity_curve = [(start_time, initial_balance)]  # List of (timestamp, equity)
        self.current_time = start_time
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        
    def process_candle(self,
                       symbol: str,
                       timestamp: int,
                       open_price: float,
                       high: float,
                       low: float,
                       close: float,
                       volume: float,
                       timeframe: str) -> None:
        """Process a candle update."""
        if self.mode != "candles":
            raise ValueError("Cannot process candles in non-candle mode")
        
        # Update current time
        self.current_time = timestamp
        
        # Process orders that could be triggered by this candle
        self._process_orders_with_candle(symbol, timestamp, open_price, high, low, close)
        
        # Update positions with the closing price
        if symbol in self.positions:
            self._update_position_price(symbol, close, timestamp)
        
        # Update equity
        self._update_equity()
        
        # Add to equity curve
        self.equity_curve.append((timestamp, self.equity))
    
    def _process_orders_with_candle(self, symbol: str, timestamp: int, open_price: float, high: float, low: float, close: float) -> None:
        """Process orders that could be triggered by a candle."""
        orders_to_process = []
        
        # Find orders for this symbol
        for i, order in enumerate(self.open_orders[:]):
            if order["symbol"] = = symbol:
                orders_to_process.append(order)
                self.open_orders.remove(order)
        
        # Process each order
        for order in orders_to_process:
            executed_price = None
            
            if order["order_type"] = = OrderType.MARKET:
                # Market orders execute at current price with slippage
                if order["side"] = = OrderSide.BUY:
                    executed_price = close * (1 + self.slippage)
                else:
                    executed_price = close * (1 - self.slippage)
            
            elif order["order_type"] = = OrderType.LIMIT:
                # Limit orders execute if price crosses the limit price
                if order["side"] = = OrderSide.BUY and low <= order["price"]:
                    executed_price = order["price"]
                elif order["side"] = = OrderSide.SELL and high >= order["price"]:
                    executed_price = order["price"]
            
            elif order["order_type"] = = OrderType.STOP_MARKET:
                # Stop orders execute if price crosses the stop price
                if order["side"] = = OrderSide.BUY and high >= order["stop_price"]:
                    executed_price = order["stop_price"] * (1 + self.slippage)
                elif order["side"] = = OrderSide.SELL and low <= order["stop_price"]:
                    executed_price = order["stop_price"] * (1 - self.slippage)
            
            elif order["order_type"] = = OrderType.STOP_LIMIT:
                # Stop limit orders: first the stop price must be reached,
                # then the limit order is placed
                if order["side"] = = OrderSide.BUY and high >= order["stop_price"] and low <= order["price"]:
                    executed_price = order["price"]
                elif order["side"] = = OrderSide.SELL and low <= order["stop_price"] and high >= order["price"]:
                    executed_price = order["price"]
            
            if executed_price is not None:
                self._execute_order(order, executed_price, timestamp)
            else:
                # Put the order back if not executed
                self.open_orders.append(order)
    
    def _execute_order(self, order: Dict[str, Any], price: float, timestamp: int) -> None:
        """Execute an order at the given price."""
        commission = price * order["amount"] * self.commission_rate
        
        # Update order info
        order["executed_price"] = price
        order["executed_at"] = timestamp
        order["status"] = "filled"
        
        if order["side"] = = OrderSide.BUY:
            # Subtract the cost plus commission from balance
            cost = price * order["amount"] + commission
            if cost > self.balance:
                logging.warning("Insufficient balance for buy order")
                order["status"] = "rejected"
                self.filled_orders.append(order)
                return
            
            self.balance -= cost
            
            # Update or create position
            if order["symbol"] in self.positions:
                position = self.positions[order["symbol"]]
                if position["amount"] < 0:
                    # If it's a short position, reduce or close it
                    realized_pnl = self._close_position(order["symbol"], order["amount"], price, timestamp)
                    self._update_stats_with_trade(realized_pnl)
                    
                    if position["amount"] = = 0:
                        # Position fully closed
                        del self.positions[order["symbol"]]
                else:
                    # If it's a long position, increase it (dollar-cost averaging)
                    position["entry_price"] = ((position["entry_price"] * position["amount"]) + (price * order["amount"])) / (position["amount"] + order["amount"])
                    position["amount"] += order["amount"]
                    position["updated_at"] = timestamp
            else:
                # Create new long position
                self.positions[order["symbol"]] = {
                    "symbol": order["symbol"],
                    "amount": order["amount"],
                    "entry_price": price,
                    "current_price": price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "opened_at": timestamp,
                    "updated_at": timestamp,
                }
        
        elif order["side"] = = OrderSide.SELL:
            sell_amount = order["amount"]
            
            if order["symbol"] in self.positions:
                position = self.positions[order["symbol"]]
                if position["amount"] > 0:
                    # Close long position (partially or fully)
                    realized_pnl = self._close_position(order["symbol"], -sell_amount, price, timestamp)
                    self._update_stats_with_trade(realized_pnl)
                    
                    # Add the proceeds minus commission to balance
                    proceeds = price * sell_amount - commission
                    self.balance += proceeds
                    
                    if position["amount"] = = 0:
                        # Position fully closed
                        del self.positions[order["symbol"]]
                else:
                    # Increase short position
                    total_amount = position["amount"] - sell_amount
                    total_cost = (position["entry_price"] * abs(position["amount"])) + (price * sell_amount)
                    position["entry_price"] = total_cost / abs(total_amount)
                    position["amount"] = total_amount
                    position["updated_at"] = timestamp
                    
                    # Add the proceeds minus commission to balance
                    proceeds = price * sell_amount - commission
                    self.balance += proceeds
            else:
                # Create new short position
                self.positions[order["symbol"]] = {
                    "symbol": order["symbol"],
                    "amount": -sell_amount,
                    "entry_price": price,
                    "current_price": price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "opened_at": timestamp,
                    "updated_at": timestamp,
                }
                
                # Add the proceeds minus commission to balance
                proceeds = price * sell_amount - commission
                self.balance += proceeds
        
        # Add to filled orders
        self.filled_orders.append(order)
        
        # Update equity
        self._update_equity()
    
    def _close_position(self, symbol: str, amount: float, price: float, timestamp: int) -> float:
        """Close part or all of a position."""
        position = self.positions[symbol]
        if abs(amount) > abs(position["amount"]):
            amount = position["amount"] if position["amount"] > 0 else -position["amount"]
        
        direction = 1 if position["amount"] > 0 else -1
        pnl = (price - position["entry_price"]) * abs(amount) * direction
        
        # Update the position
        position["amount"] -= amount
        position["realized_pnl"] += pnl
        position["updated_at"] = timestamp
        
        # If position is closed, reset unrealized PnL
        if position["amount"] = = 0:
            position["unrealized_pnl"] = 0.0
        else:
            # Recalculate unrealized PnL for remaining position
            position["unrealized_pnl"] = (position["current_price"] - position["entry_price"]) * abs(position["amount"]) * direction
        
        return pnl
    
    def _update_position_price(self, symbol: str, price: float, timestamp: int) -> None:
        """Update a position with the current price."""
        position = self.positions[symbol]
        position["current_price"] = price
        position["updated_at"] = timestamp
        
        # Calculate unrealized PnL
        direction = 1 if position["amount"] > 0 else -1
        position["unrealized_pnl"] = (price - position["entry_price"]) * abs(position["amount"]) * direction
    
    def _update_equity(self) -> None:
        """Update equity calculation based on current positions."""
        total_equity = self.balance
        
        # Add value of all positions
        for symbol, position in self.positions.items():
            total_equity += position["unrealized_pnl"]
        
        self.equity = total_equity
        
        # Update peak equity for drawdown calculations
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def _update_stats_with_trade(self, profit_loss: float) -> None:
        """Update stats with a completed trade."""
        self.total_trades += 1
        
        if profit_loss > 0:
            self.winning_trades += 1
            self.total_profit += profit_loss
        elif profit_loss < 0:
            self.losing_trades += 1
            self.total_loss += abs(profit_loss)
    
    def _calculate_drawdown(self) -> float:
        """Calculate the current drawdown."""
        if self.peak_equity > self.equity:
            return self.peak_equity - self.equity
        return 0.0
    
    def submit_market_order(self, symbol: str, side: str, amount: float) -> str:
        """Submit a market order to the backtesting engine."""
        order_id = str(uuid.uuid4())
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": OrderType.MARKET,
            "amount": amount,
            "price": None,
            "status": "created",
            "created_at": int(time.time()),
            "executed_at": None,
            "executed_price": None,
            "stop_price": None,
        }
        
        self.open_orders.append(order)
        return order_id
    
    def submit_limit_order(self, symbol: str, side: str, price: float, amount: float) -> str:
        """Submit a limit order to the backtesting engine."""
        order_id = str(uuid.uuid4())
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": OrderType.LIMIT,
            "amount": amount,
            "price": price,
            "status": "created",
            "created_at": int(time.time()),
            "executed_at": None,
            "executed_price": None,
            "stop_price": None,
        }
        
        self.open_orders.append(order)
        return order_id
    
    def submit_stop_market_order(self, symbol: str, side: str, stop_price: float, amount: float) -> str:
        """Submit a stop market order to the backtesting engine."""
        order_id = str(uuid.uuid4())
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": OrderType.STOP_MARKET,
            "amount": amount,
            "price": None,
            "status": "created",
            "created_at": int(time.time()),
            "executed_at": None,
            "executed_price": None,
            "stop_price": stop_price,
        }
        
        self.open_orders.append(order)
        return order_id
    
    def submit_stop_limit_order(self, symbol: str, side: str, stop_price: float, limit_price: float, amount: float) -> str:
        """Submit a stop limit order to the backtesting engine."""
        order_id = str(uuid.uuid4())
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": OrderType.STOP_LIMIT,
            "amount": amount,
            "price": limit_price,
            "status": "created",
            "created_at": int(time.time()),
            "executed_at": None,
            "executed_price": None,
            "stop_price": stop_price,
        }
        
        self.open_orders.append(order)
        return order_id
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        for i, order in enumerate(self.open_orders):
            if order["id"] = = order_id:
                order["status"] = "canceled"
                self.filled_orders.append(order)
                self.open_orders.pop(i)
                return
        
        raise ValueError(f"Order with ID {order_id} not found")
    
    def get_balance(self) -> float:
        """Get the current balance."""
        return self.balance
    
    def get_equity(self) -> float:
        """Get the current equity."""
        return self.equity
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get the current open positions."""
        return self.positions.copy()
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get the equity curve."""
        return [{"timestamp": ts, "equity": eq} for ts, eq in self.equity_curve]
    
    def run(self) -> PyBacktestStats:
        """Run the backtest and get the final statistics."""
        # Calculate final statistics
        max_drawdown = self._calculate_drawdown()
        max_drawdown_pct = (max_drawdown / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        # Calculate profit factor
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else None
        
        # Calculate average win/loss
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else None
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else None
        
        # Calculate Sharpe ratio using daily returns
        sharpe_ratio = None
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1][1]
                if prev_equity > 0:
                    returns.append((self.equity_curve[i][1] - prev_equity) / prev_equity)
            
            if returns:
                avg_return = sum(returns) / len(returns)
                std_dev = np.std(returns) if len(returns) > 1 else 1.0
                if std_dev > 0:
                    sharpe_ratio = (avg_return / std_dev) * np.sqrt(252)  # Annualized
        
        return PyBacktestStats(
            start_time=self.start_time,
            end_time=self.current_time,
            initial_balance=self.initial_balance,
            final_balance=self.equity,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            total_profit=self.total_profit,
            total_loss=self.total_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max([o["executed_price"] * o["amount"] - o["price"] * o["amount"] 
                            for o in self.filled_orders 
                            if o["status"] = = "filled" and o["side"] = = OrderSide.SELL and o["executed_price"]] + [0]) if self.filled_orders else None,                            if o["status"] = = "filled" and o["side"] = = OrderSide.SELL and o["executed_price"]] + [0]) if self.filled_orders else None,
            largest_loss=max([o["price"] * o["amount"] - o["executed_price"] * o["amount"] 
                             for o in self.filled_orders 
                             if o["status"] = = "filled" and o["side"] = = OrderSide.SELL and o["executed_price"]] + [0]) if self.filled_orders else None,                             if o["status"] = = "filled" and o["side"] = = OrderSide.SELL and o["executed_price"]] + [0]) if self.filled_orders else None,
        )

class BacktestEngine:
    """Main backtesting engine class that uses either Rust or Python implementation."""
    
    def __init__(self,
                 initial_balance: float,
                 symbols: List[str],
                 start_time: Union[int, datetime],
                 end_time: Union[int, datetime],
                 mode: str = "candles",
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005,
                 enable_fractional_sizing: bool = True):
        """Initialize the backtesting engine."""
        # Convert times to timestamps if they're datetime objects
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        
        # Use Rust if available, otherwise fall back to Python
        if is_rust_available():
            self.engine = get_backtest_engine(
                initial_balance=initial_balance,
                symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                mode=mode,
                commission_rate=commission_rate,
                slippage=slippage,
                enable_fractional_sizing=enable_fractional_sizing
            )
            self.using_rust = True
        else:
            self.engine = PyBacktestEngine(
                initial_balance=initial_balance,
                symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                mode=mode,
                commission_rate=commission_rate,
                slippage=slippage,
                enable_fractional_sizing=enable_fractional_sizing
            )
            self.using_rust = False
    
    def process_candle(self,
                       symbol: str,
                       timestamp: Union[int, datetime],
                       open_price: float,
                       high: float,
                       low: float,
                       close: float,
                       volume: float,
                       timeframe: str) -> None:
        """Process a candle in the backtest."""
        if isinstance(timestamp, datetime):
            timestamp = int(timestamp.timestamp())
        
        self.engine.process_candle(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=timeframe
        )
    
    def process_candles_df(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Process candles from a pandas DataFrame."""
        for _, row in df.iterrows():
            timestamp = int(row["timestamp"].timestamp()) if isinstance(row["timestamp"], datetime) else int(row["timestamp"])
            
            self.process_candle(
                symbol=symbol,
                timestamp=timestamp,
                open_price=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timeframe=timeframe
            )
    
    def submit_market_order(self, symbol: str, side: str, amount: float) -> str:
        """Submit a market order to the backtesting engine."""
        return self.engine.submit_market_order(symbol, side, amount)
    
    def submit_limit_order(self, symbol: str, side: str, price: float, amount: float) -> str:
        """Submit a limit order to the backtesting engine."""
        return self.engine.submit_limit_order(symbol, side, price, amount)
    
    def submit_stop_market_order(self, symbol: str, side: str, stop_price: float, amount: float) -> str:
        """Submit a stop market order to the backtesting engine."""
        return self.engine.submit_stop_market_order(symbol, side, stop_price, amount)
    
    def submit_stop_limit_order(self, symbol: str, side: str, stop_price: float, limit_price: float, amount: float) -> str:
        """Submit a stop limit order to the backtesting engine."""
        return self.engine.submit_stop_limit_order(symbol, side, stop_price, limit_price, amount)
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        self.engine.cancel_order(order_id)
    
    def get_balance(self) -> float:
        """Get the current balance."""
        return self.engine.get_balance()
    
    def get_equity(self) -> float:
        """Get the current equity."""
        return self.engine.get_equity()
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get the current open positions."""
        return self.engine.get_positions()
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get the equity curve."""
        return self.engine.get_equity_curve()
    
    def run(self) -> Union[PyBacktestStats, Any]:
        """Run the backtest and get the final statistics."""
        return self.engine.run()
    
    def plot_equity_curve(self, title: str = "Equity Curve") -> None:
        """Plot the equity curve."""
        equity_curve = self.get_equity_curve()
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["equity"], label="Equity")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.show()
    
    def plot_drawdown(self, title: str = "Drawdown") -> None:
        """Plot the drawdown over time."""
        equity_curve = self.get_equity_curve()
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        
        # Calculate running maximum and drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["drawdown"], label="Drawdown %", color="red")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown %")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.show()
    
    def plot_stats(self, title: str = "Backtest Results") -> None:
        """Plot various statistics from the backtest."""
        stats = self.run()
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Equity curve
        equity_curve = self.get_equity_curve()
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        axs[0, 0].plot(df.index, df["equity"])
        axs[0, 0].set_title("Equity Curve")
        axs[0, 0].set_xlabel("Date")
        axs[0, 0].set_ylabel("Equity")
        axs[0, 0].grid(True)
        
        # Drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100
        axs[0, 1].plot(df.index, df["drawdown"], color="red")
        axs[0, 1].set_title("Drawdown %")
        axs[0, 1].set_xlabel("Date")
        axs[0, 1].set_ylabel("Drawdown %")
        axs[0, 1].grid(True)
        
        # Trade statistics
        labels = ["Winning Trades", "Losing Trades"]
        values = [stats.winning_trades, stats.losing_trades]
        colors = ["green", "red"]
        axs[1, 0].bar(labels, values, color=colors)
        axs[1, 0].set_title("Trade Statistics")
        axs[1, 0].set_xlabel("Trade Outcome")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].grid(True)
        
        # Profit/Loss stats
        if hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
        else:
            stats_dict = {
                "total_profit": stats.total_profit,
                "total_loss": stats.total_loss,
                "initial_balance": stats.initial_balance,
                "final_balance": stats.final_balance,
            }
            
        profit_loss_data = [
            stats_dict["final_balance"] - stats_dict["initial_balance"],
            stats_dict["total_profit"],
            stats_dict["total_loss"]
        ]
        profit_loss_labels = ["Net P/L", "Gross Profit", "Gross Loss"]
        profit_loss_colors = ["blue", "green", "red"]
        axs[1, 1].bar(profit_loss_labels, profit_loss_data, color=profit_loss_colors)
        axs[1, 1].set_title("Profit/Loss Statistics")
        axs[1, 1].set_xlabel("Category")
        axs[1, 1].set_ylabel("Amount")
        axs[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
        plt.show()
    
    def save_results(self, filename: str) -> None:
        """Save backtest results to a file."""
        stats = self.run()
        equity_curve = self.get_equity_curve()
        
        # Convert stats to dictionary
        if hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
        else:
            # Direct conversion for Rust stats
            stats_dict = {
                "start_time": stats.start_time,
                "end_time": stats.end_time,
                "initial_balance": stats.initial_balance,
                "final_balance": stats.final_balance,
                "total_trades": stats.total_trades,
                "winning_trades": stats.winning_trades,
                "losing_trades": stats.losing_trades,
                "total_profit": stats.total_profit,
                "total_loss": stats.total_loss,
                "max_drawdown": stats.max_drawdown,
                "max_drawdown_pct": stats.max_drawdown_pct,
                "sharpe_ratio": stats.sharpe_ratio,
                "profit_factor": stats.profit_factor,
                "win_rate": stats.win_rate,
                "avg_win": stats.avg_win,
                "avg_loss": stats.avg_loss,
                "largest_win": stats.largest_win,
                "largest_loss": stats.largest_loss,
            }
        
        # Create results dictionary
        results = {
            "stats": stats_dict,
            "equity_curve": equity_curve
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Backtest results saved to {filename}")

    @staticmethod
    def load_results(filename: str) -> Dict[str, Any]:
        """Load backtest results from a file."""
        with open(filename, "r") as f:
            results = json.load(f)
        
        # Convert timestamps to datetime
        for point in results["equity_curve"]:
            point["timestamp"] = datetime.fromtimestamp(point["timestamp"])
        
        return results

# Helper function to create a backtest engine
def create_backtest_engine(
    initial_balance: float,
    symbols: List[str],
    start_time: Union[int, datetime],
    end_time: Union[int, datetime],
    mode: str = "candles",
    commission_rate: float = 0.001,
    slippage: float = 0.0005,
    enable_fractional_sizing: bool = True
) -> BacktestEngine:
    """Create a new backtest engine."""
    return BacktestEngine(
        initial_balance=initial_balance,
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        commission_rate=commission_rate,
        slippage=slippage,
        enable_fractional_sizing=enable_fractional_sizing
    ) 