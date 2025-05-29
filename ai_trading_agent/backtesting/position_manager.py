"""
Position Manager Module

This module provides position management functionality for the backtesting framework,
including position scaling, risk management, and position tracking.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import uuid
import math

from ..common.utils import get_logger

class Position:
    """
    Class representing a trading position with scaling capabilities.
    
    A position can consist of multiple entries and exits, allowing for scaling in and out.
    """
    
    def __init__(self, symbol: str, direction: str, initial_risk_pct: float = 1.0, 
                max_risk_pct: float = 2.0, position_id: str = None):
        """
        Initialize a trading position.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            initial_risk_pct: Initial risk percentage of account (0.0-100.0)
            max_risk_pct: Maximum risk percentage allowed for this position
            position_id: Unique ID for this position, generated if not provided
        """
        self.symbol = symbol
        self.direction = direction.lower()
        self.position_id = position_id or str(uuid.uuid4())
        self.initial_risk_pct = initial_risk_pct
        self.max_risk_pct = max_risk_pct
        self.status = "open"
        self.open_date = datetime.now()
        self.close_date = None
        
        # Track entries and exits
        self.entries = []  # List of entry dictionaries
        self.exits = []    # List of exit dictionaries
        
        # Position metrics
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_pnl = 0.0
        self.current_risk_pct = initial_risk_pct
        
        # Scaling metrics
        self.avg_entry_price = 0.0
        self.avg_exit_price = 0.0
        self.total_quantity = 0.0
        self.remaining_quantity = 0.0
        self.exit_quantity = 0.0
        self.pct_closed = 0.0  # Percentage of position that has been closed
        
    def add_entry(self, price: float, quantity: float, date: datetime = None, 
                 tags: List[str] = None, risk_pct: float = None) -> Dict:
        """
        Add an entry to the position (scale in).
        
        Args:
            price: Entry price
            quantity: Entry quantity
            date: Entry date, defaults to now
            tags: List of tags for this entry
            risk_pct: Risk percentage for this specific entry
            
        Returns:
            Entry dictionary
        """
        if self.status == "closed":
            raise ValueError(f"Cannot add entry to closed position {self.position_id}")
            
        entry_date = date or datetime.now()
        entry_risk_pct = risk_pct or self.initial_risk_pct
        
        # Create entry record
        entry = {
            "entry_id": str(uuid.uuid4()),
            "price": price,
            "quantity": quantity,
            "date": entry_date,
            "tags": tags or [],
            "risk_pct": entry_risk_pct,
        }
        
        # Add to entries list
        self.entries.append(entry)
        
        # Update position metrics
        self._update_position_metrics()
        
        return entry
        
    def add_exit(self, price: float, quantity: float, date: datetime = None, 
                tags: List[str] = None) -> Dict:
        """
        Add an exit to the position (scale out).
        
        Args:
            price: Exit price
            quantity: Exit quantity
            date: Exit date, defaults to now
            tags: List of tags for this exit
            
        Returns:
            Exit dictionary
        """
        if self.status == "closed":
            raise ValueError(f"Cannot add exit to closed position {self.position_id}")
            
        # Ensure we don't exit more than the remaining quantity
        if quantity > self.remaining_quantity:
            quantity = self.remaining_quantity
            
        exit_date = date or datetime.now()
        
        # Create exit record
        exit_record = {
            "exit_id": str(uuid.uuid4()),
            "price": price,
            "quantity": quantity,
            "date": exit_date,
            "tags": tags or [],
        }
        
        # Add to exits list
        self.exits.append(exit_record)
        
        # Update position metrics
        self._update_position_metrics()
        
        # Check if position is fully closed
        if self.remaining_quantity <= 0.0001:  # Allow for floating point errors
            self.status = "closed"
            self.close_date = exit_date
            
        return exit_record
        
    def _update_position_metrics(self):
        """Update all position metrics based on entries and exits."""
        # Calculate total quantity and average entry price
        total_entry_value = 0.0
        self.total_quantity = 0.0
        
        for entry in self.entries:
            entry_value = entry["price"] * entry["quantity"]
            total_entry_value += entry_value
            self.total_quantity += entry["quantity"]
            
        # Calculate average entry price
        if self.total_quantity > 0:
            self.avg_entry_price = total_entry_value / self.total_quantity
        else:
            self.avg_entry_price = 0.0
            
        # Calculate exit quantity and average exit price
        total_exit_value = 0.0
        self.exit_quantity = 0.0
        
        for exit_record in self.exits:
            exit_value = exit_record["price"] * exit_record["quantity"]
            total_exit_value += exit_value
            self.exit_quantity += exit_record["quantity"]
            
        # Calculate average exit price
        if self.exit_quantity > 0:
            self.avg_exit_price = total_exit_value / self.exit_quantity
        else:
            self.avg_exit_price = 0.0
            
        # Calculate remaining quantity
        self.remaining_quantity = max(0.0, self.total_quantity - self.exit_quantity)
        
        # Calculate percentage closed
        if self.total_quantity > 0:
            self.pct_closed = min(100.0, (self.exit_quantity / self.total_quantity) * 100.0)
        else:
            self.pct_closed = 0.0
            
        # Calculate realized P&L
        if self.direction == "long":
            self.realized_pnl = total_exit_value - (self.avg_entry_price * self.exit_quantity)
        else:  # short
            self.realized_pnl = (self.avg_entry_price * self.exit_quantity) - total_exit_value
            
        # Calculate current risk percentage
        total_risk = sum(entry["risk_pct"] for entry in self.entries)
        used_risk = total_risk * (self.exit_quantity / self.total_quantity) if self.total_quantity > 0 else 0
        self.current_risk_pct = total_risk - used_risk
        
    def update_unrealized_pnl(self, current_price: float):
        """
        Update unrealized P&L based on current market price.
        
        Args:
            current_price: Current market price
        """
        if self.remaining_quantity <= 0:
            self.unrealized_pnl = 0.0
        else:
            if self.direction == "long":
                self.unrealized_pnl = (current_price - self.avg_entry_price) * self.remaining_quantity
            else:  # short
                self.unrealized_pnl = (self.avg_entry_price - current_price) * self.remaining_quantity
                
        # Update total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this position.
        
        Returns:
            Dictionary with position details
        """
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "status": self.status,
            "open_date": self.open_date,
            "close_date": self.close_date,
            "avg_entry_price": self.avg_entry_price,
            "avg_exit_price": self.avg_exit_price,
            "total_quantity": self.total_quantity,
            "remaining_quantity": self.remaining_quantity,
            "exit_quantity": self.exit_quantity,
            "pct_closed": self.pct_closed,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "current_risk_pct": self.current_risk_pct,
            "entries_count": len(self.entries),
            "exits_count": len(self.exits),
        }

class PositionManager:
    """
    Manager for trading positions with sophisticated risk management and scaling capabilities.
    """
    
    def __init__(self, initial_capital: float = 100000.0, max_risk_per_trade: float = 1.0,
                max_risk_total: float = 5.0, position_sizing_method: str = "risk_based"):
        """
        Initialize the position manager.
        
        Args:
            initial_capital: Initial capital amount
            max_risk_per_trade: Maximum risk percentage per trade (0.0-100.0)
            max_risk_total: Maximum total risk percentage across all positions
            position_sizing_method: Method for sizing positions ('risk_based', 'fixed_lot', 'volatility_based')
        """
        self.logger = get_logger("PositionManager")
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_total = max_risk_total
        self.position_sizing_method = position_sizing_method
        
        # Position tracking
        self.positions = {}  # Map position_id -> Position
        self.open_positions = {}  # Map symbol -> position_id
        self.closed_positions = []  # List of closed position_ids
        
        # Performance tracking
        self.equity_curve = []
        self.drawdowns = []
        self.daily_returns = []
        
        # Risk tracking
        self.current_total_risk_pct = 0.0
        
    def can_open_position(self, symbol: str, risk_pct: float = None) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on risk limits.
        
        Args:
            symbol: Trading symbol
            risk_pct: Risk percentage for the new position
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check if we already have an open position for this symbol
        if symbol in self.open_positions:
            return False, f"Position already exists for {symbol}"
            
        # Check risk limit per trade
        actual_risk = risk_pct or self.max_risk_per_trade
        if actual_risk > self.max_risk_per_trade:
            return False, f"Risk {actual_risk}% exceeds max risk per trade {self.max_risk_per_trade}%"
            
        # Check total risk limit
        if self.current_total_risk_pct + actual_risk > self.max_risk_total:
            return False, f"Total risk would exceed maximum {self.max_risk_total}%"
            
        return True, "Position can be opened"
        
    def open_position(self, symbol: str, direction: str, risk_pct: float = None,
                     entry_price: float = None, stop_loss: float = None,
                     quantity: float = None, entry_date: datetime = None) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            risk_pct: Risk percentage for this position
            entry_price: Entry price
            stop_loss: Stop loss price
            quantity: Position quantity (calculated if not provided)
            entry_date: Entry date
            
        Returns:
            New Position object or None if position cannot be opened
        """
        # Use default risk if not specified
        actual_risk = risk_pct or self.max_risk_per_trade
        
        # Check if position can be opened
        can_open, reason = self.can_open_position(symbol, actual_risk)
        if not can_open:
            self.logger.warning(f"Cannot open position: {reason}")
            return None
            
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            initial_risk_pct=actual_risk,
            max_risk_pct=self.max_risk_per_trade,
        )
        
        # Add position to tracking
        self.positions[position.position_id] = position
        self.open_positions[symbol] = position.position_id
        
        # Update risk tracking
        self.current_total_risk_pct += actual_risk
        
        # Add entry if price is provided
        if entry_price is not None and (quantity is not None or stop_loss is not None):
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self._calculate_position_size(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    risk_pct=actual_risk
                )
                
            # Add entry
            position.add_entry(
                price=entry_price,
                quantity=quantity,
                date=entry_date,
                risk_pct=actual_risk
            )
            
        self.logger.info(f"Opened {direction} position for {symbol} with {actual_risk}% risk")
        return position
        
    def scale_in(self, symbol: str, price: float, quantity: float = None,
                risk_pct: float = None, stop_loss: float = None,
                date: datetime = None, tags: List[str] = None) -> Optional[Dict]:
        """
        Scale into an existing position.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            quantity: Entry quantity (calculated if not provided)
            risk_pct: Additional risk percentage for this entry
            stop_loss: Stop loss price (for calculating quantity)
            date: Entry date
            tags: Tags for this entry
            
        Returns:
            Entry dictionary or None if scaling not possible
        """
        # Check if position exists
        if symbol not in self.open_positions:
            self.logger.warning(f"Cannot scale in: No open position for {symbol}")
            return None
            
        # Get position
        position_id = self.open_positions[symbol]
        position = self.positions[position_id]
        
        # Calculate remaining risk room
        available_risk = position.max_risk_pct - position.current_risk_pct
        
        # Use provided risk or maximum available
        actual_risk = min(available_risk, risk_pct or available_risk)
        
        # Check if we have room to scale in
        if actual_risk <= 0:
            self.logger.warning(f"Cannot scale in: Maximum risk reached for {symbol}")
            return None
            
        # Calculate quantity if not provided
        if quantity is None and stop_loss is not None:
            quantity = self._calculate_position_size(
                symbol=symbol,
                direction=position.direction,
                entry_price=price,
                stop_loss=stop_loss,
                risk_pct=actual_risk
            )
        elif quantity is None:
            self.logger.warning("Cannot scale in: Either quantity or stop_loss must be provided")
            return None
            
        # Add entry
        entry = position.add_entry(
            price=price,
            quantity=quantity,
            date=date,
            tags=tags,
            risk_pct=actual_risk
        )
        
        # Update risk tracking
        self.current_total_risk_pct += actual_risk
        
        self.logger.info(f"Scaled into {position.direction} position for {symbol} with additional {actual_risk}% risk")
        return entry
        
    def scale_out(self, symbol: str, price: float, quantity: float = None,
                 percentage: float = None, date: datetime = None,
                 tags: List[str] = None) -> Optional[Dict]:
        """
        Scale out of an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            quantity: Exit quantity
            percentage: Percentage of position to exit (0-100)
            date: Exit date
            tags: Tags for this exit
            
        Returns:
            Exit dictionary or None if scaling not possible
        """
        # Check if position exists
        if symbol not in self.open_positions:
            self.logger.warning(f"Cannot scale out: No open position for {symbol}")
            return None
            
        # Get position
        position_id = self.open_positions[symbol]
        position = self.positions[position_id]
        
        # Calculate quantity if percentage is provided
        if quantity is None and percentage is not None:
            quantity = (percentage / 100.0) * position.remaining_quantity
            
        # Ensure we have a quantity
        if quantity is None:
            self.logger.warning("Cannot scale out: Either quantity or percentage must be provided")
            return None
            
        # Add exit
        exit_record = position.add_exit(
            price=price,
            quantity=quantity,
            date=date,
            tags=tags
        )
        
        # Update risk tracking
        risk_reduction = position.initial_risk_pct * (quantity / position.total_quantity)
        self.current_total_risk_pct -= risk_reduction
        
        # Check if position is fully closed
        if position.status == "closed":
            # Remove from open positions
            del self.open_positions[symbol]
            self.closed_positions.append(position_id)
            
            # Update capital
            self.current_capital += position.realized_pnl
            
        self.logger.info(f"Scaled out of {position.direction} position for {symbol}, "
                       f"quantity: {quantity}, remaining: {position.remaining_quantity}")
        return exit_record
        
    def close_position(self, symbol: str, price: float, date: datetime = None,
                      tags: List[str] = None) -> Optional[Dict]:
        """
        Close an entire position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            date: Exit date
            tags: Tags for this exit
            
        Returns:
            Exit dictionary or None if position doesn't exist
        """
        # Check if position exists
        if symbol not in self.open_positions:
            self.logger.warning(f"Cannot close position: No open position for {symbol}")
            return None
            
        # Get position
        position_id = self.open_positions[symbol]
        position = self.positions[position_id]
        
        # Close the entire position
        exit_record = position.add_exit(
            price=price,
            quantity=position.remaining_quantity,
            date=date,
            tags=tags
        )
        
        # Remove from open positions
        del self.open_positions[symbol]
        self.closed_positions.append(position_id)
        
        # Update capital
        self.current_capital += position.realized_pnl
        
        # Update risk tracking
        self.current_total_risk_pct -= position.current_risk_pct
        
        self.logger.info(f"Closed {position.direction} position for {symbol}, "
                       f"realized P&L: {position.realized_pnl:.2f}")
        return exit_record
        
    def update_positions(self, market_data: Dict[str, pd.DataFrame], current_date: datetime = None):
        """
        Update all open positions with current market data.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrame with market data
            current_date: Current simulation date
        """
        # Use latest date in data if not provided
        if current_date is None and market_data:
            # Get first symbol's data
            first_symbol = next(iter(market_data))
            if first_symbol in market_data and not market_data[first_symbol].empty:
                current_date = market_data[first_symbol].index[-1]
                
        # Update each position
        for symbol, position_id in list(self.open_positions.items()):
            position = self.positions[position_id]
            
            # Skip if we don't have data for this symbol
            if symbol not in market_data or market_data[symbol].empty:
                continue
                
            # Get current price
            current_price = market_data[symbol]["close"].iloc[-1]
            
            # Update unrealized P&L
            position.update_unrealized_pnl(current_price)
            
        # Update equity curve if we have a date
        if current_date is not None:
            self._update_equity_curve(current_date)
            
    def _update_equity_curve(self, date: datetime):
        """
        Update the equity curve with current capital and unrealized P&L.
        
        Args:
            date: Current date
        """
        # Calculate total equity (capital + unrealized P&L)
        unrealized_pnl = sum(
            self.positions[pos_id].unrealized_pnl 
            for pos_id in self.open_positions.values()
        )
        total_equity = self.current_capital + unrealized_pnl
        
        # Add to equity curve
        self.equity_curve.append({
            "date": date,
            "equity": total_equity,
            "capital": self.current_capital,
            "unrealized_pnl": unrealized_pnl
        })
        
        # Calculate daily return if we have at least two points
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (total_equity - prev_equity) / prev_equity
            
            self.daily_returns.append({
                "date": date,
                "return": daily_return
            })
            
        # Calculate drawdown
        if self.equity_curve:
            max_equity = max(point["equity"] for point in self.equity_curve)
            drawdown_pct = (max_equity - total_equity) / max_equity * 100 if max_equity > 0 else 0
            
            self.drawdowns.append({
                "date": date,
                "drawdown_pct": drawdown_pct
            })
            
    def _calculate_position_size(self, symbol: str, direction: str, entry_price: float,
                               stop_loss: float, risk_pct: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_pct: Risk percentage
            
        Returns:
            Position size (quantity)
        """
        # Calculate risk amount in currency
        risk_amount = (risk_pct / 100.0) * self.current_capital
        
        # Calculate risk per unit
        direction = direction.lower()
        if direction == "long":
            risk_per_unit = entry_price - stop_loss
        else:  # short
            risk_per_unit = stop_loss - entry_price
            
        # Handle zero or negative risk per unit
        if risk_per_unit <= 0:
            self.logger.warning(f"Invalid risk per unit for {symbol}: {risk_per_unit}")
            return 0.0
            
        # Calculate quantity
        quantity = risk_amount / risk_per_unit
        
        # Apply position sizing method adjustments
        if self.position_sizing_method == "volatility_based":
            # This would require volatility data
            # For now, we'll just use the risk-based calculation
            pass
            
        elif self.position_sizing_method == "fixed_lot":
            # Round to nearest lot size (assumed to be 100)
            lot_size = 100
            quantity = math.floor(quantity / lot_size) * lot_size
            
        return quantity
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get the Position object for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if not found
        """
        if symbol in self.open_positions:
            position_id = self.open_positions[symbol]
            return self.positions[position_id]
        return None
        
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """
        Get a Position object by its ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(position_id)
        
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position objects
        """
        return [self.positions[pos_id] for pos_id in self.open_positions.values()]
        
    def get_closed_positions(self) -> List[Position]:
        """
        Get all closed positions.
        
        Returns:
            List of Position objects
        """
        return [self.positions[pos_id] for pos_id in self.closed_positions]
        
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        return pd.DataFrame(self.equity_curve).set_index("date")
        
    def get_drawdowns(self) -> pd.DataFrame:
        """
        Get the drawdowns as a DataFrame.
        
        Returns:
            DataFrame with drawdown data
        """
        return pd.DataFrame(self.drawdowns).set_index("date")
        
    def get_daily_returns(self) -> pd.DataFrame:
        """
        Get the daily returns as a DataFrame.
        
        Returns:
            DataFrame with daily return data
        """
        return pd.DataFrame(self.daily_returns).set_index("date")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the portfolio.
        
        Returns:
            Dictionary with performance metrics
        """
        # Convert lists to DataFrames for calculations
        if not self.equity_curve:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
            
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        returns_df = pd.DataFrame(self.daily_returns).set_index("date") if self.daily_returns else None
        drawdowns_df = pd.DataFrame(self.drawdowns).set_index("date") if self.drawdowns else None
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        if not equity_df.empty:
            initial_equity = equity_df["equity"].iloc[0]
            final_equity = equity_df["equity"].iloc[-1]
            metrics["total_return_pct"] = ((final_equity / initial_equity) - 1) * 100
            
            # Annualized return (assuming 252 trading days in a year)
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = days / 365
            metrics["annualized_return_pct"] = (((final_equity / initial_equity) ** (1 / max(years, 0.01))) - 1) * 100
        else:
            metrics["total_return_pct"] = 0.0
            metrics["annualized_return_pct"] = 0.0
            
        # Max drawdown
        if drawdowns_df is not None and not drawdowns_df.empty:
            metrics["max_drawdown_pct"] = drawdowns_df["drawdown_pct"].max()
        else:
            metrics["max_drawdown_pct"] = 0.0
            
        # Sharpe ratio (assuming 0% risk-free rate)
        if returns_df is not None and not returns_df.empty:
            daily_returns_series = returns_df["return"]
            metrics["sharpe_ratio"] = (
                daily_returns_series.mean() / daily_returns_series.std() * (252 ** 0.5)
                if daily_returns_series.std() > 0 else 0.0
            )
        else:
            metrics["sharpe_ratio"] = 0.0
            
        # Win rate and profit factor
        closed_positions = self.get_closed_positions()
        if closed_positions:
            winning_positions = [p for p in closed_positions if p.realized_pnl > 0]
            losing_positions = [p for p in closed_positions if p.realized_pnl < 0]
            
            metrics["win_rate"] = len(winning_positions) / len(closed_positions) * 100
            
            gross_profit = sum(p.realized_pnl for p in winning_positions)
            gross_loss = abs(sum(p.realized_pnl for p in losing_positions))
            
            metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0
            
        # Calculate additional metrics
        metrics["total_trades"] = len(closed_positions)
        metrics["open_positions"] = len(self.open_positions)
        metrics["current_capital"] = self.current_capital
        metrics["current_risk_pct"] = self.current_total_risk_pct
        
        return metrics
