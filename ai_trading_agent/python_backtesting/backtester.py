"""
Python implementation of the backtesting functionality from the Rust extension.
This serves as a temporary replacement until the Rust extension build issues are resolved.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import uuid
from copy import deepcopy

from .models import (
    OrderSide, OrderType, OrderStatus, Order, Fill, Trade, Position,
    Portfolio, PortfolioSnapshot, OHLCVBar, BacktestConfig, PerformanceMetrics, BacktestResult
)


def calculate_execution_price(order: Order, bar: OHLCVBar) -> Optional[float]:
    """Calculate execution price based on order type and current bar"""
    if order.order_type == OrderType.Market:
        return bar.open
    
    elif order.order_type == OrderType.Limit:
        if order.side == OrderSide.Buy:
            # For buy limit orders, price must be <= limit price
            if bar.low <= order.limit_price:
                # Order will execute at limit price or better
                return min(order.limit_price, bar.open)
            return None  # Limit price not reached
        else:  # Sell
            # For sell limit orders, price must be >= limit price
            if bar.high >= order.limit_price:
                # Order will execute at limit price or better
                return max(order.limit_price, bar.open)
            return None  # Limit price not reached
    
    elif order.order_type == OrderType.Stop:
        if order.side == OrderSide.Buy:
            # For buy stop orders, price must be >= stop price
            if bar.high >= order.stop_price:
                # Once triggered, executes at market
                return max(order.stop_price, bar.open)
            return None  # Stop price not reached
        else:  # Sell
            # For sell stop orders, price must be <= stop price
            if bar.low <= order.stop_price:
                # Once triggered, executes at market
                return min(order.stop_price, bar.open)
            return None  # Stop price not reached
    
    elif order.order_type == OrderType.StopLimit:
        # First check if stop price is reached
        stop_triggered = False
        if order.side == OrderSide.Buy and bar.high >= order.stop_price:
            stop_triggered = True
        elif order.side == OrderSide.Sell and bar.low <= order.stop_price:
            stop_triggered = True
        
        if not stop_triggered:
            return None  # Stop price not reached
        
        # Then check if limit price is satisfied
        if order.side == OrderSide.Buy:
            if bar.low <= order.limit_price:
                return min(order.limit_price, max(bar.open, order.stop_price))
            return None  # Limit price not reached after stop triggered
        else:  # Sell
            if bar.high >= order.limit_price:
                return max(order.limit_price, min(bar.open, order.stop_price))
            return None  # Limit price not reached after stop triggered
    
    return None  # Unknown order type


def apply_transaction_costs(order: Order, executed_price: float, commission_rate: float, slippage: float) -> float:
    """Apply transaction costs (commission and slippage) to execution price"""
    # Apply slippage
    if order.side == OrderSide.Buy:
        executed_price *= (1.0 + slippage)
    else:  # Sell
        executed_price *= (1.0 - slippage)
    
    # Commission is handled separately when updating the portfolio
    return executed_price


def update_position_market_price(position: Position, market_price: float) -> None:
    """Update position market price and unrealized P&L"""
    position.market_price = market_price
    position.unrealized_pnl = position.quantity * (market_price - position.entry_price)


def update_portfolio_from_trade(portfolio: Portfolio, trade: Trade) -> None:
    """Update portfolio based on a trade"""
    symbol = trade.symbol
    
    # Calculate trade value
    trade_value = trade.quantity * trade.price
    
    if trade.side == OrderSide.Buy:
        # Buying: decrease cash
        portfolio.cash -= trade_value
        
        # Update or create position
        if symbol in portfolio.positions:
            position = portfolio.positions[symbol]
            
            # Calculate new average entry price
            total_quantity = position.quantity + trade.quantity
            total_cost = (position.quantity * position.entry_price) + trade_value
            
            position.entry_price = total_cost / total_quantity if total_quantity > 0 else 0
            position.quantity = total_quantity
        else:
            # Create new position
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity,
                entry_price=trade.price,
                market_price=trade.price
            )
    
    else:  # Sell
        # Selling: increase cash
        portfolio.cash += trade_value
        
        # Update position
        if symbol in portfolio.positions:
            position = portfolio.positions[symbol]
            
            # Calculate realized P&L
            realized_pnl = trade.quantity * (trade.price - position.entry_price)
            position.realized_pnl += realized_pnl
            
            # Update quantity
            position.quantity -= trade.quantity
            
            # Remove position if quantity is zero or negative
            if position.quantity <= 0:
                del portfolio.positions[symbol]
        else:
            # Short selling (not implemented in this simplified version)
            pass


def update_portfolio_value(portfolio: Portfolio) -> None:
    """Update portfolio total value"""
    position_value = sum(
        position.quantity * position.market_price
        for position in portfolio.positions.values()
    )
    portfolio.total_value = portfolio.cash + position_value


def calculate_performance_metrics(
    portfolio_history: List[PortfolioSnapshot],
    trade_history: List[Trade],
    initial_capital: float
) -> PerformanceMetrics:
    """Calculate performance metrics from backtest results"""
    metrics = PerformanceMetrics()
    
    if not portfolio_history:
        return metrics
    
    # Calculate total return
    final_value = portfolio_history[-1].total_value
    metrics.total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate daily returns for other metrics
    returns = []
    for i in range(1, len(portfolio_history)):
        prev_value = portfolio_history[i-1].total_value
        curr_value = portfolio_history[i].total_value
        daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
        returns.append(daily_return)
    
    if returns:
        # Calculate volatility (annualized)
        metrics.volatility = np.std(returns) * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
    
    # Calculate drawdowns
    max_value = initial_capital
    max_drawdown = 0.0
    drawdown_start = 0
    max_drawdown_duration = 0
    current_drawdown_duration = 0
    
    for i, snapshot in enumerate(portfolio_history):
        current_value = snapshot.total_value
        
        if current_value > max_value:
            max_value = current_value
            # Reset drawdown tracking if we're at a new high
            drawdown_start = i
            current_drawdown_duration = 0
        else:
            # Calculate drawdown
            drawdown = (max_value - current_value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            # Track drawdown duration
            current_drawdown_duration = i - drawdown_start
            if current_drawdown_duration > max_drawdown_duration:
                max_drawdown_duration = current_drawdown_duration
    
    metrics.max_drawdown = max_drawdown
    metrics.max_drawdown_duration = max_drawdown_duration
    
    # Calculate win rate and profit factor
    if trade_history:
        winning_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        profit_trades = []
        loss_trades = []
        
        for trade in trade_history:
            # Simplified P&L calculation
            trade_pnl = -trade.quantity * trade.price if trade.side == OrderSide.Buy else trade.quantity * trade.price
            
            if trade_pnl > 0.0:
                winning_trades += 1
                total_profit += trade_pnl
                profit_trades.append(trade_pnl)
            else:
                total_loss += abs(trade_pnl)
                loss_trades.append(trade_pnl)
        
        metrics.total_trades = len(trade_history)
        metrics.win_rate = winning_trades / len(trade_history)
        
        if total_loss > 0.0:
            metrics.profit_factor = total_profit / total_loss
        else:
            metrics.profit_factor = float('inf') if total_profit > 0.0 else 0.0
        
        # Calculate average profit/loss per trade
        if profit_trades:
            metrics.avg_profit_per_trade = sum(profit_trades) / len(profit_trades)
        
        if loss_trades:
            metrics.avg_loss_per_trade = sum(loss_trades) / len(loss_trades)
        
        # Calculate profit/loss ratio
        if metrics.avg_loss_per_trade != 0.0:
            metrics.avg_profit_loss_ratio = abs(metrics.avg_profit_per_trade) / abs(metrics.avg_loss_per_trade)
        else:
            metrics.avg_profit_loss_ratio = float('inf') if metrics.avg_profit_per_trade > 0.0 else 0.0
    
    # Calculate annualized return
    first_timestamp = portfolio_history[0].timestamp
    last_timestamp = portfolio_history[-1].timestamp
    days = (last_timestamp - first_timestamp) / (24.0 * 60.0 * 60.0)
    
    if days > 0.0:
        metrics.annualized_return = ((1.0 + metrics.total_return) ** (365.0 / days)) - 1.0
    
    # Calculate Sortino ratio (using downside deviation)
    downside_returns = [r for r in returns if r < 0.0]
    
    if downside_returns:
        sum_squared_downside = sum(r ** 2 for r in downside_returns)
        downside_deviation = np.sqrt(sum_squared_downside / len(downside_returns)) * np.sqrt(252.0)
        
        if downside_deviation > 0.0:
            metrics.sortino_ratio = metrics.annualized_return / downside_deviation
    
    return metrics


def run_backtest(
    data: Dict[str, List[OHLCVBar]],
    orders: List[Order],
    config: BacktestConfig
) -> Dict[str, Any]:
    """Run a backtest with the given data and configuration"""
    # Initialize portfolio
    portfolio = Portfolio(
        cash=config.initial_capital,
        total_value=config.initial_capital,
        positions={}
    )
    
    # Initialize result tracking
    portfolio_history = []
    trade_history = []
    order_history = []
    
    # Sort orders by timestamp
    sorted_orders = sorted(orders, key=lambda o: o.created_at)
    
    # Get all unique timestamps from all symbols
    all_timestamps = set()
    for symbol, bars in data.items():
        all_timestamps.update(bar.timestamp for bar in bars)
    
    # Sort timestamps
    sorted_timestamps = sorted(all_timestamps)
    
    # Create a map of timestamp -> bar for each symbol for efficient lookup
    timestamp_to_bar = {symbol: {} for symbol in data.keys()}
    for symbol, bars in data.items():
        for bar in bars:
            timestamp_to_bar[symbol][bar.timestamp] = bar
    
    # Process each timestamp
    active_orders = []
    
    for timestamp in sorted_timestamps:
        # Add new orders that were created before or at this timestamp
        while sorted_orders and sorted_orders[0].created_at <= timestamp:
            order = sorted_orders.pop(0)
            order.status = OrderStatus.Submitted
            active_orders.append(order)
        
        # Process active orders
        remaining_orders = []
        for order in active_orders:
            symbol = order.symbol
            
            # Skip if we don't have data for this symbol at this timestamp
            if symbol not in timestamp_to_bar or timestamp not in timestamp_to_bar[symbol]:
                remaining_orders.append(order)
                continue
            
            # Get current bar
            bar = timestamp_to_bar[symbol][timestamp]
            
            # Calculate execution price
            execution_price = calculate_execution_price(order, bar)
            
            if execution_price is not None:
                # Apply transaction costs
                execution_price = apply_transaction_costs(
                    order, execution_price, config.commission_rate, config.slippage
                )
                
                # Create fill
                fill = Fill(
                    quantity=order.quantity,
                    price=execution_price,
                    timestamp=timestamp
                )
                
                order.fills.append(fill)
                order.status = OrderStatus.Filled
                
                # Create trade
                trade_id = str(uuid.uuid4())
                trade = Trade(
                    trade_id=trade_id,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=execution_price,
                    timestamp=timestamp
                )
                
                # Update portfolio
                update_portfolio_from_trade(portfolio, trade)
                
                # Add to trade history
                trade_history.append(trade)
            else:
                # Order not executed, keep it active
                remaining_orders.append(order)
        
        # Update active orders
        active_orders = remaining_orders
        
        # Update market prices for all positions
        for symbol, position in list(portfolio.positions.items()):
            if symbol in timestamp_to_bar and timestamp in timestamp_to_bar[symbol]:
                bar = timestamp_to_bar[symbol][timestamp]
                update_position_market_price(position, bar.close)
        
        # Update portfolio value
        update_portfolio_value(portfolio)
        
        # Create portfolio snapshot
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=portfolio.cash,
            total_value=portfolio.total_value,
            positions=deepcopy(portfolio.positions)
        )
        
        # Add to portfolio history
        portfolio_history.append(snapshot)
    
    # Add all orders to history
    order_history.extend(sorted_orders)  # Unprocessed orders
    order_history.extend(active_orders)  # Active but unfilled orders
    
    # Calculate performance metrics
    metrics = calculate_metrics(
        portfolio_history, trade_history, config.initial_capital
    )
    
    # Create result
    result = BacktestResult(
        portfolio_history=portfolio_history,
        trade_history=trade_history,
        order_history=order_history,
        metrics=metrics
    )
    
    # Convert to dictionary for compatibility with the original Rust function
    result_dict = {
        "portfolio_history": portfolio_history,
        "trade_history": trade_history,
        "order_history": order_history,
        "metrics": metrics
    }
    
    return result_dict
