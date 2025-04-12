"""
Portfolio Manager module for AI Trading Agent.

This module provides functionality for managing a portfolio of positions,
including risk management, position sizing, and portfolio rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from .models import Order, Trade, Position, Portfolio
from .enums import OrderSide, OrderType, OrderStatus, PositionSide
from ..common import logger
from .exceptions import PortfolioUpdateError

class PortfolioManager:
    """
    Portfolio Manager class for managing a portfolio of positions.

    This class handles portfolio updates, risk management, position sizing,
    and portfolio rebalancing.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.2,
        max_correlation: float = 0.7,
        rebalance_frequency: str = "weekly",
    ):
        """
        Initialize the portfolio manager.

        Args:
            initial_capital: Initial capital for the portfolio
            risk_per_trade: Maximum risk per trade as a percentage of portfolio value
            max_position_size: Maximum position size as a percentage of portfolio value
            max_correlation: Maximum allowed correlation between positions
            rebalance_frequency: Frequency for portfolio rebalancing ('daily', 'weekly', 'monthly')
        """
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.rebalance_frequency = rebalance_frequency

        self.portfolio_history = []
        self.last_rebalance_time = None

        logger.info(f"Initialized portfolio manager with {initial_capital} initial capital")

    def update_from_trade(self, trade: Trade) -> None:
        """
        Update the portfolio based on a trade.

        Args:
            trade: Trade to update the portfolio with
        """
        try:
            current_market_prices = {trade.symbol: trade.price}
            self.portfolio.update_from_trade(trade, current_market_prices)
            self._record_portfolio_state(trade.timestamp)
            logger.info("Updated portfolio from trade",
                        trade_id=trade.trade_id,
                        symbol=trade.symbol,
                        side=trade.side.name,
                        quantity=trade.quantity,
                        price=trade.price)
        except Exception as e:
            logger.error(f"Portfolio update error for trade {trade.trade_id}: {e}", exc_info=True)
            raise PortfolioUpdateError(f"Failed to update portfolio for trade {trade.trade_id}") from e

    def update_market_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        """
        Update market prices for all positions.

        Args:
            prices: Dictionary mapping symbols to prices
            timestamp: Current timestamp
        """
        for symbol, price in prices.items():
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                position.update_market_price(price)

        self.portfolio.update_total_value(prices)
        self._record_portfolio_state(timestamp)

    def _record_portfolio_state(self, timestamp: pd.Timestamp) -> None:
        """
        Record the current portfolio state.

        Args:
            timestamp: Current timestamp
        """
        portfolio_snapshot = {
            'timestamp': timestamp,
            'cash': self.portfolio.current_balance,
            'total_value': self.portfolio.total_value,
            'positions': {
                symbol: {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl
                }
                for symbol, position in self.portfolio.positions.items()
                if position.quantity != 0
            }
        }
        self.portfolio_history.append(portfolio_snapshot)

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Symbol to trade
            price: Current price of the symbol
            stop_loss: Stop loss price (optional)
            risk_pct: Risk percentage for this trade (optional, defaults to self.risk_per_trade)

        Returns:
            float: Position size in units of the symbol
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade

        risk_amount = self.portfolio.total_value * risk_pct

        if stop_loss is not None:
            risk_per_unit = abs(price - stop_loss)
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = (self.portfolio.total_value * self.max_position_size) / price
        else:
            position_size = (self.portfolio.total_value * self.max_position_size) / price

        max_position_value = self.portfolio.total_value * self.max_position_size
        max_position_size = max_position_value / price

        current_position_size = 0
        if symbol in self.portfolio.positions:
            current_position_size = self.portfolio.positions[symbol].quantity

        available_position_size = max_position_size - current_position_size

        return min(position_size, available_position_size)

    def check_correlation(self, symbol: str, returns: pd.Series) -> bool:
        """
        Check if adding a new position would exceed correlation limits.

        Args:
            symbol: Symbol to check
            returns: Returns series for the symbol

        Returns:
            bool: True if correlation is acceptable, False otherwise
        """
        # Placeholder: always accept
        return True

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> List[Order]:
        """
        Rebalance the portfolio to match target weights.

        Args:
            target_weights: Dictionary mapping symbols to target weights
            current_prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp

        Returns:
            List[Order]: List of orders to execute for rebalancing
        """
        if self.last_rebalance_time is not None:
            if self.rebalance_frequency == "daily":
                if (timestamp - self.last_rebalance_time).days < 1:
                    return []
            elif self.rebalance_frequency == "weekly":
                if (timestamp - self.last_rebalance_time).days < 7:
                    return []
            elif self.rebalance_frequency == "monthly":
                if (timestamp - self.last_rebalance_time).days < 30:
                    return []

        self.last_rebalance_time = timestamp

        current_weights = {}
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0 and symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
                current_weights[symbol] = position_value / self.portfolio.total_value

        target_values = {}
        for symbol, weight in target_weights.items():
            if symbol in current_prices:
                target_values[symbol] = self.portfolio.total_value * weight

        orders = []
        for symbol, target_value in target_values.items():
            if symbol not in current_prices:
                continue

            current_value = 0
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                if position.quantity != 0:
                    current_value = position.quantity * current_prices[symbol]

            value_diff = target_value - current_value
            quantity_diff = value_diff / current_prices[symbol]

            if abs(quantity_diff) > 0.01:
                side = OrderSide.BUY if quantity_diff > 0 else OrderSide.SELL
                quantity = abs(quantity_diff)

                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )
                orders.append(order)
                logger.info(f"Generated rebalance order: {side.name} {quantity:.4f} {symbol}")

        return orders

    def apply_risk_management(
        self,
        current_prices: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> List[Order]:
        """
        Apply risk management rules to the portfolio.

        Args:
            current_prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp

        Returns:
            List[Order]: List of orders to execute for risk management
        """
        orders = []

        for symbol, position in self.portfolio.positions.items():
            if position.quantity == 0 or symbol not in current_prices:
                continue

            if position.quantity > 0:
                entry_value = position.quantity * position.entry_price
                current_value = position.quantity * current_prices[symbol]
                drawdown = (entry_value - current_value) / entry_value

                if drawdown > 0.1:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)
                    logger.info(f"Generated risk management order: SELL {position.quantity} {symbol} (drawdown: {drawdown:.2%})")

            elif position.quantity < 0:
                entry_value = abs(position.quantity) * position.entry_price
                current_value = abs(position.quantity) * current_prices[symbol]
                drawdown = (current_value - entry_value) / entry_value

                if drawdown > 0.1:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=abs(position.quantity),
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)
                    logger.info(f"Generated risk management order: BUY {abs(position.quantity)} {symbol} (drawdown: {drawdown:.2%})")

        return orders

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.

        Returns:
            Dict[str, Any]: Dictionary containing portfolio state
        """
        return {
            'cash': self.portfolio.current_balance,
            'total_value': self.portfolio.total_value,
            'positions': {
                symbol: {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl
                }
                for symbol, position in self.portfolio.positions.items()
                if position.quantity != 0
            }
        }

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """
        Get the portfolio history.

        Returns:
            List[Dict[str, Any]]: List of portfolio snapshots
        """
        return self.portfolio_history

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if not self.portfolio_history:
            return {}

        timestamps = [snapshot['timestamp'] for snapshot in self.portfolio_history]
        values = [snapshot['total_value'] for snapshot in self.portfolio_history]

        portfolio_values = pd.Series(values, index=timestamps)
        returns = portfolio_values.pct_change().fillna(0)

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }
