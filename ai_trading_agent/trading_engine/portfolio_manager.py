"""
Portfolio Manager module for AI Trading Agent.

This module provides functionality for managing a portfolio of positions,
including risk management, position sizing, and portfolio rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from .models import Order, Trade, Position, Portfolio
from .enums import OrderSide, OrderType, OrderStatus, PositionSide
from ..common.logging_config import logger
from .exceptions import PortfolioUpdateError

class PortfolioManager:
    """
    Portfolio Manager class for managing a portfolio of positions.

    This class handles portfolio updates, risk management, position sizing,
    and portfolio rebalancing.
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal('10000.0'),
        risk_per_trade: Decimal = Decimal('0.02'),
        max_position_size: Decimal = Decimal('0.2'),
        max_correlation: float = 0.7,  # This can remain float as it's a correlation value
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
        # Ensure all monetary values are Decimal
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.risk_per_trade = Decimal(str(risk_per_trade)) if not isinstance(risk_per_trade, Decimal) else risk_per_trade
        self.max_position_size = Decimal(str(max_position_size)) if not isinstance(max_position_size, Decimal) else max_position_size
        self.max_correlation = max_correlation  # This remains as float
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

    def update_market_prices(self, prices: Dict[str, Decimal], timestamp: pd.Timestamp) -> None:
        """
        Update market prices for all positions.

        Args:
            prices: Dictionary mapping symbols to prices
            timestamp: Current timestamp
        """
        # Convert prices to Decimal if needed
        decimal_prices = {}
        for symbol, price in prices.items():
            decimal_prices[symbol] = Decimal(str(price)) if not isinstance(price, Decimal) else price
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                position.update_market_price(decimal_prices[symbol])

        self.portfolio.update_total_value(decimal_prices)
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

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current portfolio state.
        
        Returns:
            Dict[str, Any]: Dictionary containing portfolio state
        """
        return {
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "positions": {
                symbol: {
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price if hasattr(position, 'current_price') else position.entry_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                    "market_value": position.quantity * (position.current_price if hasattr(position, 'current_price') else position.entry_price)
                }
                for symbol, position in self.portfolio.positions.items()
            },
            "last_update": self.portfolio.last_update_time.isoformat() if self.portfolio.last_update_time else None
        }
        
    def get_portfolio_value(self) -> Decimal:
        """
        Get the total value of the portfolio (cash + positions).
        
        Returns:
            Decimal: The total portfolio value
        """
        return self.portfolio.total_value

    def calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        stop_loss: Optional[Decimal] = None,
        risk_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Symbol to trade
            price: Current price of the symbol
            stop_loss: Stop loss price (optional)
            risk_pct: Risk percentage for this trade (optional, defaults to self.risk_per_trade)

        Returns:
            Decimal: Position size in units of the symbol
        """
        # Use provided risk_pct or default from manager
        risk_pct_to_use = risk_pct if risk_pct is not None else self.risk_per_trade

        logger.debug(f"Calculating position size for {symbol} at price {price}")

        # Initialize sizes to Decimal 0
        position_size = Decimal('0')
        available_position_size = Decimal('0')

        # --- Try calculating risk-based size ---
        try:
            # Ensure inputs are Decimal
            price_dec = Decimal(str(price)) if not isinstance(price, Decimal) else price
            risk_pct_dec = Decimal(str(risk_pct_to_use)) if not isinstance(risk_pct_to_use, Decimal) else risk_pct_to_use

            # Portfolio value should already be Decimal from Portfolio class
            portfolio_value = self.portfolio.total_value

            logger.debug(f"Portfolio total value: {portfolio_value}, Risk pct: {risk_pct_dec}")
            risk_amount = portfolio_value * risk_pct_dec
            logger.debug(f"Calculated risk amount: {risk_amount}")

            if stop_loss is not None:
                # Ensure stop_loss is Decimal
                stop_loss_dec = Decimal(str(stop_loss)) if not isinstance(stop_loss, Decimal) else stop_loss
                
                risk_per_unit = abs(price_dec - stop_loss_dec)
                if risk_per_unit > Decimal('0'):
                    position_size = risk_amount / risk_per_unit
                    logger.debug(f"Sizing based on stop_loss {stop_loss_dec}: risk/unit={risk_per_unit}, size={position_size}")
                else:
                    # Fallback if stop loss is exactly the price
                    if price_dec > Decimal('0'):
                        position_size = (portfolio_value * self.max_position_size) / price_dec # Use max size logic
                        logger.debug(f"Stop loss invalid (price={price_dec}), using max size logic: size={position_size}")
                    else:
                        logger.warning(f"Cannot calculate position size: price is zero or negative.")
            else:
                if price_dec > Decimal('0'):
                    position_size = (portfolio_value * self.max_position_size) / price_dec
                    logger.debug(f"Sizing based on max_position_size ({self.max_position_size}): size={position_size}")
                else:
                    logger.warning(f"Cannot calculate position size: price is zero or negative.")
        except (InvalidOperation, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error calculating risk-based position size: {e}", exc_info=True)
            position_size = Decimal('0') # Reset on error
        # ----------------------------------------

        # --- Try calculating available size based on cash/max allocation ---
        try:
            price_dec = Decimal(str(price)) if not isinstance(price, Decimal) else price
            if price_dec <= Decimal('0'):
                logger.warning(f"Cannot calculate available size: price is zero or negative.")
                available_position_size = Decimal('0')
            else:
                # Portfolio value and max_position_size should already be Decimal
                max_position_value = self.portfolio.total_value * self.max_position_size
                max_position_size_calc = max_position_value / price_dec
                logger.debug(f"Max position value: {max_position_value}, Max theoretical position size: {max_position_size_calc}")

                current_position_size = Decimal('0')
                if symbol in self.portfolio.positions:
                    # Position quantity should already be Decimal from Position class
                    current_position_size = self.portfolio.positions[symbol].quantity
                    logger.debug(f"Current position size for {symbol}: {current_position_size}")

                available_position_size = max_position_size_calc - abs(current_position_size)
                logger.debug(f"Calculated available position size (max_theoretical - current): {available_position_size}")
        except (InvalidOperation, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error calculating available position size: {e}", exc_info=True)
            available_position_size = Decimal('0') # Reset on error
        # ----------------------------------------------------------------

        final_size = min(position_size, available_position_size)
        logger.debug(f"Risk-based size: {position_size}, Cash-based size: {available_position_size}")
        logger.debug(f"Returning final size (min of above): {final_size} (type: {type(final_size)})")

        return final_size # Ensure this returns the Decimal

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
        target_weights: Dict[str, Decimal],
        current_prices: Dict[str, Decimal],
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
        current_prices: Dict[str, Decimal],
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
                # Ensure current_prices are Decimal
                current_price_dec = Decimal(str(current_prices[symbol])) if not isinstance(current_prices[symbol], Decimal) else current_prices[symbol]
                entry_value = position.quantity * position.entry_price
                current_value = position.quantity * current_price_dec
                drawdown = (entry_value - current_value) / entry_value if entry_value > Decimal('0') else Decimal('0')

                if drawdown > Decimal('0.1'):
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)
                    logger.info(f"Generated risk management order: SELL {position.quantity} {symbol} (drawdown: {drawdown:.2%})")

            elif position.quantity < 0:
                # Ensure current_prices are Decimal
                current_price_dec = Decimal(str(current_prices[symbol])) if not isinstance(current_prices[symbol], Decimal) else current_prices[symbol]
                entry_value = abs(position.quantity) * position.entry_price
                current_value = abs(position.quantity) * current_price_dec
                drawdown = (current_value - entry_value) / entry_value if entry_value > Decimal('0') else Decimal('0')

                if drawdown > Decimal('0.1'):
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
        logger.info(f"Current Cash Balance: {self.portfolio.current_balance}")
        logger.info(f"Total Portfolio Value (Equity): {self.portfolio.total_equity}")
        logger.info(f"Total Realized PnL: {self.portfolio.total_realized_pnl}")
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

        # Convert to Decimal for precise calculations
        total_return = Decimal(str((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1)) if len(portfolio_values) > 0 else Decimal('0')
        
        # Ensure returns series uses float64 for std calculation if necessary, or handle Decimal conversion
        # Pandas std operates on floats. We need to be careful here.
        # Option 1: Calculate std dev on float returns, then convert result
        float_returns = returns.astype(float)
        volatility_float = float_returns.std() * np.sqrt(252)
        volatility = Decimal(str(volatility_float)) if not np.isnan(volatility_float) else Decimal('0')
        
        # Option 2: Potentially use a Decimal-aware std dev calculation if performance allows (more complex)

        annualized_return = ((Decimal('1') + total_return) ** (Decimal('252') / Decimal(str(len(returns)))) - Decimal('1')) if len(returns) > 0 else Decimal('0')
        sharpe_ratio = (annualized_return / volatility).quantize(Decimal('0.0001')) if volatility > Decimal('0') else Decimal('0') # Added quantize for consistent precision

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = Decimal(str(drawdowns.min())) if not drawdowns.empty else Decimal('0')

        return {
            'total_return': total_return.quantize(Decimal('0.0001')),
            'annualized_return': annualized_return.quantize(Decimal('0.0001')),
            'volatility': volatility.quantize(Decimal('0.0001')),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown.quantize(Decimal('0.0001')),
        }

    def update_portfolio_value(self) -> None:
        """
        Update the total value of the portfolio.

        This method iterates over all positions, calculates their market value,
        and updates the total portfolio value.
        """
        total_position_value = Decimal('0')
        for symbol, position in self.portfolio.positions.items():
            # Ensure position quantity and entry price are Decimal
            pos_quantity = Decimal(str(position.quantity))
            pos_entry_price = Decimal(str(position.entry_price))

            if pos_quantity != Decimal('0'):
                current_price = self.get_current_price(symbol)
                if current_price is not None:
                    # Ensure current_price is Decimal
                    current_price_dec = Decimal(str(current_price))
                    # Calculate PnL and update position
                    self.calculate_position_pnl(position, current_price_dec)
                    # Calculate market value
                    market_value = pos_quantity * current_price_dec
                    total_position_value += market_value
                else:
                    # Handle missing price? Use entry price or log warning?
                    # For now, let's use entry value for total value calculation if price is missing
                    logger.warning(f"Could not retrieve current price for {symbol}. Using entry value for total value calc.")
                    total_position_value += pos_quantity * pos_entry_price # Fallback

        # Update total value using Decimal
        self.portfolio.total_value = self.portfolio.cash + total_position_value
        self.portfolio.last_update_time = datetime.utcnow()

    def calculate_position_pnl(self, position: Position, current_price: Decimal) -> None:
        """
        Calculate the PnL for a position.

        Args:
            position: Position to calculate PnL for
            current_price: Current market price of the position
        """
        # Ensure position quantity and entry price are Decimal
        pos_quantity = Decimal(str(position.quantity))
        pos_entry_price = Decimal(str(position.entry_price))

        if pos_quantity != Decimal('0'):
            # Calculate unrealized PnL
            unrealized_pnl = (current_price - pos_entry_price) * pos_quantity
            position.unrealized_pnl = unrealized_pnl
