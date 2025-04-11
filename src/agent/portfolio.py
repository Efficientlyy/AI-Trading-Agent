# src/agent/portfolio.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd # <-- Added import

# Placeholder types - these might come from existing trading_engine.models
# or be defined more formally later.
Order = Dict[str, Any] # e.g., {'symbol': 'AAPL', 'type': 'MARKET', 'quantity': 10, 'side': 'BUY'}
Fill = Dict[str, Any] # e.g., {'symbol': 'AAPL', 'price': 150.5, 'quantity': 10, 'side': 'BUY', 'timestamp': ...}
PortfolioState = Dict[str, Any] # e.g., {'cash': 10000, 'positions': {'AAPL': 10}, 'timestamp': ...}
SignalsDict = Dict[str, int] # From strategy.py

class PortfolioManagerABC(ABC):
    """Abstract base class for portfolio managers.

    Manages cash, positions, and order generation based on trading signals and market data.
    """

    @abstractmethod
    def update_market_data(self, market_data: Dict[str, pd.Series]) -> None:
        """Updates the portfolio's state based on the latest market data (e.g., mark-to-market).

        Args:
            market_data: Dictionary mapping symbols to their current market data series.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_orders(self, signals: Dict[str, int], market_data: Dict[str, pd.Series]) -> List[Order]:
        """Generates a list of Order objects based on trading signals and current market data.

        Args:
            signals: Dictionary mapping symbols to trading signals (1: buy, -1: sell, 0: hold/close).
            market_data: Dictionary mapping symbols to their current market data series.

        Returns:
            A list of Order objects to be sent to the execution handler.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_positions(self) -> Dict[str, float]:
        """Returns the current holdings for all symbols.

        Returns:
            A dictionary mapping symbols to their current quantity held.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_cash(self) -> float:
        """Returns the current cash balance."""
        raise NotImplementedError

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Calculates and returns the total current market value of the portfolio (cash + positions)."""
        raise NotImplementedError

    @abstractmethod
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Returns a dictionary representing the complete current state of the portfolio
           (e.g., cash, positions, entry prices, potentially other metrics needed by RiskManager).
        """
        raise NotImplementedError

class BasePortfolioManager(PortfolioManagerABC):
    """
    Abstract base class for Portfolio Managers.

    Responsible for translating trading signals into executable orders,
    considering risk constraints, position sizing, and current portfolio state.
    It also updates the portfolio based on execution fills.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the BasePortfolioManager.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'initial_cash' (float): The starting cash balance.
        """
        self.config = config if config is not None else {}
        self.initial_cash = self.config.get('initial_cash', 100000.0)
        self.cash = self.initial_cash
        # Positions format: {symbol: {'quantity': float, 'entry_price': float, 'timestamp': datetime}}
        self.positions: Dict[str, Dict[str, Any]] = {}
        # Holds the latest market data for mark-to-market calculations
        self.latest_market_data: Dict[str, pd.Series] = {}
        # History tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    def update_market_data(self, market_data: Dict[str, pd.Series]) -> None:
        """Stores the latest market data for internal use (e.g., calculating portfolio value).

        Args:
            market_data: Dictionary mapping symbols to their current market data series.
        """
        self.latest_market_data = market_data
        # Concrete implementations might add mark-to-market logic here or in get_portfolio_value

    def get_portfolio_value(self) -> float:
        """Calculates the total market value of the portfolio (cash + value of positions).

        Relies on self.latest_market_data having been updated.
        Assumes 'close' price exists in the market data series for valuation.
        """
        total_value = self.cash
        for symbol, position_details in self.positions.items():
            quantity = position_details.get('quantity', 0)
            if quantity != 0 and symbol in self.latest_market_data:
                current_price = self.latest_market_data[symbol].get('close')
                if current_price is not None and not pd.isna(current_price):
                    total_value += quantity * current_price
                else:
                    # Attempt fallback to 'open' or last known price if 'close' is missing/NaN?
                    # For now, log a warning if price is unavailable for an open position.
                    logger.warning(f"Could not find current price for open position in {symbol} to calculate portfolio value.")
            elif quantity != 0:
                logger.warning(f"Have position in {symbol} but no market data available for valuation.")

        return total_value

    def get_current_positions(self) -> Dict[str, float]:
        """Returns the current quantity held for each symbol."""
        return {symbol: details.get('quantity', 0) for symbol, details in self.positions.items()}

    def get_current_cash(self) -> float:
        """Returns the current cash balance."""
        return self.cash

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Returns a comprehensive state dictionary.

        Crucial for components like RiskManager that need detailed info.
        """
        return {
            'timestamp': self.latest_market_data.get(list(self.latest_market_data.keys())[0], {}).name if self.latest_market_data else None, # Get timestamp from first symbol's data
            'cash': self.cash,
            'positions': self.positions, # Includes quantity, entry_price, etc.
            'portfolio_value': self.get_portfolio_value(),
            'latest_market_data': self.latest_market_data # Provide raw data too
        }

    def _record_state(self, timestamp: datetime) -> None:
        """Records the current portfolio state at a given timestamp."""
        state = self.get_portfolio_state()
        state['timestamp'] = timestamp # Ensure correct timestamp
        self.portfolio_history.append(state)

    def _record_trade(self, order: Order, fill_price: float, fill_quantity: float) -> None:
        """Records details of an executed trade."""
        trade_record = {
            'timestamp': order.timestamp, # Or use execution timestamp if available
            'symbol': order.symbol,
            'order_type': order.order_type.name,
            'direction': 'BUY' if fill_quantity > 0 else 'SELL',
            'quantity': abs(fill_quantity),
            'price': fill_price,
            'commission': order.commission, # Assuming commission is set on the order by ExecutionHandler
            'cash_change': - (fill_quantity * fill_price) - order.commission,
            'resulting_cash': self.cash
        }
        self.trades.append(trade_record)
        logger.debug(f"Trade Recorded: {trade_record}")

    def update_fill(self, order: Order) -> None:
        """Processes an executed order (fill event) and updates cash and positions.

        Implements the method required by PortfolioManagerABC.

        Args:
            order: The Order object that has been filled (or partially filled).
                   Requires order status to be FILLED or PARTIALLY_FILLED.
                   Assumes order object contains fill details like fill_price, fill_quantity, commission.
        """
        # Extract necessary information from the Order object
        symbol = order.get('symbol')
        side = order.get('side') # 'BUY' or 'SELL'
        quantity = order.get('fill_quantity') # Quantity actually filled
        price = order.get('fill_price')    # Average price of the fill
        commission = order.get('commission', 0.0) # Execution commission
        status = order.get('status')

        if status not in ['FILLED', 'PARTIALLY_FILLED']:
            logger.warning(f"Attempted to update fill for order with status {status}. Skipping.")
            return

        if not all([symbol, side, quantity, price]):
             logger.error(f"Order object missing required fill information: {order}")
             return

        if quantity <= 0:
            logger.warning(f"Fill quantity must be positive. Received {quantity}. Skipping fill update.")
            return

        # Get current position state before update
        # Use self.positions which should be maintained by BasePortfolioManager
        position_details = self.positions.get(symbol, {'quantity': 0, 'average_price': 0.0})
        current_quantity = position_details.get('quantity', 0)
        average_price = position_details.get('average_price', 0.0)

        # Update cash (use self.cash)
        if side == 'BUY':
            cost = (quantity * price) + commission
            self.cash -= cost
        elif side == 'SELL':
            proceeds = (quantity * price) - commission
            self.cash += proceeds
        else:
             logger.error(f"Invalid fill side in order: {side}")
             return

        # Update position quantity and average price
        new_quantity = current_quantity
        new_average_price = average_price

        if side == 'BUY':
            if current_quantity >= 0: # Adding to long or opening long
                total_quantity = current_quantity + quantity
                if total_quantity != 0:
                    new_average_price = ((average_price * current_quantity) + (price * quantity)) / total_quantity
                else:
                    new_average_price = 0
            new_quantity += quantity

        elif side == 'SELL':
            if current_quantity <= 0: # Adding to short or opening short
                 total_abs_quantity = abs(current_quantity) + quantity
                 if total_abs_quantity != 0:
                     if current_quantity == 0:
                         new_average_price = price
                     else:
                         new_average_price = ((average_price * abs(current_quantity)) + (price * quantity)) / total_abs_quantity
                 else:
                     new_average_price = 0
            new_quantity -= quantity

        if abs(new_quantity) < 1e-9:
            new_quantity = 0

        # Update positions dictionary (use self.positions)
        if new_quantity == 0:
            if symbol in self.positions:
                del self.positions[symbol]
                logger.debug(f"Position closed for {symbol}")
        else:
            self.positions[symbol] = {'quantity': new_quantity, 'average_price': new_average_price}
            logger.debug(f"Position updated for {symbol}: Qty={new_quantity:.4f}, AvgPx={new_average_price:.4f}")

        logger.info(f"Fill processed: {side} {quantity} {symbol} @ {price:.4f}. Commission: {commission:.2f}. New Cash: {self.cash:.2f}")
        # Optionally record the trade if needed
        # self._record_trade(order, price, quantity)

    def get_current_state(self) -> PortfolioState:
        """ Returns the current state of the portfolio (cash, positions, valuations). """
        pass

    def generate_orders(
        self,
        signals: SignalsDict,
        timestamp: datetime,
        market_data: Dict[str, Any], # Need market data (e.g., current prices) for sizing/validation
        risk_constraints: Optional[Dict[str, Any]] = None # Constraints from RiskManager
    ) -> List[Order]:
        """
        Generates concrete orders based on signals, current state, market data, and risk.

        Args:
            signals: Signals from the StrategyManager.
            timestamp: The current timestamp for decision making.
            market_data: Dictionary containing relevant market data (e.g., current prices)
                         for symbols involved in signals or portfolio.
            risk_constraints: Optional constraints provided by the RiskManager.

        Returns:
            A list of orders to be sent to the ExecutionHandler.
        """
        pass

    # Optional methods that might be useful
    # @abstractmethod
    # def rebalance(self, target_weights: Dict[str, float], timestamp: datetime) -> List[Order]:
    #     """ Generates orders to rebalance the portfolio towards target weights. """
    #     pass
    #
    # @abstractmethod
    # def calculate_performance_metrics(self) -> Dict[str, Any]:
    #     """ Calculates and returns performance metrics based on portfolio history. """
    #     pass

    def get_current_value(self) -> float:
        """ Calculates and returns the total current market value of the portfolio. """
        pass

# --- Concrete Implementation ---

import pandas as pd
import logging
from datetime import datetime

# Placeholder for Order type - align with Execution Handler
# Order = Dict[str, Any]
# Fill = Dict[str, Any]

class SimplePortfolioManager(BasePortfolioManager):
    """
    A basic implementation of the Portfolio Manager.

    - Manages cash and positions for multiple symbols.
    - Generates simple market orders based on signals using fixed fractional allocation.
    - Updates portfolio based on fills.
    """
    def __init__(self, initial_cash: float, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SimplePortfolioManager.

        Args:
            initial_cash: The starting cash balance.
            config: Configuration dictionary. Expected keys:
                - 'allocation_fraction_per_trade' (float): Fraction of portfolio value to allocate
                                                           per trade (e.g., 0.05 for 5%). Default 0.05.
        """
        super().__init__(config)
        self.config = config if config is not None else {}
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        # Positions format: {symbol: {'quantity': int, 'average_price': float}}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._latest_market_data: Dict[str, pd.Series] = {} # Stores latest known data per symbol
        self._portfolio_history = [] # To track value over time

        # --- Configuration --- #
        # self.order_quantity = self.config.get('order_quantity', 10) # Removed
        self.allocation_fraction_per_trade = self.config.get('allocation_fraction_per_trade', 0.05)
        if not 0 < self.allocation_fraction_per_trade <= 1:
            logging.warning(f"'allocation_fraction_per_trade' should be between 0 and 1. Using default 0.05.")
            self.allocation_fraction_per_trade = 0.05

        logging.info(f"SimplePortfolioManager initialized with initial cash: {initial_cash:.2f} "
                     f"and allocation fraction: {self.allocation_fraction_per_trade*100:.2f}%")

    def update_market_data(self, market_data: Dict[str, pd.Series]) -> None:
        """
        Receives the latest market data (e.g., latest bar) for valuation.

        Args:
            market_data: Dictionary mapping symbol to a Series containing latest data
                         (e.g., 'close' price).
        """
        for symbol, data_series in market_data.items():
            if not data_series.empty:
                self._latest_market_data[symbol] = data_series
            # Optionally log if data is missing?

    def generate_orders(
        self,
        signals: Dict[str, int], # {symbol: signal}, now includes potential stop-loss overrides
        market_data: Dict[str, pd.Series], # Added market_data
        risk_constraints: Optional[Dict[str, Any]] = None # Not used yet
    ) -> List[Order]:
        """
        Generates orders based on signals, current positions, portfolio value, and allocation fraction.
        Prioritizes stop-loss signals to flatten positions.

        Args:
            signals: Dictionary mapping symbol to trading signal (1=buy, -1=sell, 0=hold).
                     Stop-loss signals are also represented here (-1 for long close, 1 for short close).
            market_data: Dictionary mapping symbol to latest market data (needs 'close' price).
            risk_constraints: Optional constraints from RiskManager (currently unused).

        Returns:
            List of order dictionaries to be sent to ExecutionHandler.
        """
        orders = []
        current_portfolio_value = self.get_current_value()
        if current_portfolio_value <= 0:
            logging.warning("Cannot generate orders: Portfolio value is zero or negative.")
            return orders

        allocation_per_trade = self.allocation_fraction_per_trade * current_portfolio_value
        logging.debug(f"Generating orders based on signals: {signals}. Portfolio Value: {current_portfolio_value:.2f}, Allocation per trade: {allocation_per_trade:.2f}")

        for symbol, signal in signals.items():
            if symbol not in market_data:
                logging.warning(f"No market data for {symbol}, cannot generate order.")
                continue

            current_price = market_data[symbol].get('close')
            if current_price is None or pd.isna(current_price) or current_price <= 0:
                logging.warning(f"Invalid or missing close price for {symbol} ({current_price}), cannot generate order.")
                continue

            current_position_data = self.positions.get(symbol, {'quantity': 0, 'average_price': 0.0})
            current_quantity = current_position_data.get('quantity', 0)

            target_quantity = 0 # Default target is flat
            is_stop_loss = False

            # --- Stop-Loss Check FIRST ---
            # A stop-loss signal means flatten the position regardless of strategy signal
            if (signal == -1 and current_quantity > 0) or (signal == 1 and current_quantity < 0):
                logging.info(f"Stop-loss triggered for {symbol}. Target quantity set to 0.")
                target_quantity = 0
                is_stop_loss = True
            # --- END Stop-Loss Check ---

            # --- Regular Signal Processing (if not a stop-loss) ---
            elif not is_stop_loss:
                if signal == 1: # Signal to Buy/Go Long
                    if current_quantity < 0:
                         target_quantity = 0 # First close short position
                    else:
                         # Calculate quantity based on fractional allocation
                         trade_value = allocation_per_trade
                         target_quantity = int(trade_value / current_price)
                         logging.debug(f"  {symbol} (BUY Signal): Current Qty={current_quantity}, Target Value={trade_value:.2f}, Target Qty={target_quantity}")
                elif signal == -1: # Signal to Sell/Go Short
                    if current_quantity > 0:
                         target_quantity = 0 # First close long position
                    else:
                         # Calculate quantity based on fractional allocation for shorting
                         trade_value = allocation_per_trade
                         target_quantity = -int(trade_value / current_price) # Negative quantity for short
                         logging.debug(f"  {symbol} (SELL Signal): Current Qty={current_quantity}, Target Value={trade_value:.2f}, Target Qty={target_quantity}")
                # If signal is 0 (Hold/Flatten), target_quantity remains 0 (unless already holding)
                elif signal == 0:
                    target_quantity = 0 # Explicitly flatten if signal is 0
                    logging.debug(f"  {symbol} (HOLD/FLATTEN Signal): Current Qty={current_quantity}, Target Qty=0")
                else:
                    logging.warning(f"Invalid signal '{signal}' for {symbol}. Ignoring.")
                    continue # Skip this symbol if signal is invalid
            # --- END Regular Signal Processing ---

            order_quantity = target_quantity - current_quantity

            if order_quantity != 0:
                side = 'BUY' if order_quantity > 0 else 'SELL'
                abs_order_quantity = abs(order_quantity)

                # Basic check: Ensure sufficient cash for buys (ignoring margin for shorts for now)
                if side == 'BUY':
                    estimated_cost = abs_order_quantity * current_price # Rough estimate
                    if estimated_cost > self.current_cash:
                        logging.warning(f"Insufficient cash for proposed BUY order: Need {estimated_cost:.2f}, Have {self.current_cash:.2f}. Reducing order size.")
                        # Simple reduction: use all available cash (can be improved)
                        available_qty = int(self.current_cash / current_price)
                        if available_qty <= 0:
                            logging.warning(f"  Cannot afford even 1 unit of {symbol}. Skipping order.")
                            continue
                        abs_order_quantity = available_qty
                        order_quantity = available_qty # Adjust order quantity
                        if target_quantity > 0 and current_quantity > target_quantity: # Sell case adjustment, should not happen with current logic but check
                             order_quantity = -available_qty # Ensure sign is correct if logic changes

                # Create the order dictionary
                order = {
                    'symbol': symbol,
                    'type': 'MARKET', # Simple market orders for now
                    'quantity': abs_order_quantity,
                    'side': side,
                    'timestamp': datetime.now(), # Or use orchestrator timestamp
                    'order_id': f"ord_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" # Unique order ID
                }
                orders.append(order)
                logging.info(f"Generated Order: {side} {abs_order_quantity} {symbol} (Target Qty: {target_quantity}, Current Qty: {current_quantity})")
            else:
                 logging.debug(f"No order generated for {symbol}: Target Qty ({target_quantity}) == Current Qty ({current_quantity}) or signal ignored.")

        return orders

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """ Returns the current positions. """
        return self.positions.copy()

    def get_cash_balance(self) -> float:
        """ Returns the current cash balance. """
        return self.current_cash

    def get_current_value(self) -> float:
        """ Calculates and returns the total current market value of the portfolio. """
        total_value = self.current_cash
        for symbol, position_details in self.positions.items():
            quantity = position_details.get('quantity', 0)
            if quantity != 0 and symbol in self._latest_market_data:
                latest_price = self._latest_market_data[symbol].get('close')
                if latest_price is not None and not pd.isna(latest_price):
                    total_value += quantity * latest_price
                else:
                    logging.warning(f"Could not get latest price for {symbol} to calculate portfolio value.")
                    # Optionally, use average entry price as fallback?
                    # total_value += quantity * position_details.get('average_price', 0.0)
            elif quantity != 0:
                logging.warning(f"No latest market data for {symbol} to calculate portfolio value.")
                # Fallback
                # total_value += quantity * position_details.get('average_price', 0.0)

        return total_value

    def get_current_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary representing the current state of the portfolio.
        """
        state = {
            'timestamp': datetime.now(),
            'cash': self.get_cash_balance(),
            'positions': self.get_positions(),
            'total_value': self.get_current_value(),
            'latest_market_data': self._latest_market_data # Include for potential debugging
        }
        return state

# --- Additional Portfolio Manager Implementations (Optional) --- #
# Example: A more complex manager handling position sizing, risk checks, etc.
