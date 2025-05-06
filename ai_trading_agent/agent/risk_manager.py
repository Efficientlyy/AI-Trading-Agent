"""
Risk manager module for the AI Trading Agent.

This module handles risk assessment, position sizing, and stop-loss management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..common import logger

# Type aliases for clarity
PortfolioState = Dict[str, Any]
Order = Dict[str, Any]

# --- Abstract Base Class --- #

class RiskManagerABC(ABC):
    """Abstract base class for risk managers.

    Responsible for assessing risk, enforcing risk limits (e.g., position sizing, stop-losses),
    and potentially adjusting trading behavior based on risk profile.
    """

    @abstractmethod
    def assess_order_risk(self, order: Order, portfolio_state: PortfolioState) -> bool:
        """Assesses if a proposed order complies with risk rules (e.g., max exposure).

        Args:
            order: The proposed Order object.
            portfolio_state: The current state of the portfolio.

        Returns:
            True if the order is acceptable from a risk perspective, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_stop_loss_signals(self, portfolio_state: PortfolioState, market_data: Dict[str, pd.Series]) -> Dict[str, int]:
        """Checks open positions against stop-loss criteria and generates signals to close them.

        Args:
            portfolio_state: The current state of the portfolio (including positions and entry prices).
            market_data: Dictionary mapping symbols to their current market data series (for current prices).

        Returns:
            A dictionary mapping symbols to signals (-1 to sell/close due to stop-loss, 0 otherwise).
            This typically overrides regular strategy signals for the affected symbols.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: int, portfolio_state: PortfolioState, market_data: Dict[str, pd.Series]) -> Optional[float]:
        """Calculates the appropriate quantity (size) for a potential trade based on risk.

        Args:
            symbol: The symbol for the potential trade.
            signal: The trading signal (1 for buy, -1 for sell).
            portfolio_state: The current state of the portfolio.
            market_data: Dictionary mapping symbols to their current market data series.

        Returns:
            The calculated position size (e.g., number of shares/contracts), or None if the trade
            should not be taken due to risk constraints (e.g., insufficient capital, max exposure reached).
        """
        raise NotImplementedError

# --- Base Class --- #

class BaseRiskManager(RiskManagerABC):
    """
    Abstract base class for Risk Managers.

    Responsible for monitoring portfolio risk exposure and providing
    constraints or directives to the PortfolioManager.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the BaseRiskManager.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'max_position_size' (Optional[int]): Max units per symbol. Default None.
                - 'max_portfolio_risk_pct' (Optional[float]): Max % of portfolio value at risk. Default None.
                - 'stop_loss_pct' (Optional[float]): Percentage drop from entry price to trigger stop-loss. Default None.
        """
        self.config = config if config is not None else {}
        self.max_position_size = self.config.get('max_position_size')
        self.max_portfolio_risk_pct = self.config.get('max_portfolio_risk_pct')
        self.stop_loss_pct = self.config.get('stop_loss_pct')
        logging.info("BaseRiskManager initialized.")
        if self.max_position_size:
             logging.info(f"  - Max position size per symbol: {self.max_position_size}")
        if self.max_portfolio_risk_pct:
             logging.info(f"  - Max portfolio risk percentage: {self.max_portfolio_risk_pct * 100:.2f}%")
        if self.stop_loss_pct:
            logging.info(f"  - Stop Loss Percentage: {self.stop_loss_pct * 100:.2f}%")

    @abstractmethod
    def assess_risk(self, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the current portfolio state and market data to return risk metrics.

        Args:
            portfolio_state: The current state of the portfolio.
            market_data: Current market data needed for valuation/risk assessment.

        Returns:
            A dictionary containing calculated risk metrics (e.g., VaR, max drawdown,
            exposure per asset/sector, volatility).
        """
        pass

    @abstractmethod
    def get_risk_constraints(self, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determines applicable risk constraints based on the current state and market data.
        These constraints are passed to the PortfolioManager during order generation.

        Args:
            portfolio_state: The current state of the portfolio.
            market_data: Current market data.

        Returns:
            A dictionary of constraints (e.g., {'max_order_value': 10000,
            'allowed_symbols': ['AAPL', 'MSFT'], 'max_drawdown_pct': 0.1,
            'max_position_pct': 0.2}) or None if no specific constraints apply.
        """
        pass

    def get_current_risk_exposure(self) -> Dict[str, Any]:
        """ Calculates and returns the current overall risk exposure. """
        pass

# --- Concrete Implementation ---

class SimpleRiskManager(BaseRiskManager):
    """
    A very basic implementation of the Risk Manager.

    Currently acts as a placeholder, performing minimal checks.
    Can be extended later with more sophisticated risk rules
    (e.g., max drawdown, VaR limits, position concentration).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SimpleRiskManager.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'max_position_size' (Optional[int]): Max units per symbol. Default None.
                - 'max_portfolio_risk_pct' (Optional[float]): Max % of portfolio value at risk. Default None.
                - 'stop_loss_pct' (Optional[float]): Percentage drop from entry price to trigger stop-loss. Default None.
        """
        super().__init__(config)

    def assess_risk(self, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the current portfolio state and market data to return risk metrics.

        Args:
            portfolio_state: The current state of the portfolio.
            market_data: Current market data needed for valuation/risk assessment.

        Returns:
            A dictionary containing calculated risk metrics (e.g., VaR, max drawdown,
            exposure per asset/sector, volatility).
        """
        pass

    def get_risk_constraints(self, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determines applicable risk constraints based on the current state and market data.
        These constraints are passed to the PortfolioManager during order generation.

        Args:
            portfolio_state: The current state of the portfolio.
            market_data: Current market data.

        Returns:
            A dictionary of constraints (e.g., {'max_order_value': 10000,
            'allowed_symbols': ['AAPL', 'MSFT'], 'max_drawdown_pct': 0.1,
            'max_position_pct': 0.2}) or None if no specific constraints apply.
        """
        pass

    def assess_order_risk(self, order: Order, portfolio_state: PortfolioState) -> bool:
        """
        Assesses if a proposed order violates basic risk rules.

        Args:
            order: The proposed order dictionary.
            portfolio_state: The current state of the portfolio.

        Returns:
            True if the order is acceptable, False otherwise.
        """
        symbol = order['symbol']
        order_quantity = order['quantity']
        order_side = order['side']
        current_quantity = portfolio_state.get(symbol, {}).get('quantity', 0)

        # 1. Max Position Size Check
        if self.max_position_size is not None:
            proposed_quantity = current_quantity
            if order_side == 'BUY':
                proposed_quantity += order_quantity
            elif order_side == 'SELL':
                # When selling to open short, check the absolute size
                 if current_quantity >= 0:
                     proposed_quantity = -order_quantity # Opening a new short
                 else:
                      proposed_quantity -= order_quantity # Adding to existing short

            if abs(proposed_quantity) > self.max_position_size:
                logging.warning(f"Risk Check Failed (Order): Proposed quantity {proposed_quantity} for {symbol} exceeds max size {self.max_position_size}.")
                return False

        # 2. Max Portfolio Risk Percentage Check (Placeholder - needs price)
        # This requires order price or estimated execution price for accuracy.
        # A simple placeholder might check if order value exceeds a percentage.
        # if self.max_portfolio_risk_pct is not None:
            # estimate_order_value = order_quantity * estimated_price
            # if estimate_order_value / current_value > self.max_portfolio_risk_pct:
            #     logging.warning(f"Risk Check Failed (Order): Estimated order value for {symbol} exceeds max risk percentage.")
            #     return False

        logging.debug(f"Risk Check Passed (Order): {order_side} {order_quantity} {symbol}")
        return True # Pass by default if no rules violated

    def generate_stop_loss_signals(self, portfolio_state: PortfolioState, market_data: Dict[str, pd.Series]) -> Dict[str, int]:
        """
        Checks open positions against stop-loss levels.

        Args:
            portfolio_state: Current state including positions with entry prices.
            market_data: Current market data (latest bar for each symbol).

        Returns:
            Dictionary of {symbol: signal} where signal is -1 (sell) or 1 (buy-to-cover)
            if a stop-loss is triggered. Empty dict otherwise.
        """
        signals = {}
        if self.stop_loss_pct is None or not isinstance(self.stop_loss_pct, (int, float)) or self.stop_loss_pct <= 0:
            return signals # Stop-loss not configured or invalid

        positions = portfolio_state.get('positions', {})
        if not positions:
            return signals # No positions to check

        logging.debug(f"Checking stop-loss for positions: {list(positions.keys())}")

        for symbol, position_data in positions.items():
            quantity = position_data.get('quantity', 0)
            avg_entry_price = position_data.get('avg_entry_price', None)

            if quantity == 0 or avg_entry_price is None or avg_entry_price <= 0: # Also check entry price validity
                logging.debug(f"Skipping stop-loss check for {symbol}: quantity={quantity}, avg_entry_price={avg_entry_price}")
                continue

            if symbol not in market_data:
                logging.warning(f"No current market data for {symbol}. Cannot check stop-loss.")
                continue

            current_price = market_data[symbol].get('close', None)
            if current_price is None or not isinstance(current_price, (int, float)) or current_price <= 0:
                logging.warning(f"Invalid or missing current close price for {symbol} ({current_price}). Cannot check stop-loss.")
                continue

            stop_loss_level = 0.0
            if quantity > 0: # Long position
                stop_loss_level = avg_entry_price * (1 - self.stop_loss_pct)
                logging.debug(f"  {symbol} (Long): Entry={avg_entry_price:.4f}, Current={current_price:.4f}, Stop Level={stop_loss_level:.4f}")
                if current_price <= stop_loss_level:
                    logging.info(f"STOP-LOSS triggered for LONG {symbol}: Price {current_price:.4f} <= Stop Level {stop_loss_level:.4f} (Entry: {avg_entry_price:.4f})")
                    signals[symbol] = -1 # Signal to sell
            elif quantity < 0: # Short position
                 stop_loss_level = avg_entry_price * (1 + self.stop_loss_pct)
                 logging.debug(f"  {symbol} (Short): Entry={avg_entry_price:.4f}, Current={current_price:.4f}, Stop Level={stop_loss_level:.4f}")
                 if current_price >= stop_loss_level:
                     logging.info(f"STOP-LOSS triggered for SHORT {symbol}: Price {current_price:.4f} >= Stop Level {stop_loss_level:.4f} (Entry: {avg_entry_price:.4f})")
                     signals[symbol] = 1 # Signal to buy-to-cover

        return signals

    def calculate_position_size(self, symbol: str, signal: int, portfolio_state: PortfolioState, market_data: Dict[str, pd.Series]) -> Optional[float]:
        """
        Calculates the appropriate quantity (size) for a potential trade based on risk.

        Args:
            symbol: The symbol for the potential trade.
            signal: The trading signal (1 for buy, -1 for sell).
            portfolio_state: The current state of the portfolio.
            market_data: Dictionary mapping symbols to their current market data series.

        Returns:
            The calculated position size (e.g., number of shares/contracts), or None if the trade
            should not be taken due to risk constraints (e.g., insufficient capital, max exposure reached).
        """
        # Placeholder implementation
        return None

    def get_current_risk_exposure(self) -> Dict[str, Any]:
        """
        Calculates and returns the current overall risk exposure.
        Placeholder implementation.
        """
        # Placeholder: Could calculate VaR, drawdown, concentration, etc.
        logging.debug("Calculating risk exposure (currently none).")
        return {"status": "Not Implemented"}
