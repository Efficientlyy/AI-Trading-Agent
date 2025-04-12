# src/agent/orchestrator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

# Import other base components - use relative imports if structure allows
try:
    from .data_manager import BaseDataManager
    from .strategy import BaseStrategyManager
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    from .risk_manager import BaseRiskManager
    from .execution_handler import BaseExecutionHandler
except ImportError:
    # Fallback for cases where direct execution might occur or structure differs
    from data_manager import BaseDataManager
    from strategy import BaseStrategyManager
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    from risk_manager import BaseRiskManager
    from execution_handler import BaseExecutionHandler


class OrchestratorABC(ABC):
    """Abstract base class for orchestrators.

    Coordinates the interaction between different agent components
    (data, strategy, portfolio, risk, execution) to run a trading
    simulation (backtest) or live trading session.
    """

    @abstractmethod
    def run(self) -> Optional[Dict[str, Any]]:
        """Executes the main trading loop (backtest or live).

        Returns:
            A dictionary containing results (e.g., performance metrics, trade history)
            in case of a backtest, or None/other status for live trading.
        """
        raise NotImplementedError


class BaseOrchestrator(OrchestratorABC):
    """
    Abstract base class for the Agent Orchestrator.

    Coordinates the flow of data and actions between all other components
    (DataManager, StrategyManager, RiskManager, PortfolioManager, ExecutionHandler).
    Drives the main event loop for backtesting or live trading.
    """

    def __init__(
        self,
        data_manager: BaseDataManager,
        strategy_manager: BaseStrategyManager,
        portfolio_manager: PortfolioManager,
        risk_manager: BaseRiskManager,
        execution_handler: BaseExecutionHandler,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the Orchestrator with all necessary components.

        Args:
            data_manager: Instance of a DataManager.
            strategy_manager: Instance of a StrategyManager.
            portfolio_manager: Instance of a PortfolioManager.
            risk_manager: Instance of a RiskManager.
            execution_handler: Instance of an ExecutionHandler.
            config: Orchestrator-specific configuration (e.g., run mode, timing, dates).
        """
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler
        self.config = config if config is not None else {}
        self._running = False
        # Add logger initialization later

    @abstractmethod
    def run(self) -> None:
        """
        Starts and runs the main loop of the trading agent.
        This loop typically involves fetching data, generating signals,
        managing the portfolio, executing orders, and handling events/time progression.
        Behavior differs significantly between backtesting and live trading.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stops the main loop gracefully.
        """
        pass

    # --- Optional methods for finer control ---

    # @abstractmethod
    # def step(self) -> bool:
    #     """
    #     Executes a single time step or event cycle of the agent.
    #     Useful for debugging or specific backtesting scenarios.
    #
    #     Returns:
    #         True if the simulation/session can continue, False otherwise.
    #     """
    #     pass
    #
    # @abstractmethod
    # def get_current_time(self) -> datetime:
    #     """ Returns the current time within the orchestrator's context (simulated or real). """
    #     pass

# --- Concrete Implementation ---

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
from ..backtesting.performance_metrics import calculate_metrics

# Type hints for Base classes will be resolved by Python if defined elsewhere
# or assumed to be defined later in the file or via __future__.annotations

# Imports for actual functionality

# Ensure all components are importable
# This try-except block seems redundant if imports are absolute
try:
    from .data_manager import SimpleDataManager # Example concrete class
    from .strategy import SimpleStrategyManager, SentimentStrategy # Example concrete classes
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager # Corrected import
    from .risk_manager import SimpleRiskManager # Example concrete class
    from .execution_handler import SimulatedExecutionHandler # Example concrete class
except ImportError as e:
    # Fallback for direct script execution or different structure
    logging.error(f"Failed to import concrete component classes: {e}. Ensure they exist and paths are correct.")
    # Assign None or raise depending on desired behavior
    SimpleDataManager = None
    SimpleStrategyManager = None
    SentimentStrategy = None
    PortfolioManager = None
    SimpleRiskManager = None
    SimulatedExecutionHandler = None

class BacktestOrchestrator(BaseOrchestrator):
    """
    Orchestrator specifically designed for running backtests.

    Coordinates the flow between components over a historical date range.
    """
    def __init__(
        self,
        data_manager: BaseDataManager,
        strategy_manager: BaseStrategyManager,
        portfolio_manager: PortfolioManager,
        risk_manager: BaseRiskManager,
        execution_handler: BaseExecutionHandler,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the BacktestOrchestrator.

        Args:
            data_manager: Instance of a DataManager.
            strategy_manager: Instance of a StrategyManager.
            portfolio_manager: Instance of a PortfolioManager.
            risk_manager: Instance of a RiskManager.
            execution_handler: Instance of an ExecutionHandler (typically SimulatedExecutionHandler).
            config: Configuration dictionary. Expected keys:
                - 'start_date' (datetime): Backtest start date.
                - 'end_date' (datetime): Backtest end date.
                - 'symbols' (List[str]): List of symbols to backtest.
                - 'data_types' (List[str]): Data types needed (e.g., ['ohlcv', 'sentiment']).
                - 'timeframe' (Optional[str]): Data timeframe (e.g., '1d').
        """
        super().__init__(data_manager, strategy_manager, portfolio_manager, risk_manager, execution_handler, config)

        # Validate required config keys
        required_keys = ['start_date', 'end_date', 'symbols']
        if not all(key in self.config for key in required_keys):
            raise ValueError(f"Missing required config keys for BacktestOrchestrator: {required_keys}")

        self.start_date: datetime = self.config['start_date']
        self.end_date: datetime = self.config['end_date']
        self.symbols: List[str] = self.config['symbols']
        self.data_types: List[str] = self.config.get('data_types', ['ohlcv'])
        self.timeframe: Optional[str] = self.config.get('timeframe')

        self.results: Dict[str, Any] = {
            'portfolio_history': [], # List of {timestamp, value, cash, positions}
            'trades': [],           # List of executed trades from ExecutionHandler
            'orders_generated': [], # List of all orders sent to ExecutionHandler
            'signals': [],          # List of {timestamp, symbol, signal_type, value}
            'performance_metrics': {}
        }
        logging.info("BacktestOrchestrator initialized.")

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Runs the backtest simulation loop.

        Returns:
            A dictionary containing the backtest results, or None if execution fails early.
        """
        if self._running:
            logging.warning("Backtest is already running.")
            return None

        self._running = True
        logging.info(f"Starting backtest run...")

        # Initialize results dictionary
        self.results = {
            'portfolio_history': [], # List of {timestamp, value, cash, positions}
            'trades': [],           # List of executed trades from ExecutionHandler
            'orders_generated': [], # List of all orders sent to ExecutionHandler
            'signals': [],          # List of {timestamp, symbol, signal_type, value}
            'performance_metrics': {}
        }

        # --- Main Backtest Loop --- #
        # Use the DataManager to drive the loop based on available timestamps
        current_time = self.data_manager.get_next_timestamp()
        if current_time is None:
            logging.error("No historical data timestamps available from DataManager. Cannot run backtest.")
            self._running = False
            return None # Return None or empty results if no data

        while self._running and current_time:
            logging.debug(f"Processing timestamp: {current_time}")

            # 0. Get current market data slice for this timestamp
            try:
                market_data = self.data_manager.get_current_data()
                if not market_data:
                    logging.warning(f"No market data available for {current_time}, skipping step.")
                    # Fetch the next timestamp and continue the loop
                    current_time = self.data_manager.get_next_timestamp()
                    continue
                logging.debug(f"Market data for {current_time}: {market_data}")
            except Exception as e:
                logging.error(f"Error retrieving market data for {current_time}: {e}", exc_info=True)
                # Decide whether to stop or continue
                self._running = False
                break # Stop the loop on data error

            # 1. Update Portfolio Manager with current market data (for valuation)
            try:
                self.portfolio_manager.update_market_data(market_data)
                portfolio_state = self.portfolio_manager.get_portfolio_state() # Corrected method name
                current_positions = portfolio_state.get('positions', {})
                # Record portfolio state at the START of the timestamp processing
                self.results['portfolio_history'].append({
                    'timestamp': current_time,
                    'value': portfolio_state.get('total_value', 0),
                    'cash': portfolio_state.get('cash', 0),
                    'positions': current_positions.copy() # Record snapshot
                })
                logging.debug(f"Portfolio state updated: {portfolio_state}")
            except Exception as e:
                logging.error(f"Error updating portfolio manager state: {e}", exc_info=True)
                self._running = False
                break # Stop if portfolio update fails

            # 2a. Get strategy signals based on current market data
            try:
                # Pass current data, portfolio state, and timestamp
                strategy_signals = self.strategy_manager.generate_signals(
                    current_data=market_data, 
                    portfolio_state=portfolio_state, 
                    timestamp=current_time
                )
                if strategy_signals:
                     self.results['signals'].extend([
                         {'timestamp': current_time, 'symbol': sym, 'signal_type': 'strategy', 'value': sig}
                         for sym, sig in strategy_signals.items()
                     ])
                logging.debug(f"Strategy signals generated: {strategy_signals}")
            except Exception as e:
                logging.error(f"Error generating strategy signals: {e}", exc_info=True)
                strategy_signals = {} # Proceed without strategy signals if error

            # 2b. Get stop-loss signals from Risk Manager
            try:
                # --- Add logging before call ---
                logging.debug(f"Calling generate_stop_loss_signals. Portfolio state type: {type(portfolio_state)}, Market data keys: {list(market_data.keys()) if market_data else 'None'}")
                logging.debug(f"Portfolio state content (sample): {str(portfolio_state)[:200]}...") # Log sample content
                # --- Correcting arguments to use 'portfolio_state' and 'market_data' --- #
                stop_loss_signals = self.risk_manager.generate_stop_loss_signals(portfolio_state=portfolio_state, market_data=market_data)
                if stop_loss_signals:
                    self.results['signals'].extend([
                        {'timestamp': current_time, 'symbol': sym, 'signal_type': 'stop_loss', 'value': sig}
                        for sym, sig in stop_loss_signals.items()
                     ])
                logging.debug(f"Stop-loss signals generated: {stop_loss_signals}")
            except Exception as e:
                logging.error(f"Failed to generate stop-loss signals: {e}", exc_info=True)
                stop_loss_signals = {}

            # --- Combine Signals (Stop-loss overrides strategy) --- #
            final_signals = strategy_signals.copy() if strategy_signals else {}
            if stop_loss_signals:
                for symbol, signal in stop_loss_signals.items():
                    if symbol in final_signals and final_signals[symbol] != signal:
                        logging.info(f"Stop-loss signal for {symbol} ({signal}) overrides strategy signal ({final_signals[symbol]}).")
                    elif symbol not in final_signals:
                        logging.info(f"Adding stop-loss signal for {symbol} ({signal}).")
                    final_signals[symbol] = signal

            # 3. Get Risk Constraints (optional, based on current state)
            try:
                # Use the already fetched state
                current_portfolio_value = portfolio_state.get('total_value')
                # Ensure value is not None before proceeding
                if current_portfolio_value is not None:
                    risk_constraints = self.risk_manager.get_risk_constraints(
                        current_positions=current_positions,
                        current_value=current_portfolio_value
                    )
                    logging.debug(f"Risk constraints generated: {risk_constraints}")
                else:
                    logging.warning("Portfolio value is None, cannot generate risk constraints.")
                    risk_constraints = None
            except Exception as e:
                logging.error(f"Failed to generate risk constraints: {e}", exc_info=True)
                risk_constraints = None

            # 4. Generate orders from Portfolio Manager based on FINAL signals and market data
            if final_signals:
                 try:
                    # Pass market_data needed for order calculation (e.g., price)
                    proposed_orders = self.portfolio_manager.generate_orders(
                         signals=final_signals,
                         market_data=market_data, # Pass current market data
                         risk_constraints=risk_constraints
                     )
                    logging.debug(f"Proposed orders: {proposed_orders}")
                 except Exception as e:
                    logging.error(f"Failed to generate proposed orders: {e}", exc_info=True)
                    proposed_orders = []
            else:
                proposed_orders = []
                logging.debug("No final signals, no orders proposed.")

            # 5. (Optional) Risk Manager approves/modifies orders
            # --- Simplified Risk Approval: Check each order individually --- #
            approved_orders = []
            if proposed_orders and current_portfolio_value is not None:
                for order in proposed_orders:
                    try:
                        # Pass necessary state to assess_risk
                        if self.risk_manager.assess_risk(order, current_positions, current_portfolio_value):
                            approved_orders.append(order)
                        else:
                            logging.warning(f"Order rejected by Risk Manager: {order}")
                    except Exception as e:
                        logging.error(f"Error during risk assessment for order {order.order_id}: {e}", exc_info=True)
            elif not proposed_orders:
                logging.debug("No proposed orders to approve.")
            else:
                logging.warning("Cannot approve orders because current portfolio value is None.")

            # 6. Submit approved orders to Execution Handler
            if approved_orders:
                self.results['orders_generated'].extend(approved_orders)
                # --- Add logging before loop ---
                logging.debug(f"Processing {len(approved_orders)} orders for execution.")
                for order in approved_orders:
                    try:
                        # --- Add logging before call ---
                        logging.debug(f"Calling execute_order for order: {order}")
                        executed_order = self.execution_handler.execute_order(order, market_data)
                        if executed_order:
                            logging.debug(f"Order execution result: {executed_order}")
                            # Check if the order resulted in a fill (status might be FILLED, PARTIALLY_FILLED)
                            if executed_order.status in ['FILLED', 'PARTIALLY_FILLED']:
                                self.results['trades'].append(executed_order)
                                # Update portfolio based on executed trade
                                # --- Use the existing update_fill method --- #
                                self.portfolio_manager.update_fill(executed_order)
                            # --- Removed call to non-existent update_from_executed_trade --- #
                        # self.execution_handler.submit_order(order)
                        # logging.debug(f"Submitted order: {order.order_id}")
                    except Exception as e:
                        logging.error(f"Error submitting order {order.order_id} to execution handler: {e}", exc_info=True)
            else:
                 logging.debug("No approved orders to submit.")

            # Fetch the next timestamp for the loop
            current_time = self.data_manager.get_next_timestamp()

        # --- Post-Backtest Analysis --- #
        # Calculate final portfolio value after last trades
        if self.portfolio_manager:
             final_state = self.portfolio_manager.get_portfolio_state() # Corrected method name
             # --- Logging for final timestamp access --- #
             dm_type = type(self.data_manager)
             index_attr = getattr(self.data_manager, 'combined_index', 'AttributeMissing') # Corrected attribute name
             index_type = type(index_attr)
             index_len = len(index_attr) if hasattr(index_attr, '__len__') else 'N/A'
             logging.debug(f"Final timestamp access: DM Type={dm_type}, Index Attr Type={index_type}, Index Length={index_len}")
 
             # Access the internal combined index for the last timestamp (Using correct attribute name)
             # The check 'hasattr...not empty' handles the conditions previously checked in the removed block.
             last_timestamp = self.data_manager.combined_index[-1] \
                 if hasattr(self.data_manager, 'combined_index') and self.data_manager.combined_index is not None and not self.data_manager.combined_index.empty else None # Check existence and non-empty
 
             if last_timestamp and final_state:
                 logging.info(f"Final portfolio state at {last_timestamp}:")
                 # Log final cash and positions
                 logging.info(f"  Final Cash: {final_state.get('cash', 'N/A'):.2f}")
                 final_positions = final_state.get('positions', {})
                 if final_positions:
                     logging.info("  Final Positions:")
                     for symbol, details in final_positions.items():
                         logging.info(f"    {symbol}: Qty={details.get('quantity', 'N/A')}, AvgPx={details.get('average_price', 'N/A'):.4f}")
                 else:
                     logging.info("  No final positions held.")

        # --- Calculate Performance Metrics --- #
        logging.debug("Attempting to calculate performance metrics...")
        if self.results.get('portfolio_history'):
            logging.debug(f"Portfolio history type: {type(self.results['portfolio_history'])}, Length: {len(self.results['portfolio_history'])}")
            if self.results['portfolio_history']:
                logging.debug(f"First history element: {self.results['portfolio_history'][0]}")
            history_df = pd.DataFrame(self.results['portfolio_history'])
            # --- Ensure timestamp column is datetime type BEFORE setting index --- #
            logging.debug(f"History DF columns before conversion: {history_df.columns}")
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            logging.debug(f"History DF columns after conversion: {history_df.columns}")
            logging.debug(f"History DF head:\n{history_df.head()}")
            history_df = history_df.set_index('timestamp').sort_index()

            if not history_df.empty:
                try:
                    # Pass the entire DataFrame as expected by the function
                    logging.debug(f"Calling calculate_metrics with history_df (Type: {type(history_df)}, Columns: {history_df.columns})")
                    self.results['performance_metrics'] = calculate_metrics(history_df)
                    logging.debug(f"calculate_metrics call successful.")
                    logging.info(f"Performance metrics calculated: {self.results['performance_metrics']}")
                except Exception as e:
                    logging.error(f"Failed to calculate performance metrics: {e}", exc_info=True)
                    self.results['performance_metrics'] = {'error': str(e)}
            else:
                 logging.warning("Portfolio history DataFrame is empty after conversion. Cannot calculate metrics.")
                 self.results['performance_metrics'] = {'error': 'Empty history DataFrame'}
        else:
            logging.warning("No portfolio history recorded. Cannot calculate metrics.")
            self.results['performance_metrics'] = {'error': 'No history'}

        self._running = False
        logging.info("Backtest run finished.")
        return self.results

    # --- Add stop method implementation --- #
    def stop(self) -> None:
        """ Stops the backtest loop gracefully. """
        if self._running:
            logging.info("Attempting to stop backtest orchestrator...")
            self._running = False
            logging.info("Backtest orchestrator stop signal received.")
        else:
            logging.info("Backtest orchestrator is not currently running.")

# --- Example Usage --- #
# (Typically run from a script like scripts/run_backtest.py)
