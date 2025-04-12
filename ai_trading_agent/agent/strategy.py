import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from .data_manager import DataManagerABC # <-- Added import

# Configure logging for this module
logger = logging.getLogger(__name__)

# Placeholder for a more structured Signal object later if needed
Signal = int # e.g., 1 (Buy), -1 (Sell), 0 (Hold)
SignalsDict = Dict[str, Signal] # e.g., {'AAPL': 1, 'GOOG': -1}

class BaseStrategy(ABC):
    """
    Abstract base class for a single trading strategy.
    Encapsulates the logic to generate trading signals based on data.
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the strategy.

        Args:
            name: A unique name for the strategy instance.
            config: Strategy-specific configuration parameters.
        """
        self.name = name
        self.config = config if config is not None else {}
        self._initialize_state() # Allow subclasses to set up

    def _initialize_state(self):
        """Optional method for subclasses to initialize internal state."""
        pass

    @abstractmethod
    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_portfolio: Optional[Dict[str, Any]] = None # Optional access to current holdings/cash
    ) -> SignalsDict:
        """
        Generate trading signals based on the provided data and optional portfolio state.

        Args:
            data: Aligned market/alternative data from the DataManager.
                  Could be a single DataFrame or Dict[symbol, DataFrame].
            current_portfolio: Optional dictionary representing current portfolio state
                               (e.g., {'cash': 10000, 'positions': {'AAPL': 10}}).

        Returns:
            A dictionary mapping symbols to signals (e.g., {'AAPL': 1, 'GOOG': -1, 'MSFT': 0}).
        """
        pass

    @abstractmethod
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the strategy's configuration parameters dynamically.

        Args:
            config_updates: A dictionary containing parameters to update.
        """
        pass


class StrategyManagerABC(ABC):
    """Abstract base class for managing multiple trading strategies.

    Coordinates signal generation across different strategies and symbols.
    """

    @abstractmethod
    def generate_signals(self, current_data: Dict[str, pd.Series], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generates trading signals for all managed symbols based on current and historical data.

        Args:
            current_data: Dictionary mapping symbols to their current market data (e.g., latest OHLCV row).
            historical_data: Dictionary mapping symbols to their historical market data (DataFrame).
                           The amount of history depends on the strategies' lookback requirements.

        Returns:
            A dictionary mapping symbols to their final trading signal (1, -1, or 0).
            If multiple strategies exist, this manager might combine or prioritize signals.
        """
        raise NotImplementedError


class BaseStrategyManager(StrategyManagerABC):
    """
    Base class for managing one or more trading strategies.
    Responsible for orchestrating strategies and potentially combining signals.
    Provides common structure, subclasses implement specific logic.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        """
        Initializes the Strategy Manager.

        Args:
            config: Configuration for the manager.
            data_manager: The data manager instance.
        """
        self.config = config
        self.data_manager = data_manager # Store data manager if needed by base logic
        self._strategies: Dict[str, BaseStrategy] = {} # Internal storage for strategies
        logger.info(f"{self.__class__.__name__} initialized.")
        # Example: Load strategies specified in config if applicable
        # self._load_strategies_from_config()

    # --- Methods from original BaseStrategyManager (now concrete) --- #

    # @abstractmethod # Removed decorator
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Adds a strategy instance to the manager. Ensures unique names.
        (Concrete implementation might be simple storage)
        """
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must be an instance of BaseStrategy")
        if strategy.name in self._strategies:
            logger.warning(f"Strategy with name '{strategy.name}' already exists. Overwriting.")
        self._strategies[strategy.name] = strategy
        logger.info(f"Strategy '{strategy.name}' added to {self.__class__.__name__}.")

    # @abstractmethod # Removed decorator
    def remove_strategy(self, strategy_name: str) -> None:
        """
        Removes a strategy instance from the manager by its name.
        """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            logger.info(f"Strategy '{strategy_name}' removed from {self.__class__.__name__}.")
        else:
            logger.warning(f"Strategy '{strategy_name}' not found in {self.__class__.__name__}.")

    # @abstractmethod # Removed decorator
    def process_data_and_generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]: # Changed return type to Dict[str, int]
        """
        Receives data, passes it to managed strategies, combines their signals
        according to configured logic, and returns the final aggregated signals.
        (Base implementation might just call generate_signals or be strategy-specific)
        """
        # This base implementation assumes generate_signals is the primary method
        # to be overridden by specific managers like SentimentStrategyManager.
        # A more complex base might iterate through self._strategies here.
        logger.warning("BaseStrategyManager.process_data_and_generate_signals called. "
                       "This should ideally be handled by subclass's generate_signals.")
        # Returning empty signals as a safe default
        return {}

    # @abstractmethod # Removed decorator
    def update_strategy_config(self, strategy_name: str, config_updates: Dict[str, Any]) -> None:
        """
        Updates the configuration for a specific managed strategy.
        """
        if strategy_name in self._strategies:
            # Assuming strategies have an 'update_config' method
            if hasattr(self._strategies[strategy_name], 'update_config') and callable(getattr(self._strategies[strategy_name], 'update_config')):
                self._strategies[strategy_name].update_config(config_updates)
                logger.info(f"Configuration updated for strategy '{strategy_name}'.")
            else:
                logger.warning(f"Strategy '{strategy_name}' does not have an 'update_config' method.")
        else:
            logger.warning(f"Strategy '{strategy_name}' not found for config update.")

    # @abstractmethod # Removed decorator
    def get_strategy_names(self) -> List[str]:
        """
        Returns the names of all managed strategies.
        """
        return list(self._strategies.keys())

    # @abstractmethod # This one comes from StrategyManagerABC, MUST be implemented by subclasses
    @abstractmethod
    def generate_signals(self, current_data: Dict[str, pd.Series], portfolio_state: Dict[str, Any], **kwargs) -> Dict[str, int]:
        """Generates trading signals for all managed strategies.
 
        Args:
            current_data: Dictionary mapping symbols to their current market data.
            portfolio_state: Current portfolio state.
            kwargs: Additional arguments.
 
        Returns:
            A dictionary mapping symbols to final trading signals (e.g., 1, -1, 0).
        """
        raise NotImplementedError("generate_signals must be implemented by concrete subclass")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """ Retrieves a strategy instance by name. """
        strategy = self._strategies.get(name)
        if strategy is None:
            logger.warning(f"Strategy '{name}' not found in {self.__class__.__name__}.")
        return strategy
 
 
# --- Concrete Strategy Implementation --- 
 
import pandas as pd
import logging 
 
# Attempt VADER import
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    logging.error("VADER Sentiment library not found. Please install it: pip install vaderSentiment")
    SentimentIntensityAnalyzer = None
 
class SentimentStrategy(BaseStrategy):
    """
    A strategy that generates trading signals based on sentiment scores.
 
    Requires 'sentiment_score' column in the input data from DataManager.
    Uses VADER for sentiment analysis if text is provided instead of score.
    """
    def __init__(self, name: str = "SentimentStrategy", config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SentimentStrategy.
 
        Args:
            name (str): The name of the strategy instance.
            config: Configuration dictionary. Expected keys:
                - 'buy_threshold' (float): Sentiment score above which to buy. Default 0.1.
                - 'sell_threshold' (float): Sentiment score below which to sell. Default -0.1.
        """
        super().__init__(name=name, config=config)
        self.buy_threshold = self.config.get('buy_threshold', 0.1)
        self.sell_threshold = self.config.get('sell_threshold', -0.1)
        self.analyzer = None
        if SentimentIntensityAnalyzer:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            logging.warning(f"{self.name}: VADER analyzer not available.")
 
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame], # Historical data window
        current_positions: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, int]:
        """
        Generates buy/sell/hold signals based on the latest sentiment score.

        Args:
            data: Dictionary mapping symbol to a DataFrame containing historical data.
                  Expected columns: 'sentiment_score'.
            current_positions: Dictionary mapping symbol to current position details.
            kwargs: Additional arguments, potentially including `current_data` (a Dict[str, pd.Series])
                    and `timestamp`.

        Returns:
            Dictionary mapping symbol to signal (1 for Buy, -1 for Sell, 0 for Hold).
        """
        logger.info(f"{self.name}: Entering generate_signals at timestamp {kwargs.get('timestamp', 'N/A')}")
        signals = {}
        timestamp = kwargs.get('timestamp')
        current_data_arg = kwargs.get('current_data') # Check if current data is passed

        # Determine the set of symbols to process
        # Use symbols from historical data dict primarily, but consider current_data if passed
        symbols_to_process = set(data.keys())
        if isinstance(current_data_arg, dict):
            symbols_to_process.update(current_data_arg.keys())

        for symbol in symbols_to_process:
            latest_score = None
            score_source = "N/A"

            # 1. Prioritize current sentiment score if available in kwargs['current_data']
            if isinstance(current_data_arg, dict) and symbol in current_data_arg:
                current_symbol_series = current_data_arg[symbol]
                if isinstance(current_symbol_series, pd.Series) and 'sentiment_score' in current_symbol_series.index:
                    current_score = current_symbol_series['sentiment_score']
                    if not pd.isna(current_score):
                        latest_score = current_score
                        score_source = "current_data"
                    else:
                        logger.debug(f"{self.name}: Current sentiment score for {symbol} is NaN.")
                elif isinstance(current_symbol_series, pd.Series) and 'text' in current_symbol_series.index and self.analyzer:
                     current_text = current_symbol_series['text']
                     if not pd.isna(current_text) and isinstance(current_text, str):
                         try:
                             vs = self.analyzer.polarity_scores(current_text)
                             latest_score = vs['compound']
                             score_source = "current_data_text"
                         except Exception as e:
                             logging.error(f"{self.name}: Error analyzing current text for {symbol}: {e}", exc_info=True)
                     else:
                         logger.debug(f"{self.name}: Current text for {symbol} is invalid.")

            # 2. Fallback to the last value in the historical data window if current is not found/valid
            if latest_score is None and symbol in data:
                symbol_data = data[symbol]
                if not symbol_data.empty:
                    if 'sentiment_score' in symbol_data.columns:
                        hist_score = symbol_data['sentiment_score'].iloc[-1]
                        if not pd.isna(hist_score):
                            latest_score = hist_score
                            score_source = "historical_data_last"
                        else:
                             logger.debug(f"{self.name}: Last historical sentiment score for {symbol} is NaN.")
                    elif 'text' in symbol_data.columns and self.analyzer:
                        hist_text = symbol_data['text'].iloc[-1]
                        if not pd.isna(hist_text) and isinstance(hist_text, str):
                             try:
                                 vs = self.analyzer.polarity_scores(hist_text)
                                 latest_score = vs['compound']
                                 score_source = "historical_data_text_last"
                             except Exception as e:
                                logging.error(f"{self.name}: Error analyzing last historical text for {symbol}: {e}", exc_info=True)
                        else:
                             logger.debug(f"{self.name}: Last historical text for {symbol} is invalid.")

            # 3. If no valid score found, generate HOLD
            if latest_score is None:
                logging.warning(f"{self.name}: Could not determine a valid sentiment score for {symbol} at {timestamp}. Generating HOLD signal.")
                signals[symbol] = 0
                continue

            # --- Debug Logging --- 
            logger.debug(f"{self.name} - Symbol: {symbol}, Timestamp: {timestamp}, Score: {latest_score:.4f} (Source: {score_source}), Buy: {self.buy_threshold}, Sell: {self.sell_threshold}")
            # --- End Debug Logging ---

            # Generate signal based on score and thresholds
            if latest_score > self.buy_threshold:
                signals[symbol] = 1  # Buy signal
            elif latest_score < self.sell_threshold:
                signals[symbol] = -1 # Sell signal
            else:
                signals[symbol] = 0  # Hold signal

            # --- Added Debug Logging for Signal --- 
            logger.debug(f"{self.name} - Symbol: {symbol}, Generated Signal: {signals[symbol]}")
            # --- End Added Debug Logging ---

            # Simple logging
            # logging.debug(f"{self.name} - {symbol}: Score={latest_score:.2f}, Signal={signals[symbol]}")
 
        return signals
 
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """ Updates the strategy's configuration. """
        super().update_config(new_config)
        # Re-read thresholds from the updated config
        self.buy_threshold = self.config.get('buy_threshold', 0.1)
        self.sell_threshold = self.config.get('sell_threshold', -0.1)
        logging.info(f"{self.name}: Configuration updated. New thresholds: Buy > {self.buy_threshold}, Sell < {self.sell_threshold}")
 
 
# --- Concrete Strategy Manager Implementation --- #
 
class SimpleStrategyManager(BaseStrategyManager):
    """
    A basic implementation of the Strategy Manager.
 
    Manages a collection of strategies and orchestrates signal generation.
    In this simple version, it handles one strategy directly.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, data_manager: DataManagerABC = None):
        super().__init__(config=config, data_manager=data_manager)
        # self._strategies is already initialized in BaseStrategyManager's __init__
        # No need to re-initialize here: self._strategies: Dict[str, BaseStrategy] = {}
        self.data_manager = data_manager # Store the data manager
 
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """ Registers a strategy instance with the manager. """
        if strategy.name in self._strategies:
            logging.warning(f"Strategy '{strategy.name}' already exists and will be overwritten.")
        self._strategies[strategy.name] = strategy
        logging.info(f"Strategy '{strategy.name}' added to StrategyManager.")
 
    def remove_strategy(self, strategy_name: str) -> bool:
        """ Removes a strategy instance from the manager. """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            logging.info(f"Strategy '{strategy_name}' removed from StrategyManager.")
            return True
        else:
            logging.warning(f"Attempted to remove non-existent strategy: {strategy_name}")
            return False
 
    def process_data_and_generate_signals(
        self,
        data: Dict[str, pd.DataFrame], # Data from DataManager
        current_positions: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Dict[str, int]]: # {strategy_name: {symbol: signal}}
        """
        Processes data through managed strategies and aggregates signals.
 
        Args:
            data: Dictionary mapping symbol to DataFrame of market/alternative data.
            current_positions: Current portfolio positions.
            kwargs: Additional arguments.
 
        Returns:
            Dictionary mapping strategy name to its generated signals ({symbol: signal}).
            In this simple version, assumes one strategy.
        """
        all_signals: Dict[str, Dict[str, int]] = {}
 
        if not self._strategies:
            logging.warning("StrategyManager has no strategies loaded. No signals generated.")
            return all_signals
 
        for strategy_name, strategy_instance in self._strategies.items():
            try:
                # Pass the data and current positions to the strategy
                symbols = list(data.keys())
                lookback = 5 # Define lookback period
                historical_data_window = self.data_manager.get_historical_data(symbols=symbols, lookback=lookback)
 
                if historical_data_window is None:
                     logger.warning(f"Could not retrieve historical data window (lookback={lookback}) at timestamp {kwargs.get('timestamp')}. Skipping signal generation.")
                     strategy_signals = {symbol: 0 for symbol in symbols} # Default to HOLD
                else:
                     # --- Call strategy with historical DataFrame dictionary ---            
                     strategy_signals = strategy_instance.generate_signals(
                         data=historical_data_window, # Pass historical DataFrames
                         current_positions=current_positions, # Pass actual positions
                         **kwargs # Pass down timestamp etc.
                     )
                     # --- End Call ---            
                all_signals[strategy_name] = strategy_signals
                logger.debug(f"Signals generated by '{strategy_name}': {strategy_signals}")
            except Exception as e:
                logging.error(f"Error generating signals from strategy '{strategy_name}': {e}", exc_info=True)
                all_signals[strategy_name] = {symbol: 0 for symbol in data.keys()} # Default to HOLD on error
 
        # Future work: Implement logic to combine signals from multiple strategies if needed.
        # For now, just return the signals per strategy.
        return all_signals
 
    def update_strategy_config(self, strategy_name: str, new_config: Dict[str, Any]) -> None:
        """ Updates the configuration of a specific managed strategy. """
        if strategy_name in self._strategies:
            self._strategies[strategy_name].update_config(new_config)
        else:
            logging.warning(f"Attempted to update config for non-existent strategy: {strategy_name}")
 
    def get_strategy_names(self) -> List[str]:
        """ Returns the names of all managed strategies. """
        return list(self._strategies.keys())
 
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """ Retrieves a strategy instance by name. """
        return self._strategies.get(name)
 
    def generate_signals(self, current_data: Dict[str, pd.Series], portfolio_state: Dict[str, Any], **kwargs) -> Dict[str, int]:
        """Generates trading signals by delegating to managed strategies.
 
        This simple manager currently assumes it manages only one strategy
        and returns its signals directly. A more complex manager might combine
        signals from multiple strategies.
 
        Args:
            current_data: Dictionary mapping symbols to their current market data.
            portfolio_state: Current portfolio state.
            kwargs: Additional arguments.
 
        Returns:
            A dictionary mapping symbols to their final trading signal (1, -1, or 0).
        """
        final_signals: Dict[str, int] = {}
 
        if not self._strategies:
            logger.warning(f"{self.__class__.__name__}: No strategies loaded. Returning HOLD signals.")
            # Determine symbols from data if possible, otherwise return empty
            symbols = list(current_data.keys()) if current_data else []
            return {symbol: 0 for symbol in symbols}
 
        # Simple approach: Use the signals from the *first* strategy found.
        # A more robust implementation would handle multiple strategies (e.g., averaging, voting).
        first_strategy_name = next(iter(self._strategies)) # Get the name of the first strategy
        strategy_instance = self._strategies[first_strategy_name]
 
        try:
            logger.debug(f"Generating signals using strategy: {first_strategy_name}")
            # Ensure the managed strategy's generate_signals method exists and is callable
            if hasattr(strategy_instance, 'generate_signals') and callable(getattr(strategy_instance, 'generate_signals')):
                # Note: The managed strategy (e.g., SentimentStrategy) might have a different
                # generate_signals signature. Adapt the call:
                # SentimentStrategy expects: data, current_positions, **kwargs
                current_positions = portfolio_state.get('positions', {})
                symbols = list(current_data.keys())
                lookback = 5 # Define lookback period
                historical_data_window = self.data_manager.get_historical_data(symbols=symbols, lookback=lookback)
 
                if historical_data_window is None:
                     logger.warning(f"Could not retrieve historical data window (lookback={lookback}) at timestamp {kwargs.get('timestamp')}. Skipping signal generation.")
                     strategy_signals = {symbol: 0 for symbol in symbols} # Default to HOLD
                else:
                     # --- Call strategy with historical DataFrame dictionary ---            
                     strategy_signals = strategy_instance.generate_signals(
                         data=historical_data_window, # Pass historical DataFrames
                         current_positions=current_positions, # Pass actual positions
                         **kwargs # Pass down timestamp etc.
                     )
                     # --- End Call ---            
                final_signals = strategy_signals
                logger.debug(f"Signals from '{first_strategy_name}': {final_signals}")
            else:
                 logger.error(f"Strategy '{first_strategy_name}' is missing the 'generate_signals' method.")
                 symbols = list(current_data.keys()) if current_data else []
                 final_signals = {symbol: 0 for symbol in symbols} # Default to HOLD
 
        except Exception as e:
            logger.error(f"Error generating signals from strategy '{first_strategy_name}': {e}", exc_info=True)
            symbols = list(current_data.keys()) if current_data else []
            final_signals = {symbol: 0 for symbol in symbols} # Default to HOLD on error
 
        return final_signals
 
    def update_strategy_config(self, strategy_name: str, new_config: Dict[str, Any]) -> None:
        """ Updates the configuration of a specific managed strategy. """
        if strategy_name in self._strategies:
            self._strategies[strategy_name].update_config(new_config)
        else:
            logging.warning(f"Attempted to update config for non-existent strategy: {strategy_name}")
 
class SentimentStrategyManager(BaseStrategyManager):
    """Manages sentiment-based strategies.
 
    This specific implementation generates signals based on sentiment scores.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        super().__init__(config, data_manager)
        # Add strategy-specific parameters from config
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5) # Example threshold
        self.lookback = config.get('lookback', 20) # For potential technical indicators
        logger.info(f"SentimentStrategyManager initialized with threshold: {self.sentiment_threshold}")
 
    def generate_signals(self, current_data: Dict[str, pd.Series], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generates trading signals based on sentiment data.
 
        Args:
            current_data: Dictionary mapping symbol to a Series of current data (OHLCV, sentiment, etc.).
            historical_data: Dictionary mapping symbol to a DataFrame of historical data.
 
        Returns:
            A dictionary mapping symbol to a signal (-1 for SELL, 0 for HOLD, 1 for BUY).
        """
        signals = {}
        symbols = list(current_data.keys()) # Get symbols from current data keys
 
        for symbol in symbols:
            # Ensure data exists for the symbol
            if symbol not in current_data or current_data[symbol] is None or current_data[symbol].empty:
                logger.warning(f"No current data available for {symbol} at this timestamp.")
                signals[symbol] = 0 # HOLD if no data
                continue
 
            # Safely get the sentiment score
            current_sentiment = current_data[symbol].get('sentiment_score')
 
            # Check if sentiment score is valid
            if current_sentiment is None or pd.isna(current_sentiment):
                # If sentiment is missing, try getting it from historical data's last row (if available)
                # This handles cases where sentiment might lag or not be present every tick
                if symbol in historical_data and not historical_data[symbol].empty and 'sentiment_score' in historical_data[symbol].columns:
                     last_historical_sentiment = historical_data[symbol]['sentiment_score'].iloc[-1]
                     if last_historical_sentiment is not None and not pd.isna(last_historical_sentiment):
                         current_sentiment = last_historical_sentiment
                         logger.debug(f"Using last historical sentiment ({current_sentiment}) for {symbol}.")
                     else:
                         logger.debug(f"Sentiment score missing for {symbol} (current & historical). Holding.")
                         signals[symbol] = 0 # HOLD if no sentiment found
                         continue
                else:
                    logger.debug(f"Sentiment score not available for {symbol}. Holding.")
                    signals[symbol] = 0 # HOLD if no sentiment found
                    continue
 
            # --- Basic Sentiment Strategy Logic --- #
            try:
                # Convert sentiment to float for comparison, handle potential errors
                sentiment_float = float(current_sentiment)
 
                if sentiment_float > self.sentiment_threshold:
                    logger.debug(f"BUY signal generated for {symbol} (Sentiment: {sentiment_float} > {self.sentiment_threshold})")
                    signals[symbol] = 1 # BUY
                elif sentiment_float < -self.sentiment_threshold: # Example: Symmetric threshold
                    logger.debug(f"SELL signal generated for {symbol} (Sentiment: {sentiment_float} < {-self.sentiment_threshold})")
                    signals[symbol] = -1 # SELL
                else:
                    # logger.debug(f"HOLD signal for {symbol} (Sentiment: {sentiment_float} within threshold)")
                    signals[symbol] = 0 # HOLD
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting sentiment '{current_sentiment}' to float for {symbol}: {e}. Holding.")
                signals[symbol] = 0 # HOLD on error
 
        # Ensure all symbols managed by the strategy have a signal (default to HOLD)
        # This covers cases where a symbol might be in the config but lacks data initially
        for symbol in self.config.get('symbols', []):
             if symbol not in signals:
                 signals[symbol] = 0
                 logger.warning(f"No data found for configured symbol {symbol}, setting HOLD signal.")
 
 
        return signals
 
    # Potentially override other methods if needed
