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
                - 'lookback_period' (int): Number of periods to lookback for sentiment analysis. Default 3.
                - 'smoothing_factor' (float): Weight for exponential smoothing. Default 0.7.
                - 'use_volume_weighting' (bool): Whether to weight sentiment by volume. Default False.
                - 'signal_scaling' (bool): Whether to scale signal strength by sentiment. Default False.
                - 'min_data_points' (int): Minimum data points needed for signal generation. Default 2.
        """
        super().__init__(name=name, config=config)
        self.config = config if config is not None else {} 
        self.buy_threshold = self.config.get('buy_threshold', 0.1)
        self.sell_threshold = self.config.get('sell_threshold', -0.1)
        self.lookback_period = self.config.get('lookback_period', 3)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.7)
        self.use_volume_weighting = self.config.get('use_volume_weighting', False)
        self.signal_scaling = self.config.get('signal_scaling', False)
        self.min_data_points = self.config.get('min_data_points', 2)
        self.analyzer = None
        if SentimentIntensityAnalyzer:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            logging.warning(f"{self.name}: VADER analyzer not available.")
        logging.info(f"{self.name} initialized with thresholds - Buy: {self.buy_threshold}, Sell: {self.sell_threshold}, Lookback: {self.lookback_period}")
 
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame], # Historical data window
        current_positions: Dict[str, Any] = None,
        **kwargs: Any
    ) -> Dict[str, int]:
        """
        Generates buy/sell/hold signals based on sentiment data with optional smoothing.

        Args:
            data: Dictionary mapping symbol to a DataFrame containing historical data.
                  Expected columns: 'sentiment_score' or similar sentiment indicators.
            current_positions: Dictionary mapping symbol to current position details.
            kwargs: Additional arguments, potentially including `current_data` (a Dict[str, pd.Series])
                    and `timestamp`.

        Returns:
            Dictionary mapping symbol to signal (1 for Buy, -1 for Sell, 0 for Hold).
        """
        if current_positions is None:
            current_positions = {}
            
        logger.info(f"{self.name}: Generating signals at timestamp {kwargs.get('timestamp', 'N/A')}")
        signals = {}
        timestamp = kwargs.get('timestamp')

        # Process each symbol in the historical data
        for symbol, symbol_data in data.items():
            # Early validation of the data
            if symbol_data is None or symbol_data.empty:
                logger.warning(f"{self.name}: Empty data for symbol {symbol}. Setting HOLD signal.")
                signals[symbol] = 0  # Default to HOLD
                continue
            
            try:
                # Log all available columns to help debug
                logger.debug(f"{self.name}: Available columns for {symbol}: {list(symbol_data.columns)}")
                
                # Find any column that might contain sentiment data using more flexible matching
                sentiment_col = None
                for col in symbol_data.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['sentiment', 'score', 'polarity', 'sentiment_score']):
                        sentiment_col = col
                        break
                
                # If no sentiment column found, try common prefixed patterns
                if sentiment_col is None:
                    prefixed_patterns = [f'{symbol}_sentiment', f'{symbol.lower()}_sentiment', 
                                         f'{symbol}_score', f'{symbol.lower()}_score']
                    for pattern in prefixed_patterns:
                        matching_cols = [col for col in symbol_data.columns if pattern in col.lower()]
                        if matching_cols:
                            sentiment_col = matching_cols[0]
                            break
                
                # Still no sentiment column? Look for any numeric column that might work
                if sentiment_col is None:
                    numeric_cols = symbol_data.select_dtypes(include=['number']).columns
                    if 'sentiment' in numeric_cols:
                        sentiment_col = 'sentiment'
                    elif len(numeric_cols) > 0:
                        # Use the first numeric column as a last resort
                        sentiment_col = numeric_cols[0]
                        logger.warning(f"{self.name}: No explicit sentiment column found. Using numeric column {sentiment_col} as fallback.")
                
                # If still no sentiment column found, set HOLD and continue to next symbol
                if sentiment_col is None:
                    logger.error(f"{self.name}: No suitable sentiment or numeric column found for {symbol}. Available columns: {list(symbol_data.columns)}. Setting HOLD signal.")
                    signals[symbol] = 0  # Default to HOLD
                    continue
                
                logger.info(f"{self.name}: Using sentiment column: {sentiment_col} for {symbol}")
                
                # Make sure the data is sorted by timestamp for time series analysis
                if isinstance(symbol_data.index, pd.DatetimeIndex):
                    symbol_data = symbol_data.sort_index()
                elif 'timestamp' in symbol_data.columns:
                    symbol_data = symbol_data.sort_values('timestamp')
                
                # Get the sentiment values and ensure they are numeric
                try:
                    sentiment_values = pd.to_numeric(symbol_data[sentiment_col], errors='coerce')
                    # Drop NaN values
                    valid_sentiment = sentiment_values.dropna()
                    
                    if valid_sentiment.empty:
                        logger.warning(f"{self.name}: No valid sentiment values for {symbol} after cleaning. Setting HOLD signal.")
                        signals[symbol] = 0  # Default to HOLD
                        continue
                        
                    # Get the latest valid sentiment value for signal generation
                    latest_sentiment = valid_sentiment.iloc[-1]
                    
                    # Generate clear Buy/Sell/Hold signal based on thresholds
                    if latest_sentiment > self.buy_threshold:
                        signals[symbol] = 1  # BUY signal
                        logger.info(f"{self.name}: BUY signal for {symbol} - sentiment: {latest_sentiment:.4f} > threshold: {self.buy_threshold}")
                    elif latest_sentiment < self.sell_threshold:
                        signals[symbol] = -1  # SELL signal
                        logger.info(f"{self.name}: SELL signal for {symbol} - sentiment: {latest_sentiment:.4f} < threshold: {self.sell_threshold}")
                    else:
                        signals[symbol] = 0  # HOLD signal
                        logger.info(f"{self.name}: HOLD signal for {symbol} - sentiment: {latest_sentiment:.4f} is between thresholds")
                    
                except Exception as e:
                    logger.error(f"{self.name}: Error converting sentiment values to numeric for {symbol}: {e}")
                    signals[symbol] = 0  # Default to HOLD on error
            
            except Exception as e:
                logger.error(f"{self.name}: Error processing sentiment for {symbol}: {e}")
                signals[symbol] = 0  # Default to HOLD on error
                
        # Ensure we have signals for all symbols (final sanity check)
        for symbol in data.keys():
            if symbol not in signals:
                logger.warning(f"{self.name}: No signal generated for {symbol}. Setting HOLD as fallback.")
                signals[symbol] = 0
        
        logger.info(f"{self.name}: Generated signals for {len(signals)} symbols: {signals}")
        return signals

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Updates the strategy's configuration parameters.

        Args:
            config_updates: Dictionary containing parameters to update.
        """
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"{self.name}: Updated config {key} to {value}")
                # Update instance variables
                if key == 'buy_threshold':
                    self.buy_threshold = value
                elif key == 'sell_threshold':
                    self.sell_threshold = value
                elif key == 'lookback_period':
                    self.lookback_period = value
                elif key == 'smoothing_factor':
                    self.smoothing_factor = value
                elif key == 'use_volume_weighting':
                    self.use_volume_weighting = value
                elif key == 'signal_scaling':
                    self.signal_scaling = value
                elif key == 'min_data_points':
                    self.min_data_points = value
 
# --- Concrete Strategy Manager Implementation --- #
 
class SimpleStrategyManager(BaseStrategyManager):
    """
    A basic implementation of the Strategy Manager.
 
    Manages a collection of strategies and orchestrates signal generation.
    In this simple version, it handles one strategy directly.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        print("DEBUG: BaseStrategyManager config keys:", list(config.keys()))
        logger.info(f"DEBUG: BaseStrategyManager config: {config}")
        super().__init__(config, data_manager)
        self._strategies = {}
        self.data_manager = data_manager
        self.config = config
        self.name = config.get('name', 'BaseStrategyManager')
 
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
    Now supports advanced signal processing (noise filtering and regime detection) on sentiment and price data.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        print("DEBUG: SentimentStrategyManager config keys:", list(config.keys()))
        logger.info(f"DEBUG: SentimentStrategyManager config: {config}")
        super().__init__(config, data_manager)
        # Add strategy-specific parameters from config
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5) # Example threshold
        # Increase lookback to ensure we have sufficient data for sentiment analysis and signal generation
        self.lookback = config.get('lookback', 120) # Increased default lookback from 60 to 120
        # --- Advanced signal processing config ---
        self.signal_processing_cfg = config.get('signal_processing', {})
        logger.info(f"SentimentStrategyManager initialized with threshold: {self.sentiment_threshold}, lookback: {self.lookback}")
 
    def generate_signals(self, current_data: Dict[str, pd.Series], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generates trading signals based on sentiment data.
        Now supports configurable advanced signal processing (noise filtering and regime detection).
        Args:
            current_data: Dict of {symbol: pd.Series} for the current timestamp.
            historical_data: Dict of {symbol: pd.DataFrame} for lookback window.
        Returns:
            Dict of {symbol: signal}.
        """
        signals = {}
        for symbol, series in current_data.items():
            # --- Extract sentiment and price series ---
            sentiment_col = f'{symbol}_sentiment_score' if f'{symbol}_sentiment_score' in series else 'sentiment_score'
            price_col = 'close' if 'close' in series else series.index[0] if len(series.index) > 0 else None

            # --- Prepare historical data for processing ---
            hist = historical_data.get(symbol)
            if hist is not None and not hist.empty:
                # --- Apply signal processing if configured ---
                from ai_trading_agent.signal_processing.filters import exponential_moving_average, savitzky_golay_filter, rolling_zscore
                from ai_trading_agent.signal_processing.regime import volatility_regime
                # --- Sentiment filtering ---
                sentiment_filter = self.signal_processing_cfg.get('sentiment_filter')
                sentiment_window = self.signal_processing_cfg.get('sentiment_filter_window', 10)
                sentiment_series = hist[sentiment_col] if sentiment_col in hist else None
                if sentiment_series is not None and sentiment_filter:
                    if sentiment_filter == 'ema':
                        sentiment_series = exponential_moving_average(sentiment_series, span=sentiment_window)
                    elif sentiment_filter == 'savgol':
                        sentiment_series = savitzky_golay_filter(sentiment_series, window_length=sentiment_window)
                    elif sentiment_filter == 'zscore':
                        sentiment_series = rolling_zscore(sentiment_series, window=sentiment_window)
                    current_sentiment = sentiment_series.iloc[-1]
                else:
                    current_sentiment = series.get(sentiment_col, None)
                # --- Price filtering ---
                price_filter = self.signal_processing_cfg.get('price_filter')
                price_window = self.signal_processing_cfg.get('price_filter_window', 10)
                price_series = hist[price_col] if price_col in hist else None
                if price_series is not None and price_filter:
                    if price_filter == 'ema':
                        price_series = exponential_moving_average(price_series, span=price_window)
                    elif price_filter == 'savgol':
                        price_series = savitzky_golay_filter(price_series, window_length=price_window)
                    elif price_filter == 'zscore':
                        price_series = rolling_zscore(price_series, window=price_window)
                # --- Regime detection ---
                regime_type = self.signal_processing_cfg.get('regime_detection')
                regime_window = self.signal_processing_cfg.get('regime_window', 20)
                regime_threshold = self.signal_processing_cfg.get('regime_vol_threshold', 0.02)
                regime_label = None
                if price_series is not None and regime_type == 'volatility':
                    regime_labels = volatility_regime(price_series, window=regime_window, threshold=regime_threshold)
                    regime_label = regime_labels.iloc[-1]
                # Optionally use regime_label in your strategy logic
            else:
                current_sentiment = series.get(sentiment_col, None)
                regime_label = None

            # --- Enhanced Sentiment Strategy Logic: Regime-aware --- #
            try:
                sentiment_float = float(current_sentiment)
                # Only trade in 'low_vol' regime if regime_label is present
                if regime_label is not None and regime_label != 'low_vol':
                    signals[symbol] = 0 # HOLD in high_vol or unknown regime
                else:
                    if sentiment_float > self.sentiment_threshold:
                        signals[symbol] = 1 # BUY
                    elif sentiment_float < -self.sentiment_threshold:
                        signals[symbol] = -1 # SELL
                    else:
                        signals[symbol] = 0 # HOLD
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting sentiment '{current_sentiment}' to float for {symbol}: {e}. Holding.")
                signals[symbol] = 0 # HOLD on error
        # Ensure all symbols managed by the strategy have a signal (default to HOLD)
        for symbol in self.config.get('symbols', []):
            if symbol not in signals:
                signals[symbol] = 0
                logger.warning(f"No data found for configured symbol {symbol}, setting HOLD signal.")
        return signals
 
    # Potentially override other methods if needed
