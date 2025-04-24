# src/agent/data_manager.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataManagerABC(ABC):
    """Abstract base class for data managers.

    Defines the interface for providing market data during a backtest or live trading.
    """

    @abstractmethod
    def get_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Returns the next timestamp in the data stream or None if finished."""
        raise NotImplementedError

    @abstractmethod
    def get_current_data(self) -> Optional[Dict[str, pd.Series]]:
        """Returns the market data slice (e.g., OHLCV) for the current timestamp
           for all symbols, or None if no data is available for the current timestamp.

        Returns:
            A dictionary where keys are symbols and values are pandas Series
            containing data for the current timestamp (e.g., open, high, low, close, volume).
            Returns None if the data stream has ended or data is missing.
        """
        raise NotImplementedError

    @abstractmethod
    def get_historical_data(self, symbols: List[str], lookback: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Returns historical data for the given symbols up to the current timestamp,
           for the specified lookback period.

        Args:
            symbols: A list of symbols to retrieve data for.
            lookback: The number of historical periods (rows) to retrieve before the current timestamp.

        Returns:
            A dictionary where keys are symbols and values are pandas DataFrames
            containing the historical data. Returns None if data is unavailable.
        """
        raise NotImplementedError

class BaseDataManager(DataManagerABC):
    """Base implementation providing structure and configuration handling.

    Concrete data managers should inherit from this and implement
    the data loading and retrieval logic specific to their source.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, Any] = {}
        logger.info(f"{self.__class__.__name__} initialized with config.")
        # Common initialization logic here

    # Provide concrete implementations for methods not typically
    # overridden by simple file-based managers, but required by ABC.
    def add_data_provider(self, provider_name: str, provider_instance: Any) -> None:
        """Registers an external data provider (e.g., for live data)."""
        logger.warning("add_data_provider is not fully implemented in this base class.")
        self.providers[provider_name] = provider_instance
        # Placeholder - specific managers might need more logic

    def get_data(
        self,
        symbols: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        data_types: Union[str, List[str]] = 'ohlcv',
        timeframe: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, pd.DataFrame]:
        """Retrieves a block of historical data (implementation specific)."""
        logger.warning("get_data provides potentially broad data; use get_historical_data or get_current_data for backtest steps.")
        # Base implementation might raise error or return empty dict
        # Derived classes MUST override if this functionality is needed.
        raise NotImplementedError("get_data must be implemented by concrete subclass if needed.")

    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        data_types: Union[str, List[str]] = 'ohlcv',
        **kwargs: Any
    ) -> Optional[Dict[str, pd.Series]]:
        """Retrieves the latest available data point (implementation specific)."""
        logger.warning("get_latest_data might behave like get_current_data in backtesting contexts.")
        # Base implementation might raise error or return None
        # Derived classes MUST override if this functionality is needed.
        raise NotImplementedError("get_latest_data must be implemented by concrete subclass if needed.")

    # --- Methods to be implemented by concrete subclasses --- #

    @abstractmethod
    def get_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Returns the next timestamp in the data stream or None if finished."""
        raise NotImplementedError

    @abstractmethod
    def get_current_data(self) -> Optional[Dict[str, pd.Series]]:
        """Returns the market data slice (e.g., OHLCV) for the current timestamp
           for all symbols, or None if no data is available for the current timestamp.

        Returns:
            A dictionary where keys are symbols and values are pandas Series
            containing data for the current timestamp (e.g., open, high, low, close, volume).
            Returns None if the data stream has ended or data is missing.
        """
        raise NotImplementedError

    @abstractmethod
    def get_historical_data(self, symbols: List[str], lookback: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Returns historical data for the given symbols up to the current timestamp,
           for the specified lookback period.

        Args:
            symbols: A list of symbols to retrieve data for.
            lookback: The number of historical periods (rows) to retrieve before the current timestamp.

        Returns:
            A dictionary where keys are symbols and values are pandas DataFrames
            containing the historical data. Returns None if data is unavailable.
        """
        raise NotImplementedError

# --- Concrete Implementation ---

class SimpleDataManager(BaseDataManager):
    """Concrete implementation for managing data from CSV files.

    Handles loading historical OHLCV and potentially other data like sentiment.
    Provides data bar by bar for backtesting.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the SimpleDataManager.

        Args:
            config: Configuration dictionary containing keys like:
                - 'data_dir': Path to the directory containing data files.
                - 'symbols': List of symbols to manage.
                - 'start_date': Optional start date for filtering data.
                - 'end_date': Optional end date for filtering data.
                - 'timeframe': Data timeframe (e.g., '1d', '1h'). Used for file naming.
                - 'data_types': List of data types to load (e.g., ['ohlcv', 'sentiment']). Default ['ohlcv'].
        """
        super().__init__(config)
        self.data_dir = config.get('data_dir')
        self.symbols = config.get('symbols', [])
        self.start_date = pd.to_datetime(config.get('start_date')) if config.get('start_date') else None
        self.end_date = pd.to_datetime(config.get('end_date')) if config.get('end_date') else None
        self.timeframe = config.get('timeframe', '1d')
        self.data_types = config.get('data_types', ['ohlcv'])

        if not self.data_dir or not os.path.isdir(self.data_dir):
            raise ValueError(f"Invalid or missing data_dir: {self.data_dir}")
        if not self.symbols:
            raise ValueError("Symbols list cannot be empty.")

        self.data: Dict[str, pd.DataFrame] = {} # Stores OHLCV data per symbol
        self.sentiment_data: Optional[pd.DataFrame] = None # Stores combined sentiment data
        self.combined_index: Optional[pd.DatetimeIndex] = None
        self.current_index = 0

        self._load_data()
        self._prepare_combined_index()
        logger.info(f"{self.__class__.__name__} initialized. Data loaded for {len(self.data)} symbols.")
        if self.sentiment_data is not None:
            logger.info(f"Sentiment data loaded with shape: {self.sentiment_data.shape}")
        if self.combined_index is not None:
            logger.info(f"Combined data index prepared with {len(self.combined_index)} timestamps.")

    def _load_data(self):
        """Loads data files (OHLCV, Sentiment) based on configuration."""
        logger.info(f"Loading data from: {self.data_dir}")

        # Load OHLCV Data
        if 'ohlcv' in self.data_types:
            for symbol in self.symbols:
                file_path = os.path.join(self.data_dir, f"{symbol}_ohlcv_{self.timeframe}.csv")
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                        # Ensure standard column names
                        df.columns = [col.lower() for col in df.columns]
                        # Filter by date range if specified
                        if self.start_date:
                            df = df[df.index >= self.start_date]
                        if self.end_date:
                            df = df[df.index <= self.end_date]

                        if not df.empty:
                            self.data[symbol] = df.sort_index()
                            logger.debug(f"Loaded OHLCV for {symbol}: {len(df)} rows")
                        else:
                            logger.warning(f"OHLCV data for {symbol} is empty after date filtering or initially.")
                    except Exception as e:
                        logger.error(f"Error loading OHLCV data for {symbol} from {file_path}: {e}", exc_info=True)
                else:
                    logger.warning(f"OHLCV file not found for symbol {symbol}: {file_path}")
            if not self.data:
                 logger.error("Failed to load any OHLCV data. Cannot proceed.")
                 # raise FileNotFoundError("No OHLCV data loaded. Check file paths and configurations.")

        # Load Sentiment Data
        if 'sentiment' in self.data_types:
            sentiment_file = os.path.join(self.data_dir, 'synthetic_sentiment.csv') # Assuming single file for now
            # TODO: Handle potentially multiple sentiment files or formats
            if os.path.exists(sentiment_file):
                try:
                    sentiment_df = pd.read_csv(sentiment_file, index_col='timestamp', parse_dates=True)
                    # Filter by date range
                    if self.start_date:
                        sentiment_df = sentiment_df[sentiment_df.index >= self.start_date]
                    if self.end_date:
                        sentiment_df = sentiment_df[sentiment_df.index <= self.end_date]

                    if not sentiment_df.empty:
                        self.sentiment_data = sentiment_df.sort_index()
                        logger.debug(f"Loaded sentiment data: {len(sentiment_df)} rows")
                    else:
                        logger.warning("Sentiment data is empty after date filtering or initially.")
                except Exception as e:
                    logger.error(f"Error loading sentiment data from {sentiment_file}: {e}", exc_info=True)
            else:
                logger.warning(f"Sentiment file not found: {sentiment_file}")

    def _prepare_combined_index(self):
        """Creates a unified sorted index from all loaded data sources."""
        all_indices = set()
        if self.data:
            for df in self.data.values():
                all_indices.update(df.index)

        if self.sentiment_data is not None:
            all_indices.update(self.sentiment_data.index)

        if not all_indices:
            logger.warning("No data found to create a combined index.")
            self.combined_index = pd.DatetimeIndex([])
        else:
            # Sort the unique timestamps
            self.combined_index = pd.DatetimeIndex(sorted(list(all_indices)))
            # Apply date filtering again to the combined index
            if self.start_date:
                 self.combined_index = self.combined_index[self.combined_index >= self.start_date]
            if self.end_date:
                 self.combined_index = self.combined_index[self.combined_index <= self.end_date]

            if len(self.combined_index) == 0:
                logger.warning(f"Combined index is empty after filtering between {self.start_date} and {self.end_date}.")
            else:
                 logger.info(f"Combined index created with {len(self.combined_index)} timestamps from {self.combined_index.min()} to {self.combined_index.max()}.")


    def get_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Returns the next timestamp from the combined data index."""
        if self.combined_index is None or self.current_index >= len(self.combined_index):
            return None
        timestamp = self.combined_index[self.current_index]
        self.current_index += 1
        return timestamp

    def get_current_data(self) -> Optional[Dict[str, pd.Series]]:
        """Returns the data slice for the current timestamp.

        Includes OHLCV and merges sentiment data if available.
        Returns None if the current_index is out of bounds.
        """
        if self.combined_index is None or self.current_index == 0 or self.current_index > len(self.combined_index):
            logger.warning(f"Cannot get current data. Index out of bounds or not initialized. Current index: {self.current_index}")
            return None

        # Get the timestamp corresponding to the *previous* call to get_next_timestamp
        current_timestamp = self.combined_index[self.current_index - 1]
        logger.debug(f"Getting data for timestamp: {current_timestamp}")

        current_data_slice = {}
        # Get OHLCV data
        for symbol, df in self.data.items():
            if current_timestamp in df.index:
                current_data_slice[symbol] = df.loc[current_timestamp]
            else:
                # Handle missing OHLCV data for this timestamp (e.g., use NaN series or skip symbol)
                # Creating a NaN series ensures downstream components know the symbol exists but data is missing
                logger.debug(f"No OHLCV data for {symbol} at {current_timestamp}. Using NaNs.")
                # Determine columns from a sample DataFrame or use defaults
                cols = ['open', 'high', 'low', 'close', 'volume']
                if symbol in self.data and not self.data[symbol].empty:
                     cols = self.data[symbol].columns
                nan_series = pd.Series(index=cols, data=np.nan, name=current_timestamp)
                current_data_slice[symbol] = nan_series

        # Merge sentiment data if available
        if self.sentiment_data is not None and current_timestamp in self.sentiment_data.index:
            sentiment_row = self.sentiment_data.loc[current_timestamp]
            for symbol in self.symbols:
                # Find sentiment columns relevant to this symbol (e.g., 'AAPL_sentiment', 'sentiment_score')
                # This assumes a naming convention or a single score column
                sentiment_col_name = f'{symbol}_sentiment_score' # Example convention
                if sentiment_col_name in sentiment_row:
                    sentiment_value = sentiment_row[sentiment_col_name]
                elif 'sentiment_score' in sentiment_row: # Fallback to a generic column
                     sentiment_value = sentiment_row['sentiment_score']
                else:
                    sentiment_value = np.nan

                if symbol in current_data_slice:
                    # Add sentiment to the existing series for the symbol
                    # Use a consistent key like 'sentiment_score'
                    current_data_slice[symbol]['sentiment_score'] = sentiment_value
                else:
                    # If OHLCV was missing, create a series just for sentiment
                     logger.warning(f"Creating series for {symbol} just for sentiment data at {current_timestamp}.")
                     current_data_slice[symbol] = pd.Series({'sentiment_score': sentiment_value}, name=current_timestamp)
        elif self.sentiment_data is not None:
            # Sentiment data exists but not for this timestamp, ensure 'sentiment_score' is NaN
             for symbol in self.symbols:
                 if symbol in current_data_slice:
                     current_data_slice[symbol]['sentiment_score'] = np.nan

        if not current_data_slice:
             logger.warning(f"No data could be retrieved for any symbol at timestamp {current_timestamp}.")
             return None

        return current_data_slice

    def get_historical_data(self, symbols: List[str], lookback: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Returns historical data for the specified symbols up to the current timestamp.

        Args:
            symbols: List of symbols to retrieve data for.
            lookback: Number of historical bars to retrieve.

        Returns:
            Dictionary mapping symbols to their historical data DataFrames, or None if insufficient data.
        """
        if self.combined_index is None or self.current_index == 0 or self.current_index > len(self.combined_index):
             logger.warning("Cannot get historical data. Index out of bounds or not initialized.")
             return None

        # Current timestamp is the one most recently returned by get_next_timestamp
        current_internal_idx = self.current_index - 1
        if current_internal_idx < lookback -1:
            # Not enough historical data available yet
            logger.debug(f"Insufficient historical data for lookback {lookback} at index {current_internal_idx}.")
            return None

        # Determine the start index for the lookback period
        start_idx = max(0, current_internal_idx - lookback + 1)
        # Get the actual timestamps for the lookback period
        lookback_timestamps = self.combined_index[start_idx : current_internal_idx + 1]

        historical_data = {}
        for symbol in symbols:
            if symbol in self.data:
                # Select rows based on the lookback timestamps
                hist_df = self.data[symbol].loc[self.data[symbol].index.intersection(lookback_timestamps)]

                # Merge sentiment if available for this symbol and timestamps
                if self.sentiment_data is not None:
                    sentiment_col_name = f'{symbol}_sentiment_score' # Example convention
                    if sentiment_col_name in self.sentiment_data.columns:
                         symbol_sentiment = self.sentiment_data.loc[self.sentiment_data.index.intersection(lookback_timestamps), [sentiment_col_name]]
                         # Rename for consistency if needed, e.g., to 'sentiment_score'
                         symbol_sentiment = symbol_sentiment.rename(columns={sentiment_col_name: 'sentiment_score'})
                         hist_df = pd.merge(hist_df, symbol_sentiment, left_index=True, right_index=True, how='left')
                    elif 'sentiment_score' in self.sentiment_data.columns: # Fallback
                         generic_sentiment = self.sentiment_data.loc[self.sentiment_data.index.intersection(lookback_timestamps), ['sentiment_score']]
                         hist_df = pd.merge(hist_df, generic_sentiment, left_index=True, right_index=True, how='left')

                if len(hist_df) >= lookback: # Ensure we have the full lookback period
                    historical_data[symbol] = hist_df
                else:
                    logger.debug(f"Insufficient historical rows for {symbol} for lookback {lookback}. Got {len(hist_df)}.")
                    # Decide whether to return partial data or None
                    # Returning None ensures strategies don't operate on incomplete data
                    return None # Indicate failure to get full history
            else:
                logger.warning(f"Symbol {symbol} not found in loaded OHLCV data for historical retrieval.")
                return None # Cannot fulfill request if a symbol is missing

        if not historical_data:
            return None

        return historical_data


    # --- Synthetic Data Generation Methods --- #

    def generate_synthetic_ohlcv(self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str = '1d', base_price: float = 100.0, volatility: float = 0.01, drift: float = 0.0001):
        """Generates synthetic OHLCV data and saves it to CSV.

        Args:
            symbols: List of stock symbols.
            start_date: Start date for the data.
            end_date: End date for the data.
            timeframe: Data frequency ('1d', '1h', etc.).
            base_price: Initial price for the simulation.
            volatility: Daily volatility.
            drift: Daily drift.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

        date_rng = pd.date_range(start=start_date, end=end_date, freq=timeframe)
        num_periods = len(date_rng)

        for symbol in symbols:
            file_path = os.path.join(self.data_dir, f"{symbol}_ohlcv_{timeframe}.csv")
            if os.path.exists(file_path):
                logger.info(f"OHLCV file already exists for {symbol}, skipping generation: {file_path}")
                continue

            # Simple geometric Brownian motion simulation for close prices
            returns = np.random.normal(loc=drift, scale=volatility, size=num_periods)
            prices = base_price * np.exp(np.cumsum(returns))

            # Synthesize OHL from close prices (very basic)
            data = pd.DataFrame(index=date_rng)
            data['close'] = prices
            data['open'] = data['close'].shift(1).fillna(base_price) # Simple open as previous close
            price_diff = data['close'] - data['open']
            data['high'] = data[['open', 'close']].max(axis=1) + np.abs(price_diff * np.random.uniform(0, 0.2))
            data['low'] = data[['open', 'close']].min(axis=1) - np.abs(price_diff * np.random.uniform(0, 0.2))
            # Ensure high >= max(open, close) and low <= min(open, close)
            data['high'] = data[['high', 'open', 'close']].max(axis=1)
            data['low'] = data[['low', 'open', 'close']].min(axis=1)

            data['volume'] = np.random.randint(100000, 5000000, size=num_periods)

            data.index.name = 'timestamp'
            data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].round(2)

            try:
                data.to_csv(file_path)
                logger.info(f"Generated synthetic OHLCV data for {symbol} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save synthetic OHLCV data for {symbol} to {file_path}: {e}", exc_info=True)

    def generate_synthetic_sentiment(self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str = '1d', mean: float = 0.0, std_dev: float = 0.2):
        """Generates synthetic sentiment data and saves it to a single CSV.

        Args:
            symbols: List of symbols (used for column naming).
            start_date: Start date.
            end_date: End date.
            timeframe: Data frequency (should match OHLCV).
            mean: Mean of the sentiment score distribution.
            std_dev: Standard deviation of the sentiment score distribution.
        """
        sentiment_file_path = os.path.join(self.data_dir, 'synthetic_sentiment.csv')
        if os.path.exists(sentiment_file_path):
            logger.info(f"Sentiment file already exists, skipping generation: {sentiment_file_path}")
            return

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

        date_rng = pd.date_range(start=start_date, end=end_date, freq=timeframe)
        sentiment_data = pd.DataFrame(index=date_rng)
        sentiment_data.index.name = 'timestamp'

        for symbol in symbols:
            # Generate random sentiment scores (e.g., normal distribution clipped between -1 and 1)
            scores = np.random.normal(loc=mean, scale=std_dev, size=len(date_rng))
            scores = np.clip(scores, -1.0, 1.0)
            sentiment_data[f'{symbol}_sentiment_score'] = scores

        try:
            sentiment_data.round(4).to_csv(sentiment_file_path)
            logger.info(f"Generated synthetic sentiment data to {sentiment_file_path}")
        except Exception as e:
            logger.error(f"Failed to save synthetic sentiment data to {sentiment_file_path}: {e}", exc_info=True)

class RealTimeDataManager(BaseDataManager):
    """Data manager implementation that processes real-time market data.
    
    Handles live data from exchanges for real-time trading. Maintains both
    a real-time data feed and a historical cache for lookbacks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the RealTimeDataManager.

        Args:
            config: Configuration dictionary containing keys like:
                - 'symbols': List of symbols to monitor
                - 'data_provider': Instance of data provider (e.g., CcxtProvider)
                - 'max_history_size': Maximum number of data points to keep in history
                - 'timeframe': Data timeframe (e.g., '1m', '5m', '1h')
                - 'paper_trading': Whether to use paper trading mode
        """
        super().__init__(config)
        self.symbols = config.get('symbols', [])
        if not self.symbols:
            raise ValueError("Symbols list cannot be empty.")
            
        self.data_provider = config.get('data_provider')
        if self.data_provider is None:
            raise ValueError("Data provider must be specified")
            
        self.max_history_size = config.get('max_history_size', 1000)
        self.timeframe = config.get('timeframe', '1m')
        self.paper_trading = config.get('paper_trading', True)
        
        # Initialize data structures
        self.history: Dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.current_timestamp = None
        self.latest_data: Dict[str, pd.Series] = {}
        
        logger.info(f"{self.__class__.__name__} initialized with {len(self.symbols)} symbols "
                   f"in {'paper trading' if self.paper_trading else 'live'} mode.")
    
    async def initialize(self):
        """Initialize the real-time data feed connection."""
        if not self.data_provider:
            raise ValueError("Cannot initialize without a data provider")
            
        try:
            # Connect to real-time data feed
            await self.data_provider.connect_realtime()
            # Subscribe to the symbols
            await self.data_provider.subscribe_to_symbols(self.symbols)
            logger.info(f"Connected to real-time data feed for {len(self.symbols)} symbols")
            
            # Pre-load some historical data if available
            start_date = pd.Timestamp.now() - pd.Timedelta(days=7)  # Get one week of history
            end_date = pd.Timestamp.now()
            
            try:
                historical_data = await self.data_provider.fetch_historical_data(
                    symbols=self.symbols,
                    timeframe=self.timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Store the historical data in our cache
                for symbol, df in historical_data.items():
                    if not df.empty:
                        # Ensure we don't exceed max history size
                        if len(df) > self.max_history_size:
                            df = df.iloc[-self.max_history_size:]
                        self.history[symbol] = df
                        logger.info(f"Loaded {len(df)} historical data points for {symbol}")
            except Exception as e:
                logger.warning(f"Could not load historical data: {e}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize real-time data feed: {e}")
            return False
    
    async def update(self):
        """Update data from real-time feed.
        
        Returns:
            bool: True if new data was received, False otherwise
        """
        if not self.data_provider:
            logger.error("No data provider available")
            return False
            
        try:
            # Get latest data from provider
            realtime_update = await self.data_provider.get_realtime_data()
            
            if not realtime_update:
                # No new data available
                return False
                
            # Process the real-time update
            symbol = realtime_update.get('symbol')
            timeframe = realtime_update.get('timeframe')
            data = realtime_update.get('data')
            
            if not symbol or symbol not in self.symbols or not data:
                logger.warning(f"Received invalid real-time data update: {realtime_update}")
                return False
                
            # Convert CCXT format to expected format
            if isinstance(data, list) and len(data) >= 6:  # CCXT OHLCV format
                timestamp = milliseconds_to_datetime(data[0])
                ohlcv = {
                    'timestamp': timestamp,
                    'open': data[1],
                    'high': data[2],
                    'low': data[3],
                    'close': data[4],
                    'volume': data[5]
                }
                
                # Create Series for the current data point
                data_series = pd.Series(
                    {k: v for k, v in ohlcv.items() if k != 'timestamp'},
                    name=timestamp
                )
                
                # Update latest data
                self.latest_data[symbol] = data_series
                self.current_timestamp = timestamp
                
                # Update history
                if symbol in self.history:
                    # Check if we already have this timestamp
                    if timestamp in self.history[symbol].index:
                        # Update existing entry
                        self.history[symbol].loc[timestamp] = data_series
                    else:
                        # Append new entry
                        new_row = pd.DataFrame([data_series], index=[timestamp])
                        self.history[symbol] = pd.concat([self.history[symbol], new_row])
                        
                        # Ensure we don't exceed max history size
                        if len(self.history[symbol]) > self.max_history_size:
                            self.history[symbol] = self.history[symbol].iloc[-self.max_history_size:]
                
                logger.debug(f"Updated real-time data for {symbol}: {timestamp}, close: {ohlcv['close']}")
                return True
            else:
                logger.warning(f"Unexpected data format for {symbol}: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            return False
    
    def get_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Returns the latest timestamp from real-time data.
        
        In live trading, this is the most recent candle timestamp.
        """
        return self.current_timestamp
    
    def get_current_data(self) -> Optional[Dict[str, pd.Series]]:
        """Returns the latest data for all subscribed symbols.
        
        Returns:
            Dictionary mapping symbols to their latest data point,
            or None if no data is available yet.
        """
        if not self.latest_data:
            return None
            
        return self.latest_data
    
    def get_historical_data(self, symbols: List[str], lookback: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Returns historical data for the specified symbols.
        
        Args:
            symbols: List of symbols to retrieve data for.
            lookback: Number of historical bars to retrieve.
            
        Returns:
            Dictionary mapping symbols to their historical data,
            or None if insufficient history is available.
        """
        result = {}
        
        for symbol in symbols:
            if symbol in self.history and not self.history[symbol].empty:
                df = self.history[symbol].copy()
                
                if len(df) >= lookback:
                    result[symbol] = df.iloc[-lookback:]
                else:
                    logger.warning(f"Insufficient history for {symbol}: have {len(df)}, need {lookback}")
                    return None  # Not enough history for this symbol
            else:
                logger.warning(f"No history available for {symbol}")
                return None  # No history for this symbol
        
        return result if result else None
    
    async def close(self):
        """Cleanup resources."""
        if self.data_provider:
            try:
                await self.data_provider.disconnect_realtime()
                logger.info("Disconnected from real-time data feed")
                
                # If the provider has a close method, call it
                if hasattr(self.data_provider, 'close'):
                    await self.data_provider.close()
                    logger.info("Closed data provider")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

# Import the milliseconds_to_datetime function from ccxt_provider.py to ensure it's available
from ..data_acquisition.ccxt_provider import milliseconds_to_datetime

class InMemoryDataManager(BaseDataManager):
    """Data manager implementation for in-memory data structures.

    This class is designed to work with data provided directly as dictionaries
    of pandas DataFrames, rather than loading from files.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the InMemoryDataManager.

        Args:
            config: Configuration dictionary containing:
                - price_data: Dictionary mapping symbols to DataFrames with OHLCV data
                - sentiment_data: Dictionary mapping symbols to DataFrames with sentiment data
                - symbols: List of symbols to track
        """
        super().__init__(config)
        
        self.price_data = config.get('price_data', {})
        self.sentiment_data = config.get('sentiment_data', {})
        self.symbols = config.get('symbols', list(self.price_data.keys()))
        
        if not self.symbols:
            raise ValueError("Symbols list cannot be empty")
            
        if not self.price_data:
            raise ValueError("Price data dictionary cannot be empty")
            
        # Validate that all symbols have price data
        for symbol in self.symbols:
            if symbol not in self.price_data:
                raise ValueError(f"No price data found for symbol: {symbol}")
                
        # Create a combined index from all data sources
        self._prepare_combined_index()
        self.current_index = 0
        
        logger.info(f"{self.__class__.__name__} initialized with {len(self.symbols)} symbols and {len(self.combined_index)} data points")
        
    def _prepare_combined_index(self):
        """Create a combined index from all data sources."""
        all_indices = set()
        
        # Add price data indices
        for symbol, df in self.price_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    self.price_data[symbol] = df  # Update with datetime index
                except Exception as e:
                    logger.error(f"Failed to convert index to datetime for {symbol}: {e}")
                    raise ValueError(f"Price data index for {symbol} must be convertible to datetime")
                    
            all_indices.update(df.index)
            
        # Add sentiment data indices
        for symbol, df in self.sentiment_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    self.sentiment_data[symbol] = df  # Update with datetime index
                except Exception as e:
                    logger.error(f"Failed to convert index to datetime for sentiment data {symbol}: {e}")
                    raise ValueError(f"Sentiment data index for {symbol} must be convertible to datetime")
                    
            all_indices.update(df.index)
            
        # Sort the combined indices
        self.combined_index = pd.DatetimeIndex(sorted(list(all_indices)))
        
        if not self.combined_index.empty:
            logger.info(f"Combined index created with {len(self.combined_index)} timestamps from {self.combined_index.min()} to {self.combined_index.max()}")
        else:
            logger.warning("Combined index is empty")
            
    def get_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Returns the next timestamp from the combined data index."""
        if self.combined_index is None or self.current_index >= len(self.combined_index):
            return None
        timestamp = self.combined_index[self.current_index]
        self.current_index += 1
        return timestamp
        
    def get_current_data(self) -> Optional[Dict[str, pd.Series]]:
        """Returns the data slice for the current timestamp."""
        if self.combined_index is None or self.current_index == 0 or self.current_index > len(self.combined_index):
            logger.warning(f"Cannot get current data. Index out of bounds or not initialized. Current index: {self.current_index}")
            return None
            
        # Get the timestamp corresponding to the *previous* call to get_next_timestamp
        current_timestamp = self.combined_index[self.current_index - 1]
        logger.debug(f"Getting data for timestamp: {current_timestamp}")
        
        current_data_slice = {}
        
        # Get OHLCV data for each symbol
        for symbol in self.symbols:
            if symbol in self.price_data:
                df = self.price_data[symbol]
                if current_timestamp in df.index:
                    current_data_slice[symbol] = df.loc[current_timestamp].copy()
                else:
                    # Create a NaN series for this symbol if data is missing
                    logger.debug(f"No price data for {symbol} at {current_timestamp}. Using NaNs.")
                    cols = ['open', 'high', 'low', 'close', 'volume']
                    nan_series = pd.Series(index=cols, data=np.nan, name=current_timestamp)
                    current_data_slice[symbol] = nan_series
                    
            # Add sentiment data if available
            if symbol in self.sentiment_data:
                df = self.sentiment_data[symbol]
                if current_timestamp in df.index:
                    sentiment_value = df.loc[current_timestamp, 'sentiment']
                    if symbol in current_data_slice:
                        current_data_slice[symbol]['sentiment_score'] = sentiment_value
                    else:
                        current_data_slice[symbol] = pd.Series({'sentiment_score': sentiment_value}, name=current_timestamp)
                else:
                    # Add NaN sentiment if missing
                    if symbol in current_data_slice:
                        current_data_slice[symbol]['sentiment_score'] = np.nan
        
        if not current_data_slice:
            logger.warning(f"No data could be retrieved for any symbol at timestamp {current_timestamp}")
            return None
            
        return current_data_slice
        
    def get_historical_data(self, symbols: List[str], lookback: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Returns historical data for the specified symbols up to the current timestamp."""
        if self.combined_index is None or self.current_index == 0 or self.current_index > len(self.combined_index):
            logger.warning("Cannot get historical data. Index out of bounds or not initialized.")
            return None
            
        # Current timestamp is the one most recently returned by get_next_timestamp
        current_internal_idx = self.current_index - 1
        if current_internal_idx < lookback - 1:
            # Not enough historical data available yet
            logger.debug(f"Insufficient historical data for lookback {lookback} at index {current_internal_idx}")
            return None
            
        # Determine the start index for the lookback period
        start_idx = max(0, current_internal_idx - lookback + 1)
        # Get the actual timestamps for the lookback period
        lookback_timestamps = self.combined_index[start_idx: current_internal_idx + 1]
        
        historical_data = {}
        for symbol in symbols:
            if symbol in self.price_data:
                # Select rows based on the lookback timestamps
                price_df = self.price_data[symbol]
                hist_df = price_df.loc[price_df.index.intersection(lookback_timestamps)]
                
                # Merge sentiment if available
                if symbol in self.sentiment_data:
                    sentiment_df = self.sentiment_data[symbol]
                    sentiment_slice = sentiment_df.loc[sentiment_df.index.intersection(lookback_timestamps)]
                    
                    # Rename column to sentiment_score for consistency
                    sentiment_slice = sentiment_slice.rename(columns={'sentiment': 'sentiment_score'})
                    
                    # Merge with price data
                    hist_df = pd.merge(hist_df, sentiment_slice, left_index=True, right_index=True, how='left')
                
                if len(hist_df) > 0:
                    historical_data[symbol] = hist_df
                else:
                    logger.debug(f"No historical data for {symbol} in the lookback period")
            else:
                logger.warning(f"Symbol {symbol} not found in price data")
                
        if not historical_data:
            return None
            
        return historical_data
