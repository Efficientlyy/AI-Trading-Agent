"""
Indicator Engine module for calculating technical indicators.
This module provides functionality for calculating various technical indicators
based on market data, with support for both Python and Rust implementations.
"""

import os
import logging
import traceback
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from functools import lru_cache

from ai_trading_agent.utils.logging import get_logger
# Inline implementation of CacheManager
class CacheManager:
    """
    Simple cache manager for indicator calculations.
    Provides a dictionary-like interface with size, memory, and TTL limits.
    """
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 50, ttl_seconds: int = 300):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        
    def __getitem__(self, key):
        return self._cache.get(key)
        
    def __setitem__(self, key, value):
        # Simple implementation without memory or TTL checks
        if len(self._cache) >= self.max_size:
            # Remove oldest item if we're at capacity
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._timestamps:
                del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = pd.Timestamp.now()
        
    def __contains__(self, key):
        return key in self._cache

# Import Rust accelerated functions if available
_RUST_AVAILABLE_FLAG = False
_REQUIRED_RUST_FUNCTIONS = [
    'create_lag_features_rs',
    'create_bollinger_bands_rs',
    'create_rsi_features_rs', 
    'create_ema_features_rs'
]

def _check_rust_functions(module):
    """Check if the required Rust functions are available in the module."""
    missing_functions = []
    for func_name in _REQUIRED_RUST_FUNCTIONS:
        if not hasattr(module, func_name):
            missing_functions.append(func_name)
    
    if missing_functions:
        logging.warning(f"Rust module found, but missing required functions: {', '.join(missing_functions)}")
        return False
    return True

try:
    # Try importing the new module name first
    import ai_trading_agent_rs as rs_features_imported
    
    # Verify all required functions are available
    if _check_rust_functions(rs_features_imported):
        _rs_features_module = rs_features_imported
        _RUST_AVAILABLE_FLAG = True
        logging.info(f"Successfully imported 'ai_trading_agent_rs' v{getattr(rs_features_imported, '__version__', 'unknown')} with all required functions.")
    else:
        _rs_features_module = rs_features_imported  # Still assign the module even if missing some functions
        _RUST_AVAILABLE_FLAG = False
        logging.warning("Some required functions missing from 'ai_trading_agent_rs'. Using Python fallbacks.")
        
except ImportError as e:
    try:
        # Fallback to older module name if it exists
        import rust_lag_features as rs_features_imported
        
        # Verify all required functions are available
        if _check_rust_functions(rs_features_imported):
            _rs_features_module = rs_features_imported
            _RUST_AVAILABLE_FLAG = True
            logging.info("Successfully imported 'rust_lag_features' for Rust acceleration.")
        else:
            _rs_features_module = rs_features_imported  # Still assign the module even if missing some functions
            _RUST_AVAILABLE_FLAG = False
            logging.warning("Some required functions missing from 'rust_lag_features'. Using Python fallbacks.")
            
    except ImportError:
        _rs_features_module = None  # Ensure rs_features is defined for type hinting or attribute access attempts
        _RUST_AVAILABLE_FLAG = False
        logging.warning(f"Neither 'ai_trading_agent_rs' nor 'rust_lag_features' available. Using Python fallbacks. Error: {e}")
        
# Log detailed information about the Rust module if available
if _RUST_AVAILABLE_FLAG and _rs_features_module is not None:
    try:
        available_functions = [name for name in dir(_rs_features_module) 
                              if not name.startswith('_') and callable(getattr(_rs_features_module, name))]
        logging.info(f"Available Rust functions: {', '.join(available_functions)}")
    except Exception as log_err:
        logging.warning(f"Could not enumerate Rust functions: {log_err}")

class IndicatorCategory(Enum):
    """Enum for categorizing indicators."""
    TREND = auto()
    MOMENTUM = auto()
    VOLATILITY = auto()
    VOLUME = auto()
    CUSTOM = auto()
    FEATURES = auto()

class IndicatorEngine:
    """
    Engine for calculating technical indicators based on market data.
    
    This class provides methods for calculating various technical indicators,
    including trend, momentum, volatility, and volume indicators, as well as
    custom indicators and feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any], 
                 enable_cache: bool = True, 
                 max_cache_size: int = 100, 
                 max_cache_memory_mb: int = 50,
                 cache_ttl_seconds: int = 300):
        self.config = config
        self.logger = get_logger("IndicatorEngine") # Get base logger
        self.logger.info("Initializing IndicatorEngine") 
        
        # Initialize the Rust module if available
        self.rs_features = _rs_features_module if _RUST_AVAILABLE_FLAG else None
        if self.rs_features is not None:
            self.logger.info(f"Rust acceleration enabled with module: {self.rs_features.__name__}")
            try:
                rust_version = getattr(self.rs_features, '__version__', 'unknown')
                self.logger.info(f"Rust module version: {rust_version}")
            except Exception as ver_err:
                self.logger.warning(f"Could not determine Rust module version: {ver_err}")
        else:
            self.logger.warning("Rust acceleration not available. Using Python fallbacks for all calculations.")

        # Setup file logging if configured
        logging_config = self.config.get("logging", {})
        log_file = logging_config.get("log_file")
        log_level_str = logging_config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        if log_file:
            try:
                # Ensure the directory for the log file exists
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # Add file handler
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"Added file logging to {log_file} at level {log_level_str}")
            except Exception as e:
                self.logger.error(f"Failed to setup file logging: {e}")
        
        # Store Rust module if available
        self.rs_features = _rs_features_module
        self.rust_available = _RUST_AVAILABLE_FLAG
        if not self.rust_available:
            self.logger.warning("Rust acceleration not available - falling back to Python implementations.")
        
        # Set up caching
        self.cache_enabled = enable_cache
        if self.cache_enabled:
            self.cache_manager = CacheManager(
                max_size=max_cache_size, 
                max_memory_mb=max_cache_memory_mb,
                ttl_seconds=cache_ttl_seconds
            )
            self.logger.info(f"Caching enabled with {self.cache_manager.max_size} max items, {self.cache_manager.max_memory_mb}MB max memory, {self.cache_manager.ttl_seconds}s TTL")
        
        # Performance metrics
        self.metrics = {
            "calculations_performed": 0,
            "calculation_errors": 0,
            "indicator_errors": 0,
            "indicators_calculated": 0,
            "indicator_calculation_time_ms": {}
        }
        
        # Initialize indicators
        self._init_indicators()
    
    def _init_indicators(self) -> None:
        """Initialize indicator factories based on configuration."""
        # VERY EARLY LOGGING TEST
        try:
            self.logger.critical("ENTERING _init_indicators - VERY EARLY TEST")
        except Exception as e:
            # Fallback print if logger itself is problematic here
            print(f"CRITICAL: FAILED TO LOG ENTRY TO _init_indicators: {e}")
        
        self.logger.debug(f"IndicatorEngine._init_indicators: Logger ID is {id(self.logger)}")
        self.logger.debug(f"_init_indicators called. Full config received: {self.config}")
        self.indicators: Dict[str, Dict[str, Any]] = {}
        
        # Trend Indicators
        trend_config = self.config.get("trend", {})
        sma_settings = trend_config.get("sma", {})
        if sma_settings.get("enabled", False):
            periods = sma_settings.get("periods", [10, 20])
            self.indicators["sma"] = {
                "calculator": self._calculate_sma,
                "periods": periods,
                "enabled": True,
                "category": IndicatorCategory.TREND,
                "config": sma_settings
            }
            self.logger.debug(f"Registered SMA indicator with periods: {periods}")
        
        ema_settings = trend_config.get("ema", {})
        if ema_settings.get("enabled", False):
            periods = ema_settings.get("periods", [10, 20])
            self.indicators["ema"] = {
                "calculator": self._calculate_ema,
                "periods": periods,
                "enabled": True,
                "category": IndicatorCategory.TREND,
                "config": ema_settings
            }
            self.logger.debug(f"Registered EMA indicator with periods: {periods}")
        
        # Momentum Indicators
        momentum_config = self.config.get("momentum", {})
        rsi_settings = momentum_config.get("rsi", {})
        if rsi_settings.get("enabled", False):
            period = rsi_settings.get("period", 14)
            self.indicators["rsi"] = {
                "calculator": self._calculate_rsi,
                "period": period,
                "enabled": True,
                "category": IndicatorCategory.MOMENTUM,
                "config": rsi_settings
            }
            self.logger.debug(f"Registered RSI indicator with period: {period}")
        
        # Volatility Indicators
        volatility_config = self.config.get("volatility", {})
        bb_settings = volatility_config.get("bollinger_bands", {})
        if bb_settings.get("enabled", False):
            periods = bb_settings.get("periods", [20])
            deviations = bb_settings.get("deviations", 2)
            self.indicators["bollinger_bands"] = {
                "calculator": self._calculate_bollinger_bands,
                "periods": periods,
                "deviations": deviations,
                "enabled": True,
                "category": IndicatorCategory.VOLATILITY,
                "config": bb_settings
            }
            self.logger.debug(f"Registered Bollinger Bands indicator with periods: {periods}, deviations: {deviations}")
        
        # Feature Engineering
        features_config = self.config.get("features", {})
        
        # Standard lag features
        lag_features_settings = features_config.get("lag_features", {})
        if lag_features_settings.get("enabled", False):
            lags = lag_features_settings.get("lags", [1, 2, 3, 5, 10])
            source_column = lag_features_settings.get("source_column", "close")
            self.indicators["lag_features"] = {
                "calculator": self._calculate_lag_features,
                "lags_to_calculate": lags,
                "source_column": source_column,
                "enabled": True,
                "category": IndicatorCategory.FEATURES,
                "config": lag_features_settings
            }
            self.logger.debug(f"Registered lag features with lags: {lags}, source: {source_column}")
        
        # Process additional features that might have specialized calculators
        for feature_name, feature_settings in features_config.items():
            # Skip lag_features as it's handled above
            if feature_name == "lag_features":
                continue
            # Handle Rust lag features
            elif feature_name.startswith("LAG_FEATURES_RS_") and feature_settings.get("enabled", True):
                lags = feature_settings.get("lags_to_calculate", [1, 2, 3, 5, 10])
                source_column = feature_settings.get("source_column", "close")
                self.indicators[feature_name] = {
                    "calculator": self._calculate_lag_features_rs,
                    "lags_to_calculate": lags,
                    "source_column": source_column,
                    "enabled": True,
                    "category": IndicatorCategory.FEATURES,
                    "config": feature_settings
                }
                self.logger.debug(f"Registered Rust lag features {feature_name} with lags: {lags}, source: {source_column}")
        
        self.logger.debug(f"_init_indicators finished. Registered indicators: {list(self.indicators.keys())}")
    
    def calculate_all_indicators(self, market_data: Dict[str, pd.DataFrame], symbols: List[str]) -> Dict[str, Dict]:
        """
        Calculate all enabled indicators for the provided market data.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            symbols: List of symbols to calculate indicators for
            
        Returns:
            Dictionary of indicator values organized by symbol and indicator type
        """
        self.logger.debug(f"IndicatorEngine.calculate_all_indicators: Logger ID is {id(self.logger)}")
        overall_start_time = datetime.now()
        all_results: Dict[str, Dict[str, Any]] = {symbol: {} for symbol in symbols} # Initialize results for all symbols
        self.metrics["calculations_performed"] = 0 # Reset for this call
        self.metrics["calculation_errors"] = 0 # Reset for this call
        self.metrics["indicator_errors"] = 0 # Reset for this call
        self.metrics["indicators_calculated"] = 0 # Reset for this call
        self.metrics["indicator_calculation_time_ms"] = {} # Reset for this call
        
        for symbol in symbols:
            if symbol not in market_data:
                self.logger.warning(f"Symbol {symbol} not found in market data, skipping")
                continue
            
            df = market_data[symbol]
            if df.empty:
                self.logger.warning(f"Empty DataFrame for symbol {symbol}, skipping")
                continue
            
            for indicator_name, indicator_settings in self.indicators.items():
                if not indicator_settings.get("enabled", True):
                    self.logger.debug(f"Indicator {indicator_name} is disabled, skipping")
                    continue
                
                try:
                    # Check cache if enabled
                    cache_key = None
                    cache_hit = False
                    
                    if self.cache_enabled:
                        # Create a cache key based on symbol, indicator, and DataFrame hash
                        df_hash = hash(tuple(map(tuple, df.values.tolist())))
                        cache_key = f"{symbol}_{indicator_name}_{df_hash}"
                        
                        if cache_key in self.cache_manager:
                            cache_result = self.cache_manager[cache_key]
                            all_results[symbol][indicator_name] = cache_result
                            cache_hit = True
                            self.logger.debug(f"Cache hit for {symbol} {indicator_name}")
                            continue
                    
                    # Not in cache, calculate the indicator
                    start_time = datetime.now()
                    calculator = indicator_settings["calculator"]
                    
                    # Call the calculator with the right parameters based on indicator type
                    indicator_result = self._calculate_indicator(calculator, df, indicator_settings)
                    
                    # Measure calculation time
                    end_time = datetime.now()
                    calculation_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # Store the result
                    all_results[symbol][indicator_name] = indicator_result
                    
                    # Update metrics
                    self.metrics["calculations_performed"] += 1
                    self.metrics["indicators_calculated"] += 1
                    if indicator_name not in self.metrics["indicator_calculation_time_ms"]:
                        self.metrics["indicator_calculation_time_ms"][indicator_name] = []
                    self.metrics["indicator_calculation_time_ms"][indicator_name].append(calculation_time_ms)
                    
                    # Add to cache if enabled
                    if self.cache_enabled and cache_key is not None and not cache_hit:
                        self.cache_manager[cache_key] = indicator_result
                    
                    self.logger.debug(f"Calculated {indicator_name} for {symbol} in {calculation_time_ms:.2f}ms")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    self.metrics["indicator_errors"] += 1
                    self.metrics["calculation_errors"] += 1
        
        # Calculate overall time
        overall_time_ms = (datetime.now() - overall_start_time).total_seconds() * 1000
        self.logger.info(f"Calculated {self.metrics['indicators_calculated']} indicators for {len(symbols)} symbols in {overall_time_ms:.2f}ms")
        
        return all_results
    
    def _calculate_indicator(self, calculator: Callable, df: pd.DataFrame, settings: Dict[str, Any]) -> Any:
        """
        Call the appropriate calculator function with the right parameters based on indicator type.
        
        Args:
            calculator: The calculator function to call
            df: DataFrame with market data
            settings: Indicator settings dictionary
            
        Returns:
            The calculated indicator values
        """
        # Dispatch to the right calculator based on indicator type
        if calculator == self._calculate_sma:
            return calculator(df, settings["periods"])
        elif calculator == self._calculate_ema:
            return calculator(df, settings["periods"])
        elif calculator == self._calculate_rsi:
            return calculator(df, settings["period"])
        elif calculator == self._calculate_bollinger_bands:
            return calculator(df, settings["periods"], settings["deviations"])
        elif calculator == self._calculate_lag_features:
            return calculator(df, settings["lags_to_calculate"], settings["source_column"])
        elif calculator == self._calculate_lag_features_rs:
            return calculator(df, settings["lags_to_calculate"], settings["source_column"])
        else:
            raise ValueError(f"Unknown calculator function: {calculator.__name__}")
    
    def _calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Average (SMA) for the given periods.
        
        Args:
            df: DataFrame with market data
            periods: List of periods to calculate SMA for
            
        Returns:
            Dictionary mapping period strings to SMA Series
        """
        results = {}
        for period in periods:
            results[str(period)] = df['close'].rolling(window=period).mean()
        return results
    
    def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Average (EMA) for the given periods.
        
        Args:
            df: DataFrame with market data
            periods: List of periods to calculate EMA for
            
        Returns:
            Dictionary mapping period strings to EMA Series
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_ema_features_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for EMA with periods: {periods}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Convert to list of integers for Rust
                periods_int = [int(p) for p in periods]
                
                # Call Rust function
                ema_results = self.rs_features.create_ema_features_rs(close_values.tolist(), periods_int)
                
                # Process results
                results = {}
                for i, period in enumerate(periods):
                    # Convert None values to NaN
                    ema_values = [float('nan') if val is None else val for val in ema_results[i]]
                    results[str(period)] = pd.Series(ema_values, index=df.index)
                
                return results
            
            except Exception as e:
                self.logger.error(f"Error in Rust EMA calculation: {e}")
                self.logger.info("Falling back to Python implementation for EMA")
        
        # Python implementation (fallback)
        self.logger.debug(f"Using Python implementation for EMA with periods: {periods}")
        results = {}
        for period in periods:
            results[str(period)] = df['close'].ewm(span=period, adjust=False).mean()
        return results
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """
        Calculate Relative Strength Index (RSI) for the given period.
        
        Args:
            df: DataFrame with market data
            period: Period to calculate RSI for
            
        Returns:
            Dictionary mapping period string to RSI Series
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_rsi_features_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for RSI with period: {period}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Call Rust function with single period as a list
                rsi_result = self.rs_features.create_rsi_features_rs(close_values.tolist(), [period])
                
                # Convert None values to NaN
                rsi_values = [float('nan') if val is None else val for val in rsi_result[0]]
                
                return {str(period): pd.Series(rsi_values, index=df.index)}
            
            except Exception as e:
                self.logger.error(f"Error in Rust RSI calculation: {e}")
                self.logger.info("Falling back to Python implementation for RSI")
        
        # Python implementation (fallback)
        self.logger.debug(f"Using Python implementation for RSI with period: {period}")
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Make losses positive
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {str(period): rsi}
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, periods: List[int], deviations: float) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate Bollinger Bands for the given periods and deviations.
        
        Args:
            df: DataFrame with market data
            periods: List of periods to calculate Bollinger Bands for
            deviations: Number of standard deviations for the bands
            
        Returns:
            Dictionary mapping period strings to dictionaries of upper, middle, and lower bands
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_bollinger_bands_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for Bollinger Bands with periods: {periods}, deviations: {deviations}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Convert to list of integers for Rust
                periods_int = [int(p) for p in periods]
                
                # Call Rust function
                bb_results = self.rs_features.create_bollinger_bands_rs(close_values.tolist(), periods_int, float(deviations))
                
                # Process results
                results = {}
                for i, period in enumerate(periods):
                    period_key = str(period)
                    results[period_key] = {
                        'upper': pd.Series([float('nan') if val is None else val for val in bb_results[i][0]], index=df.index),
                        'middle': pd.Series([float('nan') if val is None else val for val in bb_results[i][1]], index=df.index),
                        'lower': pd.Series([float('nan') if val is None else val for val in bb_results[i][2]], index=df.index)
                    }
                
                return results
            
            except Exception as e:
                self.logger.error(f"Error in Rust Bollinger Bands calculation: {e}")
                self.logger.info("Falling back to Python implementation for Bollinger Bands")
        
        # Python implementation (fallback)
        self.logger.debug(f"Using Python implementation for Bollinger Bands with periods: {periods}, deviations: {deviations}")
        results = {}
        for period in periods:
            # Calculate middle band (SMA)
            middle = df['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = df['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper = middle + (std * deviations)
            lower = middle - (std * deviations)
            
            results[str(period)] = {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        
        return results
    
    def _calculate_lag_features(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate lag features using Python implementation.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        self.logger.debug(f"Calculating lag features using Python implementation for lags: {lags}")
        result = {}
        for lag in lags:
            key = f'lag_{lag}'
            result[key] = df[source_column].shift(lag)
        return result
    
    def _calculate_lag_features_rs(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate lag features using Rust implementation with fallback to Python.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        # Add debug logging
        self.logger.debug(f"_calculate_lag_features_rs called with lags: {lags}, rust_available: {rust_available}")
        
        if rust_available:
            try:
                # Check if create_lag_features_rs exists in the module
                if not hasattr(self.rs_features, 'create_lag_features_rs'):
                    self.logger.warning("create_lag_features_rs function not found in Rust module")
                    return self._calculate_lag_features(df, lags, source_column)
                
                # Get the values from the source column - this matches the Rust function signature
                # The Rust function only takes the series values, not timestamps
                values = df[source_column].to_numpy()
                
                # Convert lag values to integers
                lags_int = [int(lag) for lag in lags]
                
                self.logger.debug(f"Calling Rust function with values shape: {values.shape}, lags: {lags_int}")
                
                # Call the Rust function with the correct signature
                # For debugging
                rust_func = getattr(self.rs_features, 'create_lag_features_rs')
                self.logger.debug(f"Rust function: {rust_func}, callable: {callable(rust_func)}")
                
                # Convert numpy array to list to ensure compatibility
                values_list = values.tolist()
                self.logger.debug(f"Input data type: {type(values_list)}, length: {len(values_list)}")
                
                # Actual call to Rust function
                lag_results = self.rs_features.create_lag_features_rs(values_list, lags_int)
                
                # Log the result structure
                self.logger.debug(f"Rust result type: {type(lag_results)}, length: {len(lag_results) if lag_results else 'None'}")
                
                # Convert results to dictionary of Series
                result = {}
                if lag_results and len(lag_results) == len(lags):
                    for i, lag in enumerate(lags):
                        key = f'lag_{lag}'
                        # Convert None values to NaN
                        lag_values = [float('nan') if val is None else val for val in lag_results[i]]
                        result[key] = pd.Series(lag_values, index=df.index)
                    
                    self.logger.debug(f"Successfully calculated lag features using Rust for lags: {lags}")
                    return result
                else:
                    self.logger.error(f"Unexpected result structure from Rust function: {lag_results}")
                    return self._calculate_lag_features(df, lags, source_column)
            except Exception as e:
                self.logger.error(f"Error in Rust lag features calculation: {e}\n{traceback.format_exc()}")
                self.logger.info("Falling back to Python implementation after Rust error.")
                return self._calculate_lag_features(df, lags, source_column)
        else:
            self.logger.info(f"Rust lag features not available. Falling back to Python implementation.")
            return self._calculate_lag_features(df, lags, source_column)
