"""
Technical Analysis Agent - Unified agent for all technical analysis operations.

This agent consolidates chart analysis, technical indicators, and trading strategies
into a single coherent component that leverages both Python and Rust implementations
for optimal performance.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from ai_trading_agent.common.utils import get_logger
from ai_trading_agent.data_acquisition.market_data_provider import MarketDataProvider, get_market_data_provider
from ai_trading_agent.data_acquisition.mexc_spot_v3_client import MexcSpotV3Client
from ai_trading_agent.config.mexc_config import MEXC_CONFIG, TRADING_PAIRS, SUPPORTED_TIMEFRAMES
from .indicator_engine import IndicatorEngine
from .strategy_manager import StrategyManager
from .agent_definitions import AgentStatus, AgentRole, BaseAgent
from .mock_data_generator import MockDataGenerator, MarketPattern
from .pattern_types import PatternType
from .pattern_detector import PatternDetector

class DataSource(Enum):
    MEXC = "mexc"
    MOCK = "mock"

class DataMode(Enum):
    MOCK = "mock"
    REAL = "real"

class TrendType(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"

class TechnicalAnalysisAgent(BaseAgent):
    """
    Comprehensive technical analysis agent that integrates all technical indicators, 
    pattern detection, and strategy signal generation capabilities.
    
    This agent uses a hybrid Python-Rust architecture for optimal performance,
    with compute-intensive operations accelerated in Rust.
    """
    
    AGENT_ID_PREFIX = "tech_analysis_"
    
    def __init__(self, agent_id_suffix: str, name: str, symbols: List[str], 
                 config_details: Optional[Dict] = None, 
                 data_mode: DataMode = DataMode.REAL,
                 data_source: DataSource = DataSource.MEXC):
        
        agent_id = f"{name.replace(' ', '_')}_{agent_id_suffix}"

        # Call BaseAgent's __init__ first.
        # BaseAgent expects agent_id, name, agent_role, agent_type, etc.
        # We need to define agent_role and agent_type for TechnicalAnalysisAgent.
        super().__init__(
            agent_id=agent_id,
            name=name,
            agent_role=AgentRole.SPECIALIZED_TECHNICAL,  # Corrected role
            agent_type="TechnicalAnalysis", # Example, can be more specific
            symbols=symbols,
            config_details=config_details # Pass along config_details
        )

        # Now load the configuration for TechnicalAnalysisAgent
        # self.config_details is available from BaseAgent
        self.config = self._load_config(self.config_details) 

        # Initialize logger after config is loaded, so log level can be set from config
        log_level = self.config.get('logging', {}).get('level', 'INFO').upper()
        self.logger = get_logger(self.agent_id, level=log_level)

        # Now it's safe to use the logger and self.config
        self.logger.info(f"TechnicalAnalysisAgent '{self.name}' initialized. Log level: {log_level}")
        self.logger.debug(f"Agent configuration: {self.config}")

        # Configure data_mode from the final self.config
        self._configure_data_mode_from_config(data_mode) # Pass initial data_mode for potential override by config
        
        # Configure data source
        self._configure_data_source(data_source)

        # Initialize the component engines using self.config
        indicator_engine_cfg = self.config.get("indicator_engine", self._get_default_indicator_config())
        strategy_manager_cfg = self.config.get("strategy_manager", self._get_default_strategy_config())

        self.indicator_engine = IndicatorEngine(indicator_engine_cfg)
        self.strategy_manager = StrategyManager(strategy_manager_cfg)

        self.status = AgentStatus.IDLE # Set initial status after full initialization
        self.logger.info(f"TechnicalAnalysisAgent '{self.name}' setup complete.")

        # Initialize the data providers
        self.mexc_client = None
        self.market_data_provider = None
        self._initialize_data_providers()
        
        # Initialize the mock data generator
        try:
            self.mock_data_generator = MockDataGenerator(seed=42)  # Use seed for reproducibility
        except Exception as e:
            self.logger.warning(f"Failed to initialize MockDataGenerator: {e}. Mock data will not be available.")
            self.mock_data_generator = None

        # Initialize the pattern detector
        self.pattern_detector = PatternDetector({
            "peak_prominence": 0.005,  # 0.5% of price level
            "peak_distance": 5,        # Minimum 5 bars between peaks
            "shoulder_height_diff_pct": 0.10,  # Maximum 10% difference between shoulders
            "neckline_slope_threshold": 0.05,  # Maximum 5% slope for neckline
            "trendline_min_points": 3,  # Minimum points for trendline construction
            "trendline_r_squared_threshold": 0.7,  # Minimum R² for valid trendline
        })

        self.metrics = {
            "processing_errors": 0,
            "signals_generated": 0,
            "indicator_calculations": 0,
            "avg_processing_time_ms": 0.0,
            "last_processing_time_ms": 0.0,
            "patterns_detected": 0
        }

        self.market_data_cache = {}
        self.technical_state = {}
        self.current_signals = {}
        
        # Last data fetch time to avoid redundant API calls
        self.last_data_fetch = {}
        # Data fetch cooldown period (in seconds)
        self.data_fetch_cooldown = self.config.get('data_fetch_cooldown', 60)

    def update_status(self, new_status: AgentStatus):
        if self.status != new_status:
            self.logger.info(f"Agent status changed from {self.status.value} to {new_status.value}")
            self.status = new_status

    def update_metrics(self, new_metrics: Dict[str, Any]):
        for key, value in new_metrics.items():
            if key in self.metrics:
                if isinstance(value, (int, float)) and isinstance(self.metrics[key], (int, float)):
                    if key == "avg_processing_time_ms": # This might need a more sophisticated averaging
                        # For now, let's assume it's the latest processing time and we store it
                        # A true average would require total_time and count
                        self.metrics["last_processing_time_ms"] = value 
                        # A simple moving average or cumulative average could be implemented here
                        # For now, just updating if it's provided, or could be set to last_processing_time_ms
                        self.metrics[key] = value 
                    elif key in ["processing_errors", "signals_generated", "indicator_calculations", "patterns_detected"]:
                        self.metrics[key] += value # Accumulate counts
                    else:
                        self.metrics[key] = value # Default update for other numeric types
                else:
                    self.metrics[key] = value # Update for non-numeric or type mismatch
            else:
                self.metrics[key] = value # Add new metric
        self.logger.debug(f"Metrics updated: {self.metrics}")

    def _configure_data_mode_from_config(self, initial_data_mode: DataMode):
        # Prioritize config file's data_mode, then constructor arg, then default to REAL
        config_data_mode_str = self.config.get('data_mode')
        if config_data_mode_str:
            try:
                self.data_mode = DataMode[config_data_mode_str.upper()]
                self.logger.info(f"DataMode set to '{self.data_mode.name}' from configuration file.")
            except KeyError:
                self.logger.warning(f"Invalid DataMode '{config_data_mode_str}' in config. Using initial: {initial_data_mode.name}.")
                self.data_mode = initial_data_mode
        else:
            self.data_mode = initial_data_mode
            self.logger.info(f"DataMode set to '{self.data_mode.name}' from constructor argument.")
            
    def _configure_data_source(self, initial_data_source: DataSource):
        # Prioritize config file's data_source, then constructor arg, then default to MEXC
        config_data_source_str = self.config.get('data_source')
        if config_data_source_str:
            try:
                self.data_source = DataSource[config_data_source_str.upper()]
                self.logger.info(f"DataSource set to '{self.data_source.name}' from configuration file.")
            except KeyError:
                self.logger.warning(f"Invalid DataSource '{config_data_source_str}' in config. Using initial: {initial_data_source.name}.")
                self.data_source = initial_data_source
        else:
            self.data_source = initial_data_source
            self.logger.info(f"DataSource set to '{self.data_source.name}' from constructor argument.")

    def _load_config(self, config_details: Optional[Dict]) -> Dict:
        """Load technical analysis agent configuration, using defaults if not provided."""
        if config_details:
            self.logger.debug("Using provided config details")
            return config_details
        
        # Use default config
        self.logger.debug("No config details provided, using defaults")
        return self._get_default_config()
        
    def _initialize_data_providers(self):
        """Initialize the appropriate data providers based on configuration."""
        # Initialize MEXC client if using real data
        if self.data_mode == DataMode.REAL and self.data_source == DataSource.MEXC:
            try:
                self.mexc_client = MexcSpotV3Client()
                self.logger.info("MEXC client initialized successfully.")
            except Exception as e:
                self.logger.error(f"Error initializing MEXC client: {e}")
                self.logger.warning("Falling back to mock data due to MEXC client initialization failure.")
                self.data_mode = DataMode.MOCK
        
        # Get the appropriate market data provider
        try:
            self.market_data_provider = get_market_data_provider()
            self.logger.info("Market data provider initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing market data provider: {e}")
            self.logger.warning("Using basic implementation of market data provider.")
            self.market_data_provider = MarketDataProvider()
            
    async def get_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data based on current data mode and source configuration.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDC')
            interval: Timeframe interval (e.g., '1h', '4h', '1d')
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data or None if data retrieval fails
        """
        # Skip if we've fetched this data recently (unless in mock mode)
        if self.data_mode == DataMode.REAL:
            cache_key = f"{symbol}_{interval}"
            current_time = datetime.now()
            last_fetch = self.last_data_fetch.get(cache_key)
            
            if last_fetch and (current_time - last_fetch).seconds < self.data_fetch_cooldown:
                self.logger.debug(f"Using cached data for {symbol} ({interval}) - fetched {(current_time - last_fetch).seconds}s ago")
                return self.market_data_cache.get(cache_key)
        
        try:
            if self.data_mode == DataMode.MOCK:
                # Generate mock data if MockDataGenerator is available
                if self.mock_data_generator:
                    self.logger.info(f"Generating mock data for {symbol} ({interval})")
                    
                    # Convert interval to appropriate time delta
                    interval_map = {
                        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
                    }
                    minutes = interval_map.get(interval, 60)  # Default to 1h
                    
                    # Generate mock data with a realistic pattern
                    pattern = MarketPattern.UPTREND_WITH_PULLBACKS  # Can be made configurable
                    mock_data = self.mock_data_generator.generate_ohlcv(
                        start_price=10000.0 if 'BTC' in symbol else 1000.0,
                        periods=limit,
                        interval_minutes=minutes,
                        pattern=pattern,
                        volatility=0.015 if 'BTC' in symbol else 0.025  # Higher volatility for altcoins
                    )
                    
                    # Cache the mock data
                    cache_key = f"{symbol}_{interval}"
                    self.market_data_cache[cache_key] = mock_data
                    self.last_data_fetch[cache_key] = datetime.now()
                    
                    return mock_data
                else:
                    self.logger.warning("MockDataGenerator not available. Cannot generate mock data.")
                    return None
            
            elif self.data_mode == DataMode.REAL:
                if self.data_source == DataSource.MEXC:
                    if not self.mexc_client:
                        self.logger.error("MEXC client not initialized. Cannot fetch real data.")
                        return None
                    
                    self.logger.info(f"Fetching real MEXC data for {symbol} ({interval})")
                    
                    # Ensure the symbol format is correct for MEXC
                    formatted_symbol = symbol
                    if '/' not in symbol:
                        formatted_symbol = f"{symbol[:3]}/{symbol[3:]}"
                    
                    # Check if interval is supported
                    if interval not in SUPPORTED_TIMEFRAMES:
                        self.logger.warning(f"Interval {interval} not supported by MEXC. Using 1h instead.")
                        interval = '1h'
                    
                    try:
                        # Fetch klines from MEXC
                        klines = await self.mexc_client.get_klines(formatted_symbol, interval, limit=limit)
                        
                        if not klines:
                            self.logger.warning(f"No data returned from MEXC for {symbol} ({interval})")
                            return None
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # Convert types
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Set timestamp as index
                        df.set_index('timestamp', inplace=True)
                        
                        # Cache the data
                        cache_key = f"{symbol}_{interval}"
                        self.market_data_cache[cache_key] = df
                        self.last_data_fetch[cache_key] = datetime.now()
                        
                        self.logger.debug(f"Successfully fetched and processed {len(df)} candles for {symbol} ({interval})")
                        return df
                    
                    except Exception as e:
                        self.logger.error(f"Error fetching data from MEXC: {e}")
                        self.update_metrics({"processing_errors": 1})
                        
                        # Try to use cached data if available
                        cache_key = f"{symbol}_{interval}"
                        if cache_key in self.market_data_cache:
                            self.logger.warning(f"Using cached data for {symbol} due to MEXC fetch error")
                            return self.market_data_cache[cache_key]
                        
                        # Fall back to mock data if no cache available
                        if self.mock_data_generator:
                            self.logger.warning("Falling back to mock data due to MEXC fetch error")
                            return await self.get_market_data(symbol, interval, limit)  # Will use mock mode now
                        
                        return None
                else:
                    self.logger.error(f"Unsupported data source: {self.data_source}")
                    return None
            else:
                self.logger.error(f"Unsupported data mode: {self.data_mode}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in get_market_data: {e}")
            self.update_metrics({"processing_errors": 1})
            return None
            
    async def toggle_data_mode(self):
        """Toggle between real and mock data modes."""
        if self.data_mode == DataMode.REAL:
            self.data_mode = DataMode.MOCK
            self.logger.info("Switched to MOCK data mode")
        else:
            self.data_mode = DataMode.REAL
            self.logger.info("Switched to REAL data mode")
            
        # Clear cache on mode switch
        self.market_data_cache = {}
        self.last_data_fetch = {}
        
        return {"success": True, "new_mode": self.data_mode.value}
    
    async def set_data_source(self, source: str) -> Dict[str, Any]:
        """Set the data source to use for real data.
        
        Args:
            source: The data source to use (e.g., 'mexc', 'mock')
            
        Returns:
            Dict with success status and new source
        """
        try:
            new_source = DataSource[source.upper()]
            self.data_source = new_source
            self.logger.info(f"Data source set to {new_source.name}")
            
            # Reinitialize data providers
            self._initialize_data_providers()
            
            # Clear cache on source switch
            self.market_data_cache = {}
            self.last_data_fetch = {}
            
            return {"success": True, "new_source": self.data_source.value}
        except KeyError:
            self.logger.error(f"Invalid data source: {source}")
            return {"success": False, "error": f"Invalid data source: {source}"}

    def _get_default_config(self) -> Dict:
        """Default configuration for TechnicalAnalysisAgent."""
        # Override this in subclasses if needed
        return {
            "logging": {
                "level": "INFO",
                "log_to_file": False,
                "log_dir": str(Path.home() / ".ai_trading_agent" / "logs")
            },
            "data_mode": "REAL",  # Use real data by default
            "data_source": "MEXC",  # Use MEXC by default
            "data_fetch_cooldown": 60,  # Seconds between API calls for the same data
            "historical_periods": 200,  # Number of historical periods to analyze
            "update_interval_seconds": 300,  # 5 minutes between updates by default
            "indicator_engine": self._get_default_indicator_config(),
            "strategy_manager": self._get_default_strategy_config(),
            "pattern_detection": {
                "enabled": True,
                "min_pattern_bars": 10,
                "confirmations_required": 2
            }
        }
    
    def _get_default_indicator_config(self) -> Dict[str, Any]:
        """Get default configuration for indicators when none is provided."""
        return {
            "trend": {
                "sma": {"enabled": True, "periods": [20, 50, 200]},
                "ema": {"enabled": True, "periods": [9, 21]},
                "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9}
            },
            "momentum": {
                "rsi": {"enabled": True, "period": 14}
            },
            "volatility": {
                "bollinger_bands": {"enabled": True, "period": 20, "deviations": 2},
                "atr": {"enabled": True, "period": 14}
            }
        }
    
    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Get default configuration for strategies when none is provided."""
        return {
            "strategies": {
                "ma_cross": {
                    "enabled": True,
                    "fast_ma": {"type": "ema", "period": 9},
                    "slow_ma": {"type": "ema", "period": 21},
                    "signal_threshold": 0.001
                },
                "rsi_ob_os": {
                    "enabled": True,
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            }
        }
    
    def _extract_market_data(self, data: Optional[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Extract market data from input or fetch appropriate data based on the current data mode.
        
        Args:
            data: Optional input data dictionary
            
        Returns:
            Dictionary mapping symbols to DataFrames of market data
        """
        # If data is provided directly, use it
        if data and isinstance(data, dict) and any(s in data for s in self.symbols):
            self.logger.debug("Using provided market data")
            return data
        
        # Otherwise, use the appropriate data source based on the data mode
        if self.data_mode == DataMode.MOCK:
            self.logger.info("Generating mock market data")
            return {symbol: self._generate_mock_data(symbol) for symbol in self.symbols}
        else:
            # For real data mode, use the MEXC market data provider
            self.logger.info("Fetching real market data from MEXC")
            try:
                # Initialize market data provider if needed
                market_data_provider = get_market_data_provider(self.symbols)
                
                # Get the data using asyncio
                loop = asyncio.get_event_loop()
                interval = self.config.get("data", {}).get("default_interval", "1h")
                limit = self.config.get("data", {}).get("default_candles", 100)
                
                # Fetch the data
                market_data = loop.run_until_complete(
                    market_data_provider.get_market_data(self.symbols, interval, limit)
                )
                
                # Check if we got valid data
                if market_data and any(not df.empty for df in market_data.values()):
                    self.logger.info(f"Successfully fetched market data for {len(market_data)} symbols")
                    return market_data
                else:
                    self.logger.warning("Failed to fetch market data, falling back to mock data")
                    return {symbol: self._generate_mock_data(symbol) for symbol in self.symbols}
            except Exception as e:
                self.logger.error(f"Error fetching market data: {e}", exc_info=True)
                self.logger.warning("Falling back to mock data due to error")
                return {symbol: self._generate_mock_data(symbol) for symbol in self.symbols}
    
    def _generate_mock_data(self, symbol: str) -> pd.DataFrame:
        """
        Generate mock market data for testing and development.
        
        This implementation uses the MockDataGenerator to create realistic
        market data with configurable patterns and trends.
        
        Args:
            symbol: The trading symbol to generate data for
            
        Returns:
            DataFrame with mock OHLCV data
        """
        self.logger.info(f"Generating mock data for {symbol}")
        
        # Determine an appropriate pattern based on the symbol
        # This allows for more interesting and varied test data
        pattern_map = {
            "BTC": MarketPattern.UPTREND,
            "ETH": MarketPattern.TRIANGLE_ASCENDING,
            "SOL": MarketPattern.FLAG_BULLISH,
            "XRP": MarketPattern.SIDEWAYS,
            "ADA": MarketPattern.DOUBLE_BOTTOM,
            "DOGE": MarketPattern.HEAD_SHOULDERS
        }
        
        # Get pattern from config or use default from map
        config_pattern = self.config.get("mock_data", {}).get("pattern")
        if config_pattern:
            # Try to convert string to MarketPattern enum
            try:
                pattern = MarketPattern(config_pattern)
            except ValueError:
                self.logger.warning(f"Invalid pattern '{config_pattern}' specified in config, using default")
                pattern = pattern_map.get(symbol, MarketPattern.SIDEWAYS)
        else:
            pattern = pattern_map.get(symbol, MarketPattern.SIDEWAYS)
        
        # Determine appropriate base price for the symbol
        base_price_map = {
            "BTC": 30000.0,
            "ETH": 2000.0,
            "SOL": 100.0,
            "XRP": 0.5,
            "ADA": 0.3,
            "DOGE": 0.1
        }
        
        base_price = base_price_map.get(symbol, 100.0)
        
        # Get mock data parameters from config
        mock_config = self.config.get("mock_data", {})
        periods = mock_config.get("periods", 200)  # Default to 200 periods
        interval = mock_config.get("interval", "1d")  # Default to daily data
        
        # Generate data using the MockDataGenerator
        data = self.mock_data_generator.generate_ohlcv_data(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=periods),
            periods=periods,
            interval=interval,
            base_price=base_price,
            pattern_type=pattern.value
        )
        
        self.logger.debug(f"Generated mock data for {symbol} with {len(data)} rows using pattern {pattern.value}")
        return data
    
    def process(self, data: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Process market data and generate technical signals.
        
        This is the main entry point for the agent's functionality. It processes
        data through the complete technical analysis pipeline:
        1. Extract/generate market data based on current mode
        2. Calculate technical indicators
        3. Detect patterns (when implemented)
        4. Classify market regimes (when implemented)
        5. Generate trading signals from strategies
        
        Args:
            data: Optional input data dictionary
            
        Returns:
            List of signal dictionaries or None if processing fails
        """
        self.update_status(AgentStatus.RUNNING) # Changed from PROCESSING to RUNNING
        self.logger.info(f"Processing data in {self.data_mode.value} mode")
        start_time = datetime.now()
        
        try:
            # Extract market data appropriate for the current data mode
            market_data = self._extract_market_data(data)
            if not market_data:
                self.logger.warning("No valid market data provided or extracted")
                return None
            
            # Step 1: Calculate technical indicators
            self.logger.info(f"Calculating indicators for {len(market_data)} symbols")
            indicators = self.indicator_engine.calculate(market_data, self.symbols)
            
            # Store calculated indicators in the technical state
            self.technical_state["indicators"] = indicators
            
            # Step 2: Detect chart patterns
            self.logger.info(f"Detecting patterns for {len(market_data)} symbols")
            patterns = self.pattern_detector.detect_patterns(market_data, self.symbols)
            
            # Count total patterns detected
            total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
            self.update_metrics({"patterns_detected": total_patterns})
            
            # Add patterns to technical state
            self.technical_state["patterns"] = patterns
            
            # Step 3: Classify market regimes (will be implemented in future)
            # regimes = self.regime_classifier.classify_regimes(market_data, indicators, self.symbols)
            # self.technical_state["regimes"] = regimes
            
            # Step 4: Generate strategy signals
            signals = self.strategy_manager.generate_signals(
                market_data, indicators, self.symbols
            )
            
            # Store the generated signals
            self.current_signals = {
                s["payload"]["symbol"]: s for s in signals
            }
            
            # Add data mode information to each signal
            for signal in signals:
                signal["payload"]["data_mode"] = self.data_mode.value
            
            # Track processing time and metrics
            process_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update agent metrics
            self.update_metrics({
                "avg_processing_time_ms": process_time, # This will be treated as last_processing_time_ms by current update_metrics
                "signals_generated": len(signals),
                # Assuming indicator_engine.metrics exists and has this key
                "indicator_calculations": self.indicator_engine.metrics.get("calculations_performed", 0) 
            })
            
            self.logger.info(
                f"Processed data in {process_time:.2f}ms, generated {len(signals)} signals"
            )
            self.update_status(AgentStatus.IDLE)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}", exc_info=True)
            self.update_metrics({"processing_errors": 1})
            self.update_status(AgentStatus.ERROR)
            return None
    
    def get_technical_state(self) -> Dict[str, Any]:
        """Get the current technical analysis state."""
        return self.technical_state
    
    def get_data_source_type(self) -> str:
        """Return the current data source type (mock or real)."""
        return self.data_mode.value

    def toggle_data_source(self) -> str:
        """Toggle between mock and real data sources."""
        if self.data_mode == DataMode.MOCK:
            self.data_mode = DataMode.REAL
            # Initialize MEXC connection if switching to real mode
            try:
                market_data_provider = get_market_data_provider(self.symbols)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(market_data_provider.initialize())
                self.logger.info("Initialized MEXC connection for real-time data")
            except Exception as e:
                self.logger.error(f"Error initializing MEXC connection: {e}", exc_info=True)
        else:
            self.data_mode = DataMode.MOCK
        
        self.logger.info(f"Switched data mode to {self.data_mode.value}")
        return self.data_mode.value

    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all components."""
        metrics = {
            "agent": self.metrics,
            "indicator_engine": self.indicator_engine.metrics if self.indicator_engine else {},
            "strategy_manager": self.strategy_manager.metrics if self.strategy_manager else {},
            "pattern_detector": {"patterns_detected": self.metrics.get("patterns_detected", 0)},
            "data_source": {
                "type": self.data_mode.value,
                "symbols": self.symbols,
                "provider": "MEXC" if self.data_mode == DataMode.REAL else "MockDataGenerator"
            }
        }
        
        # Add strategy-specific metrics
        if self.strategy_manager:
            metrics["strategies"] = self.strategy_manager.get_strategy_metrics()
            
        return metrics
