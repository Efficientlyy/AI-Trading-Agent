#!/usr/bin/env python
"""
Comprehensive Autonomous System Test

This script provides a complete end-to-end test of the AI Trading Agent system,
demonstrating all advanced autonomous capabilities including:

1. Multi-agent specialized ecosystem
2. Advanced market regime detection with ML techniques
3. Adaptive strategy optimization with genetic algorithms
4. LLM-powered oversight and decision verification
5. Self-healing system with fault detection and recovery
6. Portfolio-level risk management across multiple assets
7. Performance attribution and continuous improvement
8. Reinforcement learning for strategy selection
9. Regime transition detection with early warning signals

The test simulates various market conditions, component failures, and real-world
scenarios to demonstrate the system's resilience and adaptability.
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
import threading
import queue
import json
import pickle
import collections
import traceback
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path to help with imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the fixed MockLLMOversight class
try:
    from mock_llm_oversight import MockLLMOversight
    logger.info("Using imported MockLLMOversight class")
except ImportError:
    logger.warning("Could not import MockLLMOversight, will use internal definition")
    pass  # Will use internal definition if available

# Try importing the actual components if available, otherwise use mocks
try:
    # Import system components if available
    from ai_trading_agent.common.health_monitoring.core_definitions import HealthStatus, AlertSeverity
    from ai_trading_agent.market_regime import MarketRegimeType, VolatilityRegimeType
    from ai_trading_agent.market_regime.regime_classifier import MarketRegimeClassifier
    actual_components_available = True
    logger.info("Using actual system components")
except ImportError:
    logger.warning("Using mock components as actual components could not be imported")
    actual_components_available = False
    
    # Define mock enums for testing
    class HealthStatus(Enum):
        """Health status for system components."""
        HEALTHY = auto()
        DEGRADED = auto()
        CRITICAL = auto()
        FAILED = auto()
    
    class AlertSeverity(Enum):
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
    class MarketRegimeType(Enum):
        UNKNOWN = "unknown"
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"
        VOLATILE = "volatile"
        VOLATILE_BULL = "volatile_bull"
        VOLATILE_BEAR = "volatile_bear"
        RECOVERY = "recovery"
        BREAKDOWN = "breakdown"
        TRENDING = "trending"
    
    class VolatilityRegimeType(Enum):
        UNKNOWN = "unknown"
        VERY_LOW = "very_low"
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"
        VERY_HIGH = "very_high"
        EXTREME = "extreme"
        CRISIS = "crisis"

# Constants for the simulation
SIMULATION_DAYS = 90  # Simulate 90 days
HISTORICAL_DAYS = 60  # Generate 60 days of historical data first
TOTAL_ASSETS = 8      # Number of assets to simulate
MAX_TEST_CYCLES = 1000  # Maximum number of cycles to run
CYCLE_INTERVAL_SEC = 0.5  # How many seconds between cycles
FAILURE_PROBABILITY = 0.02  # 2% chance of component failure per cycle
RECOVERY_PROBABILITY = 0.7  # 70% chance of auto-recovery per cycle 
MARKET_SHIFT_PROBABILITY = 0.05  # 5% chance of market regime shift per day

# Asset types for simulation
class AssetClass(Enum):
    CRYPTO = "cryptocurrency"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"

#=============================================================================
# Market Simulation Components
#=============================================================================

class MarketSimulator:
    """Simulates market conditions with regime shifts and realistic price patterns."""
    
    def __init__(self, start_date: datetime = None, seed: int = None):
        """Initialize the market simulator.
        
        Args:
            start_date: Starting date for the simulation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.start_date = start_date or datetime.now() - timedelta(days=HISTORICAL_DAYS)
        self.current_date = self.start_date
        self.assets = {}
        self.regimes = {}
        self.volatility_regimes = {}
        self.correlations = {}
        self.trend_factors = {}
        self.market_events = []
        
        # Global market factors (affects all assets)
        self.global_regime = MarketRegimeType.SIDEWAYS
        self.global_volatility = VolatilityRegimeType.MODERATE
        self.global_sentiment = 0.0  # -1.0 to 1.0
        
        # For simulating correlated asset movements
        self.correlation_matrix = None
        
        logger.info(f"Market simulator initialized starting from {self.start_date}")
    
    def add_asset(self, symbol: str, asset_class: AssetClass, base_price: float, 
                  initial_regime: Optional[MarketRegimeType] = None,
                  initial_volatility: Optional[VolatilityRegimeType] = None,
                  correlation_group: Optional[str] = None):
        """Add an asset to the simulation.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC/USD')
            asset_class: Type of asset
            base_price: Starting price
            initial_regime: Initial market regime
            initial_volatility: Initial volatility regime
            correlation_group: Group for correlation (assets in same group correlate more)
        """
        if symbol in self.assets:
            logger.warning(f"Asset {symbol} already exists in simulation")
            return
        
        # Set default regimes if not specified
        if initial_regime is None:
            initial_regime = random.choice(list(MarketRegimeType))
        
        if initial_volatility is None:
            initial_volatility = random.choice(list(VolatilityRegimeType))
        
        # Initialize asset data
        self.assets[symbol] = {
            'symbol': symbol,
            'asset_class': asset_class,
            'base_price': base_price,
            'current_price': base_price,
            'price_history': [],
            'returns_history': [],
            'volume_history': [],
            'sentiment_history': [],
            'correlation_group': correlation_group or asset_class.value,
            'events': [],
        }
        
        # Set initial regimes
        self.regimes[symbol] = initial_regime
        self.volatility_regimes[symbol] = initial_volatility
        
        # Set initial trend factor (-1.0 to 1.0)
        self.trend_factors[symbol] = self._regime_to_trend(initial_regime)
        
        logger.info(f"Added asset {symbol} ({asset_class.value}) to simulation with initial regime {initial_regime.value}")
    
    def _regime_to_trend(self, regime: MarketRegimeType) -> float:
        """Convert a market regime to a trend factor."""
        trend_map = {
            MarketRegimeType.BULL: 0.7,
            MarketRegimeType.BEAR: -0.7,
            MarketRegimeType.SIDEWAYS: 0.0,
            MarketRegimeType.VOLATILE: 0.0,
            MarketRegimeType.VOLATILE_BULL: 0.5,
            MarketRegimeType.VOLATILE_BEAR: -0.5,
            MarketRegimeType.RECOVERY: 0.8,
            MarketRegimeType.BREAKDOWN: -0.8,
            MarketRegimeType.TRENDING: 0.6,
            MarketRegimeType.UNKNOWN: 0.0
        }
        return trend_map.get(regime, 0.0) + (random.random() - 0.5) * 0.2  # Add slight randomness
    
    def _volatility_to_factor(self, volatility: VolatilityRegimeType) -> float:
        """Convert a volatility regime to a factor."""
        vol_map = {
            VolatilityRegimeType.VERY_LOW: 0.005,
            VolatilityRegimeType.LOW: 0.01,
            VolatilityRegimeType.MODERATE: 0.02,
            VolatilityRegimeType.HIGH: 0.03,
            VolatilityRegimeType.VERY_HIGH: 0.05,
            VolatilityRegimeType.EXTREME: 0.07,
            VolatilityRegimeType.CRISIS: 0.10,
            VolatilityRegimeType.UNKNOWN: 0.02
        }
        return vol_map.get(volatility, 0.02)
        
    def _generate_correlated_returns(self, symbols: List[str]) -> Dict[str, float]:
        """Generate correlated returns for a set of assets."""
        # Group symbols by correlation group
        groups = {}
        for symbol in symbols:
            group = self.assets[symbol]['correlation_group']
            if group not in groups:
                groups[group] = []
            groups[group].append(symbol)
        
        returns = {}
        
        # Generate base random values (uncorrelated)
        base_randoms = {symbol: random.normalvariate(0, 1) for symbol in symbols}
        
        # For each group, generate correlated returns
        for group, group_symbols in groups.items():
            # Generate a group factor (affects all assets in the group)
            group_factor = random.normalvariate(0, 1)
            
            for symbol in group_symbols:
                # Mix individual factor with group factor (70% group, 30% individual)
                mixed_random = 0.7 * group_factor + 0.3 * base_randoms[symbol]
                
                # Add global market factor (affects all assets)
                global_factor = random.normalvariate(0, 1)
                final_random = 0.5 * mixed_random + 0.3 * global_factor
                
                # Apply regime trend and volatility
                trend = self.trend_factors[symbol]
                vol = self._volatility_to_factor(self.volatility_regimes[symbol])
                
                # Calculate return: trend + volatility*random
                daily_return = trend * 0.001 + vol * final_random
                returns[symbol] = daily_return
        
        return returns
    
    def _check_regime_transition(self, symbol: str) -> bool:
        """Check if an asset should transition to a different regime."""
        # Basic probability-based check
        if random.random() < MARKET_SHIFT_PROBABILITY:
            current_regime = self.regimes[symbol]
            
            # Define possible transitions based on current regime
            transitions = {
                MarketRegimeType.BULL: [MarketRegimeType.VOLATILE_BULL, MarketRegimeType.SIDEWAYS, MarketRegimeType.BREAKDOWN],
                MarketRegimeType.BEAR: [MarketRegimeType.VOLATILE_BEAR, MarketRegimeType.SIDEWAYS, MarketRegimeType.RECOVERY],
                MarketRegimeType.SIDEWAYS: [MarketRegimeType.BULL, MarketRegimeType.BEAR, MarketRegimeType.VOLATILE],
                MarketRegimeType.VOLATILE: [MarketRegimeType.VOLATILE_BULL, MarketRegimeType.VOLATILE_BEAR],
                MarketRegimeType.VOLATILE_BULL: [MarketRegimeType.BULL, MarketRegimeType.BREAKDOWN],
                MarketRegimeType.VOLATILE_BEAR: [MarketRegimeType.BEAR, MarketRegimeType.RECOVERY],
                MarketRegimeType.RECOVERY: [MarketRegimeType.BULL, MarketRegimeType.SIDEWAYS],
                MarketRegimeType.BREAKDOWN: [MarketRegimeType.BEAR, MarketRegimeType.SIDEWAYS],
                MarketRegimeType.TRENDING: [MarketRegimeType.SIDEWAYS, MarketRegimeType.VOLATILE],
                MarketRegimeType.UNKNOWN: list(MarketRegimeType)
            }
            
            # Select new regime from possible transitions
            possible_transitions = transitions.get(current_regime, list(MarketRegimeType))
            new_regime = random.choice(possible_transitions)
            
            # Update regime and trend factor
            self.regimes[symbol] = new_regime
            self.trend_factors[symbol] = self._regime_to_trend(new_regime)
            
            # Add a market event
            self.market_events.append({
                'date': self.current_date,
                'symbol': symbol,
                'event_type': 'regime_change',
                'from_regime': current_regime.value,
                'to_regime': new_regime.value,
                'description': f"Market regime changed from {current_regime.value} to {new_regime.value}"
            })
            
            logger.info(f"Regime transition for {symbol}: {current_regime.value} → {new_regime.value}")
            return True
        
        return False
    
    def _check_volatility_transition(self, symbol: str) -> bool:
        """Check if an asset should transition to a different volatility regime."""
        # Volatility changes less frequently than market regime
        if random.random() < MARKET_SHIFT_PROBABILITY / 2:
            current_volatility = self.volatility_regimes[symbol]
            
            # Define possible transitions (can only move one step at a time)
            volatility_levels = list(VolatilityRegimeType)
            current_index = volatility_levels.index(current_volatility)
            
            # Can move up, down, or stay the same
            possible_indices = [max(0, current_index - 1), current_index, min(len(volatility_levels) - 1, current_index + 1)]
            new_index = random.choice(possible_indices)
            
            if new_index != current_index:
                new_volatility = volatility_levels[new_index]
                self.volatility_regimes[symbol] = new_volatility
                
                # Add a market event
                self.market_events.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'event_type': 'volatility_change',
                    'from_volatility': current_volatility.value,
                    'to_volatility': new_volatility.value,
                    'description': f"Volatility changed from {current_volatility.value} to {new_volatility.value}"
                })
                
                logger.info(f"Volatility transition for {symbol}: {current_volatility.value} → {new_volatility.value}")
                return True
        
        return False
        
    def generate_sentiment(self, symbol: str) -> float:
        """Generate sentiment data for an asset."""
        # Base sentiment influenced by regime
        regime = self.regimes[symbol]
        base_sentiment = {
            MarketRegimeType.BULL: 0.6,
            MarketRegimeType.BEAR: -0.6,
            MarketRegimeType.SIDEWAYS: 0.0,
            MarketRegimeType.VOLATILE: 0.0,
            MarketRegimeType.VOLATILE_BULL: 0.3,
            MarketRegimeType.VOLATILE_BEAR: -0.3,
            MarketRegimeType.RECOVERY: 0.7,
            MarketRegimeType.BREAKDOWN: -0.7,
            MarketRegimeType.TRENDING: 0.4,
            MarketRegimeType.UNKNOWN: 0.0
        }.get(regime, 0.0)
        
        # Add randomness
        noise = random.normalvariate(0, 0.3)
        
        # Add global sentiment influence (30%)
        sentiment = 0.7 * (base_sentiment + noise) + 0.3 * self.global_sentiment
        
        # Clamp to [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        return sentiment
        
    def advance_day(self) -> Dict[str, Dict[str, Any]]:
        """Advance the simulation by one day and generate new data."""
        self.current_date += timedelta(days=1)
        
        # Check for global market regime shifts (less frequent)
        if random.random() < MARKET_SHIFT_PROBABILITY / 3:
            regimes = list(MarketRegimeType)
            new_global_regime = random.choice(regimes)
            
            if new_global_regime != self.global_regime:
                logger.info(f"Global market regime shift: {self.global_regime.value} → {new_global_regime.value}")
                self.global_regime = new_global_regime
                
                # Global sentiment shifts with regime
                self.global_sentiment = self._regime_to_trend(new_global_regime)
                
                # Add a market event
                self.market_events.append({
                    'date': self.current_date,
                    'event_type': 'global_regime_change',
                    'from_regime': self.global_regime.value,
                    'to_regime': new_global_regime.value,
                    'description': f"Global market regime changed to {new_global_regime.value}"
                })
        
        # Generate correlated returns for all assets
        daily_returns = self._generate_correlated_returns(list(self.assets.keys()))
        
        # Update each asset
        daily_data = {}
        
        for symbol, asset_data in self.assets.items():
            # Check for regime transitions
            regime_changed = self._check_regime_transition(symbol)
            volatility_changed = self._check_volatility_transition(symbol)
            
            # Calculate new price based on return
            daily_return = daily_returns[symbol]
            old_price = asset_data['current_price']
            new_price = old_price * (1 + daily_return)
            
            # Update price and store history
            asset_data['current_price'] = new_price
            asset_data['price_history'].append((self.current_date, new_price))
            asset_data['returns_history'].append((self.current_date, daily_return))
            
            # Generate volume (correlated with volatility and absolute return)
            base_volume = asset_data['base_price'] * 1000  # Base volume proportional to price
            vol_factor = self._volatility_to_factor(self.volatility_regimes[symbol])
            return_factor = abs(daily_return) * 10  # Higher volume on bigger moves
            
            volume = base_volume * (1 + vol_factor * 10 + return_factor)
            volume *= random.uniform(0.8, 1.2)  # Add some randomness
            asset_data['volume_history'].append((self.current_date, volume))
            
            # Generate sentiment
            sentiment = self.generate_sentiment(symbol)
            asset_data['sentiment_history'].append((self.current_date, sentiment))
            
            # Create today's data point
            daily_data[symbol] = {
                'date': self.current_date,
                'symbol': symbol,
                'open': old_price,
                'high': max(old_price, new_price) * random.uniform(1.0, 1.02),
                'low': min(old_price, new_price) * random.uniform(0.98, 1.0),
                'close': new_price,
                'volume': volume,
                'return': daily_return,
                'sentiment': sentiment,
                'regime': self.regimes[symbol].value,
                'volatility_regime': self.volatility_regimes[symbol].value
            }
        
        logger.debug(f"Generated market data for {self.current_date}")
        return daily_data
    
    def generate_historical_data(self, days: int) -> Dict[str, List[Dict[str, Any]]]:
        """Generate historical data for the specified number of days."""
        if not self.assets:
            raise ValueError("No assets defined. Add assets before generating historical data.")
        
        # Store the original current date
        original_date = self.current_date
        
        # Reset to historical start date
        self.current_date = self.start_date - timedelta(days=days)
        
        # Generate data for each day
        historical_data = {symbol: [] for symbol in self.assets}
        
        for _ in range(days):
            daily_data = self.advance_day()
            for symbol, data in daily_data.items():
                historical_data[symbol].append(data)
        
        # Restore the original date
        self.current_date = original_date
        
        logger.info(f"Generated {days} days of historical data from {self.current_date - timedelta(days=days)} to {self.current_date}")
        return historical_data
    
    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """Get historical data for a symbol as a pandas DataFrame."""
        if symbol not in self.assets:
            raise ValueError(f"Symbol {symbol} not found in simulation")
        
        # Extract price history and convert to DataFrame
        price_history = self.assets[symbol]['price_history']
        dates, prices = zip(*price_history) if price_history else ([], [])
        
        # Extract volume history
        volume_history = self.assets[symbol]['volume_history']
        _, volumes = zip(*volume_history) if volume_history else ([], [])
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes if volumes else [0] * len(dates)
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the market simulation."""
        return {
            'current_date': self.current_date,
            'global_regime': self.global_regime.value,
            'global_sentiment': self.global_sentiment,
            'assets': {symbol: {
                'current_price': data['current_price'],
                'regime': self.regimes[symbol].value,
                'volatility': self.volatility_regimes[symbol].value,
                'sentiment': data['sentiment_history'][-1][1] if data['sentiment_history'] else 0.0
            } for symbol, data in self.assets.items()},
            'recent_events': self.market_events[-10:] if self.market_events else []
        }

#=============================================================================
# Mock Data Providers for Trading System
#=============================================================================

class MockMarketDataProvider:
    """Mock data provider for market data."""
    
    def __init__(self, market_simulator: MarketSimulator):
        self.market_simulator = market_simulator
        self.subscribers = set()
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue(maxsize=100)
        self.health_status = HealthStatus.HEALTHY
        self.failure_mode = False
        
        logger.info("Initialized Mock Market Data Provider")
    
    def subscribe(self, symbol: str) -> bool:
        """Subscribe to data updates for a symbol."""
        if symbol in self.market_simulator.assets:
            self.subscribers.add(symbol)
            logger.info(f"Subscribed to market data for {symbol}")
            return True
        else:
            logger.warning(f"Cannot subscribe to {symbol}: symbol not found in simulation")
            return False
    
    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from data updates for a symbol."""
        if symbol in self.subscribers:
            self.subscribers.remove(symbol)
            logger.info(f"Unsubscribed from market data for {symbol}")
            return True
        else:
            logger.warning(f"Cannot unsubscribe from {symbol}: not subscribed")
            return False
    
    def start(self) -> bool:
        """Start the data provider simulation."""
        if self.running:
            logger.warning("Market data provider already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._data_generation_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started market data provider simulation")
        return True
    
    def stop(self) -> bool:
        """Stop the data provider simulation."""
        if not self.running:
            logger.warning("Market data provider not running")
            return False
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        logger.info("Stopped market data provider simulation")
        return True
    
    def inject_failure(self):
        """Inject a failure into the data provider for testing resilience."""
        self.failure_mode = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning("Failure injected into market data provider")
    
    def recover(self):
        """Recover from a failure."""
        self.failure_mode = False
        self.health_status = HealthStatus.HEALTHY
        logger.info("Market data provider recovered from failure")
    
    def _data_generation_loop(self) -> None:
        """Background thread for generating market data."""
        while self.running:
            try:
                # If in failure mode, either delay heavily or drop data
                if self.failure_mode:
                    if random.random() < 0.7:  # 70% chance to skip data point
                        time.sleep(random.uniform(0.5, 2.0))
                        continue
                    time.sleep(random.uniform(0.5, 1.5))  # Slower response
                else:
                    time.sleep(random.uniform(0.1, 0.3))  # Normal response time
                
                # Generate new data point (simplified - in reality would come from external source)
                daily_data = self.market_simulator.advance_day()
                
                # Put data in queue for subscribed symbols
                for symbol in self.subscribers:
                    if symbol in daily_data:
                        self.data_queue.put({
                            'type': 'market_data',
                            'symbol': symbol,
                            'data': daily_data[symbol]
                        })
            except Exception as e:
                logger.error(f"Error in market data generation: {e}")
                self.health_status = HealthStatus.DEGRADED
                time.sleep(1.0)  # Slow down on error
    
    def get_latest_data(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get the latest data from the queue."""
        try:
            return self.data_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for a symbol."""
        if self.failure_mode and random.random() < 0.4:
            # Simulate data corruption or unavailability
            logger.warning(f"Failed to retrieve historical data for {symbol} due to simulated failure")
            return pd.DataFrame()  # Empty dataframe
        
        return self.market_simulator.get_dataframe(symbol).tail(days)
    
    def check_health(self) -> HealthStatus:
        """Check the health of the data provider."""
        return self.health_status

class MockSentimentDataProvider:
    """Mock sentiment data provider to generate sentiment data."""
    
    def __init__(self, market_simulator):
        """Initialize the mock sentiment data provider.
        
        Args:
            market_simulator: Market simulator instance for coordinating data.
        """
        self.market_simulator = market_simulator
        self.subscribed_symbols = set()
        self.data_queue = collections.deque(maxlen=100)
        self.latest_data = {}
        self.sentiment_history = {}
        self.news_history = {}
        self.failure_state = False
        self.data_thread = None
        self.running = False
        self.update_interval = 5.0  # seconds
        self.historical_days = 60
        self.health_status = HealthStatus.HEALTHY
        
        logger.info("Initialized Mock Sentiment Data Provider")
    
    def start(self):
        """Start the sentiment data provider."""
        if self.running:
            logger.warning("Sentiment data provider already running")
            return False
        
        self.running = True
        self.data_thread = threading.Thread(target=self._sentiment_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        logger.info("Started sentiment data provider")
        return True
    
    def stop(self):
        """Stop the sentiment data provider."""
        if not self.running:
            logger.warning("Sentiment data provider not running")
            return False
        
        self.running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=3.0)
        
        logger.info("Stopped sentiment data provider")
        return True
    
    def subscribe(self, symbol):
        """Subscribe to sentiment updates for a symbol."""
        self.subscribed_symbols.add(symbol)
        return True
        
    def get_latest_data(self, timeout=0.1):
        """Get the latest sentiment data."""
        try:
            if self.data_queue:
                return self.data_queue.pop()
            return None
        except (IndexError, KeyError):
            return None
            
    def get_historical_data(self, symbol, days=30):
        """Get historical sentiment data for a symbol."""
        # Create some basic synthetic data
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        sentiment_scores = [random.uniform(-1.0, 1.0) for _ in range(days)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'confidence': [random.uniform(0.7, 0.95) for _ in range(days)]
        })
        df.set_index('date', inplace=True)
        return df
    
    def check_health(self):
        """Check the health of the sentiment data provider."""
        return self.health_status
    
    def inject_failure(self):
        """Inject a failure into the sentiment data provider."""
        self.failure_state = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning("Injected failure into sentiment data provider")
        
    def recover(self):
        """Recover from failure."""
        self.failure_state = False
        self.health_status = HealthStatus.HEALTHY
        logger.info("Recovered sentiment data provider from failure")
        
    def _sentiment_data_loop(self):
        """Main loop to generate sentiment data."""
        while self.running:
            try:
                # Skip if in failure state
                if self.failure_state:
                    time.sleep(1.0)
                    continue
                
                # Generate sentiment data for each subscribed symbol
                for symbol in self.subscribed_symbols:
                    # Generate random sentiment data
                    sentiment_data = {
                        'sentiment_score': random.uniform(-1.0, 1.0),
                        'confidence': random.uniform(0.7, 0.95),
                        'source': random.choice(['news', 'social_media', 'analyst_report']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store the data
                    self.latest_data[symbol] = sentiment_data
                    
                    # Add to the queue
                    self.data_queue.append({
                        'symbol': symbol,
                        'data': sentiment_data,
                        'timestamp': time.time()
                    })
                
                # Sleep for the update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in sentiment data loop: {e}")
                time.sleep(1.0)

#=============================================================================
# Mock LLM Oversight System
#=============================================================================

# Note: The MockLLMOversight class is now imported from mock_llm_oversight.py
# This prevents issues with the old implementation that had syntax errors

#=============================================================================
# Mock Market Regime Classifier
#=============================================================================

class MockMarketRegimeClassifier:
    """Mock market regime classifier that identifies market regimes."""
    
    def __init__(self):
        """Initialize the mock market regime classifier."""
        self.regime_history = {}  # Symbol -> list of regime classifications
        self.transition_history = {}  # Symbol -> list of transition signals
        self.ml_models = {}  # Simulated ML models for each symbol
        self.stats = {}
        self.failure_state = False
        self.running = False
        self.update_thread = None
        
        logger.info("Initialized Mock Market Regime Classifier")
        
    def start(self):
        """Start the market regime classifier."""
        if self.running:
            logger.warning("Market regime classifier already running")
            return False
        
        self.running = True
        self.update_thread = threading.Thread(target=self._model_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started market regime classifier")
        return True
    
    def stop(self):
        """Stop the market regime classifier."""
        if not self.running:
            logger.warning("Market regime classifier not running")
            return False
        
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=3.0)
        
        logger.info("Stopped market regime classifier")
        return True
    
    def _model_update_loop(self):
        """Periodically update ML models (simulated)."""
        while self.running:
            try:
                # Skip if in failure state
                if self.failure_state:
                    time.sleep(1.0)
                    continue
                
                # Update models for each symbol with history
                for symbol in self.regime_history.keys():
                    if random.random() < 0.05:  # 5% chance of model update
                        logger.debug(f"Updating ML model for {symbol}")
                        # Simulated model improvement
                        if symbol not in self.ml_models:
                            self.ml_models[symbol] = {
                                'version': 1,
                                'accuracy': 0.75 + random.random() * 0.1,
                                'last_update': time.time()
                            }
                        else:
                            self.ml_models[symbol]['version'] += 1
                            self.ml_models[symbol]['accuracy'] = min(0.95, self.ml_models[symbol]['accuracy'] + random.random() * 0.02)
                            self.ml_models[symbol]['last_update'] = time.time()
                
                # Sleep for a while
                time.sleep(60.0)  # Check for model updates every minute
            except Exception as e:
                logger.error(f"Error in model update loop: {e}")
                time.sleep(10.0)
    
    def classify_regime(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify the market regime for a given symbol using its price data."""
        if self.failure_state:
            if random.random() < 0.6:
                logger.warning(f"Market regime classification failed for {symbol} due to simulated failure")
                return {
                    'status': 'error',
                    'regime': MarketRegimeType.UNKNOWN.value,
                    'confidence': 0.0,
                    'volatility_regime': VolatilityRegimeType.UNKNOWN.value,
                    'message': 'Classification failed due to system error'
                }
        
        # Need at least some data points for classification
        if data.empty or len(data) < 5:
            logger.warning(f"Insufficient data for regime classification of {symbol}")
            return {
                'status': 'error',
                'regime': MarketRegimeType.UNKNOWN.value,
                'confidence': 0.0,
                'volatility_regime': VolatilityRegimeType.UNKNOWN.value,
                'message': 'Insufficient data for classification'
            }
        
        # Get previous classification if available
        prev_regime = None
        if self.classification_history[symbol]:
            prev_regime = self.classification_history[symbol][-1]['regime']
        
        # In a real implementation, this would use ML models to classify regime
        # Here we use a simplified approach based on price trends and volatility
        
        # Calculate returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
        else:
            # If we only have a single value column
            returns = data.iloc[:, 0].pct_change().dropna()
        
        if len(returns) < 3:
            logger.warning(f"Not enough return data for regime classification of {symbol}")
            regime = MarketRegimeType.UNKNOWN.value
            vol_regime = VolatilityRegimeType.UNKNOWN.value
            confidence = 0.5
        else:        
            # Calculate trend and volatility metrics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Classify volatility regime first
            if std_return < 0.01:
                vol_regime = VolatilityRegimeType.VERY_LOW.value
            elif std_return < 0.015:
                vol_regime = VolatilityRegimeType.LOW.value
            elif std_return < 0.025:
                vol_regime = VolatilityRegimeType.MODERATE.value
            elif std_return < 0.04:
                vol_regime = VolatilityRegimeType.HIGH.value
            elif std_return < 0.06:
                vol_regime = VolatilityRegimeType.VERY_HIGH.value
            elif std_return < 0.1:
                vol_regime = VolatilityRegimeType.EXTREME.value
            else:
                vol_regime = VolatilityRegimeType.CRISIS.value
            
            # Now classify market regime
            if mean_return > 0.005:
                if std_return > 0.03:
                    regime = MarketRegimeType.VOLATILE_BULL.value
                else:
                    regime = MarketRegimeType.BULL.value
            elif mean_return < -0.005:
                if std_return > 0.03:
                    regime = MarketRegimeType.VOLATILE_BEAR.value
                else:
                    regime = MarketRegimeType.BEAR.value
            else:
                if std_return > 0.03:
                    regime = MarketRegimeType.VOLATILE.value
                else:
                    regime = MarketRegimeType.SIDEWAYS.value
            
            # Add transition regimes if we detect a change from previous state
            if prev_regime and prev_regime != regime:
                if prev_regime == MarketRegimeType.BEAR.value and regime in [MarketRegimeType.SIDEWAYS.value, MarketRegimeType.BULL.value]:
                    regime = MarketRegimeType.RECOVERY.value
                elif prev_regime in [MarketRegimeType.BULL.value, MarketRegimeType.VOLATILE_BULL.value] and regime in [MarketRegimeType.BEAR.value, MarketRegimeType.SIDEWAYS.value]:
                    regime = MarketRegimeType.BREAKDOWN.value
            
            # Calculate a confidence level (in a real implementation this would come from the ML model)
            if len(returns) > 20:
                confidence = random.uniform(0.75, 0.95)  # More data = higher confidence
            else:
                confidence = random.uniform(0.6, 0.85)  # Less data = lower confidence
        
        # Store the classification result
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'regime': regime,
            'volatility_regime': vol_regime,
            'confidence': confidence,
            'transition_signal': self.transition_signals.get(symbol, 0.0),
            'metrics': {
                'mean_return': float(mean_return) if 'mean_return' in locals() else 0.0,
                'volatility': float(std_return) if 'std_return' in locals() else 0.0,
                'data_points': len(data)
            }
        }
        
        # Update history
        self.classification_history[symbol].append(result)
        if len(self.classification_history[symbol]) > 100:
            self.classification_history[symbol].pop(0)  # Keep history from growing too large
        
        # Update confidence levels
        self.confidence_levels[symbol] = confidence
        
        logger.info(f"Classified market regime for {symbol}: {regime} (vol: {vol_regime}), confidence: {confidence:.2f}")
        return result
    
    def detect_early_transition_signals(self, symbol: str, data: pd.DataFrame) -> float:
        """Detect early signals of regime transitions using advanced indicators."""
        if self.failure_mode or data.empty or len(data) < 10:
            return 0.0
        
        # In a real implementation, this would use specialized ML models for early detection
        # Here we simulate this with a simple random value weighted by recent volatility
        
        # Calculate returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
        else:
            returns = data.iloc[:, 0].pct_change().dropna()
        
        if len(returns) < 5:
            return 0.0
            
        # Compare recent volatility to overall volatility
        recent_vol = returns.tail(5).std()
        overall_vol = returns.std()
        
        vol_ratio = recent_vol / overall_vol if overall_vol > 0 else 1.0
        
        # Higher ratio = higher chance of transition
        transition_signal = min(1.0, max(0.0, (vol_ratio - 1.0) * random.uniform(0.5, 1.5)))
        
        # Store the signal
        self.transition_signals[symbol] = transition_signal
        
        if transition_signal > 0.7:
            logger.info(f"Detected strong early transition signal for {symbol}: {transition_signal:.2f}")
        
        return transition_signal
    
    def check_health(self) -> HealthStatus:
        """Check the health of the market regime classifier."""
        return self.health_status
    
    def inject_failure(self):
        """Inject a failure into the classifier for testing resilience."""
        self.failure_mode = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning("Failure injected into market regime classifier")
    
    def recover(self):
        """Recover from a failure."""
        self.failure_mode = False
        self.health_status = HealthStatus.HEALTHY
        logger.info("Market regime classifier recovered from failure")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the classifier's performance."""
        return {
            'symbols_tracked': len(self.classification_history),
            'average_confidence': sum(self.confidence_levels.values()) / max(1, len(self.confidence_levels)),
            'transitions_detected': sum(1 for v in self.transition_signals.values() if v > 0.7),
            'health_status': self.health_status.value
        }

#=============================================================================
# Mock Trading Agents
#=============================================================================

class AgentRole(Enum):
    """Defines the roles of agents in the trading system."""
    TECHNICAL = "technical"  # Technical analysis agent
    SENTIMENT = "sentiment"  # Sentiment analysis agent
    RISK = "risk"            # Risk management agent
    DECISION = "decision"    # Decision making agent
    EXECUTION = "execution"  # Order execution agent
    PORTFOLIO = "portfolio"  # Portfolio management agent
    MONITORING = "monitoring"  # System monitoring agent
    OVERSIGHT = "oversight"  # Oversight and compliance agent

class TradingAction(Enum):
    """Defines the possible trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class BaseAgent:
    """Base class for all trading agents."""
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.health_status = HealthStatus.HEALTHY
        self.signals = []
        self.running = False
        self.failure_mode = False
        
        logger.info(f"Initialized {role.value} agent: {name}")
    
    def start(self) -> bool:
        """Start the agent."""
        if self.running:
            logger.warning(f"Agent {self.name} already running")
            return False
            
        self.running = True
        logger.info(f"Started agent {self.name} ({self.role.value})")
        return True
    
    def stop(self) -> bool:
        """Stop the agent."""
        if not self.running:
            logger.warning(f"Agent {self.name} not running")
            return False
            
        self.running = False
        logger.info(f"Stopped agent {self.name} ({self.role.value})")
        return True
    
    def check_health(self) -> HealthStatus:
        """Check the health of the agent."""
        return self.health_status
    
    def inject_failure(self):
        """Inject a failure into the agent for testing resilience."""
        self.failure_mode = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning(f"Failure injected into agent {self.name} ({self.role.value})")
    
    def recover(self):
        """Recover from a failure."""
        self.failure_mode = False
        self.health_status = HealthStatus.HEALTHY
        logger.info(f"Agent {self.name} ({self.role.value}) recovered from failure")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and generate signals."""
        raise NotImplementedError("Subclasses must implement this method")

class TechnicalAgent(BaseAgent):
    """Agent that analyzes technical indicators and generates trading signals."""
    
    def __init__(self, name: str, strategy_type: str = "trend_following"):
        super().__init__(name, AgentRole.TECHNICAL)
        self.strategy_type = strategy_type
        self.last_signals = {}
        
        # Strategy-specific parameters
        if strategy_type == "trend_following":
            self.short_window = 20
            self.long_window = 50
        elif strategy_type == "mean_reversion":
            self.lookback_period = 30
            self.z_score_threshold = 2.0
        else:
            self.strategy_params = {}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate trading signals."""
        if self.failure_mode and random.random() < 0.7:
            logger.warning(f"Technical agent {self.name} failed to process data due to simulated failure")
            return {
                'status': 'error',
                'message': 'Agent in failure mode',
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.running:
            logger.warning(f"Technical agent {self.name} not running, cannot process data")
            return {
                'status': 'error',
                'message': 'Agent not running',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract relevant data
        symbol = data.get('symbol')
        market_data = data.get('market_data')
        
        if not symbol or not market_data:
            logger.warning(f"Technical agent {self.name} received invalid data")
            return {
                'status': 'error',
                'message': 'Invalid data format',
                'timestamp': datetime.now().isoformat()
            }
        
        # In a real implementation, we would compute various technical indicators
        # Here we simulate this with a simplified approach
        
        # Generate a trading signal based on the strategy
        if self.strategy_type == "trend_following":
            signal = self._generate_trend_following_signal(symbol, market_data)
        elif self.strategy_type == "mean_reversion":
            signal = self._generate_mean_reversion_signal(symbol, market_data)
        else:
            signal = self._generate_random_signal(symbol)
        
        # Store the signal
        self.last_signals[symbol] = signal
        
        # Add to signal history
        self.signals.append(signal)
        if len(self.signals) > 100:
            self.signals.pop(0)  # Keep history from growing too large
        
        logger.info(f"Technical agent {self.name} generated {signal['action']} signal for {symbol} with strength {signal['strength']:.2f}")
        return signal
    
    def _generate_trend_following_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trend following signal."""
        # Simulate a trend following strategy
        # In a real implementation, this would compute moving averages, etc.
        
        # Get the current market regime
        regime = market_data.get('regime', 'unknown')
        
        # Trend-following strategies work well in trending markets
        if regime in ['bull', 'bear', 'trending', 'volatile_bull', 'volatile_bear']:
            strength_multiplier = 1.2  # Stronger signals in trending markets
        else:
            strength_multiplier = 0.7  # Weaker signals in non-trending markets
        
        # Generate a directional bias based on the regime
        if regime in ['bull', 'volatile_bull', 'recovery']:
            directional_bias = 0.7  # Bullish bias
        elif regime in ['bear', 'volatile_bear', 'breakdown']:
            directional_bias = -0.7  # Bearish bias
        else:
            directional_bias = 0.0  # Neutral
        
        # Add some randomness
        random_factor = random.normalvariate(0, 0.3)
        
        # Combine factors to determine signal strength (-1.0 to 1.0)
        signal_strength = (directional_bias + random_factor) * strength_multiplier
        
        # Determine action based on signal strength
        if signal_strength > 0.3:
            action = TradingAction.BUY.value
        elif signal_strength < -0.3:
            action = TradingAction.SELL.value
        else:
            action = TradingAction.HOLD.value
        
        # Create the signal
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'strength': abs(signal_strength),  # Strength is always positive
            'confidence': random.uniform(0.6, 0.9),
            'strategy': f"trend_following_{self.name}",
            'market_regime': regime,
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window
            },
            'metrics': {
                'signal_strength': signal_strength,
                'directional_bias': directional_bias,
                'random_factor': random_factor
            }
        }
    
    def _generate_mean_reversion_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mean reversion signal."""
        # Simulate a mean reversion strategy
        # In a real implementation, this would compute z-scores, etc.
        
        # Get the current market regime
        regime = market_data.get('regime', 'unknown')
        
        # Mean reversion strategies work well in sideways or volatile markets
        if regime in ['sideways', 'volatile']:
            strength_multiplier = 1.3  # Stronger signals in sideways markets
        else:
            strength_multiplier = 0.6  # Weaker signals in trending markets
        
        # Generate a mean reversion signal
        # For mean reversion, the signal is often the opposite of the recent trend
        recent_return = market_data.get('return', 0.0)
        
        # Invert the return to get a mean reversion bias
        mean_reversion_bias = -np.sign(recent_return) * min(1.0, abs(recent_return) * 10)
        
        # Add some randomness
        random_factor = random.normalvariate(0, 0.3)
        
        # Combine factors to determine signal strength (-1.0 to 1.0)
        signal_strength = (mean_reversion_bias + random_factor) * strength_multiplier
        
        # Determine action based on signal strength
        if signal_strength > 0.3:
            action = TradingAction.BUY.value
        elif signal_strength < -0.3:
            action = TradingAction.SELL.value
        else:
            action = TradingAction.HOLD.value
        
        # Create the signal
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'strength': abs(signal_strength),  # Strength is always positive
            'confidence': random.uniform(0.6, 0.9),
            'strategy': f"mean_reversion_{self.name}",
            'market_regime': regime,
            'parameters': {
                'lookback_period': self.lookback_period,
                'z_score_threshold': self.z_score_threshold
            },
            'metrics': {
                'signal_strength': signal_strength,
                'mean_reversion_bias': mean_reversion_bias,
                'recent_return': recent_return
            }
        }
    
    def _generate_random_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate a random trading signal (fallback)."""
        signal_strength = random.uniform(-1.0, 1.0)
        
        if signal_strength > 0.3:
            action = TradingAction.BUY.value
        elif signal_strength < -0.3:
            action = TradingAction.SELL.value
        else:
            action = TradingAction.HOLD.value
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'strength': abs(signal_strength),
            'confidence': random.uniform(0.5, 0.7),
            'strategy': f"random_{self.name}",
            'parameters': {}
        }

class SentimentAgent(BaseAgent):
    """Agent that analyzes sentiment data and generates trading signals."""
    
    def __init__(self, name: str):
        super().__init__(name, AgentRole.SENTIMENT)
        self.last_signals = {}
        self.sentiment_threshold = 0.3  # Threshold for generating signals
        self.confidence_threshold = 0.6  # Minimum confidence level to generate signals
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment data and generate trading signals."""
        if self.failure_mode and random.random() < 0.7:
            logger.warning(f"Sentiment agent {self.name} failed to process data due to simulated failure")
            return {
                'status': 'error',
                'message': 'Agent in failure mode',
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.running:
            logger.warning(f"Sentiment agent {self.name} not running, cannot process data")
            return {
                'status': 'error',
                'message': 'Agent not running',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract relevant data
        symbol = data.get('symbol')
        sentiment_data = data.get('sentiment_data', {})
        market_data = data.get('market_data', {})
        
        if not symbol or not sentiment_data:
            logger.warning(f"Sentiment agent {self.name} received invalid data")
            return {
                'status': 'error',
                'message': 'Invalid data format',
                'timestamp': datetime.now().isoformat()
            }
        
        # Get sentiment score and confidence
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
        confidence = sentiment_data.get('confidence', 0.5)
        
        # Get market regime for context
        market_regime = market_data.get('regime', 'unknown')
        
        # Adjust sentiment threshold based on market regime
        adjusted_threshold = self.sentiment_threshold
        if market_regime in ['volatile', 'volatile_bull', 'volatile_bear']:
            adjusted_threshold *= 1.5  # Require stronger sentiment in volatile markets
        
        # Generate signal if sentiment is strong enough and confidence is sufficient
        if abs(sentiment_score) >= adjusted_threshold and confidence >= self.confidence_threshold:
            if sentiment_score > 0:
                action = TradingAction.BUY.value
                strength = min(1.0, sentiment_score)
            else:
                action = TradingAction.SELL.value
                strength = min(1.0, abs(sentiment_score))
        else:
            action = TradingAction.HOLD.value
            strength = 0.0
        
        # Create the signal
        signal = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'strategy': f"sentiment_{self.name}",
            'market_regime': market_regime,
            'parameters': {
                'sentiment_threshold': adjusted_threshold,
                'confidence_threshold': self.confidence_threshold
            },
            'source': sentiment_data.get('source', 'unknown')
        }
        
        # Store the signal
        self.last_signals[symbol] = signal
        
        # Add to signal history
        self.signals.append(signal)
        if len(self.signals) > 100:
            self.signals.pop(0)
        
        if action != TradingAction.HOLD.value:
            logger.info(f"Sentiment agent {self.name} generated {action} signal for {symbol} with strength {strength:.2f}")
        
        return signal

class RiskManagementAgent(BaseAgent):
    """Agent that manages risk and position sizing."""
    
    def __init__(self, name: str):
        super().__init__(name, AgentRole.RISK)
        self.max_position_size = 0.05  # Max 5% of portfolio per position
        self.max_total_risk = 0.2  # Max 20% of portfolio at risk
        self.stop_loss_pct = 0.02  # 2% stop loss by default
        self.volatility_multiplier = 1.0  # Adjusts position size based on volatility
        self.current_positions = {}
        self.portfolio_value = 100000.0  # Default starting value
        self.cash = 100000.0  # Initial cash
        
        logger.info(f"Initialized Risk Management Agent: {name}")
    
    def set_portfolio_value(self, value: float):
        """Set the current portfolio value."""
        self.portfolio_value = value
    
    def set_cash(self, cash: float):
        """Set the current cash value."""
        self.cash = cash
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update a position in the portfolio."""
        if quantity == 0:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            self.current_positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': quantity * price
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trading signals and apply risk management."""
        if self.failure_mode and random.random() < 0.7:
            logger.warning(f"Risk management agent {self.name} failed to process data due to simulated failure")
            return {
                'status': 'error',
                'message': 'Agent in failure mode',
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.running:
            logger.warning(f"Risk management agent {self.name} not running, cannot process data")
            return {
                'status': 'error',
                'message': 'Agent not running',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract relevant data
        symbol = data.get('symbol')
        signal = data.get('signal', {})
        market_data = data.get('market_data', {})
        
        if not symbol or not signal or not market_data:
            logger.warning(f"Risk management agent {self.name} received invalid data")
            return {
                'status': 'error',
                'message': 'Invalid data format',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract signal details
        action = signal.get('action', TradingAction.HOLD.value)
        strength = signal.get('strength', 0.0)
        confidence = signal.get('confidence', 0.5)
        
        # Extract market details
        current_price = market_data.get('close', 0.0)
        regime = market_data.get('regime', 'unknown')
        volatility_regime = market_data.get('volatility_regime', 'unknown')
        
        # Skip if it's a hold or we have no price
        if action == TradingAction.HOLD.value or current_price <= 0:
            return {
                'status': 'success',
                'symbol': symbol,
                'action': TradingAction.HOLD.value,
                'quantity': 0,
                'risk_level': 0.0,
                'timestamp': datetime.now().isoformat(),
                'message': 'No action required'
            }
        
        # Adjust volatility multiplier based on volatility regime
        vol_adjustment = {
            'very_low': 1.5,
            'low': 1.2,
            'moderate': 1.0,
            'high': 0.7,
            'very_high': 0.5,
            'extreme': 0.3,
            'crisis': 0.1,
            'unknown': 0.8
        }.get(volatility_regime, 1.0)
        
        # Adjust for market regime
        regime_adjustment = {
            'bull': 1.2,
            'bear': 0.8,
            'sideways': 1.0,
            'volatile': 0.6,
            'volatile_bull': 0.9,
            'volatile_bear': 0.7,
            'recovery': 1.1,
            'breakdown': 0.5,
            'trending': 1.1,
            'unknown': 0.9
        }.get(regime, 1.0)
        
        # Calculate base position size
        base_position_value = self.portfolio_value * self.max_position_size
        
        # Adjust by signal strength and confidence
        signal_adjustment = strength * confidence
        
        # Calculate final position size
        adjusted_size = base_position_value * signal_adjustment * vol_adjustment * regime_adjustment
        
        # Ensure we don't exceed max position size
        max_position_value = self.portfolio_value * self.max_position_size
        position_value = min(adjusted_size, max_position_value)
        
        # Check cash constraints
        if action == TradingAction.BUY.value and position_value > self.cash:
            position_value = self.cash  # Can't spend more than available cash
        
        # Calculate quantity
        quantity = position_value / current_price if current_price > 0 else 0
        
        # Calculate stop loss price
        stop_loss_pct = self.stop_loss_pct
        
        # Adjust stop loss based on volatility
        if volatility_regime in ['high', 'very_high', 'extreme', 'crisis']:
            stop_loss_pct *= 1.5  # Wider stops in volatile markets
        
        if action == TradingAction.BUY.value:
            stop_loss_price = current_price * (1 - stop_loss_pct)
        else:  # SELL - for short positions
            stop_loss_price = current_price * (1 + stop_loss_pct)
        
        # Create risk assessment result
        risk_result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'original_action': action,  # Keep track of original recommendation
            'quantity': quantity,
            'position_value': position_value,
            'current_price': current_price,
            'stop_loss_price': stop_loss_price,
            'risk_level': position_value / self.portfolio_value,
            'risk_adjustments': {
                'volatility_adjustment': vol_adjustment,
                'regime_adjustment': regime_adjustment,
                'signal_adjustment': signal_adjustment
            },
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_total_risk': self.max_total_risk,
                'stop_loss_pct': stop_loss_pct
            }
        }
        
        # Log decision
        logger.info(f"Risk manager {self.name} processed {action} signal for {symbol}: "
                   f"quantity={quantity:.2f}, risk={position_value/self.portfolio_value:.2%}")
        
        return risk_result

class DecisionAgent(BaseAgent):
    """Agent that integrates signals from multiple sources and makes final trading decisions."""
    
    def __init__(self, name: str, llm_oversight=None):
        super().__init__(name, AgentRole.DECISION)
        self.technical_signals = {}  # Latest signals from technical agents
        self.sentiment_signals = {}  # Latest signals from sentiment agents
        self.risk_assessments = {}   # Latest risk assessments
        self.regime_classifications = {}  # Latest market regime classifications
        self.llm_oversight = llm_oversight  # LLM oversight system
        self.decisions = []  # Historical decisions
        self.signal_weights = {}  # Weights for different signal sources
        self.adaptive_weights = True  # Whether to adapt weights based on performance
        self.performance_history = {}  # Track performance of different signal sources
        
        # Default signal weights
        self.default_weights = {
            'technical': 0.6,
            'sentiment': 0.4,
            'trend_following': 0.7,
            'mean_reversion': 0.3
        }
        
        # Initially set weights to defaults
        self.signal_weights = self.default_weights.copy()
        
        logger.info(f"Initialized Decision Agent: {name}")
    
    def register_technical_signal(self, symbol: str, signal: Dict[str, Any]):
        """Register a new technical analysis signal."""
        if symbol not in self.technical_signals:
            self.technical_signals[symbol] = []
        self.technical_signals[symbol].append(signal)
        
        # Keep only the most recent signals (last 5)
        if len(self.technical_signals[symbol]) > 5:
            self.technical_signals[symbol].pop(0)
    
    def register_sentiment_signal(self, symbol: str, signal: Dict[str, Any]):
        """Register a new sentiment analysis signal."""
        if symbol not in self.sentiment_signals:
            self.sentiment_signals[symbol] = []
        self.sentiment_signals[symbol].append(signal)
        
        # Keep only the most recent signals (last 5)
        if len(self.sentiment_signals[symbol]) > 5:
            self.sentiment_signals[symbol].pop(0)
    
    def register_risk_assessment(self, symbol: str, assessment: Dict[str, Any]):
        """Register a new risk assessment."""
        self.risk_assessments[symbol] = assessment
    
    def register_regime_classification(self, symbol: str, classification: Dict[str, Any]):
        """Register a new market regime classification."""
        self.regime_classifications[symbol] = classification
    
    def adapt_weights(self, symbol: str):
        """Adapt signal weights based on recent performance."""
        if not self.adaptive_weights or symbol not in self.performance_history:
            return
        
        # Get recent performance data
        perf_data = self.performance_history.get(symbol, [])
        if not perf_data or len(perf_data) < 5:
            return  # Not enough data to adapt weights
        
        # Calculate success rate for different signal sources
        tech_success = sum(1 for p in perf_data if p.get('source') == 'technical' and p.get('success', False)) / \
                      max(1, sum(1 for p in perf_data if p.get('source') == 'technical'))
        sent_success = sum(1 for p in perf_data if p.get('source') == 'sentiment' and p.get('success', False)) / \
                      max(1, sum(1 for p in perf_data if p.get('source') == 'sentiment'))
        
        # Adjust weights based on success rates
        total_success = tech_success + sent_success
        if total_success > 0:
            self.signal_weights['technical'] = tech_success / total_success
            self.signal_weights['sentiment'] = sent_success / total_success
        
        # Also adjust strategy weights if we have data
        trend_success = sum(1 for p in perf_data if p.get('strategy') == 'trend_following' and p.get('success', False)) / \
                      max(1, sum(1 for p in perf_data if p.get('strategy') == 'trend_following'))
        mean_rev_success = sum(1 for p in perf_data if p.get('strategy') == 'mean_reversion' and p.get('success', False)) / \
                      max(1, sum(1 for p in perf_data if p.get('strategy') == 'mean_reversion'))
        
        total_strat_success = trend_success + mean_rev_success
        if total_strat_success > 0:
            self.signal_weights['trend_following'] = trend_success / total_strat_success
            self.signal_weights['mean_reversion'] = mean_rev_success / total_strat_success
        
        logger.info(f"Adapted signal weights for {symbol}: technical={self.signal_weights['technical']:.2f}, "
                  f"sentiment={self.signal_weights['sentiment']:.2f}")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all available signals and make a trading decision."""
        if self.failure_mode and random.random() < 0.7:
            logger.warning(f"Decision agent {self.name} failed to process data due to simulated failure")
            return {
                'status': 'error',
                'message': 'Agent in failure mode',
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.running:
            logger.warning(f"Decision agent {self.name} not running, cannot process data")
            return {
                'status': 'error',
                'message': 'Agent not running',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extract relevant data
        symbol = data.get('symbol')
        market_data = data.get('market_data', {})
        
        if not symbol or not market_data:
            logger.warning(f"Decision agent {self.name} received invalid data")
            return {
                'status': 'error',
                'message': 'Invalid data format',
                'timestamp': datetime.now().isoformat()
            }
        
        # Adapt weights based on performance
        self.adapt_weights(symbol)
        
        # Get the most recent signals
        tech_signals = self.technical_signals.get(symbol, [])
        sent_signals = self.sentiment_signals.get(symbol, [])
        risk_assessment = self.risk_assessments.get(symbol, {})
        regime_classification = self.regime_classifications.get(symbol, {})
        
        # Skip if we don't have signals or regime classification
        if not tech_signals and not sent_signals:
            logger.warning(f"No signals available for {symbol}, skipping decision")
            return {
                'status': 'warning',
                'message': 'No signals available',
                'symbol': symbol,
                'action': TradingAction.HOLD.value,
                'timestamp': datetime.now().isoformat()
            }
        
        # Get the current market regime
        regime = regime_classification.get('regime', 'unknown')
        volatility_regime = regime_classification.get('volatility_regime', 'unknown')
        regime_confidence = regime_classification.get('confidence', 0.5)
        
        # Adjust weights based on market regime
        regime_adjusted_weights = self.signal_weights.copy()
        
        # In trending markets, favor trend following and technical analysis
        if regime in ['bull', 'bear', 'trending', 'recovery', 'breakdown']:
            regime_adjusted_weights['technical'] *= 1.2
            regime_adjusted_weights['trend_following'] *= 1.3
            regime_adjusted_weights['mean_reversion'] *= 0.7
        
        # In sideways or volatile markets, favor mean reversion and sentiment
        elif regime in ['sideways', 'volatile', 'volatile_bull', 'volatile_bear']:
            regime_adjusted_weights['sentiment'] *= 1.2
            regime_adjusted_weights['mean_reversion'] *= 1.3
            regime_adjusted_weights['trend_following'] *= 0.7
        
        # Normalize weights to sum to 1.0
        tech_weight = regime_adjusted_weights['technical']
        sent_weight = regime_adjusted_weights['sentiment']
        total_weight = tech_weight + sent_weight
        if total_weight > 0:
            tech_weight /= total_weight
            sent_weight /= total_weight
        
        # Calculate weighted signal scores for each action
        action_scores = {
            TradingAction.BUY.value: 0.0,
            TradingAction.SELL.value: 0.0,
            TradingAction.HOLD.value: 0.0
        }
        
        # Process technical signals
        if tech_signals:
            for signal in tech_signals[-3:]:  # Use the last 3 signals
                action = signal.get('action', TradingAction.HOLD.value)
                strength = signal.get('strength', 0.0)
                confidence = signal.get('confidence', 0.5)
                strategy = signal.get('strategy', '').split('_')[0]  # Extract strategy type
                
                # Get strategy-specific weight
                strategy_weight = regime_adjusted_weights.get(strategy, 1.0)
                
                # Calculate signal weight (0.0 to 1.0)
                signal_weight = strength * confidence * strategy_weight
                
                # Add to action score
                action_scores[action] += signal_weight * tech_weight
        
        # Process sentiment signals
        if sent_signals:
            for signal in sent_signals[-3:]:  # Use the last 3 signals
                action = signal.get('action', TradingAction.HOLD.value)
                strength = signal.get('strength', 0.0)
                confidence = signal.get('confidence', 0.5)
                
                # Calculate signal weight (0.0 to 1.0)
                signal_weight = strength * confidence
                
                # Add to action score
                action_scores[action] += signal_weight * sent_weight
        
        # Determine the highest-scoring action
        best_action = TradingAction.HOLD.value
        best_score = action_scores[TradingAction.HOLD.value]
        
        for action, score in action_scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        
        # Apply risk management constraints
        quantity = 0.0
        risk_level = 0.0
        position_value = 0.0
        stop_loss_price = 0.0
        
        if best_action != TradingAction.HOLD.value and risk_assessment:
            # Check if risk assessment has a different action
            risk_action = risk_assessment.get('action', best_action)
            
            # If risk assessment overrides, use its recommendation
            if risk_action != best_action:
                logger.info(f"Risk management overrode {best_action} with {risk_action} for {symbol}")
                best_action = risk_action
            
            # Extract risk parameters
            quantity = risk_assessment.get('quantity', 0.0)
            risk_level = risk_assessment.get('risk_level', 0.0)
            position_value = risk_assessment.get('position_value', 0.0)
            stop_loss_price = risk_assessment.get('stop_loss_price', 0.0)
        
        # Prepare decision data
        decision = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': best_action,
            'confidence': best_score,
            'quantity': quantity,
            'risk_level': risk_level,
            'position_value': position_value,
            'market_price': market_data.get('close', 0.0),
            'stop_loss_price': stop_loss_price,
            'market_regime': regime,
            'volatility_regime': volatility_regime,
            'scores': action_scores,
            'weights': regime_adjusted_weights,
            'llm_approved': False,  # Will be updated after LLM review
            'llm_feedback': None,   # Will be populated after LLM review
            'execution_status': 'pending'
        }
        
        # If we have LLM oversight, submit decision for review
        if self.llm_oversight and best_action != TradingAction.HOLD.value:
            try:
                # Prepare data for LLM review
                llm_data = {
                    'symbol': symbol,
                    'action': best_action,
                    'confidence': best_score,
                    'quantity': quantity,
                    'risk_level': risk_level,
                    'market_price': market_data.get('close', 0.0),
                    'market_regime': regime,
                    'volatility_regime': volatility_regime,
                    'sentiment_score': sent_signals[-1].get('sentiment_score', 0.0) if sent_signals else 0.0
                }
                
                # Submit for review
                llm_result = self.llm_oversight.review_decision(llm_data)
                
                # Update decision with LLM result
                decision['llm_approved'] = llm_result.get('approved', False)
                decision['llm_feedback'] = llm_result.get('reasoning', '')
                decision['llm_confidence'] = llm_result.get('confidence', 0.0)
                
                # If LLM rejected, switch to HOLD
                if not decision['llm_approved']:
                    logger.info(f"LLM oversight rejected {best_action} for {symbol}: {decision['llm_feedback']}")
                    decision['original_action'] = best_action
                    decision['action'] = TradingAction.HOLD.value
                    decision['quantity'] = 0.0
                else:
                    logger.info(f"LLM oversight approved {best_action} for {symbol}")
            except Exception as e:
                logger.error(f"Error during LLM oversight: {e}")
                # If LLM fails, default to HOLD to be safe
                decision['original_action'] = best_action
                decision['action'] = TradingAction.HOLD.value
                decision['quantity'] = 0.0
                decision['llm_feedback'] = f"LLM oversight error: {str(e)}"
        
        # Add to decision history
        self.decisions.append(decision)
        if len(self.decisions) > 100:
            self.decisions.pop(0)
        
        # Log decision
        action_str = decision['action']
        if decision.get('original_action') and decision['original_action'] != decision['action']:
            action_str = f"{decision['original_action']} -> {decision['action']} (LLM override)"
        
        logger.info(f"Decision agent {self.name} decided {action_str} for {symbol} "
                   f"with confidence {best_score:.2f}, quantity={quantity:.2f}")
        
        return decision

class PortfolioManager:
    """Manages the portfolio and tracks performance."""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}  # Current positions
        self.transactions = []  # Transaction history
        self.portfolio_history = []  # Portfolio value history
        self.current_nav = initial_cash  # Net asset value
        self.cash_history = [(datetime.now(), initial_cash)]  # Cash history
        self.nav_history = [(datetime.now(), initial_cash)]  # NAV history
        self.performance_metrics = {}
        self.health_status = HealthStatus.HEALTHY
        
        logger.info(f"Initialized Portfolio Manager with ${initial_cash:.2f}")
    
    def execute_decision(self, decision: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Execute a trading decision."""
        symbol = decision.get('symbol')
        action = decision.get('action')
        quantity = decision.get('quantity', 0.0)
        
        if not symbol or not action or current_price <= 0:
            logger.warning("Invalid decision or price for execution")
            return {
                'status': 'error',
                'message': 'Invalid decision or price',
                'timestamp': datetime.now().isoformat()
            }
        
        # Initialize execution result
        execution_result = {
            'status': 'pending',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': 0.0,  # Will be updated with actual quantity
            'price': current_price,
            'value': 0.0,
            'fees': 0.0,
            'cash_change': 0.0
        }
        
        # Skip if it's a HOLD action
        if action == TradingAction.HOLD.value:
            execution_result['status'] = 'skipped'
            execution_result['message'] = 'Hold decision, no action taken'
            return execution_result
        
        # Calculate transaction cost (simplified)
        fee_rate = 0.001  # 0.1% fee
        transaction_value = quantity * current_price
        fee = transaction_value * fee_rate
        
        # Check if we already have a position for this symbol
        current_position = self.positions.get(symbol, {'quantity': 0.0, 'cost_basis': 0.0})
        current_quantity = current_position.get('quantity', 0.0)
        
        # Execute the trade
        if action == TradingAction.BUY.value:
            # Check if we have enough cash
            total_cost = transaction_value + fee
            if total_cost > self.cash:
                # Adjust quantity to match available cash
                adjusted_quantity = (self.cash - fee) / current_price if current_price > 0 else 0
                if adjusted_quantity <= 0:
                    logger.warning(f"Insufficient cash to execute buy for {symbol}")
                    execution_result['status'] = 'failed'
                    execution_result['message'] = 'Insufficient cash'
                    return execution_result
                
                logger.warning(f"Adjusted buy quantity for {symbol} from {quantity} to {adjusted_quantity} due to cash constraints")
                quantity = adjusted_quantity
                transaction_value = quantity * current_price
                fee = transaction_value * fee_rate
                total_cost = transaction_value + fee
            
            # Update cash
            self.cash -= total_cost
            
            # Update position
            new_quantity = current_quantity + quantity
            new_cost_basis = ((current_quantity * current_position.get('cost_basis', 0.0)) + transaction_value) / new_quantity if new_quantity > 0 else 0
            
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': new_quantity,
                'cost_basis': new_cost_basis,
                'current_price': current_price,
                'market_value': new_quantity * current_price,
                'unrealized_pnl': (current_price - new_cost_basis) * new_quantity
            }
            
            # Record cash change (negative for buys)
            cash_change = -total_cost
            
        elif action == TradingAction.SELL.value:
            # Check if we have enough quantity to sell
            if quantity > current_quantity:
                logger.warning(f"Insufficient quantity to execute sell for {symbol}")
                if current_quantity <= 0:
                    execution_result['status'] = 'failed'
                    execution_result['message'] = 'No position to sell'
                    return execution_result
                
                logger.warning(f"Adjusted sell quantity for {symbol} from {quantity} to {current_quantity}")
                quantity = current_quantity
                transaction_value = quantity * current_price
                fee = transaction_value * fee_rate
            
            # Calculate realized P&L
            cost_basis = current_position.get('cost_basis', current_price)
            realized_pnl = (current_price - cost_basis) * quantity
            
            # Update cash
            net_proceeds = transaction_value - fee
            self.cash += net_proceeds
            
            # Update position
            new_quantity = current_quantity - quantity
            if new_quantity <= 0:
                # Position closed
                if symbol in self.positions:
                    del self.positions[symbol]
            else:
                # Partial sell, cost basis remains the same
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': new_quantity,
                    'cost_basis': cost_basis,
                    'current_price': current_price,
                    'market_value': new_quantity * current_price,
                    'unrealized_pnl': (current_price - cost_basis) * new_quantity
                }
            
            # Record cash change (positive for sells)
            cash_change = net_proceeds
        
        # Update execution result
        execution_result['status'] = 'executed'
        execution_result['quantity'] = quantity
        execution_result['value'] = transaction_value
        execution_result['fees'] = fee
        execution_result['cash_change'] = cash_change
        
        # Record the transaction
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': current_price,
            'value': transaction_value,
            'fees': fee,
            'cash_after': self.cash
        }
        self.transactions.append(transaction)
        
        # Update cash history
        self.cash_history.append((datetime.now(), self.cash))
        
        # Update NAV
        self._update_nav()
        
        logger.info(f"Executed {action} for {symbol}: {quantity} shares at ${current_price:.2f}, fees=${fee:.2f}")
        return execution_result
    
    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                old_price = position['current_price']
                position['current_price'] = price
                position['market_value'] = position['quantity'] * price
                position['unrealized_pnl'] = (price - position['cost_basis']) * position['quantity']
                position['price_change'] = price - old_price
                position['price_change_pct'] = (price / old_price - 1) if old_price > 0 else 0.0
        
        # Update NAV after price updates
        self._update_nav()
    
    def _update_nav(self):
        """Update the portfolio net asset value."""
        # Sum up the market value of all positions
        total_market_value = sum(position['market_value'] for position in self.positions.values())
        
        # Add cash
        new_nav = total_market_value + self.cash
        
        # Record NAV history if it changed
        if new_nav != self.current_nav:
            self.current_nav = new_nav
            self.nav_history.append((datetime.now(), new_nav))
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics."""
        # Only calculate if we have some history
        if len(self.nav_history) < 2:
            return
        
        # Get dates and values
        dates = [date for date, _ in self.nav_history]
        values = [value for _, value in self.nav_history]
        
        # Calculate returns
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
        
        # Calculate metrics
        total_return = (self.current_nav / self.initial_cash - 1) if self.initial_cash > 0 else 0.0
        
        # Annualized return (simplified)
        days = (dates[-1] - dates[0]).days
        annualized_return = ((1 + total_return) ** (365 / max(1, days)) - 1) if days > 0 else 0.0
        
        # Volatility (standard deviation of returns)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0.0  # Annualized
        
        # Sharpe ratio (simplified, assuming risk-free rate of 0)
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0
        
        # Maximum drawdown
        max_drawdown = 0.0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Update performance metrics
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_nav': self.current_nav,
            'initial_capital': self.initial_cash,
            'cash': self.cash,
            'num_positions': len(self.positions),
            'last_update': datetime.now().isoformat()
        }
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get details for a specific position."""
        return self.positions.get(symbol, None)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current positions."""
        return self.positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio."""
        return {
            'cash': self.cash,
            'nav': self.current_nav,
            'positions': len(self.positions),
            'performance': self.performance_metrics,
            'allocation': {
                symbol: {
                    'weight': position['market_value'] / self.current_nav if self.current_nav > 0 else 0.0,
                    'market_value': position['market_value'],
                    'unrealized_pnl': position['unrealized_pnl']
                } for symbol, position in self.positions.items()
            }
        }
    
    def check_health(self) -> HealthStatus:
        """Check the health of the portfolio manager."""
        return self.health_status

#=============================================================================
# Health Monitoring System
#=============================================================================

class HealthMonitoringSystem:
    """Health monitoring system that detects component failures and coordinates recovery."""
    
    def __init__(self):
        self.components = {}  # Tracked components
        self.health_history = {}  # Health history for each component
        self.alerts = []  # Alert history
        self.running = False
        self.status_thread = None
        self.last_check = {}  # Last check time for each component
        self.recovery_attempts = {}  # Number of recovery attempts for each component
        self.max_recovery_attempts = 3  # Maximum number of recovery attempts before manual intervention
        self.check_interval = 5.0  # Seconds between health checks
        self.alert_callbacks = []  # Callbacks for alerting (can be used to send notifications)
        
        logger.info("Initialized Health Monitoring System")
    
    def register_component(self, component_id: str, component: Any, check_interval: float = 5.0):
        """Register a component for health monitoring."""
        if not hasattr(component, 'check_health'):
            logger.warning(f"Component {component_id} does not have a check_health method, cannot monitor")
            return False
        
        self.components[component_id] = {
            'id': component_id,
            'component': component,
            'check_interval': check_interval,
            'last_status': HealthStatus.UNKNOWN,
            'last_check_time': 0,
            'failure_count': 0,
            'recovery_count': 0
        }
        
        self.health_history[component_id] = []
        self.last_check[component_id] = 0
        self.recovery_attempts[component_id] = 0
        
        logger.info(f"Registered component {component_id} for health monitoring")
        return True
    
    def start(self) -> bool:
        """Start the health monitoring system."""
        if self.running:
            logger.warning("Health monitoring system already running")
            return False
        
        self.running = True
        self.status_thread = threading.Thread(target=self._monitoring_loop)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        logger.info("Started health monitoring system")
        return True
    
    def stop(self) -> bool:
        """Stop the health monitoring system."""
        if not self.running:
            logger.warning("Health monitoring system not running")
            return False
        
        self.running = False
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=2.0)
        
        logger.info("Stopped health monitoring system")
        return True
    
    def _monitoring_loop(self):
        """Background thread for monitoring component health."""
        while self.running:
            try:
                now = time.time()
                
                # Check each component's health
                for component_id, component_info in self.components.items():
                    # Only check if enough time has passed since last check
                    if now - self.last_check.get(component_id, 0) >= component_info['check_interval']:
                        self._check_component_health(component_id)
                        self.last_check[component_id] = now
                
                # Sleep a bit to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(1.0)  # Slow down on error
    
    def _check_component_health(self, component_id: str):
        """Check the health of a specific component."""
        if component_id not in self.components:
            return
        
        component_info = self.components[component_id]
        component = component_info['component']
        
        try:
            # Get current health status
            status = component.check_health()
            
            # Record the status
            previous_status = component_info['last_status']
            component_info['last_status'] = status
            
            # Add to history
            self.health_history[component_id].append({
                'timestamp': datetime.now().isoformat(),
                'status': status.value
            })
            
            # Limit history size
            if len(self.health_history[component_id]) > 100:
                self.health_history[component_id].pop(0)
            
            # Check for degradation
            if status != HealthStatus.HEALTHY and previous_status == HealthStatus.HEALTHY:
                # Component has degraded
                component_info['failure_count'] += 1
                
                # Create an alert
                self._create_alert(component_id, AlertSeverity.WARNING, 
                                   f"Component {component_id} degraded from {previous_status.value} to {status.value}")
                
                # Attempt recovery if status is critical or unhealthy
                if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    self._attempt_recovery(component_id)
            
            # Check for recovery
            elif status == HealthStatus.HEALTHY and previous_status != HealthStatus.HEALTHY:
                # Component has recovered
                component_info['recovery_count'] += 1
                
                # Create an alert
                self._create_alert(component_id, AlertSeverity.INFO, 
                                   f"Component {component_id} recovered from {previous_status.value} to {status.value}")
                
                # Reset recovery attempts
                self.recovery_attempts[component_id] = 0
            
        except Exception as e:
            logger.error(f"Error checking health of component {component_id}: {e}")
            
            # If we can't check health, assume it's degraded
            component_info['last_status'] = HealthStatus.DEGRADED
            
            # Create an alert
            self._create_alert(component_id, AlertSeverity.ERROR, 
                              f"Error checking health of component {component_id}: {str(e)}")
    
    def _attempt_recovery(self, component_id: str):
        """Attempt to recover a degraded component."""
        if component_id not in self.components:
            return
        
        component_info = self.components[component_id]
        component = component_info['component']
        
        # Check if we've exceeded the maximum number of recovery attempts
        if self.recovery_attempts.get(component_id, 0) >= self.max_recovery_attempts:
            self._create_alert(component_id, AlertSeverity.CRITICAL, 
                              f"Component {component_id} failed maximum recovery attempts, manual intervention required")
            return
        
        try:
            # Increment recovery attempts
            self.recovery_attempts[component_id] += 1
            
            # Log recovery attempt
            logger.info(f"Attempting recovery of {component_id} (attempt {self.recovery_attempts[component_id]})")
            
            # If component has a recover method, call it
            if hasattr(component, 'recover') and callable(getattr(component, 'recover')):
                component.recover()
                self._create_alert(component_id, AlertSeverity.INFO, 
                                  f"Recovery attempt {self.recovery_attempts[component_id]} initiated for {component_id}")
            else:
                # If no recover method, try stopping and starting if possible
                if hasattr(component, 'stop') and callable(getattr(component, 'stop')) and \
                   hasattr(component, 'start') and callable(getattr(component, 'start')):
                    component.stop()
                    time.sleep(0.5)  # Give it time to shut down
                    component.start()
                    self._create_alert(component_id, AlertSeverity.INFO, 
                                      f"Restart initiated for {component_id} (attempt {self.recovery_attempts[component_id]})")
                else:
                    self._create_alert(component_id, AlertSeverity.WARNING, 
                                      f"No recovery method available for {component_id}")
                    
        except Exception as e:
            logger.error(f"Error attempting to recover component {component_id}: {e}")
            self._create_alert(component_id, AlertSeverity.ERROR, 
                              f"Recovery attempt for {component_id} failed: {str(e)}")
    
    def _create_alert(self, component_id: str, severity: AlertSeverity, message: str):
        """Create an alert for a component."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'component_id': component_id,
            'severity': severity.value,
            'message': message,
            'status': self.components.get(component_id, {}).get('last_status', HealthStatus.UNKNOWN).value
        }
        
        self.alerts.append(alert)
        
        # Limit alerts history size
        if len(self.alerts) > 1000:
            self.alerts.pop(0)
        
        # Log the alert
        log_level = {
            AlertSeverity.INFO.value: logging.INFO,
            AlertSeverity.WARNING.value: logging.WARNING,
            AlertSeverity.ERROR.value: logging.ERROR,
            AlertSeverity.CRITICAL.value: logging.CRITICAL
        }.get(severity.value, logging.INFO)
        
        logger.log(log_level, f"ALERT - {severity.value}: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback to be called when an alert is created."""
        self.alert_callbacks.append(callback)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get the overall health of the system."""
        # Count components by status
        status_counts = collections.Counter()
        for component_info in self.components.values():
            status_counts[component_info['last_status'].value] += 1
        
        # Determine overall status (worst component status)
        overall_status = HealthStatus.HEALTHY
        for component_info in self.components.values():
            status = component_info['last_status']
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and overall_status not in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                overall_status = HealthStatus.DEGRADED
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status.value,
            'components_total': len(self.components),
            'components_healthy': status_counts.get(HealthStatus.HEALTHY.value, 0),
            'components_degraded': status_counts.get(HealthStatus.DEGRADED.value, 0),
            'components_unhealthy': status_counts.get(HealthStatus.UNHEALTHY.value, 0),
            'components_critical': status_counts.get(HealthStatus.CRITICAL.value, 0),
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
    
    def get_component_health(self, component_id: str) -> Dict[str, Any]:
        """Get detailed health information for a specific component."""
        if component_id not in self.components:
            return {'error': f"Component {component_id} not found"}
        
        component_info = self.components[component_id]
        
        return {
            'id': component_id,
            'status': component_info['last_status'].value,
            'failure_count': component_info['failure_count'],
            'recovery_count': component_info['recovery_count'],
            'recovery_attempts': self.recovery_attempts.get(component_id, 0),
            'history': self.health_history.get(component_id, [])[-20:],
            'last_check_time': self.last_check.get(component_id, 0)
        }

#=============================================================================
# Autonomous System Orchestrator
#=============================================================================

class AutonomousSystemOrchestrator:
    """Main orchestrator that coordinates all components of the autonomous trading system."""
    
    def __init__(self, initial_cash: float = 100000.0, seed: int = None):
        # Initialize market simulator
        self.market_simulator = MarketSimulator(start_date=datetime.now() - timedelta(days=HISTORICAL_DAYS), seed=seed)
        
        # Initialize data providers
        self.market_data_provider = MockMarketDataProvider(self.market_simulator)
        self.sentiment_data_provider = MockSentimentDataProvider(self.market_simulator)
        
        # Initialize market regime classifier
        self.regime_classifier = MockMarketRegimeClassifier()
        
        # Initialize LLM oversight
        self.llm_oversight = MockLLMOversight()
        
        # Initialize agents
        self.technical_agent1 = TechnicalAgent("TrendFollower", strategy_type="trend_following")
        self.technical_agent2 = TechnicalAgent("MeanReverter", strategy_type="mean_reversion")
        self.sentiment_agent = SentimentAgent("SentimentAnalyzer")
        self.risk_manager = RiskManagementAgent("RiskManager")
        self.decision_agent = DecisionAgent("MasterDecision", llm_oversight=self.llm_oversight)
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(initial_cash=initial_cash)
        
        # Initialize health monitoring
        self.health_monitor = HealthMonitoringSystem()
        
        # Register components for health monitoring
        self._register_components_for_monitoring()
        
        # Trading state
        self.running = False
        self.trading_thread = None
        self.cycle_interval = CYCLE_INTERVAL_SEC
        self.symbols = []  # Symbols to trade
        self.cycle_count = 0
        self.max_cycles = MAX_TEST_CYCLES
        self.data_buffer = {}  # Buffer for latest data
        self.last_regime_check = {}  # Last regime check time for each symbol
        self.regime_check_interval = 60  # Seconds between regime checks
        self.component_failure_info = {}  # Info about component failures
        
        # Add import for UUID
        import uuid
        self.uuid = uuid.uuid4
        
        logger.info("Initialized Autonomous System Orchestrator")
    
    def _register_components_for_monitoring(self):
        """Register all components for health monitoring."""
        self.health_monitor.register_component("market_data_provider", self.market_data_provider)
        self.health_monitor.register_component("sentiment_data_provider", self.sentiment_data_provider)
        self.health_monitor.register_component("regime_classifier", self.regime_classifier)
        self.health_monitor.register_component("llm_oversight", self.llm_oversight)
        self.health_monitor.register_component("technical_agent1", self.technical_agent1)
        self.health_monitor.register_component("technical_agent2", self.technical_agent2)
        self.health_monitor.register_component("sentiment_agent", self.sentiment_agent)
        self.health_monitor.register_component("risk_manager", self.risk_manager)
        self.health_monitor.register_component("decision_agent", self.decision_agent)
        self.health_monitor.register_component("portfolio_manager", self.portfolio_manager)
    
    def setup_assets(self):
        """Set up the assets to be traded."""
        # Add some crypto assets
        self.market_simulator.add_asset(
            "BTC/USD", AssetClass.CRYPTO, base_price=40000.0,
            initial_regime=MarketRegimeType.BULL,
            initial_volatility=VolatilityRegimeType.HIGH
        )
        
        self.market_simulator.add_asset(
            "ETH/USD", AssetClass.CRYPTO, base_price=2200.0,
            initial_regime=MarketRegimeType.VOLATILE_BULL,
            initial_volatility=VolatilityRegimeType.HIGH
        )
        
        # Add some stock assets
        self.market_simulator.add_asset(
            "AAPL", AssetClass.STOCK, base_price=180.0,
            initial_regime=MarketRegimeType.SIDEWAYS,
            initial_volatility=VolatilityRegimeType.MODERATE
        )
        
        self.market_simulator.add_asset(
            "MSFT", AssetClass.STOCK, base_price=350.0,
            initial_regime=MarketRegimeType.BULL,
            initial_volatility=VolatilityRegimeType.LOW
        )
        
        self.market_simulator.add_asset(
            "TSLA", AssetClass.STOCK, base_price=240.0,
            initial_regime=MarketRegimeType.VOLATILE,
            initial_volatility=VolatilityRegimeType.VERY_HIGH
        )
        
        # Add some forex assets
        self.market_simulator.add_asset(
            "EUR/USD", AssetClass.FOREX, base_price=1.09,
            initial_regime=MarketRegimeType.SIDEWAYS,
            initial_volatility=VolatilityRegimeType.LOW
        )
        
        # Add commodity
        self.market_simulator.add_asset(
            "GOLD", AssetClass.COMMODITY, base_price=2300.0,
            initial_regime=MarketRegimeType.BULL,
            initial_volatility=VolatilityRegimeType.MODERATE
        )
        
        # Generate historical data
        self.market_simulator.generate_historical_data(days=HISTORICAL_DAYS)
        
        # Update symbols list
        self.symbols = list(self.market_simulator.assets.keys())
        
        logger.info(f"Set up {len(self.symbols)} assets for trading: {', '.join(self.symbols)}")
    
    def subscribe_to_data(self):
        """Subscribe to data for all symbols."""
        # Subscribe to market data
        for symbol in self.symbols:
            self.market_data_provider.subscribe(symbol)
            self.sentiment_data_provider.subscribe(symbol)
            self.last_regime_check[symbol] = 0
        
        logger.info(f"Subscribed to data for {len(self.symbols)} symbols")
    
    def start_all_components(self):
        """Start all system components."""
        # Start health monitoring first
        self.health_monitor.start()
        
        # Start data providers
        self.market_data_provider.start()
        self.sentiment_data_provider.start()
        
        # Start agents
        self.technical_agent1.start()
        self.technical_agent2.start()
        self.sentiment_agent.start()
        self.risk_manager.start()
        self.decision_agent.start()
        
        # Start regime classifier
        self.regime_classifier.start() if hasattr(self.regime_classifier, 'start') else None
        
        logger.info("Started all system components")
    
    def stop_all_components(self):
        """Stop all system components."""
        # Stop agents first
        self.technical_agent1.stop()
        self.technical_agent2.stop()
        self.sentiment_agent.stop()
        self.risk_manager.stop()
        self.decision_agent.stop()
        
        # Stop data providers
        self.market_data_provider.stop()
        self.sentiment_data_provider.stop()
        
        # Stop regime classifier
        self.regime_classifier.stop() if hasattr(self.regime_classifier, 'stop') else None
        
        # Stop health monitoring last
        self.health_monitor.stop()
        
        logger.info("Stopped all system components")
    
    def start_autonomous_operation(self) -> bool:
        """Start autonomous operation."""
        if self.running:
            logger.warning("Autonomous operation already running")
            return False
        
        self.running = True
        self.trading_thread = threading.Thread(target=self._autonomous_operation_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("Started autonomous operation")
        return True
    
    def stop_autonomous_operation(self) -> bool:
        """Stop autonomous operation."""
        if not self.running:
            logger.warning("Autonomous operation not running")
            return False
        
        self.running = False
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5.0)
        
        logger.info("Stopped autonomous operation")
        return True
        
    def _autonomous_operation_loop(self):
        """Main autonomous operation loop."""
        self.cycle_count = 0
        
        while self.running and self.cycle_count < self.max_cycles:
            try:
                cycle_start_time = time.time()
                self.cycle_count += 1
                
                logger.info(f"\n===== TRADING CYCLE {self.cycle_count} =====")
                
                # Periodically inject failures to test resilience (after initial stability)
                if self.cycle_count > 20 and self.cycle_count % 50 == 0:
                    self._inject_random_failure()
                
                # Process market data
                self._process_market_data()
                
                # Process sentiment data
                self._process_sentiment_data()
                
                # Check and update market regimes
                self._update_market_regimes()
                
                # Update portfolio with latest prices
                self._update_portfolio_prices()
                
                # Generate signals and make decisions
                self._generate_and_process_signals()
                
                # Check system health
                system_health = self.health_monitor.get_system_health()
                if system_health['overall_status'] != HealthStatus.HEALTHY.value:
                    logger.warning(f"System health is {system_health['overall_status']}")
                    # Log component status
                    for component_id, component_info in self.health_monitor.components.items():
                        if component_info['last_status'] != HealthStatus.HEALTHY:
                            logger.warning(f"Component {component_id} status: {component_info['last_status'].value}")
                
                # Sleep to maintain cycle interval
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(0.0, self.cycle_interval - cycle_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Print cycle summary every 10 cycles
                if self.cycle_count % 10 == 0:
                    self._print_cycle_summary()
                
            except Exception as e:
                logger.error(f"Error in autonomous operation cycle {self.cycle_count}: {e}")
                # Continue operation despite errors
                time.sleep(1.0)  # Slow down on error
        
        logger.info(f"Completed {self.cycle_count} trading cycles")
    
    def _process_market_data(self):
        """Process incoming market data."""
        # Get the latest market data
        try:
            market_data = self.market_data_provider.get_latest_data(timeout=0.2)
            
            if market_data:
                symbol = market_data.get('symbol')
                data = market_data.get('data')
                
                if symbol and data:
                    # Buffer the data
                    if symbol not in self.data_buffer:
                        self.data_buffer[symbol] = {}
                    
                    self.data_buffer[symbol]['market_data'] = data
                    logger.debug(f"Processed market data for {symbol}: close=${data['close']:.2f}")
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _process_sentiment_data(self):
        """Process incoming sentiment data."""
        # Get the latest sentiment data
        try:
            sentiment_data = self.sentiment_data_provider.get_latest_data(timeout=0.2)
            
            if sentiment_data:
                symbol = sentiment_data.get('symbol')
                data = sentiment_data.get('data')
                
                if symbol and data:
                    # Buffer the data
                    if symbol not in self.data_buffer:
                        self.data_buffer[symbol] = {}
                    
                    self.data_buffer[symbol]['sentiment_data'] = data
                    logger.debug(f"Processed sentiment data for {symbol}: score={data['sentiment_score']:.2f}")
        except Exception as e:
            logger.error(f"Error processing sentiment data: {e}")
    
    def _update_market_regimes(self):
        """Update market regime classifications."""
        now = time.time()
        
        # For each symbol with market data
        for symbol in self.symbols:
            # Only check if we have enough data and enough time has passed since last check
            if symbol in self.data_buffer and now - self.last_regime_check.get(symbol, 0) >= self.regime_check_interval:
                try:
                    # Get historical data for classification
                    historical_data = self.market_data_provider.get_historical_data(symbol, days=30)
                    
                    if not historical_data.empty:
                        # Classify regime
                        classification = self.regime_classifier.classify_regime(symbol, historical_data)
                        
                        # Detect early transition signals
                        transition_signal = self.regime_classifier.detect_early_transition_signals(symbol, historical_data)
                        
                        # Buffer the classification
                        if symbol not in self.data_buffer:
                            self.data_buffer[symbol] = {}
                        
                        self.data_buffer[symbol]['regime_classification'] = classification
                        self.data_buffer[symbol]['transition_signal'] = transition_signal
                        
                        # Register with decision agent
                        self.decision_agent.register_regime_classification(symbol, classification)
                        
                        # Update last check time
                        self.last_regime_check[symbol] = now
                        
                        # Log interesting transitions and high confidence classifications
                        if classification['status'] == 'success':
                            regime = classification['regime']
                            confidence = classification['confidence']
                            vol_regime = classification['volatility_regime']
                            
                            if confidence > 0.8 or transition_signal > 0.7:
                                logger.info(f"Market regime for {symbol}: {regime} (vol: {vol_regime}) with confidence {confidence:.2f}")
                                if transition_signal > 0.7:
                                    logger.info(f"Strong transition signal detected for {symbol}: {transition_signal:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error updating market regime for {symbol}: {e}")
    
    def _update_portfolio_prices(self):
        """Update portfolio with latest prices."""
        # Collect latest prices
        prices = {}
        for symbol, data in self.data_buffer.items():
            if 'market_data' in data:
                market_data = data['market_data']
                if 'close' in market_data:
                    prices[symbol] = market_data['close']
        
        # Update portfolio
        if prices:
            self.portfolio_manager.update_prices(prices)
    
    def _generate_and_process_signals(self):
        """Generate trading signals and make decisions."""
        # Process each symbol with data
        for symbol in self.symbols:
            if symbol in self.data_buffer:
                data_bundle = self.data_buffer[symbol]
                
                # Skip if we don't have market data
                if 'market_data' not in data_bundle:
                    continue
                
                # Prepare data for agents
                agent_data = {
                    'symbol': symbol,
                    'market_data': data_bundle.get('market_data', {}),
                    'sentiment_data': data_bundle.get('sentiment_data', {}),
                    'regime_classification': data_bundle.get('regime_classification', {})
                }
                
                try:
                    # Generate technical signals
                    tech1_signal = self.technical_agent1.process(agent_data)
                    tech2_signal = self.technical_agent2.process(agent_data)
                    
                    # Register signals with decision agent
                    if tech1_signal.get('status') == 'success':
                        self.decision_agent.register_technical_signal(symbol, tech1_signal)
                    
                    if tech2_signal.get('status') == 'success':
                        self.decision_agent.register_technical_signal(symbol, tech2_signal)
                    
                    # Generate sentiment signal if data available
                    if 'sentiment_data' in data_bundle:
                        sentiment_signal = self.sentiment_agent.process(agent_data)
                        
                        # Register with decision agent
                        if sentiment_signal.get('status') == 'success':
                            self.decision_agent.register_sentiment_signal(symbol, sentiment_signal)
                    
                    # Have decision agent make a decision
                    decision_data = self.decision_agent.process({
                        'symbol': symbol,
                        'market_data': data_bundle.get('market_data', {})
                    })
                    
                    # Skip if decision was unsuccessful
                    if decision_data.get('status') != 'success':
                        continue
                    
                    # Apply risk management if decision is to trade
                    action = decision_data.get('action')
                    if action != TradingAction.HOLD.value:
                        risk_data = self.risk_manager.process({
                            'symbol': symbol,
                            'signal': decision_data,
                            'market_data': data_bundle.get('market_data', {})
                        })
                        
                        # Register risk assessment with decision agent
                        if risk_data.get('status') == 'success':
                            self.decision_agent.register_risk_assessment(symbol, risk_data)
                        
                        # Make final decision with risk assessment
                        final_decision = self.decision_agent.process({
                            'symbol': symbol,
                            'market_data': data_bundle.get('market_data', {})
                        })
                        
                        # Execute the decision if approved (or no LLM oversight)
                        if final_decision.get('status') == 'success':
                            if final_decision.get('llm_approved', True) or not self.llm_oversight:
                                # Execute the decision
                                execution_result = self.portfolio_manager.execute_decision(
                                    final_decision,
                                    current_price=data_bundle['market_data']['close']
                                )
                                
                                # Log execution result
                                if execution_result.get('status') == 'executed':
                                    logger.info(f"Executed {final_decision['action']} for {symbol}: "
                                               f"{execution_result['quantity']:.2f} units at "
                                               f"${execution_result['price']:.2f}, value=${execution_result['value']:.2f}")
                                elif execution_result.get('status') == 'failed':
                                    logger.warning(f"Failed to execute {final_decision['action']} for {symbol}: "
                                                 f"{execution_result.get('message', 'No error message')}")
                            else:
                                # Log LLM rejection
                                logger.info(f"LLM rejected {final_decision.get('original_action', action)} for {symbol}: "
                                           f"{final_decision.get('llm_feedback', 'No feedback provided')}")
                    
                except Exception as e:
                    logger.error(f"Error processing signals for {symbol}: {e}")
    
    def _inject_random_failure(self):
        """Inject a random component failure to test resilience."""
        # List of components that can fail
        components = [
            ("market_data_provider", self.market_data_provider),
            ("sentiment_data_provider", self.sentiment_data_provider),
            ("regime_classifier", self.regime_classifier),
            ("technical_agent1", self.technical_agent1),
            ("technical_agent2", self.technical_agent2),
            ("sentiment_agent", self.sentiment_agent),
            ("llm_oversight", self.llm_oversight)
        ]
        
        # Select a random component that's not already failed
        healthy_components = [(name, comp) for name, comp in components 
                              if name not in self.component_failure_info or 
                                 time.time() - self.component_failure_info.get(name, {}).get('failure_time', 0) > 120]
        
        if healthy_components:
            comp_name, component = random.choice(healthy_components)
            
            # Inject failure
            if hasattr(component, 'inject_failure'):
                component.inject_failure()
                logger.warning(f"Injected failure into {comp_name}")
                
                # Register failure
                self.component_failure_info[comp_name] = {
                    'failure_time': time.time(),
                    'recovery_scheduled': time.time() + random.uniform(30, 90)  # Schedule recovery in 30-90 seconds
                }
            else:
                logger.warning(f"Component {comp_name} does not support inject_failure")
        
        # Check for components to recover
        for comp_name, info in list(self.component_failure_info.items()):
            if time.time() >= info.get('recovery_scheduled', 0):
                # Find component
                for name, comp in components:
                    if name == comp_name and hasattr(comp, 'recover'):
                        comp.recover()
                        logger.info(f"Recovered {comp_name} from failure")
                        # Remove from failure list
                        del self.component_failure_info[comp_name]
                        break
    
    def _print_cycle_summary(self):
        """Print a summary of the current system state."""
        # Get portfolio summary
        portfolio = self.portfolio_manager.get_portfolio_summary()
        
        # Get system health
        health = self.health_monitor.get_system_health()
        
        # Get LLM oversight stats
        llm_stats = self.llm_oversight.get_stats()
        
        # Get classifier stats
        classifier_stats = self.regime_classifier.get_stats()
        
        # Print summary
        logger.info("\n-----------------------------------------")
        logger.info(f"CYCLE {self.cycle_count} SUMMARY:")
        logger.info("-----------------------------------------")
        
        # Portfolio summary
        logger.info(f"Portfolio Value: ${portfolio['nav']:.2f} (Cash: ${portfolio['cash']:.2f})")
        if 'total_return' in portfolio.get('performance', {}):
            metrics = portfolio['performance']
            logger.info(f"Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}, Max DD: {metrics['max_drawdown']:.2%}")
        
        # Positions summary
        if portfolio.get('positions', 0) > 0:
            logger.info("Current Positions:")
            for symbol, alloc in portfolio.get('allocation', {}).items():
                logger.info(f"  {symbol}: ${alloc['market_value']:.2f} ({alloc['weight']:.2%}), P&L: ${alloc['unrealized_pnl']:.2f}")
        
        # Health status
        logger.info(f"\nSystem Health: {health['overall_status']}")
        logger.info(f"  Components: {health['components_healthy']} healthy, {health['components_degraded']} degraded, "
                  f"{health['components_unhealthy']} unhealthy, {health['components_critical']} critical")
        
        # LLM oversight summary
        logger.info(f"\nLLM Oversight: {llm_stats['decisions_reviewed']} decisions reviewed, "
                  f"{llm_stats['decisions_approved']} approved ({llm_stats['approval_rate']:.2%})")
        
        # Market regime summary
        logger.info(f"\nMarket Regimes:")
        for symbol in self.symbols:
            if symbol in self.data_buffer and 'regime_classification' in self.data_buffer[symbol]:
                classification = self.data_buffer[symbol]['regime_classification']
                if 'regime' in classification:
                    transition = self.data_buffer[symbol].get('transition_signal', 0)
                    transition_str = f", transition signal: {transition:.2f}" if transition > 0.3 else ""
                    logger.info(f"  {symbol}: {classification['regime']} (vol: {classification['volatility_regime']}){transition_str}")
        
        logger.info("-----------------------------------------\n")

    def run_test(self, duration_minutes=30):
        """Run the complete autonomous test for the specified duration."""
        logger.info(f"\n==================================================")
        logger.info(f"STARTING AUTONOMOUS TRADING TEST - {duration_minutes} MINUTES")
        logger.info(f"==================================================\n")
        
        # Setup time-based test parameters
        self.max_cycles = int((duration_minutes * 60) / self.cycle_interval)
        
        try:
            # 1. Set up assets
            logger.info("Setting up market assets...")
            self.setup_assets()
            
            # 2. Subscribe to data
            logger.info("Subscribing to market and sentiment data...")
            self.subscribe_to_data()
            
            # 3. Start all components
            logger.info("Starting all system components...")
            self.start_all_components()
            
            # 4. Start autonomous operation
            logger.info("Starting autonomous operation...")
            self.start_autonomous_operation()
            
            # Wait for all cycles to complete
            if self.trading_thread:
                self.trading_thread.join()
                
            # Get final portfolio metrics
            portfolio = self.portfolio_manager.get_portfolio_summary()
            
            # Print final summary
            logger.info(f"\n==================================================")
            logger.info(f"AUTONOMOUS TRADING TEST COMPLETED - {self.cycle_count} CYCLES")
            logger.info(f"==================================================\n")
            
            logger.info(f"Final Portfolio Value: ${portfolio['nav']:.2f}")
            logger.info(f"Initial Investment: $100,000.00")
            logger.info(f"Total Return: {portfolio.get('performance', {}).get('total_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {portfolio.get('performance', {}).get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {portfolio.get('performance', {}).get('max_drawdown', 0):.2%}")
            logger.info(f"Win Rate: {portfolio.get('performance', {}).get('win_rate', 0):.2%}")
            
            # Print positions
            if portfolio.get('positions', 0) > 0:
                logger.info("\nFinal Positions:")
                for symbol, alloc in portfolio.get('allocation', {}).items():
                    logger.info(f"  {symbol}: ${alloc['market_value']:.2f} ({alloc['weight']:.2%}), P&L: ${alloc['unrealized_pnl']:.2f}")
            
            # Print transaction summary
            transactions = self.portfolio_manager.get_transaction_summary()
            if transactions:
                logger.info("\nTransaction Summary:")
                logger.info(f"  Total Trades: {transactions['total_trades']}")
                logger.info(f"  Buy Trades: {transactions['buy_trades']} (${transactions['buy_volume']:.2f})")
                logger.info(f"  Sell Trades: {transactions['sell_trades']} (${transactions['sell_volume']:.2f})")
                logger.info(f"  Realized P&L: ${transactions['realized_pnl']:.2f}")
            
            # Print component health summary
            health_summary = self.health_monitor.get_health_summary()
            logger.info("\nComponent Health Summary:")
            for component_id, status in health_summary.items():
                logger.info(f"  {component_id}: {status['status'].value}, uptime: {status['uptime']:.2f}s, failures: {status['failures']}")
            
            # Print LLM oversight summary
            llm_stats = self.llm_oversight.get_stats()
            logger.info("\nLLM Oversight Summary:")
            logger.info(f"  Decisions Reviewed: {llm_stats['decisions_reviewed']}")
            logger.info(f"  Decisions Approved: {llm_stats['decisions_approved']} ({llm_stats['approval_rate']:.2%})")
            logger.info(f"  Decisions Modified: {llm_stats['decisions_modified']} ({llm_stats.get('modification_rate', 0):.2%})")
            logger.info(f"  Decisions Rejected: {llm_stats['decisions_rejected']} ({llm_stats.get('rejection_rate', 0):.2%})")
            
            # Recovery statistics
            recovery_stats = self.health_monitor.get_recovery_stats()
            logger.info("\nResilience Summary:")
            logger.info(f"  Total Failures: {recovery_stats.get('total_failures', 0)}")
            logger.info(f"  Successful Recoveries: {recovery_stats.get('successful_recoveries', 0)}")
            logger.info(f"  Failed Recoveries: {recovery_stats.get('failed_recoveries', 0)}")
            logger.info(f"  Average Recovery Time: {recovery_stats.get('avg_recovery_time', 0):.2f}s")
            
            # Print market regime statistics
            logger.info("\nMarket Regime Statistics:")
            for symbol in self.symbols:
                regime_stats = self.regime_classifier.get_regime_history(symbol)
                if regime_stats:
                    regimes = ', '.join([f"{regime}: {count}" for regime, count in regime_stats.get('regimes', {}).items()])
                    logger.info(f"  {symbol}: {regimes}")
                    logger.info(f"    Transitions: {regime_stats.get('transitions', 0)}, Early Signals: {regime_stats.get('early_signals', 0)}")
            
            logger.info("\n===================================================")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in autonomous test: {e}")
            traceback.print_exc()
            return False
        finally:
            # Always stop all components
            self.stop_autonomous_operation()
            self.stop_all_components()
            logger.info("Stopped all system components")

#=============================================================================
# Main execution
#=============================================================================

def setup_logging():
    """Set up logging for the test script."""
    global logger
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = os.path.join('logs', f'autonomous_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = root_logger
    
    return logger

def main():
    """Main function to run the autonomous trading test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Autonomous Trading System Test')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in minutes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--cash', type=float, default=100000.0, help='Initial cash amount')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Set up logging
    logger = setup_logging()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create and run the test
    try:
        orchestrator = AutonomousSystemOrchestrator(initial_cash=args.cash, seed=args.seed)
        orchestrator.run_test(duration_minutes=args.duration)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    def subscribe(self, symbol: str) -> bool:
        """Subscribe to sentiment updates for a symbol."""
        if symbol in self.market_simulator.assets:
            self.subscribers.add(symbol)
            self.last_update[symbol] = 0
            logger.info(f"Subscribed to sentiment data for {symbol}")
            return True
        else:
            logger.warning(f"Cannot subscribe to sentiment for {symbol}: symbol not found")
            return False
    
    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from sentiment updates for a symbol."""
        if symbol in self.subscribers:
            self.subscribers.remove(symbol)
            if symbol in self.last_update:
                del self.last_update[symbol]
            logger.info(f"Unsubscribed from sentiment data for {symbol}")
            return True
        else:
            logger.warning(f"Cannot unsubscribe from sentiment for {symbol}: not subscribed")
            return False
    
    def start(self) -> bool:
        """Start the sentiment provider simulation."""
        if self.running:
            logger.warning("Sentiment data provider already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._data_generation_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started sentiment data provider simulation")
        return True
    
    def stop(self) -> bool:
        """Stop the sentiment provider simulation."""
        if not self.running:
            logger.warning("Sentiment data provider not running")
            return False
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        logger.info("Stopped sentiment data provider simulation")
        return True
    
    def inject_failure(self):
        """Inject a failure into the sentiment provider for testing resilience."""
        self.failure_mode = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning("Failure injected into sentiment data provider")
    
    def recover(self):
        """Recover from a failure."""
        self.failure_mode = False
        self.health_status = HealthStatus.HEALTHY
        logger.info("Sentiment data provider recovered from failure")
    
    def _data_generation_loop(self) -> None:
        """Background thread for generating sentiment data."""
        while self.running:
            try:
                # If in failure mode, either delay heavily or drop data
                if self.failure_mode:
                    if random.random() < 0.6:  # 60% chance to skip update
                        time.sleep(random.uniform(1.0, 3.0))
                        continue
                    time.sleep(random.uniform(1.0, 5.0))  # Slower response
                
                # Check each subscribed symbol for update
                current_time = time.time()
                
                for symbol in self.subscribers:
                    # Check if it's time to update this symbol
                    if current_time - self.last_update.get(symbol, 0) >= self.update_interval:
                        # Generate new sentiment data
                        sentiment = self.market_simulator.generate_sentiment(symbol)
                        
                        # Record update time
                        self.last_update[symbol] = current_time
                        
                        # Generate a sentiment update
                        self.data_queue.put({
                            'type': 'sentiment_data',
                            'symbol': symbol,
                            'data': {
                                'timestamp': self.market_simulator.current_date,
                                'sentiment_score': sentiment,
                                'confidence': random.uniform(0.7, 0.95),
                                'source': random.choice(['news', 'social_media', 'analyst_report'])
                            }
                        })
                
                # Sleep for a bit
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                logger.error(f"Error in sentiment data generation: {e}")
                self.health_status = HealthStatus.DEGRADED
                time.sleep(5.0)  # Slow down on error
    
    def get_latest_data(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get the latest sentiment data from the queue."""
        try:
            return self.data_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_historical_sentiment(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical sentiment data for a symbol."""
        if self.failure_mode and random.random() < 0.5:
            # Simulate data corruption or unavailability
            logger.warning(f"Failed to retrieve sentiment data for {symbol} due to simulated failure")
            return pd.DataFrame()  # Empty dataframe
            
        if symbol not in self.market_simulator.assets:
            logger.warning(f"Symbol {symbol} not found in simulation")
            return pd.DataFrame()
        
        # Get sentiment history
        sentiment_history = self.market_simulator.assets[symbol]['sentiment_history']
        
        # Convert to DataFrame
        if sentiment_history:
            dates, sentiments = zip(*sentiment_history)
            df = pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiments,
                'confidence': [random.uniform(0.7, 0.95) for _ in sentiments]
            })
            df.set_index('date', inplace=True)
            return df.tail(days)
        else:
            return pd.DataFrame()
    
    def check_health(self) -> HealthStatus:
        """Check the health of the sentiment provider."""
        return self.health_status
