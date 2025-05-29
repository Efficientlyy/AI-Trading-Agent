"""
Market Data Validation System for the AI Trading Agent.

This module provides comprehensive validation for market data feeds to ensure data integrity,
detect anomalies, and provide circuit breaker functionality for unreliable data conditions.
Critical for real-time trading with production data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from ai_trading_agent.utils.circuit_breaker import EnhancedCircuitBreaker
from ai_trading_agent.data.models import MarketData, OHLCV, Tick
from ai_trading_agent.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataValidator:
    """Validates market data and detects anomalies in real-time data feeds."""
    
    def __init__(self):
        # Circuit breaker for each data source/symbol combination
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        
        # Historical stats for each symbol to detect anomalies
        self.historical_stats: Dict[str, Dict] = {}
        
        # Validation thresholds
        self.max_price_change_pct = float(settings.PRICE_MOVE_CIRCUIT_BREAKER)
        self.max_volatility_z_score = float(settings.VOLATILITY_CIRCUIT_BREAKER)
        
        # Cache of last valid data points 
        self.last_valid_data: Dict[str, Dict] = {}
        
        # Set up reconciliation sources
        self.reconciliation_sources = settings.RECONCILIATION_SOURCES.split(',') if hasattr(settings, 'RECONCILIATION_SOURCES') else []
        
        logger.info(f"Market Data Validator initialized with {len(self.reconciliation_sources)} reconciliation sources")
    
    def get_circuit_breaker(self, symbol: str, source: str) -> EnhancedCircuitBreaker:
        """Get or create a circuit breaker for a specific symbol and source."""
        key = f"{source}:{symbol}"
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = EnhancedCircuitBreaker()
        return self.circuit_breakers[key]
    
    def validate_tick(self, tick: Tick) -> Tuple[bool, str]:
        """
        Validate a single tick data point.
        
        Args:
            tick: The tick data to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        symbol = tick.symbol
        source = tick.source
        
        # 1. Check if circuit breaker is open
        circuit_breaker = self.get_circuit_breaker(symbol, source)
        if not circuit_breaker.is_allowed():
            return False, f"Circuit breaker open for {symbol} from {source}"
        
        # 2. Basic data integrity checks
        if tick.price <= 0:
            circuit_breaker.record_failure()
            return False, f"Invalid price: {tick.price}"
            
        if tick.timestamp > datetime.now() + timedelta(seconds=5):
            circuit_breaker.record_failure()
            return False, f"Future timestamp: {tick.timestamp}"
        
        # 3. Check for unrealistic price changes
        key = f"{source}:{symbol}"
        if key in self.last_valid_data:
            last_price = self.last_valid_data[key].get('price')
            if last_price:
                change_pct = abs(tick.price / last_price - 1) * 100
                if change_pct > self.max_price_change_pct:
                    circuit_breaker.record_failure()
                    return False, f"Excessive price change: {change_pct:.2f}%"
        
        # 4. Store as last valid data point
        self.last_valid_data[key] = {
            'price': tick.price,
            'timestamp': tick.timestamp,
            'volume': tick.volume
        }
        
        return True, "Valid"
    
    def validate_ohlcv(self, ohlcv: OHLCV) -> Tuple[bool, str]:
        """
        Validate OHLCV data for anomalies and integrity issues.
        
        Args:
            ohlcv: The OHLCV data to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        symbol = ohlcv.symbol
        source = ohlcv.source
        
        # 1. Check if circuit breaker is open
        circuit_breaker = self.get_circuit_breaker(symbol, source)
        if not circuit_breaker.is_allowed():
            return False, f"Circuit breaker open for {symbol} from {source}"
        
        # 2. Basic data integrity checks
        if ohlcv.close <= 0 or ohlcv.open <= 0 or ohlcv.high <= 0 or ohlcv.low <= 0:
            circuit_breaker.record_failure()
            return False, "Non-positive prices"
            
        if ohlcv.high < ohlcv.low:
            circuit_breaker.record_failure()
            return False, f"High ({ohlcv.high}) < Low ({ohlcv.low})"
            
        if ohlcv.high < ohlcv.close or ohlcv.high < ohlcv.open:
            circuit_breaker.record_failure()
            return False, "High price not >= all other prices"
            
        if ohlcv.low > ohlcv.close or ohlcv.low > ohlcv.open:
            circuit_breaker.record_failure()
            return False, "Low price not <= all other prices"
            
        if ohlcv.volume < 0:
            circuit_breaker.record_failure()
            return False, f"Negative volume: {ohlcv.volume}"
        
        # 3. Check for unusual volatility
        key = f"{source}:{symbol}"
        if key in self.historical_stats:
            typical_volatility = self.historical_stats[key]['typical_volatility']
            current_volatility = (ohlcv.high - ohlcv.low) / ohlcv.low * 100
            
            # Calculate z-score of current volatility
            z_score = (current_volatility - self.historical_stats[key]['mean_volatility']) / typical_volatility
            
            if abs(z_score) > self.max_volatility_z_score:
                circuit_breaker.record_failure()
                return False, f"Unusual volatility: z-score = {z_score:.2f}"
        
        return True, "Valid"
    
    def update_historical_stats(self, symbol: str, source: str, data: pd.DataFrame):
        """
        Update the historical statistics for a symbol.
        
        Args:
            symbol: The trading symbol
            source: Data source identifier
            data: Pandas DataFrame with historical OHLCV data
        """
        key = f"{source}:{symbol}"
        
        # Calculate historical volatility stats
        if 'high' in data.columns and 'low' in data.columns:
            # Calculate historical volatility for each period
            volatility = (data['high'] - data['low']) / data['low'] * 100
            
            # Store statistics
            self.historical_stats[key] = {
                'mean_volatility': volatility.mean(),
                'typical_volatility': volatility.std(),
                'max_volatility': volatility.max(),
                'data_points': len(data),
                'last_updated': datetime.now()
            }
            
            logger.info(f"Updated historical stats for {symbol} from {source}: "
                       f"Mean volatility: {volatility.mean():.2f}%, "
                       f"Std: {volatility.std():.2f}%")
    
    def reconcile_data_sources(self, symbol: str, timestamp: datetime, 
                              prices: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        Reconcile data from multiple sources to detect outliers.
        
        Args:
            symbol: The trading symbol
            timestamp: The timestamp of the data point
            prices: Dict mapping source names to price values
            
        Returns:
            Tuple of (is_reconciled, reconciled_prices)
        """
        if len(prices) < 2:
            # Not enough sources to reconcile
            return True, prices
        
        # Calculate mean and std of prices
        price_values = list(prices.values())
        mean_price = np.mean(price_values)
        std_price = np.std(price_values)
        
        # Detect and remove outliers (more than 3 standard deviations from mean)
        if std_price > 0:
            reconciled_prices = {}
            outliers = []
            
            for source, price in prices.items():
                z_score = abs(price - mean_price) / std_price
                
                if z_score > 3.0:
                    # This is an outlier
                    outliers.append((source, price, z_score))
                    circuit_breaker = self.get_circuit_breaker(symbol, source)
                    circuit_breaker.record_failure()
                else:
                    reconciled_prices[source] = price
            
            if outliers:
                outlier_info = ", ".join([f"{s}:{p:.2f} (z:{z:.2f})" for s, p, z in outliers])
                logger.warning(f"Detected outliers for {symbol} at {timestamp}: {outlier_info}")
                
                # If we've removed all prices as outliers, this is a problem
                if len(reconciled_prices) == 0:
                    logger.error(f"All price sources for {symbol} were outliers")
                    return False, prices
                
                return True, reconciled_prices
                
        return True, prices
    
    def implement_circuit_breakers(self, symbol: str, source: str, is_valid: bool):
        """
        Update the appropriate circuit breaker based on data validation.
        
        Args:
            symbol: The trading symbol
            source: Data source identifier
            is_valid: Whether the data point was valid
        """
        circuit_breaker = self.get_circuit_breaker(symbol, source)
        
        if not is_valid:
            circuit_breaker.record_failure()
        else:
            # If we were in HALF_OPEN state and got valid data, reset the circuit breaker
            if circuit_breaker.state == "HALF_OPEN":
                circuit_breaker.reset()
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict]:
        """Get the status of all circuit breakers."""
        status = {}
        for key, cb in self.circuit_breakers.items():
            status[key] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }
        return status
