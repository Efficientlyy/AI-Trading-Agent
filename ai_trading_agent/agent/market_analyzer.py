"""
Market analyzer module for analyzing market conditions and regimes.

This module provides tools for:
1. Detecting market regimes (trending, volatile, normal, crisis)
2. Calculating volatility metrics
3. Measuring trend strength
4. Identifying market turning points
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..common import logger

class MarketAnalyzer:
    """
    Analyzes market conditions and identifies market regimes.
    
    This class provides methods to analyze price data and determine
    market conditions such as volatility, trend strength, and overall
    market regime.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market analyzer.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - 'volatility_window': Window size for volatility calculation (default: 20)
                - 'trend_window': Window size for trend calculation (default: 50)
                - 'volatility_thresholds': Dictionary with 'low' and 'high' thresholds
                - 'trend_thresholds': Dictionary with 'weak' and 'strong' thresholds
        """
        self.config = config or {}
        
        # Set default parameters
        self.volatility_window = self.config.get('volatility_window', 20)
        self.trend_window = self.config.get('trend_window', 50)
        
        # Set default thresholds
        self.volatility_thresholds = self.config.get('volatility_thresholds', {
            'low': 0.15,  # Annualized volatility below 15%
            'high': 0.30  # Annualized volatility above 30%
        })
        
        self.trend_thresholds = self.config.get('trend_thresholds', {
            'weak': 0.001,  # Normalized slope below 0.1%
            'strong': 0.003  # Normalized slope above 0.3%
        })
        
        logger.info("MarketAnalyzer initialized")
    
    def calculate_volatility(self, prices: pd.Series) -> float:
        """
        Calculate the annualized volatility of a price series.
        
        Args:
            prices: Series of price data
        
        Returns:
            Annualized volatility as a float
        """
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Use the most recent window for calculation
        if len(returns) > self.volatility_window:
            returns = returns[-self.volatility_window:]
        
        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(252)  # Assuming daily data
        
        return volatility
    
    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """
        Calculate the strength of the trend in a price series.
        
        Args:
            prices: Series of price data
        
        Returns:
            Trend strength as a float (normalized slope)
        """
        if len(prices) < 2:
            return 0.0
        
        # Use the most recent window for calculation
        if len(prices) > self.trend_window:
            prices = prices[-self.trend_window:]
        
        # Calculate linear regression
        x = np.arange(len(prices))
        y = prices.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope by average price
        normalized_slope = slope / prices.mean()
        
        return normalized_slope
    
    def determine_volatility_level(self, volatility: float) -> str:
        """
        Determine the volatility level based on thresholds.
        
        Args:
            volatility: Calculated volatility value
        
        Returns:
            Volatility level as a string ('low', 'medium', or 'high')
        """
        if volatility < self.volatility_thresholds['low']:
            return 'low'
        elif volatility > self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def determine_trend_level(self, trend_strength: float) -> str:
        """
        Determine the trend strength level based on thresholds.
        
        Args:
            trend_strength: Calculated trend strength value
        
        Returns:
            Trend level as a string ('weak', 'medium', or 'strong')
        """
        # Use absolute value to ignore direction
        abs_trend = abs(trend_strength)
        
        if abs_trend < self.trend_thresholds['weak']:
            return 'weak'
        elif abs_trend > self.trend_thresholds['strong']:
            return 'strong'
        else:
            return 'medium'
    
    def determine_market_regime(self, volatility_level: str, trend_level: str) -> str:
        """
        Determine the market regime based on volatility and trend levels.
        
        Args:
            volatility_level: Volatility level ('low', 'medium', or 'high')
            trend_level: Trend level ('weak', 'medium', or 'strong')
        
        Returns:
            Market regime as a string ('normal', 'trending', 'volatile', or 'crisis')
        """
        if volatility_level == 'high' and trend_level == 'weak':
            return 'volatile'
        elif trend_level == 'strong' and volatility_level != 'high':
            return 'trending'
        elif volatility_level == 'high' and trend_level == 'strong':
            return 'crisis'
        else:
            return 'normal'
    
    def analyze_price_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze price data for multiple symbols and determine market conditions.
        
        Args:
            price_data: Dictionary mapping symbols to their price DataFrames
        
        Returns:
            Dictionary with market condition analysis
        """
        market_conditions = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            # Extract close prices
            close_prices = df['close']
            
            # Calculate metrics
            volatility = self.calculate_volatility(close_prices)
            trend_strength = self.calculate_trend_strength(close_prices)
            
            # Determine levels
            volatility_level = self.determine_volatility_level(volatility)
            trend_level = self.determine_trend_level(trend_strength)
            
            # Determine market regime
            regime = self.determine_market_regime(volatility_level, trend_level)
            
            # Store results
            market_conditions[symbol] = {
                'volatility': volatility,
                'volatility_level': volatility_level,
                'trend_strength': trend_strength,
                'trend_level': trend_level,
                'regime': regime
            }
        
        # Determine overall market regime (majority vote)
        regimes = [cond['regime'] for cond in market_conditions.values()]
        if regimes:
            from collections import Counter
            overall_regime = Counter(regimes).most_common(1)[0][0]
        else:
            overall_regime = 'normal'  # Default if no data
        
        return {
            'symbol_conditions': market_conditions,
            'overall_regime': overall_regime,
            'timestamp': datetime.now()
        }
    
    def detect_regime_change(self, 
                           current_conditions: Dict[str, Any],
                           previous_conditions: Dict[str, Any]) -> bool:
        """
        Detect if there has been a regime change between two market condition analyses.
        
        Args:
            current_conditions: Current market condition analysis
            previous_conditions: Previous market condition analysis
        
        Returns:
            True if there has been a regime change, False otherwise
        """
        if not previous_conditions:
            return False
        
        current_regime = current_conditions.get('overall_regime')
        previous_regime = previous_conditions.get('overall_regime')
        
        return current_regime != previous_regime
