"""
Simple data manager module for the AI Trading Agent.

This module provides a simple data manager implementation that can work
with both real and synthetic data.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..common import logger
from ..data_acquisition.synthetic_data_generator import generate_synthetic_data_for_backtest


class SimpleDataManager:
    """
    Simple data manager implementation that can work with both real and synthetic data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simple data manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.data = {}  # Dictionary to store data for each symbol
        self.current_index = 0
        self.timestamps = []
        
        logger.info("SimpleDataManager initialized")
    
    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with price data
        """
        if symbol in self.data:
            logger.warning(f"Data for {symbol} already exists. Overwriting.")
        
        self.data[symbol] = data
        
        # Update timestamps
        if not self.timestamps:
            self.timestamps = data.index.tolist()
        else:
            # Merge timestamps
            self.timestamps = sorted(list(set(self.timestamps) | set(data.index.tolist())))
        
        logger.info(f"Added data for {symbol} with {len(data)} rows")
    
    def load_synthetic_data(self, symbols: List[str], 
                           start_date: datetime, 
                           end_date: datetime,
                           timeframe: str = '1d',
                           include_sentiment: bool = True,
                           seed: Optional[int] = None) -> None:
        """
        Load synthetic data for backtesting.
        
        Args:
            symbols: List of symbols to generate data for
            start_date: Start date for the data
            end_date: End date for the data
            timeframe: Timeframe for the data ('1m', '5m', '15m', '1h', '4h', '1d')
            include_sentiment: Whether to include sentiment data
            seed: Random seed for reproducibility
        """
        logger.info(f"Generating synthetic data for {symbols} from {start_date} to {end_date}")
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data_for_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            include_sentiment=include_sentiment,
            include_regimes=True,
            seed=seed
        )
        
        # Add price data for each symbol
        price_data = synthetic_data['price_data']
        for symbol, data in price_data.items():
            self.add_data(symbol, data)
        
        # Store sentiment data if available
        if include_sentiment and 'sentiment_data' in synthetic_data:
            self.sentiment_data = synthetic_data['sentiment_data']
        
        # Store market regimes if available
        if 'market_regimes' in synthetic_data:
            self.market_regimes = synthetic_data['market_regimes']
        
        logger.info(f"Loaded synthetic data for {len(symbols)} symbols")
    
    def get_dates_in_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all dates in the specified range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of dates in the range
        """
        if not self.timestamps:
            logger.warning("No data available")
            return []
        
        # Filter timestamps in the range
        dates = [ts for ts in self.timestamps if start_date <= ts <= end_date]
        
        return dates
    
    def get_current_data(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get current data for all symbols at the specified timestamp.
        
        Args:
            timestamp: Timestamp to get data for. If None, use the current index.
        
        Returns:
            Dictionary mapping symbols to their current data
        """
        if not self.timestamps:
            logger.warning("No data available")
            return {}
        
        # If timestamp is provided, find the closest timestamp in the data
        if timestamp is not None:
            # Find the closest timestamp
            closest_idx = min(range(len(self.timestamps)), 
                             key=lambda i: abs(self.timestamps[i] - timestamp))
            current_ts = self.timestamps[closest_idx]
        else:
            # Use the current index
            if self.current_index >= len(self.timestamps):
                logger.warning("End of data reached")
                return {}
            
            current_ts = self.timestamps[self.current_index]
            self.current_index += 1
        
        # Get data for each symbol at the current timestamp
        current_data = {}
        
        for symbol, data in self.data.items():
            if current_ts in data.index:
                # Get the row for the current timestamp
                row = data.loc[current_ts]
                
                # Convert to dictionary if it's a Series
                if isinstance(row, pd.Series):
                    current_data[symbol] = row.to_dict()
                else:
                    current_data[symbol] = row
            else:
                logger.debug(f"No data for {symbol} at {current_ts}")
        
        # Add market regime if available
        if hasattr(self, 'market_regimes') and current_ts in self.market_regimes.index:
            regime_row = self.market_regimes.loc[current_ts]
            current_data['market_regime'] = regime_row.to_dict()
        
        return current_data
    
    def get_historical_data(self, end_date: Optional[datetime] = None, 
                           lookback_periods: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for all symbols up to the specified end date.
        
        Args:
            end_date: End date for the historical data. If None, use the current index.
            lookback_periods: Number of periods to look back
        
        Returns:
            Dictionary mapping symbols to their historical data
        """
        if not self.timestamps:
            logger.warning("No data available")
            return {}
        
        # If end_date is provided, find the closest timestamp in the data
        if end_date is not None:
            # Find the closest timestamp
            closest_idx = min(range(len(self.timestamps)), 
                             key=lambda i: abs(self.timestamps[i] - end_date))
            end_ts = self.timestamps[closest_idx]
        else:
            # Use the current index
            if self.current_index >= len(self.timestamps):
                logger.warning("End of data reached")
                return {}
            
            end_ts = self.timestamps[self.current_index - 1]
        
        # Find the start timestamp
        end_idx = self.timestamps.index(end_ts)
        start_idx = max(0, end_idx - lookback_periods + 1)
        start_ts = self.timestamps[start_idx]
        
        # Get historical data for each symbol
        historical_data = {}
        
        for symbol, data in self.data.items():
            # Filter data between start_ts and end_ts
            symbol_data = data[(data.index >= start_ts) & (data.index <= end_ts)]
            
            if not symbol_data.empty:
                historical_data[symbol] = symbol_data
            else:
                logger.debug(f"No historical data for {symbol} between {start_ts} and {end_ts}")
        
        return historical_data
    
    def get_latest_data(self) -> Dict[str, Any]:
        """
        Get the latest data for all symbols.
        
        Returns:
            Dictionary mapping symbols to their latest data
        """
        if not self.timestamps:
            logger.warning("No data available")
            return {}
        
        # Get the latest timestamp
        latest_ts = self.timestamps[-1]
        
        # Get data for each symbol at the latest timestamp
        latest_data = {}
        
        for symbol, data in self.data.items():
            if latest_ts in data.index:
                # Get the row for the latest timestamp
                row = data.loc[latest_ts]
                
                # Convert to dictionary if it's a Series
                if isinstance(row, pd.Series):
                    latest_data[symbol] = row.to_dict()
                else:
                    latest_data[symbol] = row
            else:
                logger.debug(f"No data for {symbol} at {latest_ts}")
        
        # Add market regime if available
        if hasattr(self, 'market_regimes') and latest_ts in self.market_regimes.index:
            regime_row = self.market_regimes.loc[latest_ts]
            latest_data['market_regime'] = regime_row.to_dict()
        
        return latest_data
    
    def reset(self) -> None:
        """Reset the data manager to the beginning of the data."""
        self.current_index = 0
        logger.info("Data manager reset to the beginning of the data")
