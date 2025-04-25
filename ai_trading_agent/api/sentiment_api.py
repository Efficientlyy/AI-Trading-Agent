"""
Sentiment API Module

This module provides API routes for sentiment data that can be consumed by the frontend.
It integrates with the Alpha Vantage sentiment connector to provide real data,
and also includes mock data capabilities for development and testing.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import pandas as pd

from ..sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector

logger = logging.getLogger(__name__)

class SentimentAPI:
    """API for sentiment data that can be consumed by the frontend."""
    
    def __init__(self, use_mock: bool = False, mock_data_path: str = "mock_data/sentiment"):
        """
        Initialize the Sentiment API.
        
        Args:
            use_mock: Whether to use mock data instead of real API calls
            mock_data_path: Path to mock data directory
        """
        self.use_mock = use_mock
        self.mock_data_path = mock_data_path
        
        # If using real data, initialize the Alpha Vantage connector
        if not use_mock:
            self.connector = AlphaVantageSentimentConnector()
        
        # Ensure mock data directory exists
        if use_mock and not os.path.exists(mock_data_path):
            os.makedirs(mock_data_path, exist_ok=True)
    
    def _get_mock_file_path(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the path to a mock data file.
        
        Args:
            endpoint: API endpoint name
            params: Optional parameters to include in the filename
            
        Returns:
            Path to mock data file
        """
        if params:
            # Create a deterministic filename based on parameters
            param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
            filename = f"{endpoint}_{param_str}.json"
        else:
            filename = f"{endpoint}.json"
        
        return os.path.join(self.mock_data_path, filename)
    
    def _load_mock_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load mock data for a specific endpoint.
        
        Args:
            endpoint: API endpoint name
            params: Optional parameters to include in the filename
            
        Returns:
            Mock data as a dictionary
        """
        file_path = self._get_mock_file_path(endpoint, params)
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load mock data from {file_path}: {e}")
            # Return empty data structure
            if endpoint == 'sentiment_summary':
                return {'sentimentData': {}, 'timestamp': datetime.now().isoformat()}
            elif endpoint == 'historical_sentiment':
                return []
            return {}
    
    def _save_mock_data(self, endpoint: str, data: Dict[str, Any], 
                       params: Optional[Dict[str, Any]] = None) -> None:
        """
        Save data to a mock data file.
        
        Args:
            endpoint: API endpoint name
            data: Data to save
            params: Optional parameters to include in the filename
        """
        file_path = self._get_mock_file_path(endpoint, params)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved mock data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save mock data to {file_path}: {e}")
    
    def get_sentiment_summary(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get sentiment summary for multiple symbols.
        
        Args:
            symbols: List of symbols to get sentiment for (default: predefined list)
            
        Returns:
            Dictionary with sentiment summary
        """
        # Default symbols if none provided
        if symbols is None or len(symbols) == 0:
            symbols = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOGE"]
        
        if self.use_mock:
            # Load mock data
            mock_data = self._load_mock_data('sentiment_summary', {'symbols': '_'.join(symbols)})
            return mock_data
        
        try:
            # Get real data from Alpha Vantage
            data = self.connector.get_sentiment_summary(symbols)
            
            # Save as mock data for future use
            self._save_mock_data('sentiment_summary', data, {'symbols': '_'.join(symbols)})
            
            return data
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            # Return empty data
            return {'sentimentData': {}, 'timestamp': datetime.now().isoformat()}
    
    def get_historical_sentiment(self, symbol: str, timeframe: str = '1M') -> List[Dict[str, Any]]:
        """
        Get historical sentiment data for a specific symbol.
        
        Args:
            symbol: Symbol to get historical sentiment for
            timeframe: Timeframe ('1D', '1W', '1M', '3M', '1Y')
            
        Returns:
            List of dictionaries with historical sentiment data
        """
        if self.use_mock:
            # Load mock data
            mock_data = self._load_mock_data('historical_sentiment', 
                                           {'symbol': symbol, 'timeframe': timeframe})
            return mock_data
        
        try:
            # Get real data from Alpha Vantage
            data = self.connector.get_historical_sentiment(symbol, timeframe)
            
            # Save as mock data for future use
            self._save_mock_data('historical_sentiment', data, 
                               {'symbol': symbol, 'timeframe': timeframe})
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical sentiment for {symbol}: {e}")
            # Return empty data
            return []
    
    def generate_mock_data(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Generate mock data for all endpoints and symbols.
        This is useful for development and testing without hitting the API.
        
        Args:
            symbols: List of symbols to generate mock data for
            
        Returns:
            Dictionary with generated mock data references
        """
        # Default symbols if none provided
        if symbols is None or len(symbols) == 0:
            symbols = ["BTC", "ETH", "XRP", "ADA", "SOL", "DOGE"]
        
        # Temporarily disable mock mode to fetch real data
        original_mode = self.use_mock
        self.use_mock = False
        
        try:
            # Generate sentiment summary for all symbols
            summary = self.get_sentiment_summary(symbols)
            
            # Generate historical sentiment for each symbol
            historical = {}
            for symbol in symbols:
                for timeframe in ['1D', '1W', '1M', '3M']:
                    data = self.get_historical_sentiment(symbol, timeframe)
                    self._save_mock_data('historical_sentiment', data, 
                                       {'symbol': symbol, 'timeframe': timeframe})
                    historical[f"{symbol}_{timeframe}"] = len(data)
            
            result = {
                'summary': {
                    'file': self._get_mock_file_path('sentiment_summary', {'symbols': '_'.join(symbols)}),
                    'symbols': symbols,
                    'count': len(summary.get('sentimentData', {}))
                },
                'historical': historical
            }
            
            return result
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            return {'error': str(e)}
        finally:
            # Restore original mock mode
            self.use_mock = original_mode