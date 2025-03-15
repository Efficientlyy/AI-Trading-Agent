"""Utility functions for the Market Regime Detection API."""

import uuid
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Union
import os
from pathlib import Path
import matplotlib.pyplot as plt

from api.models import MarketData
from api import config

# Initialize logger
logger = logging.getLogger(__name__)

def prepare_market_data(market_data: Union[MarketData, List[MarketData]]) -> Dict:
    """
    Convert API market data to internal format.
    
    Args:
        market_data: Market data from API request
        
    Returns:
        Dict with prepared data
    """
    # Handle single asset
    if isinstance(market_data, MarketData):
        dates = [point.date for point in market_data.data]
        prices = [point.price for point in market_data.data]
        volumes = [point.volume if point.volume is not None else 0 for point in market_data.data]
        
        # Calculate returns if not provided
        if market_data.data[0].return_value is None:
            returns = [0.0]  # Start with 0.0 as float
            for i in range(1, len(prices)):
                ret = (prices[i] / prices[i-1]) - 1 if prices[i-1] > 0 else 0.0
                returns.append(ret)
        else:
            returns = [float(point.return_value) if point.return_value is not None else 0.0 for point in market_data.data]
        
        return {
            "symbol": market_data.symbol,
            "dates": dates,
            "prices": prices,
            "volumes": volumes,
            "returns": returns
        }
    
    # Handle multiple assets
    else:
        result = {}
        for asset in market_data:
            asset_data = prepare_market_data(asset)
            result[asset.symbol] = asset_data
        return result

def save_visualization(fig, request_id: str, method_name: str) -> str:
    """
    Save visualization to file and return URL.
    
    Args:
        fig: Matplotlib figure
        request_id: Request ID
        method_name: Method name
        
    Returns:
        URL to visualization
    """
    filename = f"{request_id}_{method_name}.png"
    filepath = config.VISUALIZATION_DIR / filename
    fig.savefig(filepath)
    plt.close(fig)
    return f"/static/visualizations/{filename}"

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

def log_request(endpoint: str, request_data: Any) -> None:
    """Log API request details."""
    logger.info(f"Request to {endpoint}: {request_data}")

def log_response(endpoint: str, response_data: Any, execution_time: float) -> None:
    """Log API response details."""
    logger.info(f"Response from {endpoint} (execution time: {execution_time:.4f}s)")

def format_error(error: Exception) -> Dict:
    """
    Format error for API response.
    
    Args:
        error: Exception
        
    Returns:
        Formatted error dict
    """
    return {
        "error": str(error),
        "type": type(error).__name__,
        "timestamp": datetime.now().isoformat()
    }

def validate_market_data(market_data: Union[MarketData, List[MarketData]]) -> bool:
    """
    Validate market data for completeness and consistency.
    
    Args:
        market_data: Market data from API request
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(market_data, MarketData):
            # Check if data is not empty
            if not market_data.data:
                logger.error("Market data is empty")
                return False
            
            # Check if symbol is provided
            if not market_data.symbol:
                logger.error("Symbol is missing")
                return False
            
            # Check if dates and prices are provided
            for i, point in enumerate(market_data.data):
                if point.date is None or point.price is None:
                    logger.error(f"Date or price missing at index {i}")
                    return False
            
            return True
        
        # Handle list of market data
        else:
            if not market_data:
                logger.error("Market data list is empty")
                return False
            
            return all(validate_market_data(asset) for asset in market_data)
    
    except Exception as e:
        logger.error(f"Error validating market data: {str(e)}")
        return False 