"""
Mock/Real Data Toggle Demo

This example demonstrates how to use the mock/real data toggle functionality
in the Technical Analysis Agent framework.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.config.data_source_config import get_data_source_config
from ai_trading_agent.data.data_source_factory import get_data_source_factory
from ai_trading_agent.data.mock_data_generator import MockDataGenerator, TrendType


def generate_test_data(symbols):
    """Generate test market data for demonstration."""
    # Create a mock data generator
    mock_gen = MockDataGenerator(seed=42)
    
    # Generate data for different timeframes
    timeframes = ["1h", "4h", "1d"]
    end_date = datetime.now()
    
    market_data = {}
    for symbol in symbols:
        market_data[symbol] = {}
        
        # Generate different trend types for different symbols
        if symbol == "BTC-USD":
            trend = TrendType.BULLISH
        elif symbol == "ETH-USD":
            trend = TrendType.BEARISH
        else:
            trend = TrendType.VOLATILE
            
        for tf in timeframes:
            # Generate more periods for smaller timeframes
            if tf == "1h":
                periods = 500
            elif tf == "4h":
                periods = 250
            else:
                periods = 200
                
            # Generate data with specific trend characteristics
            data = mock_gen.generate_data(
                symbol=symbol,
                periods=periods,
                trend_type=trend,
                volatility=0.02,
                end_date=end_date
            )
            
            market_data[symbol][tf] = data
    
    return market_data


def print_data_source_info(agent):
    """Print information about the current data source."""
    data_source = agent.get_data_source_type()
    print(f"\n=== Using {data_source.upper()} Data ===")
    print(f"Data source type: {data_source}")
    metrics = agent.get_metrics()
    if "data_source" in metrics:
        print(f"Metrics data source: {metrics['data_source']}")


def run_demo():
    """Run the mock/real data toggle demonstration."""
    print("\n=== Mock/Real Data Toggle Demo ===\n")
    
    # Define test symbols
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    # Create test configuration
    config = {
        "strategies": [
            {
                "name": "MA Crossover",
                "indicators": ["sma_fast", "sma_slow", "rsi"],
                "parameters": {
                    "sma_fast_period": 10,
                    "sma_slow_period": 30,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            }
        ],
        "timeframes": ["1h", "4h", "1d"],
        "ml_validator": {
            "enabled": True,
            "min_confidence": 0.6
        },
        "data_source": {
            "use_mock_data": True,  # Start with mock data
            "mock_data_settings": {
                "volatility": 0.02,
                "trend_strength": 0.4,
                "seed": 42
            }
        }
    }
    
    # Create the technical analysis agent
    agent = AdvancedTechnicalAnalysisAgent(config)
    
    # Get data source factory and config
    data_source_factory = get_data_source_factory()
    data_source_config = get_data_source_config()
    
    # Generate test data
    print("Generating test market data...")
    market_data = generate_test_data(symbols)
    print(f"Generated data for {len(symbols)} symbols with {len(config['timeframes'])} timeframes")
    
    # Run analysis with mock data
    print_data_source_info(agent)
    print("Running analysis with current data source...")
    signals = agent.analyze(market_data, symbols)
    print(f"Generated {len(signals)} signals\n")
    
    # Display some basic metrics
    metrics = agent.get_metrics()
    print("Current metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Toggle to real data
    print("\n--- Toggling data source ---")
    new_source = agent.toggle_data_source()
    print(f"Data source toggled to: {new_source}")
    
    # Run analysis again with the new data source
    print_data_source_info(agent)
    print("Running analysis with new data source...")
    signals = agent.analyze(market_data, symbols)
    print(f"Generated {len(signals)} signals\n")
    
    # Display updated metrics
    metrics = agent.get_metrics()
    print("Updated metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Toggle back to original state
    print("\n--- Toggling data source back ---")
    new_source = agent.toggle_data_source()
    print(f"Data source toggled to: {new_source}")
    print_data_source_info(agent)
    
    # Demonstrate updating configuration
    print("\n--- Updating mock data settings ---")
    new_settings = {
        "mock_data_settings": {
            "volatility": 0.03,  # Increase volatility
            "trend_strength": 0.6,  # Increase trend strength
            "seed": 42
        }
    }
    data_source_config.update_config(new_settings)
    print("Mock data settings updated")
    print("  volatility: 0.03")
    print("  trend_strength: 0.6")
    
    # Run analysis with updated settings
    print("\nRunning analysis with updated mock data settings...")
    signals = agent.analyze(market_data, symbols)
    print(f"Generated {len(signals)} signals\n")
    
    print("Demo complete!")


if __name__ == "__main__":
    run_demo()
