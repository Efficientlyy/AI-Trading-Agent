"""
Simplified test script that directly forces signal generation to validate our strategies.

This script uses a highly simplified approach that bypasses most of the complex
infrastructure to focus purely on verifying signal generation logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging

# Ensure the ai_trading_agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading_agent.agent.strategy_manager import (
    MovingAverageCrossStrategy, 
    RSIOverboughtOversoldStrategy,
    SignalDirection
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SignalTest")

def test_ma_cross_strategy():
    """
    Test the MA Cross strategy by directly creating the conditions for a signal.
    """
    print("\n" + "=" * 50)
    print("Testing MA Cross Strategy with Forced Conditions")
    
    # Create strategy instance
    strategy = MovingAverageCrossStrategy({
        "fast_ma": {"type": "ema", "period": 10},
        "slow_ma": {"type": "sma", "period": 30},
        "signal_threshold": 0.0  # Accept any signal
    })
    
    # Create a simple dataset with 100 days of data
    dates = [datetime.now() - timedelta(days=i) for i in range(100)]
    dates.reverse()
    
    # Create a basic price series
    close_prices = np.ones(100) * 100.0
    
    # Create a dataframe
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.01,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.ones(100) * 1000000
    })
    df.set_index('date', inplace=True)
    
    # Create market data dictionary
    market_data = {"TEST": df}
    
    # Create indicator values that will force a buy signal
    # Fast MA crosses above slow MA
    ema_values = pd.Series(np.ones(100) * 100.0, index=dates)
    sma_values = pd.Series(np.ones(100) * 100.0, index=dates)
    
    # Create crossover at second-to-last position
    ema_values.iloc[-2] = 99.0  # Fast MA below slow MA
    ema_values.iloc[-1] = 101.0  # Fast MA above slow MA (crossover)
    
    # Create indicators dictionary
    indicators = {
        "TEST": {
            "ema": {
                "10": ema_values
            },
            "sma": {
                "30": sma_values
            }
        }
    }
    
    # Generate signals
    signals = strategy.generate_signals(market_data, indicators, ["TEST"])
    
    # Print results
    print(f"Generated {len(signals)} signals from MA Cross Strategy")
    for signal in signals:
        symbol = signal["payload"]["symbol"]
        direction = signal["payload"]["signal"]
        confidence = signal["payload"]["confidence"]
        print(f"  Signal: {symbol}, Direction: {direction:+.4f}, Confidence: {confidence:.4f}")
    
    return signals

def test_rsi_strategy():
    """
    Test the RSI strategy by directly creating the conditions for a signal.
    """
    print("\n" + "=" * 50)
    print("Testing RSI Strategy with Forced Conditions")
    
    # Create strategy instance
    strategy = RSIOverboughtOversoldStrategy({
        "period": 14,
        "overbought": 70,
        "oversold": 30
    })
    
    # Create a simple dataset with 100 days of data
    dates = [datetime.now() - timedelta(days=i) for i in range(100)]
    dates.reverse()
    
    # Create a basic price series
    close_prices = np.ones(100) * 100.0
    
    # Create a dataframe
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.01,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.ones(100) * 1000000
    })
    df.set_index('date', inplace=True)
    
    # Create market data dictionary
    market_data = {
        "OVERSOLD_TEST": df.copy(),
        "OVERBOUGHT_TEST": df.copy()
    }
    
    # Create RSI values that will force signals
    # For oversold buy signal: RSI crosses from below 30 to above 30
    oversold_rsi = pd.Series(np.ones(100) * 40.0, index=dates)
    oversold_rsi.iloc[-2] = 25.0  # Below oversold threshold
    oversold_rsi.iloc[-1] = 35.0  # Above oversold threshold (crossover)
    
    # For overbought sell signal: RSI crosses from above 70 to below 70
    overbought_rsi = pd.Series(np.ones(100) * 60.0, index=dates)
    overbought_rsi.iloc[-2] = 75.0  # Above overbought threshold
    overbought_rsi.iloc[-1] = 65.0  # Below overbought threshold (crossover)
    
    # Create indicators dictionary
    indicators = {
        "OVERSOLD_TEST": {
            "rsi": oversold_rsi
        },
        "OVERBOUGHT_TEST": {
            "rsi": overbought_rsi
        }
    }
    
    # Generate signals
    signals = strategy.generate_signals(market_data, indicators, ["OVERSOLD_TEST", "OVERBOUGHT_TEST"])
    
    # Print results
    print(f"Generated {len(signals)} signals from RSI Strategy")
    for signal in signals:
        symbol = signal["payload"]["symbol"]
        direction = signal["payload"]["signal"]
        confidence = signal["payload"]["confidence"]
        condition = signal["payload"]["metadata"]["condition"]
        print(f"  Signal: {symbol}, Direction: {direction:+.4f}, Confidence: {confidence:.4f}, Condition: {condition}")
    
    return signals

if __name__ == "__main__":
    print("Forced Signal Generation Test")
    print("=" * 50)
    
    # Test individual strategies
    ma_signals = test_ma_cross_strategy()
    rsi_signals = test_rsi_strategy()
    
    total_signals = len(ma_signals) + len(rsi_signals)
    
    print("\n" + "=" * 50)
    print(f"Test Summary: Generated {total_signals} trading signals")
    print(f"  Moving Average Crossover: {len(ma_signals)} signals")
    print(f"  RSI Overbought/Oversold: {len(rsi_signals)} signals")
    
    # Add more detailed analysis of signals if needed
    if total_signals > 0:
        print("\nSignal Analysis:")
        
        if len(ma_signals) > 0:
            print("  MA Cross Signals:")
            for i, signal in enumerate(ma_signals):
                print(f"    Signal {i+1}: {signal['payload']['symbol']} - Direction: {signal['payload']['signal']:+.4f}")
                
        if len(rsi_signals) > 0:
            print("  RSI Signals:")
            for i, signal in enumerate(rsi_signals):
                print(f"    Signal {i+1}: {signal['payload']['symbol']} - Direction: {signal['payload']['signal']:+.4f}")
    else:
        print("\nNo signals were generated. Strategy implementation may need to be reviewed.")
