"""
Test script specifically focused on generating and validating trading signals.

This script creates precisely engineered test data designed to trigger
specific trading signals in our strategy implementations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import time
import logging

# Ensure the ai_trading_agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent
from ai_trading_agent.agent.strategy_manager import StrategyManager, MovingAverageCrossStrategy, RSIOverboughtOversoldStrategy
from ai_trading_agent.agent.indicator_engine import IndicatorEngine
from ai_trading_agent.agent.signal_test_data import generate_ma_crossover_data, generate_rsi_signal_data

# Create output directory
os.makedirs("output", exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalTest")

def test_ma_cross_strategy():
    """Test that MA Cross strategy generates signals correctly."""
    print("\n" + "=" * 50)
    print("Testing Moving Average Crossover Strategy")
    
    # Initialize components directly for precise testing
    strategy = MovingAverageCrossStrategy({
        "fast_ma": {"type": "ema", "period": 10},
        "slow_ma": {"type": "sma", "period": 30},
        "signal_threshold": 0.0  # Accept any signal
    })
    
    # Create test data that should trigger a buy signal
    buy_data = generate_ma_crossover_data(
        symbol="TEST_MA_BUY",
        fast_period=10,
        slow_period=30,
        signal_type="buy"
    )
    
    # Create test data that should trigger a sell signal
    sell_data = generate_ma_crossover_data(
        symbol="TEST_MA_SELL",
        fast_period=10,
        slow_period=30,
        signal_type="sell"
    )
    
    # Initialize indicator engine
    indicator_config = {
        "trend": {
            "ema": {"enabled": True, "periods": [10]},
            "sma": {"enabled": True, "periods": [30]},
        }
    }
    
    # Define a custom indicator engine that returns pandas Series
    class TestIndicatorEngine:
        def __init__(self):
            self.logger = logging.getLogger("TestIndicatorEngine")
            
        def calculate_all_indicators(self, market_data, symbols):
            results = {}
            
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                    
                df = market_data[symbol]
                results[symbol] = {}
                
                # Calculate EMA
                results[symbol]["ema"] = {}
                results[symbol]["ema"]["10"] = df["close"].ewm(span=10, adjust=False).mean()
                
                # Calculate SMA
                results[symbol]["sma"] = {}
                results[symbol]["sma"]["30"] = df["close"].rolling(window=30).mean()
                
            return results
    
    # Use our custom engine instead of the standard one
    indicator_engine = TestIndicatorEngine()
    # indicator_engine = IndicatorEngine(indicator_config)  # Commented out
    
    # Create market data dictionary
    market_data = {
        "TEST_MA_BUY": buy_data,
        "TEST_MA_SELL": sell_data
    }
    
    # Calculate indicators
    symbols = list(market_data.keys())
    indicators = indicator_engine.calculate_all_indicators(market_data, symbols)
    
    # Generate signals
    signals = strategy.generate_signals(market_data, indicators, symbols)
    
    # Display results
    print(f"Generated {len(signals)} signals from Moving Average Cross strategy")
    for signal in signals:
        symbol = signal["payload"]["symbol"]
        direction = signal["payload"]["signal"]
        confidence = signal["payload"]["confidence"]
        print(f"  Signal: {symbol}, Direction: {direction:+.4f}, Confidence: {confidence:.4f}")
    
    # Plot test data with signals
    for symbol in symbols:
        df = market_data[symbol]
        
        # Get matching signals for this symbol
        symbol_signals = [s for s in signals if s["payload"]["symbol"] == symbol]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price and moving averages
        ax.plot(df.index, df["close"], label="Price", linewidth=2)
        
        # Plot MA lines if available in indicators
        if symbol in indicators and "ema" in indicators[symbol] and "10" in indicators[symbol]["ema"]:
            ax.plot(df.index, indicators[symbol]["ema"]["10"], label="EMA(10)", linewidth=1.5, linestyle="--")
        if symbol in indicators and "sma" in indicators[symbol] and "30" in indicators[symbol]["sma"]:
            ax.plot(df.index, indicators[symbol]["sma"]["30"], label="SMA(30)", linewidth=1.5, linestyle="-.")
        
        # Mark signals on the chart
        for signal in symbol_signals:
            idx = df.index[-5]  # Signal position
            price = df.loc[idx, "close"]
            direction = signal["payload"]["signal"]
            
            if direction > 0:
                ax.scatter(idx, price * 0.98, s=200, marker="^", color="green", zorder=5)
                ax.annotate("BUY", (idx, price * 0.96), ha="center", fontweight="bold", color="green")
            else:
                ax.scatter(idx, price * 1.02, s=200, marker="v", color="red", zorder=5)
                ax.annotate("SELL", (idx, price * 1.04), ha="center", fontweight="bold", color="red")
        
        ax.set_title(f"MA Cross Strategy Test - {symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"output/signal_test_ma_cross_{symbol}.png")
        print(f"Chart saved to output/signal_test_ma_cross_{symbol}.png")
    
    return signals

def test_rsi_strategy():
    """Test that RSI strategy generates signals correctly."""
    print("\n" + "=" * 50)
    print("Testing RSI Overbought/Oversold Strategy")
    
    # Initialize strategy component directly
    strategy = RSIOverboughtOversoldStrategy({
        "period": 14,
        "overbought": 70,  # Standard overbought level
        "oversold": 30,    # Standard oversold level
    })
    
    # Create test data that should trigger an oversold (buy) signal
    oversold_data = generate_rsi_signal_data(
        symbol="TEST_RSI_OVERSOLD",
        rsi_period=14,
        signal_type="oversold"
    )
    
    # Create test data that should trigger an overbought (sell) signal
    overbought_data = generate_rsi_signal_data(
        symbol="TEST_RSI_OVERBOUGHT",
        rsi_period=14,
        signal_type="overbought"
    )
    
    # Initialize indicator engine with correct config structure
    indicator_config = {
        "momentum": {
            "rsi": {"enabled": True, "params": {"period": 14}},
        }
    }
    
    # Define a custom indicator engine that returns pandas Series
    class TestIndicatorEngine:
        def __init__(self):
            self.logger = logging.getLogger("TestIndicatorEngine")
            
        def calculate_all_indicators(self, market_data, symbols):
            results = {}
            
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                    
                df = market_data[symbol]
                results[symbol] = {}
                
                # Calculate RSI
                delta = df["close"].diff()
                gain = delta.copy()
                loss = delta.copy()
                gain[gain < 0] = 0
                loss[loss > 0] = 0
                loss = abs(loss)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # Prevent division by zero
                avg_loss = avg_loss.replace(0, 0.001)
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                results[symbol]["rsi"] = rsi
                
            return results
    
    # Use our custom engine
    indicator_engine = TestIndicatorEngine()
    # indicator_engine = IndicatorEngine(indicator_config)  # Commented out
    
    # Create market data dictionary
    market_data = {
        "TEST_RSI_OVERSOLD": oversold_data,
        "TEST_RSI_OVERBOUGHT": overbought_data
    }
    
    # Calculate indicators
    symbols = list(market_data.keys())
    indicators = indicator_engine.calculate_all_indicators(market_data, symbols)
    
    # Generate signals
    signals = strategy.generate_signals(market_data, indicators, symbols)
    
    # Display results
    print(f"Generated {len(signals)} signals from RSI strategy")
    for signal in signals:
        symbol = signal["payload"]["symbol"]
        direction = signal["payload"]["signal"]
        confidence = signal["payload"]["confidence"]
        condition = signal["payload"]["metadata"]["condition"]
        print(f"  Signal: {symbol}, Direction: {direction:+.4f}, Confidence: {confidence:.4f}, Condition: {condition}")
    
    # Plot test data with signals
    for symbol in symbols:
        df = market_data[symbol]
        
        # Get matching signals for this symbol
        symbol_signals = [s for s in signals if s["payload"]["symbol"] == symbol]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df["close"], label="Price", linewidth=2)
        
        # Mark signals on the chart
        for signal in symbol_signals:
            idx = df.index[-5]  # Signal position
            price = df.loc[idx, "close"]
            direction = signal["payload"]["signal"]
            
            if direction > 0:
                ax1.scatter(idx, price * 0.98, s=200, marker="^", color="green", zorder=5)
                ax1.annotate("BUY", (idx, price * 0.96), ha="center", fontweight="bold", color="green")
            else:
                ax1.scatter(idx, price * 1.02, s=200, marker="v", color="red", zorder=5)
                ax1.annotate("SELL", (idx, price * 1.04), ha="center", fontweight="bold", color="red")
        
        ax1.set_title(f"RSI Strategy Test - {symbol}")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        
        # Plot RSI on bottom subplot
        if "rsi" in indicators[symbol]:
            rsi_values = indicators[symbol]["rsi"]
            ax2.plot(df.index, rsi_values, color="purple", linewidth=1.5)
            
            # Add overbought/oversold lines
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='k', linestyle='-', alpha=0.2)
            
            # Add annotations
            ax2.text(df.index[0], 70, 'Overbought', verticalalignment='bottom', 
                     horizontalalignment='left', color='r')
            ax2.text(df.index[0], 30, 'Oversold', verticalalignment='top', 
                     horizontalalignment='left', color='g')
            
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI")
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Date")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"output/signal_test_rsi_{symbol}.png")
        print(f"Chart saved to output/signal_test_rsi_{symbol}.png")
    
    return signals

def test_technical_agent_with_signal_data():
    """Test the full Technical Analysis Agent with signal-generating data."""
    print("\n" + "=" * 50)
    print("Testing Technical Analysis Agent with Signal Data")
    
    # Create both types of test data
    test_data = {}
    
    # Moving average crossover test data
    test_data["AGENT_MA_BUY"] = generate_ma_crossover_data(
        symbol="AGENT_MA_BUY",
        fast_period=10,
        slow_period=30,
        signal_type="buy"
    )
    
    test_data["AGENT_MA_SELL"] = generate_ma_crossover_data(
        symbol="AGENT_MA_SELL",
        fast_period=10,
        slow_period=30,
        signal_type="sell"
    )
    
    # RSI test data
    test_data["AGENT_RSI_OVERSOLD"] = generate_rsi_signal_data(
        symbol="AGENT_RSI_OVERSOLD",
        rsi_period=14,
        signal_type="oversold"
    )
    
    test_data["AGENT_RSI_OVERBOUGHT"] = generate_rsi_signal_data(
        symbol="AGENT_RSI_OVERBOUGHT",
        rsi_period=14,
        signal_type="overbought"
    )
    
    # Initialize Technical Analysis Agent with default settings
    agent = TechnicalAnalysisAgent(
        agent_id_suffix="signal_test",
        name="Signal Test Agent",
        symbols=list(test_data.keys()),
        config_details={}
    )
    
    # Override the _extract_market_data method to use our test data
    original_extract_method = agent._extract_market_data
    
    def mock_extract_method(data):
        print(f"Using specialized signal test data with {len(test_data)} symbols")
        return test_data
    
    # Apply the override
    agent._extract_market_data = mock_extract_method
    
    # Process with our test data
    start_time = time.time()
    signals = agent.process()
    process_time = time.time() - start_time
    
    # Restore the original method
    agent._extract_market_data = original_extract_method
    
    # Display results
    print(f"\nGenerated {len(signals) if signals else 0} signals from Technical Analysis Agent")
    print(f"Processing time: {process_time:.4f} seconds")
    
    if signals:
        for signal in signals:
            symbol = signal["payload"]["symbol"]
            direction = signal["payload"]["signal"]
            confidence = signal["payload"]["confidence"]
            strategy = signal["payload"]["strategy"]
            print(f"  Signal: {symbol}, Direction: {direction:+.4f}, Confidence: {confidence:.4f}, Strategy: {strategy}")
    
    # Display metrics
    metrics = agent.get_component_metrics()
    print("\nComponent Metrics:")
    for component, comp_metrics in metrics.items():
        print(f"\n{component.upper()} METRICS:")
        for metric_name, metric_value in comp_metrics.items():
            print(f"  {metric_name}: {metric_value}")
    
    return agent, signals

if __name__ == "__main__":
    print("Technical Analysis Signal Generation Test")
    print("=" * 50)
    
    # Test individual strategies first
    ma_signals = test_ma_cross_strategy()
    rsi_signals = test_rsi_strategy()
    
    # Test the full Technical Analysis Agent
    agent, agent_signals = test_technical_agent_with_signal_data()
    
    total_signals = len(ma_signals) + len(rsi_signals) + len(agent_signals)
    
    print("\n" + "=" * 50)
    print(f"Test Summary: Generated {total_signals} trading signals across all tests")
    print(f"  Moving Average Crossover: {len(ma_signals)} signals")
    print(f"  RSI Overbought/Oversold: {len(rsi_signals)} signals")
    print(f"  Technical Analysis Agent: {len(agent_signals)} signals")
    print("\nAll output images saved to the 'output' directory")
