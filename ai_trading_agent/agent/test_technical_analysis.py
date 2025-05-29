"""
Test module for the Technical Analysis Agent.

This script demonstrates the usage of the Technical Analysis Agent
and tests the mock/real data toggle functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import os
import sys
import json
import time

# Ensure the ai_trading_agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  

from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent, DataMode
from ai_trading_agent.data.mock_data_generator import MockDataGenerator, PatternType, TrendType
from ai_trading_agent.agent.signal_test_data import generate_signal_test_data

# Create output directory for visualizations
os.makedirs("output", exist_ok=True)


def test_mock_data_generation():
    """Test the mock data generation functionality with various patterns and trends."""
    print("\n" + "=" * 50)
    print("Testing Mock Data Generation...")
    
    # Initialize mock data generator with fixed seed for reproducibility
    generator = MockDataGenerator(seed=42)
    
    # Define patterns to test
    patterns = [
        (PatternType.HEAD_AND_SHOULDERS, "Head and Shoulders Pattern", TrendType.BULLISH),
        (PatternType.DOUBLE_TOP, "Double Top Pattern", TrendType.BULLISH),
        (PatternType.DOUBLE_BOTTOM, "Double Bottom Pattern (Placeholder)", TrendType.BEARISH),
        (PatternType.NONE, "Bullish Trend (No Pattern)", TrendType.BULLISH),
        (PatternType.NONE, "Bearish Trend (No Pattern)", TrendType.BEARISH),
        (PatternType.NONE, "Volatile Market (No Pattern)", TrendType.VOLATILE)
    ]
    
    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(len(patterns), 1, figure=fig)
    
    all_mock_data = {}
    
    for i, (pattern, title, trend) in enumerate(patterns):
        # Generate data with the pattern and trend
        df = generator.generate_data(
            symbol=f"TEST{i+1}",
            periods=200,
            trend_type=trend,
            pattern=pattern,
            volatility=0.015,
            end_date=datetime.now(),
            ensure_enough_data=True
        )
        
        # Store the generated data
        all_mock_data[f"TEST{i+1}"] = df
        
        # Plot the data
        ax = fig.add_subplot(gs[i])
        ax.plot(df.index, df['close'], linewidth=2)
        
        # Add volume as bars at the bottom
        volume_ax = ax.twinx()
        volume_ax.bar(df.index, df['volume'], alpha=0.3, color='gray', width=1.5)
        volume_ax.set_ylim(0, df['volume'].max() * 3)
        volume_ax.set_ylabel('Volume')
        
        # Set title and labels
        ax.set_title(f"{title} - {trend.value.capitalize()}")
        ax.set_ylabel('Price')
        
        # Only show x-label for the last plot
        if i == len(patterns) - 1:
            ax.set_xlabel('Date')
        
        # Format x-axis to show dates nicely
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        print(f"Generated {len(df)} periods of mock data with {pattern.value} pattern and {trend.value} trend")
    
    plt.tight_layout()
    plt.savefig("output/mock_data_patterns.png", dpi=300)
    print("Mock data visualization saved to output/mock_data_patterns.png")
    
    return generator, all_mock_data


def visualize_technical_analysis(market_data, signals, mode_name, filename):
    """Create a visualization of the technical analysis results."""
    if not market_data or not signals:
        print(f"No data to visualize for {mode_name} mode")
        return
    
    # Create a figure for each symbol
    for symbol, df in market_data.items():
        # Filter signals for this symbol
        symbol_signals = [s for s in signals if s['payload']['symbol'] == symbol]
        if not symbol_signals:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(df.index, df['close'], label='Close Price')
        
        # Add signals to the chart
        # Handle ISO format timestamps and ensure they're compatible with matplotlib
        signal_dates = []
        for s in symbol_signals:
            try:
                # Parse the ISO format timestamp
                date = datetime.fromisoformat(s['payload']['timestamp'])
                signal_dates.append(date)
            except (ValueError, TypeError):
                # If timestamp isn't valid ISO format, use the most recent date in the dataframe
                signal_dates.append(df.index[-1])
                
        signal_prices = [s['payload']['price_at_signal'] for s in symbol_signals]
        signal_values = [s['payload']['signal'] for s in symbol_signals]
        
        # Plot buy signals (positive) and sell signals (negative)
        buy_dates = [date for i, date in enumerate(signal_dates) if signal_values[i] > 0]
        buy_prices = [price for i, price in enumerate(signal_prices) if signal_values[i] > 0]
        
        sell_dates = [date for i, date in enumerate(signal_dates) if signal_values[i] < 0]
        sell_prices = [price for i, price in enumerate(signal_prices) if signal_values[i] < 0]
        
        ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', label='Buy Signal')
        ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', label='Sell Signal')
        
        # Plot signal confidence as a heatmap in the second subplot
        confidence_values = [abs(s['payload']['confidence']) for s in symbol_signals]
        colors = ['red' if v < 0 else 'green' for v in signal_values]
        
        ax2.bar(signal_dates, confidence_values, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('Signal Confidence')
        ax2.set_title('Signal Confidence (Red = Sell, Green = Buy)')
        
        # Set titles and labels
        ax1.set_title(f"{symbol} Technical Analysis - {mode_name.upper()} DATA MODE")
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Format x-axis
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"output/{filename}_{symbol}.png", dpi=300)
        plt.close()
        
        print(f"Visualization for {symbol} in {mode_name} mode saved to output/{filename}_{symbol}.png")


def create_test_data_for_signals():
    """Create specific test data designed to trigger trading signals."""
    print("\n" + "=" * 50)
    print("Creating Test Data Specifically for Signal Generation...")
    
    # Generate our specialized test data
    test_data = generate_signal_test_data()
    
    # Log what we created
    for symbol, df in test_data.items():
        print(f"Created {symbol} test data with {len(df)} periods designed to trigger signals")
    
    return test_data


def test_data_mode_toggle(test_data=None):
    """Test the Technical Analysis Agent with both mock and real data, demonstrating the toggle functionality."""
    print("\n" + "=" * 50)
    print("Testing Technical Analysis Agent with Data Mode Toggle...")
    
    # Determine symbols based on input
    if test_data is None:
        # Default symbols if nothing provided
        symbols = ["AAPL", "MSFT", "GOOGL"]
        test_data = None
    else:
        # Extract symbols from test data dictionary
        symbols = list(test_data.keys())
    
    print(f"Testing with symbols: {symbols}")
    
    # Initialize agent with real data mode
    agent = TechnicalAnalysisAgent(
        agent_id_suffix="test",
        name="Test Technical Agent",
        symbols=symbols,
        config_details={
            "data_mode": "real",  # Start with real data mode
            "indicators": {
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
            },
            "strategies": {
                "strategies": {
                    "ma_cross": {
                        "enabled": True,
                        "fast_ma": {"type": "ema", "period": 9},
                        "slow_ma": {"type": "ema", "period": 21},
                        "signal_threshold": 0.0005  # Lower threshold to catch more signals
                    },
                    "rsi_ob_os": {
                        "enabled": True,
                        "period": 14,
                        "overbought": 70,
                        "oversold": 30
                    }
                }
            }
        }
    )
    
    print(f"Agent initialized in {agent.data_mode.value} data mode")
    print(f"Active strategies: {list(agent.strategy_manager.strategies.keys())}")
    
    # Process with real data mode (will likely use placeholder signals since we don't have real data)
    start_time = time.time()
    signals_real = agent.process()
    real_processing_time = time.time() - start_time
    
    # Get the market data that was used (could be empty in this test)
    real_market_data = agent._extract_market_data(None)
    
    print(f"Generated {len(signals_real) if signals_real else 0} signals in real mode")
    print(f"Real mode processing time: {real_processing_time:.4f} seconds")
    
    # Visualize the real data results
    visualize_technical_analysis(real_market_data, signals_real, "real", "real_mode")
    
    # Switch to mock data mode - simulating UI toggle action
    print("\nSwitching to mock data mode... (UI toggle action)")
    agent.update_data_mode("mock")
    print(f"Data mode changed to: {agent.data_mode.value}")
    
    # If we have specific test data, use it instead of the generator
    if test_data is not None:
        # Process with mock data mode but use our specific test data
        start_time = time.time()
        
        # We need to override the _extract_market_data method temporarily to use our test data
        original_extract_method = agent._extract_market_data
        
        # Define a replacement method that returns our test data
        def mock_extract_method(data):
            print(f"Using specialized signal test data with {len(test_data)} symbols")
            return test_data
        
        # Apply the override
        agent._extract_market_data = mock_extract_method
        
        # Process with our test data
        signals_mock = agent.process()
        mock_processing_time = time.time() - start_time
        
        # Restore the original method
        agent._extract_market_data = original_extract_method
        
        # Use our test data
        mock_market_data = test_data
    else:
        # Process with the agent's built-in mock data generator
        start_time = time.time()
        signals_mock = agent.process()
        mock_processing_time = time.time() - start_time
        
        # Get the mock market data that was generated
        mock_market_data = agent._extract_market_data(None)
    
    print(f"Generated {len(signals_mock) if signals_mock else 0} signals in mock mode")
    print(f"Mock mode processing time: {mock_processing_time:.4f} seconds")
    
    # Display signal details
    if signals_mock:
        print("\nSample signals generated:")
        for i, signal in enumerate(signals_mock[:2]):  # Show first 2 signals
            print(f"\nSignal {i+1}:")
            print(f"  Symbol: {signal['payload']['symbol']}")
            print(f"  Strategy: {signal['payload']['strategy']}")
            print(f"  Direction: {'BUY' if signal['payload']['signal'] > 0 else 'SELL'}")
            print(f"  Confidence: {signal['payload']['confidence']:.4f}")
            print(f"  Price: {signal['payload']['price_at_signal']:.2f}")
            if 'indicators_used' in signal['payload']:
                print("  Indicators Used:")
                for indicator, details in signal['payload']['indicators_used'].items():
                    print(f"    {indicator}: {details}")
            if 'metadata' in signal['payload']:
                print("  Metadata:")
                for key, value in signal['payload']['metadata'].items():
                    print(f"    {key}: {value}")
    
    # Visualize the mock data results
    visualize_technical_analysis(mock_market_data, signals_mock, "mock", "mock_mode")
    
    # Switch back to real data mode - simulating UI toggle action again
    print("\nSwitching back to real data mode... (UI toggle action)")
    agent.update_data_mode("real")
    print(f"Data mode changed to: {agent.data_mode.value}")
    
    # Show component metrics
    print("\nComponent Metrics:")
    component_metrics = agent.get_component_metrics()
    for component, metrics in component_metrics.items():
        print(f"\n{component.upper()} METRICS:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    return agent, real_market_data, mock_market_data, signals_real, signals_mock


def create_toggle_demonstration():
    """Create a visual demonstration of the mock/real data toggle functionality."""
    print("\n" + "=" * 50)
    print("Creating Mock/Real Toggle Demonstration...")
    
    # Create a figure to demonstrate the toggle
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create two example datasets - one for mock and one for real data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=99)  # 100 days of data
    dates = pd.date_range(start=start_date, end=end_date, periods=100)
    
    # Real data - more noisy and irregular
    real_data = np.cumsum(np.random.normal(0, 1, 100)) + 100
    real_data += np.sin(np.linspace(0, 4*np.pi, 100)) * 5  # Add some cyclicality
    
    # Mock data - cleaner with a recognizable pattern
    x = np.linspace(0, 4*np.pi, 100)
    mock_data = 100 + 10 * np.sin(x) + np.cumsum(np.random.normal(0, 0.3, 100))
    
    # Plot both lines
    ax.plot(dates, real_data, 'b-', linewidth=2, label='Real Market Data')
    ax.plot(dates, mock_data, 'r-', linewidth=2, label='Mock Data (Patterns for Testing)')
    
    # Add toggle switch illustration - use a specific date to avoid issues
    toggle_x = dates[30]  # Use the 30th day in the range
    toggle_y = max(np.max(real_data), np.max(mock_data)) + 10
    
    # Add toggle annotation with fixed offsets
    ax.annotate('UI Toggle Switch', 
                xy=(toggle_x, toggle_y), 
                xytext=(0, 15),  # Offset in points
                textcoords='offset points',
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7))
    
    # Add Mock/Real labels with date-independent positioning
    mock_x = dates[25]  # Slightly to the left of toggle
    real_x = dates[35]  # Slightly to the right of toggle
    ax.text(mock_x, toggle_y, 'Mock', fontsize=10, ha='right', va='center')
    ax.text(real_x, toggle_y, 'Real', fontsize=10, ha='left', va='center')
    
    # Add explanatory text
    ax.text(0.5, 0.02, 
            "The Technical Analysis Agent supports toggling between real market data and \n"
            "mock data with known patterns for testing and demonstration purposes.\n"
            "This allows developers to test pattern detection and signal generation \n"
            "against predictable data before deploying to real markets.",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    # Set title and labels
    ax.set_title('Technical Analysis Agent: Mock/Real Data Toggle Functionality')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("output/mock_real_toggle_demonstration.png", dpi=300)
    print("Toggle demonstration saved to output/mock_real_toggle_demonstration.png")


if __name__ == "__main__":
    print("Technical Analysis Agent Test")
    print("=" * 50)
    
    # Create a visual demonstration of the toggle functionality
    create_toggle_demonstration()
    
    # Test general mock data generation with various patterns
    generator, general_mock_data = test_mock_data_generation()
    
    # Create specialized test data designed to trigger signals
    signal_test_data = create_test_data_for_signals()
    
    # Test technical agent with toggle between mock and real data
    # Use our specially crafted signal test data
    agent, real_data, mock_data, real_signals, mock_signals = test_data_mode_toggle(signal_test_data)
    
    print("\nTest completed successfully!")
    print("All output images saved to the 'output' directory")
    print(f"Generated {len(mock_signals) if mock_signals else 0} trading signals with the Technical Analysis Agent.")
    print("\nNext steps for Technical Analysis Agent development:")
    print("1. Implement Pattern Detection in Phase 4")
    print("2. Implement Market Regime Classification in Phase 5")
    print("3. Develop Advanced UI Integration in Phase 6")
    print("4. Complete the production-ready documentation in Phase 7")
    print("\nThe agent is now ready for testing and further development!")
