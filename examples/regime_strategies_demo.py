"""
Regime-Specific Strategies Demo

This example demonstrates how different trading strategies perform
in various market regimes (trending, range-bound, volatile, etc.).
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import time

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.strategies.regime_strategies import BaseRegimeStrategy
from ai_trading_agent.strategies.range_bound_strategy import RangeBoundStrategy
from ai_trading_agent.strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy
from ai_trading_agent.strategies.mean_reversion_strategy import MeanReversionStrategy
from ai_trading_agent.strategies.regime_transition_strategy import RegimeTransitionStrategy
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier
from ai_trading_agent.data.mock_data_generator import MockDataGenerator, TrendType, PatternType


def generate_regime_specific_data():
    """Generate test data for different market regimes."""
    # Create a mock data generator
    mock_gen = MockDataGenerator(seed=42)
    
    # Generate data for different market regimes
    end_date = datetime.now()
    
    market_data = {}
    
    # Trending market (bullish)
    market_data["TRENDING_BULLISH"] = mock_gen.generate_data(
        symbol="TRENDING_BULL",
        periods=200,
        trend_type=TrendType.BULLISH,
        volatility=0.015,
        end_date=end_date
    )
    
    # Trending market (bearish)
    market_data["TRENDING_BEARISH"] = mock_gen.generate_data(
        symbol="TRENDING_BEAR",
        periods=200,
        trend_type=TrendType.BEARISH,
        volatility=0.015,
        end_date=end_date
    )
    
    # Range-bound market
    market_data["RANGE_BOUND"] = mock_gen.generate_data(
        symbol="RANGING",
        periods=200,
        trend_type=TrendType.SIDEWAYS,
        volatility=0.01,
        end_date=end_date
    )
    
    # Volatile market
    market_data["VOLATILE"] = mock_gen.generate_data(
        symbol="VOLATILE",
        periods=200,
        trend_type=TrendType.VOLATILE,
        volatility=0.025,
        end_date=end_date
    )
    
    # Transition market (starts range-bound, then trends)
    # Create manually to ensure the transition
    dates = pd.date_range(end=end_date, periods=200, freq='D')
    df = pd.DataFrame(index=dates)
    
    # First part: ranging
    first_half = len(dates) // 2
    df.loc[dates[:first_half], 'close'] = 150 + np.random.normal(0, 5, first_half)
    
    # Second part: trending
    df.loc[dates[first_half:], 'close'] = np.linspace(
        df.iloc[first_half-1]['close'], 
        df.iloc[first_half-1]['close'] + 50, 
        len(dates) - first_half
    ) + np.random.normal(0, 3, len(dates) - first_half)
    
    # Fill in other columns
    df['open'] = df['close'].shift(1).fillna(df['close'][0])
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(1, 3, len(dates))
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(1, 3, len(dates))
    df['volume'] = np.random.uniform(1000, 5000, len(dates))
    
    market_data["TRANSITION"] = df
    
    return market_data


def run_strategy_comparison(market_data):
    """Run different strategies on various market regimes and compare results."""
    # Create strategy configurations
    strategy_config = {
        "lookback_window": 20,
        "sensitivity": 1.0,
        "confirmation_threshold": 1,
        "enable_filters": True
    }
    
    # Initialize strategies
    strategies = {
        "Range Bound": RangeBoundStrategy(strategy_config),
        "Volatility Breakout": VolatilityBreakoutStrategy(strategy_config),
        "Mean Reversion": MeanReversionStrategy(strategy_config),
        "Regime Transition": RegimeTransitionStrategy(strategy_config)
    }
    
    # Initialize regime classifier
    regime_classifier = MarketRegimeClassifier({
        "lookback_window": 50,
        "smoothing_window": 3
    })
    
    # Store results for each regime and strategy
    results = {}
    
    # Process each market regime
    for regime_name, data in market_data.items():
        print(f"\nAnalyzing {regime_name} market...")
        
        # Detect regime
        detected_regime = regime_classifier.classify_regime(data)
        print(f"Detected regime: {detected_regime}")
        
        # Dictionary to store signals for this regime
        regime_signals = {}
        
        # Run each strategy
        for strategy_name, strategy in strategies.items():
            # Generate signals
            signals = strategy.generate_signals({regime_name: data})
            
            if regime_name in signals:
                signal_data = signals[regime_name]
                signal_strength = signal_data.get("signal", 0)
                signal_direction = signal_data.get("direction", "neutral")
                
                print(f"  {strategy_name} Strategy: {signal_direction.upper()} signal with strength {signal_strength:.4f}")
                
                # Store signal for later comparison
                regime_signals[strategy_name] = {
                    "strength": abs(signal_strength),
                    "direction": signal_direction,
                    "signal_type": signal_data.get("signal_type", "Unknown"),
                    "metadata": signal_data.get("metadata", {})
                }
            else:
                print(f"  {strategy_name} Strategy: No signal generated")
                regime_signals[strategy_name] = {
                    "strength": 0,
                    "direction": "neutral",
                    "signal_type": "No Signal",
                    "metadata": {}
                }
        
        # Store results for this regime
        results[regime_name] = {
            "data": data,
            "detected_regime": detected_regime,
            "signals": regime_signals
        }
    
    return results


def plot_results(results):
    """Plot the market data and strategy performance for each regime."""
    regimes = list(results.keys())
    strategies = ["Range Bound", "Volatility Breakout", "Mean Reversion", "Regime Transition"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(regimes), 2, figsize=(15, 4 * len(regimes)))
    
    # Set the figure title
    fig.suptitle("Market Regimes and Strategy Performance", fontsize=16)
    
    # Process each regime
    for i, regime_name in enumerate(regimes):
        regime_results = results[regime_name]
        data = regime_results["data"]
        
        # Left subplot: Price chart
        ax1 = axes[i, 0]
        ax1.plot(data.index, data['close'], label='Close Price')
        ax1.set_title(f"{regime_name} Market (Detected: {regime_results['detected_regime']})")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Right subplot: Strategy performance bar chart
        ax2 = axes[i, 1]
        
        # Extract signal strengths
        strengths = [regime_results["signals"][strategy]["strength"] for strategy in strategies]
        
        # Create color map based on signal direction
        colors = []
        for strategy in strategies:
            direction = regime_results["signals"][strategy]["direction"]
            if direction == "buy":
                colors.append("green")
            elif direction == "sell":
                colors.append("red")
            else:
                colors.append("gray")
        
        # Create bar chart
        ax2.bar(strategies, strengths, color=colors)
        ax2.set_title(f"Strategy Signal Strength for {regime_name} Market")
        ax2.set_ylabel("Signal Strength")
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the figure
    plt.savefig("regime_strategies_comparison.png", dpi=300, bbox_inches="tight")
    print("\nSaved comparison plot to 'regime_strategies_comparison.png'")
    
    # Show the plot
    plt.show()


def print_strategy_insights(results):
    """Print insights about which strategies work best in each regime."""
    print("\n=== Strategy Effectiveness by Market Regime ===\n")
    
    for regime_name, regime_results in results.items():
        print(f"Market Regime: {regime_name} (Detected as: {regime_results['detected_regime']})")
        
        # Find the best strategy for this regime
        signals = regime_results["signals"]
        best_strategy = max(signals.items(), key=lambda x: x[1]["strength"])
        
        print(f"  Best Strategy: {best_strategy[0]} (Signal Strength: {best_strategy[1]['strength']:.4f})")
        print(f"  Signal Direction: {best_strategy[1]['direction'].upper()}")
        
        # Print some strategy-specific insights
        if best_strategy[0] == "Range Bound":
            print("  Key Components:")
            metadata = best_strategy[1]["metadata"]
            print(f"    Oscillator Component: {metadata.get('oscillator_component', 'N/A'):.4f}")
            print(f"    Support/Resistance Component: {metadata.get('support_resistance_component', 'N/A'):.4f}")
        
        elif best_strategy[0] == "Volatility Breakout":
            print("  Key Components:")
            metadata = best_strategy[1]["metadata"]
            print(f"    Volatility Expansion Component: {metadata.get('volatility_expansion_component', 'N/A'):.4f}")
            print(f"    Bollinger Band Breakout Component: {metadata.get('bb_breakout_component', 'N/A'):.4f}")
        
        elif best_strategy[0] == "Mean Reversion":
            print("  Key Components:")
            metadata = best_strategy[1]["metadata"]
            print(f"    Z-Score Component: {metadata.get('zscore_component', 'N/A'):.4f}")
            print(f"    MA Deviation Component: {metadata.get('ma_deviation_component', 'N/A'):.4f}")
        
        elif best_strategy[0] == "Regime Transition":
            print("  Key Components:")
            metadata = best_strategy[1]["metadata"]
            print(f"    Volatility Change Component: {metadata.get('volatility_change_component', 'N/A'):.4f}")
            print(f"    Trend Reversal Component: {metadata.get('trend_reversal_component', 'N/A'):.4f}")
        
        print()


def run_demo():
    """Run the regime strategies demonstration."""
    print("\n=== Regime-Specific Strategies Demo ===\n")
    
    print("Generating market data for different regimes...")
    market_data = generate_regime_specific_data()
    print(f"Generated data for {len(market_data)} market regimes")
    
    print("\nRunning strategies on each market regime...")
    results = run_strategy_comparison(market_data)
    
    # Print insights
    print_strategy_insights(results)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
