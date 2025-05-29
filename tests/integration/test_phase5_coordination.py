"""
Simplified Test for Phase 5 Coordination Components

This test script validates the functionality of Phase 5 coordination components:
1. Cross-Strategy Coordination
2. Performance Attribution

The test uses simulated market data and mocks the ML components.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import coordination components
from ai_trading_agent.coordination.strategy_coordinator import StrategyCoordinator
from ai_trading_agent.coordination.performance_attribution import PerformanceAttributor
from ai_trading_agent.agent.coordination_manager import CoordinationManager

# Create a RichSignal class for testing
# Function to create rich signals for testing (matching the Dict[str, Any] type in the real codebase)
def create_rich_signal(action, quantity, price, metadata=None):
    """Create a RichSignal dictionary with the given parameters."""
    signal = {
        'action': action,  # 1 for buy, -1 for sell, 0 for hold
        'quantity': quantity,
        'price': price,
    }
    
    if metadata:
        signal['metadata'] = metadata
    else:
        signal['metadata'] = dict()
        
    return signal

# Helper function to generate synthetic market data
def generate_market_data(symbols, days=100):
    """Generate synthetic market data for testing."""
    data = {}
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for symbol in symbols:
        # Set initial price
        initial_price = np.random.randint(50, 200)
        
        # Create price series with random walk
        returns = np.random.normal(0.0005, 0.015, days)
        
        # Convert returns to prices
        prices = initial_price * (1 + returns).cumprod()
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.006, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.006, days))),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, days),
            'returns': returns
        }, index=dates)
        
        # Ensure high > low
        df['high'] = np.maximum(df[['open', 'close', 'high']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close', 'low']].min(axis=1), df['low'])
        
        data[symbol] = df
        
    return data

# Generate signals for testing
def generate_test_signals(data, strategy_type="trend"):
    signals = {}
    
    for symbol, df in data.items():
        signals[symbol] = {}
        latest_date = df.index[-1]
        
        if strategy_type == "trend":
            # Trend strategy: buy if price increasing
            action = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
        else:
            # Mean reversion strategy: buy if price decreased
            action = 1 if df['close'].iloc[-1] < df['close'].iloc[-2] else -1
            
        signal = RichSignal(
            action=action,
            quantity=100,
            price=df['close'].iloc[-1],
            metadata={
                "strategy": strategy_type,
                "confidence": 0.7
            }
        )
        
        signals[symbol][latest_date] = signal
            
    return signals

def main():
    # Set up test environment
    print("\n===== Testing Phase 5 Coordination Components =====")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic market data
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    market_data = generate_market_data(symbols, days=100)
    
    print(f"Generated synthetic market data for {len(symbols)} symbols")
    
    # Configure coordination components
    coord_config = {
        "strategies": ["TrendStrategy", "MeanReversionStrategy"],
        "lookback_periods": 20,
        "conflict_resolution_method": "performance_weighted",
        "capital_allocation_method": "dynamic"
    }
    
    attr_config = {
        "strategies": ["TrendStrategy", "MeanReversionStrategy"],
        "metrics": ["returns", "sharpe_ratio", "max_drawdown", "win_rate"],
        "output_path": os.path.join(output_dir, "attribution")
    }
    
    # Create coordination components
    strategy_coordinator = StrategyCoordinator(coord_config)
    performance_attributor = PerformanceAttributor(attr_config)
    
    # Create coordination manager
    mgr_config = {
        "strategies": [
            {"name": "TrendStrategy"},
            {"name": "MeanReversionStrategy"}
        ],
        "coordination_config": coord_config,
        "attribution_config": attr_config,
        "output_path": os.path.join(output_dir, "coordinator")
    }
    coordination_manager = CoordinationManager(mgr_config)
    
    print("Initialized coordination components")
    
    # 1. Test strategy coordination
    print("\n----- Testing Strategy Coordination -----")
    
    # Generate signals for both strategies
    trend_signals = generate_test_signals(market_data, "trend")
    reversion_signals = generate_test_signals(market_data, "mean_reversion")
    
    print(f"Generated signals for {len(trend_signals)} symbols")
    
    # Record signals with coordination manager
    coordination_manager.record_strategy_signals("TrendStrategy", trend_signals)
    coordination_manager.record_strategy_signals("MeanReversionStrategy", reversion_signals)
    
    # Generate coordinated signals
    coordinated_signals = coordination_manager.coordinate_signals()
    
    print(f"Generated coordinated signals for {len(coordinated_signals)} symbols")
    
    # 2. Test performance attribution
    print("\n----- Testing Performance Attribution -----")
    
    # Record performance metrics
    timestamp = datetime.now().isoformat()
    
    for symbol in symbols:
        # Trend strategy performance
        trend_metrics = {
            "returns": 0.02 + (np.random.random() - 0.5) * 0.01,
            "sharpe_ratio": 1.8 + (np.random.random() - 0.5) * 0.5,
            "max_drawdown": 0.03 + (np.random.random() - 0.5) * 0.01,
            "win_rate": 0.6 + (np.random.random() - 0.5) * 0.1
        }
        coordination_manager.record_performance(
            "TrendStrategy", symbol, trend_metrics, timestamp
        )
        
        # Mean reversion strategy performance
        reversion_metrics = {
            "returns": 0.015 + (np.random.random() - 0.5) * 0.01,
            "sharpe_ratio": 1.5 + (np.random.random() - 0.5) * 0.5,
            "max_drawdown": 0.04 + (np.random.random() - 0.5) * 0.01,
            "win_rate": 0.55 + (np.random.random() - 0.5) * 0.1
        }
        coordination_manager.record_performance(
            "MeanReversionStrategy", symbol, reversion_metrics, timestamp
        )
        
        # Combined performance
        combined_metrics = {
            "returns": 0.025 + (np.random.random() - 0.5) * 0.01,
            "sharpe_ratio": 2.0 + (np.random.random() - 0.5) * 0.5,
            "max_drawdown": 0.025 + (np.random.random() - 0.5) * 0.01,
            "win_rate": 0.65 + (np.random.random() - 0.5) * 0.1
        }
        coordination_manager.record_combined_performance(
            symbol, combined_metrics, timestamp
        )
    
    print(f"Recorded performance metrics for {len(symbols)} symbols")
    
    # Generate attribution report
    report_path = coordination_manager.generate_attribution_report("json")
    print(f"Generated attribution report: {report_path}")
    
    # Get strategy allocations
    trend_allocation = coordination_manager.get_strategy_allocation("TrendStrategy")
    reversion_allocation = coordination_manager.get_strategy_allocation("MeanReversionStrategy")
    
    print(f"Capital allocation:")
    print(f"  - TrendStrategy: {trend_allocation:.2f}")
    print(f"  - MeanReversionStrategy: {reversion_allocation:.2f}")
    
    # Get strategy recommendations
    recommendations = coordination_manager.get_strategy_recommendations()
    
    print("\nStrategy improvement recommendations:")
    for strategy, recs in recommendations.items():
        if recs:
            print(f"  {strategy}:")
            for rec in recs:
                print(f"    - {rec}")
    
    # Test complete
    print("\n===== Phase 5 Coordination Components Test Complete =====")
    
    # Return success
    return coordinated_signals, report_path


if __name__ == '__main__':
    main()
