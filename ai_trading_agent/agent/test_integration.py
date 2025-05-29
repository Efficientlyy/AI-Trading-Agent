"""
Integration test for the enhanced Technical Analysis Agent components.

This script tests the integration of multi-timeframe analysis, ML signal validation,
and adaptive parameter tuning with the existing Technical Analysis Agent.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root to the path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.agent.multi_timeframe import MultiTimeframeStrategy
from ai_trading_agent.ml.signal_validator import MLSignalValidator
from ai_trading_agent.ml.adaptive_parameters import AdaptiveParameterManager, MarketRegimeClassifier
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.common.utils import get_logger
from ai_trading_agent.agent.indicator_engine import IndicatorEngine


def generate_simple_mock_data(days=100, volatility=0.02, trend_strength=0.5, start_price=50000.0):
    """Generate simple mock OHLCV data for testing."""
    # Set up date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Generate price data
    returns = np.random.normal(0, volatility, days)
    if trend_strength > 0:
        # Add trend component
        direction = np.random.choice([-1, 1])
        trend = np.linspace(0, trend_strength * direction * 0.2, days)
        returns += trend
    
    # Calculate price from returns
    price = start_price * (1 + returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame(index=dates)
    df['close'] = price
    
    # Generate open, high, low
    daily_volatility = volatility * price
    df['open'] = df['close'].shift(1).fillna(start_price)
    
    # Generate high/low with some randomness
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, daily_volatility)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, daily_volatility)
    
    # Generate volume
    df['volume'] = np.random.uniform(800000, 1200000, days) * (1 + np.abs(returns) / volatility)
    
    # Round values
    df['open'] = np.round(df['open'], 2)
    df['high'] = np.round(df['high'], 2)
    df['low'] = np.round(df['low'], 2)
    df['close'] = np.round(df['close'], 2)
    df['volume'] = np.round(df['volume']).astype(int)
    
    return df


def test_ml_signal_validator():
    """Test the ML signal validator component."""
    print("\n=== Testing ML Signal Validator ===")
    
    # Generate mock data
    mock_data = generate_simple_mock_data(days=50)
    
    # Initialize validator
    validator = MLSignalValidator({
        "min_confidence": 0.6
    })
    
    # Create a mock signal
    mock_signal = {
        "type": "technical_signal",
        "payload": {
            "symbol": "BTC/USD",
            "signal": 0.75,  # Strong buy signal
            "strategy": "MA_Cross",
            "price_at_signal": 50000.0,
            "timestamp": datetime.now().isoformat(),
            "indicators_used": {
                "sma_9": 49500.0,
                "sma_21": 49000.0
            },
            "metadata": {
                "volatility_percent": 0.02,
                "volume_ratio": 1.5,
                "volume_confirmation": "high",
                "confirmation_count": 2
            }
        }
    }
    
    # Calculate indicators for validation
    indicator_engine = IndicatorEngine({})
    indicators = indicator_engine.calculate_all_indicators({"BTC/USD": mock_data}, [
        {"name": "rsi", "params": {"window": 14}},
        {"name": "ema", "params": {"window": 9}},
        {"name": "sma", "params": {"window": 21}}
    ])
    
    # Validate the signal
    is_valid, confidence, validation_metadata = validator.validate_signal(
        mock_signal, {"BTC/USD": mock_data}, {"BTC/USD": indicators}
    )
    
    # Print results
    print(f"Signal validated: {is_valid}")
    print(f"Confidence score: {confidence:.4f}")
    print("Validation metadata:")
    for key, value in validation_metadata.items():
        print(f"  {key}: {value}")
    
    return validator


def test_adaptive_parameters():
    """Test the adaptive parameter functionality."""
    print("\n=== Testing Adaptive Parameters ===")
    
    # Generate different types of market data
    trend_data = generate_simple_mock_data(days=50, volatility=0.01, trend_strength=0.8)
    range_data = generate_simple_mock_data(days=50, volatility=0.02, trend_strength=0.1)
    volatile_data = generate_simple_mock_data(days=50, volatility=0.04, trend_strength=0.3)
    
    # Initialize parameter manager
    param_manager = AdaptiveParameterManager()
    
    # Initialize indicator engine
    indicator_engine = IndicatorEngine({})
    
    # Calculate indicators for each market type
    indicator_list = [
        {"name": "adx", "params": {"window": 14}},
        {"name": "atr", "params": {"window": 14}},
        {"name": "sma", "params": {"window": 20}}
    ]
    
    trend_indicators = indicator_engine.calculate_all_indicators({"BTC/USD": trend_data}, indicator_list)
    range_indicators = indicator_engine.calculate_all_indicators({"BTC/USD": range_data}, indicator_list)
    volatile_indicators = indicator_engine.calculate_all_indicators({"BTC/USD": volatile_data}, indicator_list)
    
    # Test market regime classification
    regime_classifier = MarketRegimeClassifier()
    
    trend_regime = regime_classifier.classify_regime(trend_data, trend_indicators.get("BTC/USD", {}))
    range_regime = regime_classifier.classify_regime(range_data, range_indicators.get("BTC/USD", {}))
    volatile_regime = regime_classifier.classify_regime(volatile_data, volatile_indicators.get("BTC/USD", {}))
    
    print("Market Regime Classification:")
    print(f"  Trend data: {trend_regime['regime']} (confidence: {trend_regime['confidence']:.4f})")
    print(f"  Range data: {range_regime['regime']} (confidence: {range_regime['confidence']:.4f})")
    print(f"  Volatile data: {volatile_regime['regime']} (confidence: {volatile_regime['confidence']:.4f})")
    
    # Get adapted parameters for each market regime
    trend_params = param_manager.get_strategy_parameters("MA_Cross", trend_data, trend_indicators.get("BTC/USD", {}))
    range_params = param_manager.get_strategy_parameters("MA_Cross", range_data, range_indicators.get("BTC/USD", {}))
    volatile_params = param_manager.get_strategy_parameters("MA_Cross", volatile_data, volatile_indicators.get("BTC/USD", {}))
    
    print("\nAdaptive Parameters for MA_Cross:")
    
    # Compare key parameters across regimes
    param_keys = ["fast_period", "slow_period", "confirmation_periods", "min_divergence", "max_volatility"]
    print(f"{'Parameter':<20} {'Trending':<15} {'Ranging':<15} {'Volatile':<15}")
    print("-" * 65)
    
    for key in param_keys:
        trend_val = trend_params.get(key, "N/A")
        range_val = range_params.get(key, "N/A")
        volatile_val = volatile_params.get(key, "N/A")
        
        # Format numbers nicely
        if isinstance(trend_val, (int, float)):
            trend_val = f"{trend_val:.4f}" if isinstance(trend_val, float) else f"{trend_val}"
        if isinstance(range_val, (int, float)):
            range_val = f"{range_val:.4f}" if isinstance(range_val, float) else f"{range_val}"
        if isinstance(volatile_val, (int, float)):
            volatile_val = f"{volatile_val:.4f}" if isinstance(volatile_val, float) else f"{volatile_val}"
            
        print(f"{key:<20} {trend_val:<15} {range_val:<15} {volatile_val:<15}")
    
    return param_manager


def test_multi_timeframe():
    """Test the multi-timeframe analysis component."""
    print("\n=== Testing Multi-Timeframe Analysis ===")
    
    # Generate mock data for different timeframes
    daily_data = generate_simple_mock_data(days=50)
    hourly_data = generate_simple_mock_data(days=50, volatility=0.015)  # More volatile at shorter timeframes
    
    print("Generated mock data for timeframe testing")
    
    # Create data structure
    market_data = {
        "BTC/USD": {
            "1d": daily_data,
            "1h": hourly_data
        }
    }
    
    # Initialize timeframe strategy
    strategy_config = {
        "base_strategy": "MA_Cross",
        "timeframes": ["1h", "1d"],
        "min_confirmations": 2,
        "weighting": "longer_higher",
        "base_config": {
            "fast_period": 9,
            "slow_period": 21,
            "signal_period": 9,
            "confirmation_periods": 1
        }
    }
    
    multi_tf_strategy = MultiTimeframeStrategy(strategy_config)
    
    # Calculate indicators
    indicator_engine = IndicatorEngine({})
    indicator_list = [
        {"name": "sma", "params": {"window": 9}},
        {"name": "sma", "params": {"window": 21}},
        {"name": "ema", "params": {"window": 9}},
        {"name": "ema", "params": {"window": 21}},
        {"name": "rsi", "params": {"window": 14}},
        {"name": "atr", "params": {"window": 14}}
    ]
    
    indicators = {}
    for symbol, timeframe_data in market_data.items():
        indicators[symbol] = {}
        for tf, data in timeframe_data.items():
            indicators[symbol][tf] = indicator_engine.calculate_all_indicators({symbol: data}, indicator_list).get(symbol, {})
    
    # Generate signals
    signals = multi_tf_strategy.generate_signals(market_data, indicators, ["BTC/USD"])
    
    # Print results
    print(f"Generated {len(signals)} multi-timeframe signals")
    for i, signal in enumerate(signals):
        if i < 3:  # Show details for at most 3 signals
            print(f"\nSignal {i+1}:")
            print(f"  Symbol: {signal['payload']['symbol']}")
            print(f"  Direction: {'BUY' if signal['payload']['signal'] > 0 else 'SELL'}")
            print(f"  Strength: {abs(signal['payload']['signal']):.4f}")
            print(f"  Confidence: {signal['payload']['confidence']:.4f}")
            print(f"  Timeframes confirmed: {signal['payload']['metadata']['timeframes_confirmed']}")
            print(f"  Confirmation count: {signal['payload']['metadata']['confirmation_count']}")
    
    return multi_tf_strategy, signals


def test_integration():
    """Test the integration of all components."""
    print("\n=== Testing Advanced TA Integration ===")
    
    # Generate data for multiple symbols with different characteristics
    btc_data = generate_simple_mock_data(days=50, volatility=0.02, trend_strength=0.7)
    eth_data = generate_simple_mock_data(days=50, volatility=0.03, trend_strength=0.1)
    xrp_data = generate_simple_mock_data(days=50, volatility=0.04, trend_strength=0.3)
    
    # Create market data structure
    market_data = {
        "BTC/USD": btc_data,
        "ETH/USD": eth_data,
        "XRP/USD": xrp_data
    }
    
    # Create configuration for advanced TA agent
    config = {
        "strategies": [
            {
                "type": "MA_Cross",
                "name": "MA_Cross_Standard",
                "fast_period": 9,
                "slow_period": 21,
                "signal_period": 9,
                "confirmation_periods": 1
            },
            {
                "type": "RSI_OB_OS",
                "name": "RSI_Standard",
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30,
                "confirmation_periods": 1
            }
        ],
        "indicators": [
            {"name": "sma", "params": {"window": 9}},
            {"name": "sma", "params": {"window": 21}},
            {"name": "ema", "params": {"window": 9}},
            {"name": "ema", "params": {"window": 21}},
            {"name": "rsi", "params": {"window": 14}},
            {"name": "atr", "params": {"window": 14}},
            {"name": "adx", "params": {"window": 14}}
        ],
        "timeframes": ["1d"],  # Using only daily for simplicity
        "ml_validator": {
            "min_confidence": 0.5  # Lower threshold for testing
        },
        "adaptive_parameters": {
            "optimization_method": "regime_based"
        }
    }
    
    # Initialize advanced TA agent
    advanced_ta = AdvancedTechnicalAnalysisAgent(config)
    
    # Generate signals
    signals = advanced_ta.analyze(market_data, list(market_data.keys()))
    
    # Print results
    print(f"Generated {len(signals)} validated signals")
    for i, signal in enumerate(signals):
        if i < 5:  # Show details for at most 5 signals
            print(f"\nSignal {i+1}:")
            print(f"  Symbol: {signal['payload']['symbol']}")
            print(f"  Strategy: {signal['payload']['strategy']}")
            print(f"  Direction: {'BUY' if signal['payload']['signal'] > 0 else 'SELL'}")
            print(f"  Strength: {abs(signal['payload']['signal']):.4f}")
            print(f"  Confidence: {signal['payload']['confidence']:.4f}")
            
            # Print regime information if available
            if "market_regime" in signal["payload"]:
                regime = signal["payload"]["market_regime"]["regime"]
                regime_conf = signal["payload"]["market_regime"]["confidence"]
                print(f"  Market Regime: {regime} (confidence: {regime_conf:.4f})")
            
            # Print validation info if available
            if "validation" in signal["payload"]:
                print(f"  Validation Method: {signal['payload']['validation'].get('validation_method', 'unknown')}")
                if "reasons" in signal["payload"]["validation"]:
                    print(f"  Validation Reasons: {signal['payload']['validation']['reasons']}")
    
    # Get and print metrics
    metrics = advanced_ta.get_metrics()
    print("\nAdvanced TA Metrics:")
    print(f"  Signals Generated: {metrics['signals_generated']}")
    print(f"  Signals Validated: {metrics['signals_validated']}")
    print(f"  Signals Rejected: {metrics['signals_rejected']}")
    print(f"  Avg Signal Confidence: {metrics['avg_signal_confidence']:.4f}")
    print(f"  Current Market Regime: {metrics['current_regime']}")
    
    return advanced_ta, signals


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    print("Testing Technical Analysis Agent Enhancements")
    print("=" * 50)
    
    try:
        # Test ML signal validation
        validator = test_ml_signal_validator()
        
        # Test adaptive parameters
        param_manager = test_adaptive_parameters()
        
        # Test multi-timeframe analysis
        multi_tf_strategy, tf_signals = test_multi_timeframe()
        
        # Test full integration
        try:
            advanced_ta, signals = test_integration()
        except Exception as e:
            print(f"Integration test error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
