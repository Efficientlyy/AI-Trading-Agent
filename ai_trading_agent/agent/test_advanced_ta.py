"""
Test script for Advanced Technical Analysis components.

This script demonstrates how to use the multi-timeframe analysis, ML signal validation,
and adaptive parameter tuning features integrated with the Technical Analysis Agent.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.agent.multi_timeframe import MultiTimeframeStrategy, TimeframeManager
from ai_trading_agent.ml.signal_validator import MLSignalValidator
from ai_trading_agent.ml.adaptive_parameters import AdaptiveParameterManager, MarketRegimeClassifier
from ai_trading_agent.common.utils import get_logger
from ai_trading_agent.data_sources.mock_data_generator import generate_mock_data


def generate_multi_timeframe_mock_data(days=100, volatility=0.02, trend_strength=0.5):
    """
    Generate mock data for multiple timeframes.
    
    Args:
        days: Number of days of data to generate
        volatility: Volatility of the price data
        trend_strength: Strength of the trend (0-1)
        
    Returns:
        Dictionary mapping timeframes to market data DataFrames
    """
    # Generate daily data
    daily_data = generate_mock_data(
        days=days,
        volatility=volatility,
        trend_strength=trend_strength,
        pattern_type="trend"
    )
    
    # Create 4-hour data by upsampling and adding noise
    hours = days * 24
    four_hour_periods = hours // 4
    
    # Start with the daily data and add noise
    four_hour_data = pd.DataFrame(index=range(four_hour_periods))
    
    # Map from 4-hour periods to daily periods
    for i in range(four_hour_periods):
        day_idx = i // 6  # 6 four-hour periods per day
        
        if day_idx < len(daily_data):
            # Add some random noise to make it look like intraday data
            noise = np.random.normal(0, volatility * 0.5)
            
            # Time within the day affects price pattern
            time_of_day = i % 6
            time_factor = np.sin(time_of_day / 6 * np.pi) * volatility
            
            # Copy the daily data and add noise
            four_hour_data.loc[i, 'open'] = daily_data.loc[day_idx, 'open'] * (1 + noise + time_factor)
            four_hour_data.loc[i, 'high'] = daily_data.loc[day_idx, 'high'] * (1 + noise + time_factor + 0.002)
            four_hour_data.loc[i, 'low'] = daily_data.loc[day_idx, 'low'] * (1 + noise + time_factor - 0.002)
            four_hour_data.loc[i, 'close'] = daily_data.loc[day_idx, 'close'] * (1 + noise + time_factor)
            four_hour_data.loc[i, 'volume'] = daily_data.loc[day_idx, 'volume'] * (1 + np.random.normal(0, 0.2))
    
    # Create 1-hour data with more noise
    hour_periods = hours
    one_hour_data = pd.DataFrame(index=range(hour_periods))
    
    # Map from 1-hour periods to daily periods
    for i in range(hour_periods):
        day_idx = i // 24  # 24 one-hour periods per day
        
        if day_idx < len(daily_data):
            # Add more random noise for hourly data
            noise = np.random.normal(0, volatility * 0.7)
            
            # Time within the day affects price pattern
            time_of_day = i % 24
            time_factor = np.sin(time_of_day / 24 * np.pi * 2) * volatility * 0.8
            
            # Copy the daily data and add noise
            one_hour_data.loc[i, 'open'] = daily_data.loc[day_idx, 'open'] * (1 + noise + time_factor)
            one_hour_data.loc[i, 'high'] = daily_data.loc[day_idx, 'high'] * (1 + noise + time_factor + 0.003)
            one_hour_data.loc[i, 'low'] = daily_data.loc[day_idx, 'low'] * (1 + noise + time_factor - 0.003)
            one_hour_data.loc[i, 'close'] = daily_data.loc[day_idx, 'close'] * (1 + noise + time_factor)
            one_hour_data.loc[i, 'volume'] = daily_data.loc[day_idx, 'volume'] * (1 + np.random.normal(0, 0.3))
    
    # Create realistic timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Set timestamps for each timeframe
    daily_data.index = pd.date_range(start=start_date, periods=len(daily_data), freq='D')
    four_hour_data.index = pd.date_range(start=start_date, periods=len(four_hour_data), freq='4H')
    one_hour_data.index = pd.date_range(start=start_date, periods=len(one_hour_data), freq='1H')
    
    # Return data organized by timeframe
    return {
        "1d": daily_data,
        "4h": four_hour_data,
        "1h": one_hour_data
    }


def test_multi_timeframe_analysis():
    """Test the multi-timeframe analysis functionality."""
    print("\n=== Testing Multi-Timeframe Analysis ===")
    
    # Generate mock data for different timeframes
    timeframes = ["1h", "4h", "1d"]
    mock_data = generate_multi_timeframe_mock_data(days=100, volatility=0.02, trend_strength=0.5)
    
    # Initialize timeframe manager
    tf_manager = TimeframeManager(timeframes)
    
    # Create symbol data structure
    symbol_data = {"BTC/USD": mock_data}
    organized_data = {}
    
    for symbol, tf_data in symbol_data.items():
        organized_data[symbol] = {}
        for tf, data in tf_data.items():
            organized_data[symbol][tf] = data
    
    # Create a multi-timeframe strategy
    strategy_config = {
        "base_strategy": "MA_Cross",
        "timeframes": timeframes,
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
    
    # Initialize indicator engine
    from ai_trading_agent.agent.indicator_engine import IndicatorEngine
    indicator_engine = IndicatorEngine()
    
    # Calculate indicators for each timeframe
    indicator_list = [
        {"name": "sma", "params": {"window": 9}},
        {"name": "sma", "params": {"window": 21}},
        {"name": "ema", "params": {"window": 9}},
        {"name": "ema", "params": {"window": 21}},
        {"name": "rsi", "params": {"window": 14}},
        {"name": "atr", "params": {"window": 14}}
    ]
    
    # Calculate indicators for all symbols and timeframes
    organized_indicators = {}
    for symbol, tf_data in organized_data.items():
        organized_indicators[symbol] = {}
        for tf, data in tf_data.items():
            organized_indicators[symbol][tf] = indicator_engine.calculate_all_indicators(data, indicator_list)
    
    # Generate signals
    signals = multi_tf_strategy.generate_signals(
        organized_data, organized_indicators, ["BTC/USD"]
    )
    
    # Print results
    print(f"Generated {len(signals)} multi-timeframe signals")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Symbol: {signal['payload']['symbol']}")
        print(f"  Direction: {'BUY' if signal['payload']['signal'] > 0 else 'SELL'}")
        print(f"  Strength: {abs(signal['payload']['signal']):.4f}")
        print(f"  Confidence: {signal['payload']['confidence']:.4f}")
        print(f"  Timeframes confirmed: {signal['payload']['metadata']['timeframes_confirmed']}")
        print(f"  Confirmation count: {signal['payload']['metadata']['confirmation_count']}")
    
    return signals


def test_ml_signal_validation():
    """Test the ML signal validation functionality."""
    print("\n=== Testing ML Signal Validation ===")
    
    # Generate some mock data
    mock_data = generate_mock_data(days=100, volatility=0.02, trend_strength=0.5)
    
    # Create a signal validator
    validator_config = {
        "min_confidence": 0.6,
        "features": [
            "signal_strength", 
            "volatility_percent", 
            "volume_ratio", 
            "price_change_percent",
            "rsi_value",
            "ma_diff_percent",
            "confirmation_count"
        ]
    }
    
    signal_validator = MLSignalValidator(validator_config)
    
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
    
    # Calculate some indicators for validation
    from ai_trading_agent.agent.indicator_engine import IndicatorEngine
    indicator_engine = IndicatorEngine()
    indicators = indicator_engine.calculate_all_indicators(mock_data, [
        {"name": "rsi", "params": {"window": 14}},
        {"name": "ema", "params": {"window": 9}},
        {"name": "sma", "params": {"window": 21}}
    ])
    
    # Validate the signal
    is_valid, confidence, validation_metadata = signal_validator.validate_signal(
        mock_signal, {"BTC/USD": mock_data}, {"BTC/USD": indicators}
    )
    
    # Print results
    print(f"Signal validated: {is_valid}")
    print(f"Confidence score: {confidence:.4f}")
    print("Validation metadata:")
    for key, value in validation_metadata.items():
        print(f"  {key}: {value}")
    
    # Generate training data for the validator
    # In a real scenario, you would have historical signals and their outcomes
    outcomes = {
        "signal1": True,   # Success
        "signal2": False,  # Failure
        "signal3": True    # Success
    }
    
    mock_signals = [
        {
            "id": "signal1",
            "payload": {
                "signal": 0.8,
                "metadata": {
                    "volatility_percent": 0.02,
                    "volume_ratio": 1.7,
                    "confirmation_count": 3
                }
            }
        },
        {
            "id": "signal2",
            "payload": {
                "signal": 0.3,
                "metadata": {
                    "volatility_percent": 0.05,
                    "volume_ratio": 0.8,
                    "confirmation_count": 1
                }
            }
        },
        {
            "id": "signal3",
            "payload": {
                "signal": 0.65,
                "metadata": {
                    "volatility_percent": 0.03,
                    "volume_ratio": 1.2,
                    "confirmation_count": 2
                }
            }
        }
    ]
    
    training_data = signal_validator.generate_training_data(mock_signals, outcomes)
    print("\nGenerated training data:")
    print(training_data)
    
    return signal_validator


def test_adaptive_parameters():
    """Test the adaptive parameter functionality."""
    print("\n=== Testing Adaptive Parameters ===")
    
    # Generate different types of market data
    trend_data = generate_mock_data(days=50, volatility=0.01, trend_strength=0.8, pattern_type="trend")
    range_data = generate_mock_data(days=50, volatility=0.02, trend_strength=0.1, pattern_type="range")
    volatile_data = generate_mock_data(days=50, volatility=0.04, trend_strength=0.3, pattern_type="volatile")
    
    # Initialize parameter manager
    param_manager = AdaptiveParameterManager()
    
    # Initialize indicator engine
    from ai_trading_agent.agent.indicator_engine import IndicatorEngine
    indicator_engine = IndicatorEngine()
    
    # Calculate indicators for each market type
    indicator_list = [
        {"name": "adx", "params": {"window": 14}},
        {"name": "atr", "params": {"window": 14}},
        {"name": "sma", "params": {"window": 20}}
    ]
    
    trend_indicators = indicator_engine.calculate_all_indicators(trend_data, indicator_list)
    range_indicators = indicator_engine.calculate_all_indicators(range_data, indicator_list)
    volatile_indicators = indicator_engine.calculate_all_indicators(volatile_data, indicator_list)
    
    # Test market regime classification
    regime_classifier = MarketRegimeClassifier()
    
    trend_regime = regime_classifier.classify_regime(trend_data, trend_indicators)
    range_regime = regime_classifier.classify_regime(range_data, range_indicators)
    volatile_regime = regime_classifier.classify_regime(volatile_data, volatile_indicators)
    
    print("Market Regime Classification:")
    print(f"  Trend data: {trend_regime['regime']} (confidence: {trend_regime['confidence']:.4f})")
    print(f"  Range data: {range_regime['regime']} (confidence: {range_regime['confidence']:.4f})")
    print(f"  Volatile data: {volatile_regime['regime']} (confidence: {volatile_regime['confidence']:.4f})")
    
    # Get adapted parameters for each market regime
    trend_params = param_manager.get_strategy_parameters("MA_Cross", trend_data, trend_indicators)
    range_params = param_manager.get_strategy_parameters("MA_Cross", range_data, range_indicators)
    volatile_params = param_manager.get_strategy_parameters("MA_Cross", volatile_data, volatile_indicators)
    
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


def test_advanced_ta_integration():
    """Test the integration of all advanced TA components."""
    print("\n=== Testing Advanced Technical Analysis Integration ===")
    
    # Create configuration for the advanced TA agent
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
            },
            {
                "type": "Multi_TF",
                "name": "MA_Cross_Multi_TF",
                "base_strategy": "MA_Cross",
                "timeframes": ["1h", "4h", "1d"],
                "min_confirmations": 2,
                "weighting": "longer_higher",
                "base_config": {
                    "fast_period": 9,
                    "slow_period": 21,
                    "signal_period": 9,
                    "confirmation_periods": 1
                }
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
        "timeframes": ["1h", "4h", "1d"],
        "ml_validator": {
            "min_confidence": 0.6
        },
        "adaptive_parameters": {
            "optimization_method": "regime_based"
        }
    }
    
    # Initialize the advanced TA agent
    advanced_ta = AdvancedTechnicalAnalysisAgent(config)
    
    # Generate mock data for different market regimes
    print("Generating mock data for different market regimes...")
    
    # Trending market data
    trending_tf_data = generate_multi_timeframe_mock_data(
        days=50, volatility=0.01, trend_strength=0.8
    )
    
    # Ranging market data
    ranging_tf_data = generate_multi_timeframe_mock_data(
        days=50, volatility=0.02, trend_strength=0.1
    )
    
    # Volatile market data
    volatile_tf_data = generate_multi_timeframe_mock_data(
        days=50, volatility=0.04, trend_strength=0.3
    )
    
    # Create market data structure for different symbols with different regimes
    market_data = {
        "BTC/USD": trending_tf_data["1d"],    # Trending
        "ETH/USD": ranging_tf_data["1d"],     # Ranging
        "XRP/USD": volatile_tf_data["1d"]     # Volatile
    }
    
    # Analyze each market and generate signals
    symbols = list(market_data.keys())
    signals = advanced_ta.analyze(market_data, symbols)
    
    # Print results
    print(f"\nGenerated {len(signals)} validated signals")
    for i, signal in enumerate(signals):
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
    print(f"  Regime Changes: {metrics['regime_changes']}")
    print(f"  Parameter Adaptations: {metrics['parameter_adaptations']}")
    
    # Save agent state
    save_dir = os.path.join(os.path.dirname(__file__), "advanced_ta_state")
    advanced_ta.save_state(save_dir)
    print(f"\nSaved agent state to {save_dir}")
    
    return advanced_ta, signals


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    print("Testing Advanced Technical Analysis Components")
    print("=" * 50)
    
    # Test multi-timeframe analysis
    multi_tf_signals = test_multi_timeframe_analysis()
    
    # Test ML signal validation
    signal_validator = test_ml_signal_validation()
    
    # Test adaptive parameters
    param_manager = test_adaptive_parameters()
    
    # Test the full integration
    advanced_ta, signals = test_advanced_ta_integration()
    
    print("\nAll tests completed successfully!")
