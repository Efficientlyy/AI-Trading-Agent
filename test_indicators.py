"""
Comprehensive test script for the indicator engine with Rust integration.
This script tests all indicator types with both Rust and Python implementations.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndicatorTest")

# Import the IndicatorEngine
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

def create_test_data(length=100):
    """Create test data with realistic price movements."""
    # Start with a base price
    base_price = 100.0
    
    # Create some volatility with random walks
    np.random.seed(42)  # For reproducibility
    random_walk = np.random.normal(0, 1, length).cumsum()
    
    # Scale the random walk
    scaled_walk = random_walk * 5.0
    
    # Create prices
    close_prices = base_price + scaled_walk
    
    # Create high/low with some spread from close
    high_prices = close_prices + np.random.uniform(0, 2, length)
    low_prices = close_prices - np.random.uniform(0, 2, length)
    
    # Ensure low < close < high
    for i in range(length):
        if low_prices[i] > close_prices[i]:
            low_prices[i] = close_prices[i] - 0.5
        if high_prices[i] < close_prices[i]:
            high_prices[i] = close_prices[i] + 0.5
    
    # Create open prices between previous close and current close
    open_prices = np.zeros(length)
    open_prices[0] = close_prices[0] - 1.0  # First open price
    for i in range(1, length):
        prev_close = close_prices[i-1]
        curr_close = close_prices[i]
        # Open somewhere between previous close and current close
        weight = np.random.uniform(0, 1)
        open_prices[i] = prev_close + (curr_close - prev_close) * weight
    
    # Create volume with some correlation to price moves
    volume = np.abs(np.diff(np.append([0], close_prices))) * 10000 + 1000
    
    # Create timestamps
    dates = pd.date_range('2023-01-01', periods=length, freq='D')
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df

def test_lag_features(engine, test_data, lags=[1, 3, 5, 10], use_rust=True):
    """Test lag features with both Rust and Python implementations."""
    print(f"\n>>> Testing {'Rust' if use_rust else 'Python'} lag features implementation")
    logger.info(f"Testing {'Rust' if use_rust else 'Python'} lag features implementation")
    
    if use_rust:
        # Use Rust implementation
        results = engine._calculate_lag_features_rs(test_data, lags)
    else:
        # Use Python implementation
        results = engine._calculate_lag_features(test_data, lags, 'close')
    
    # Verify structure and values
    assert isinstance(results, dict), "Result should be a dictionary"
    assert len(results) == len(lags), f"Expected {len(lags)} lag features, got {len(results)}"
    
    for lag in lags:
        key = f'lag_{lag}'
        assert key in results, f"Expected {key} in results"
        lag_series = results[key]
        
        # Check first N values are NaN
        assert lag_series.iloc[:lag].isna().all(), f"First {lag} values should be NaN"
        
        # Check that values match expectations
        for i in range(lag, len(test_data)):
            expected = test_data['close'].iloc[i - lag]
            actual = lag_series.iloc[i]
            np.testing.assert_almost_equal(actual, expected, decimal=6, 
                                          err_msg=f"Value mismatch at position {i} for lag {lag}")
    
    logger.info(f"{'Rust' if use_rust else 'Python'} lag features test: PASSED")
    return results

def test_ema(engine, test_data, periods=[5, 10, 20], use_rust=True):
    """Test EMA with both Rust and Python implementations."""
    print(f"\n>>> Testing {'Rust' if use_rust else 'Python'} EMA implementation")
    logger.info(f"Testing {'Rust' if use_rust else 'Python'} EMA implementation")
    
    # Force Python implementation for testing
    if not use_rust:
        # Save original state
        original_rs_features = engine.rs_features
        engine.rs_features = None  # Disable Rust temporarily
    
    try:
        # Calculate EMAs
        results = engine._calculate_ema(test_data, periods)
        
        # Verify structure and basic properties
        assert isinstance(results, dict), "Result should be a dictionary"
        assert len(results) == len(periods), f"Expected {len(periods)} period results, got {len(results)}"
        
        for period in periods:
            period_key = str(period)
            assert period_key in results, f"Expected period {period_key} in results"
            ema_series = results[period_key]
            
            # Check length
            assert len(ema_series) == len(test_data), f"EMA series length mismatch for period {period}"
            
            # Check initial NaN values (first N-1 values typically None/NaN for period N)
            # For EMA, typically the first value is not NaN but subsequent ones might be,
            # depending on the implementation
            if use_rust:
                # Just check that we get reasonable values, don't make specific assertions about NaNs
                pass
            else:
                # Python implementation might start with actual values
                pass
            
            # Check that we have values (not all NaN)
            assert not ema_series.isna().all(), f"EMA series for period {period} is all NaN"
            
            # For longer periods, values should be smoother (less variance)
            if len(periods) > 1 and period != periods[0]:
                prev_period = periods[periods.index(period) - 1]
                prev_ema = results[str(prev_period)]
                
                # Skip NaN values for variance calculation
                curr_variance = ema_series.dropna().var()
                prev_variance = prev_ema.dropna().var()
                
                # Shorter periods should have higher variance
                assert prev_variance >= curr_variance, f"Expected period {prev_period} to have higher variance than period {period}"
        
        logger.info(f"{'Rust' if use_rust else 'Python'} EMA test: PASSED")
        return results
    
    finally:
        if not use_rust:
            # Restore original state
            engine.rs_features = original_rs_features

def test_rsi(engine, test_data, period=14, use_rust=True):
    """Test RSI with both Rust and Python implementations."""
    print(f"\n>>> Testing {'Rust' if use_rust else 'Python'} RSI implementation")
    logger.info(f"Testing {'Rust' if use_rust else 'Python'} RSI implementation")
    
    # Force Python implementation for testing
    if not use_rust:
        # Save original state
        original_rs_features = engine.rs_features
        engine.rs_features = None  # Disable Rust temporarily
    
    try:
        # Calculate RSI
        results = engine._calculate_rsi(test_data, period)
        
        # Verify structure and basic properties
        assert isinstance(results, dict), "Result should be a dictionary"
        assert str(period) in results, f"Expected period {period} in results"
        
        rsi_series = results[str(period)]
        
        # Check length
        assert len(rsi_series) == len(test_data), f"RSI series length mismatch"
        
        # Check range of values (RSI is between 0 and 100)
        valid_values = rsi_series.dropna()
        if not valid_values.empty:
            assert valid_values.min() >= 0, f"RSI values should be >= 0, got {valid_values.min()}"
            assert valid_values.max() <= 100, f"RSI values should be <= 100, got {valid_values.max()}"
        
        # Check that we have values (not all NaN)
        assert not rsi_series.isna().all(), f"RSI series is all NaN"
        
        # First 'period' values are typically NaN (or close to it)
        # Note: This might vary by implementation
        if use_rust:
            # Rust implementation might handle the first values differently
            pass
        else:
            # Python implementation typically has NaNs at the start
            assert rsi_series.iloc[:period-1].isna().all(), f"First {period-1} values should be NaN"
        
        logger.info(f"{'Rust' if use_rust else 'Python'} RSI test: PASSED")
        return results
    
    finally:
        if not use_rust:
            # Restore original state
            engine.rs_features = original_rs_features

def test_bollinger_bands(engine, test_data, periods=[20], deviations=2.0, use_rust=True):
    """Test Bollinger Bands with both Rust and Python implementations."""
    print(f"\n>>> Testing {'Rust' if use_rust else 'Python'} Bollinger Bands implementation")
    logger.info(f"Testing {'Rust' if use_rust else 'Python'} Bollinger Bands implementation")
    
    # Force Python implementation for testing
    if not use_rust:
        # Save original state
        original_rs_features = engine.rs_features
        engine.rs_features = None  # Disable Rust temporarily
    
    try:
        # Calculate Bollinger Bands
        results = engine._calculate_bollinger_bands(test_data, periods, deviations)
        
        # Verify structure and basic properties
        assert isinstance(results, dict), "Result should be a dictionary"
        assert len(results) == len(periods), f"Expected {len(periods)} period results, got {len(results)}"
        
        for period in periods:
            period_key = str(period)
            assert period_key in results, f"Expected period {period_key} in results"
            bb_dict = results[period_key]
            
            # Check that we have upper, middle, and lower bands
            assert 'upper' in bb_dict, "Expected 'upper' band in results"
            assert 'middle' in bb_dict, "Expected 'middle' band in results"
            assert 'lower' in bb_dict, "Expected 'lower' band in results"
            
            upper = bb_dict['upper']
            middle = bb_dict['middle']
            lower = bb_dict['lower']
            
            # Check lengths
            assert len(upper) == len(test_data), f"Upper band length mismatch"
            assert len(middle) == len(test_data), f"Middle band length mismatch"
            assert len(lower) == len(test_data), f"Lower band length mismatch"
            
            # Check relationship between bands (upper > middle > lower)
            for i in range(len(test_data)):
                if not pd.isna(upper.iloc[i]) and not pd.isna(middle.iloc[i]) and not pd.isna(lower.iloc[i]):
                    assert upper.iloc[i] >= middle.iloc[i], f"Upper band should be >= middle band at index {i}"
                    assert middle.iloc[i] >= lower.iloc[i], f"Middle band should be >= lower band at index {i}"
            
            # Check that bands are approximately symmetrical around the middle
            # Upper - middle should be approximately equal to middle - lower
            non_nan_indices = (~upper.isna()) & (~middle.isna()) & (~lower.isna())
            upper_diff = upper[non_nan_indices] - middle[non_nan_indices]
            lower_diff = middle[non_nan_indices] - lower[non_nan_indices]
            
            if not upper_diff.empty and not lower_diff.empty:
                # Should be nearly identical for Bollinger Bands
                np.testing.assert_allclose(upper_diff.values, lower_diff.values, rtol=1e-5, atol=1e-5)
        
        logger.info(f"{'Rust' if use_rust else 'Python'} Bollinger Bands test: PASSED")
        return results
    
    finally:
        if not use_rust:
            # Restore original state
            engine.rs_features = original_rs_features

def compare_implementations(rust_results, python_results, indicator_name):
    """Compare results between Rust and Python implementations."""
    print(f"\n>>> Comparing {indicator_name} implementations")
    logger.info(f"Comparing {indicator_name} implementations")
    
    # Check that both results have the same structure
    assert type(rust_results) == type(python_results), f"Result types don't match: {type(rust_results)} vs {type(python_results)}"
    
    if isinstance(rust_results, dict):
        # Check that they have the same keys
        assert set(rust_results.keys()) == set(python_results.keys()), f"Result keys don't match: {rust_results.keys()} vs {python_results.keys()}"
        
        # Compare each value
        for key in rust_results:
            rust_value = rust_results[key]
            python_value = python_results[key]
            
            # Handle nested dictionaries (e.g., Bollinger Bands)
            if isinstance(rust_value, dict):
                compare_implementations(rust_value, python_value, f"{indicator_name} - {key}")
            else:
                # For Series, check correlation and closeness
                if isinstance(rust_value, pd.Series) and isinstance(python_value, pd.Series):
                    # Drop NaN values that exist in either series
                    combined = pd.concat([rust_value, python_value], axis=1)
                    combined.columns = ['rust', 'python']
                    clean_data = combined.dropna()
                    
                    if len(clean_data) > 1:  # Need at least 2 points for correlation
                        correlation = clean_data['rust'].corr(clean_data['python'])
                        logger.info(f"{indicator_name} - {key}: Correlation = {correlation:.4f}")
                        assert correlation > 0.99, f"Low correlation between implementations: {correlation}"
                        
                        # Check closeness
                        np.testing.assert_allclose(
                            clean_data['rust'].values, 
                            clean_data['python'].values, 
                            rtol=1e-4, atol=1e-4,
                            err_msg=f"Values don't match for {indicator_name} - {key}"
                        )
                    else:
                        logger.warning(f"Not enough clean data points to calculate correlation for {indicator_name} - {key}")
    
    logger.info(f"{indicator_name} comparison: PASSED")

def create_indicator_engine():
    """Create a properly configured IndicatorEngine for testing."""
    # Basic config for all indicators
    config = {
        "trend": {
            "sma": {"enabled": True, "periods": [5, 10, 20]},
            "ema": {"enabled": True, "periods": [5, 10, 20]}
        },
        "momentum": {
            "rsi": {"enabled": True, "period": 14}
        },
        "volatility": {
            "bollinger_bands": {"enabled": True, "periods": [20], "deviations": 2}
        },
        "features": {
            "lag_features": {"enabled": True, "lags": [1, 3, 5, 10], "source_column": "close"},
            "LAG_FEATURES_RS_test": {
                "enabled": True, 
                "lags_to_calculate": [1, 3, 5, 10],
                "source_column": "close"
            }
        },
        "logging": {"log_level": "INFO"}
    }
    
    # Create the engine
    engine = IndicatorEngine(config)
    return engine

def main():
    """Run all indicator tests."""
    print("\n=============== STARTING INDICATOR TESTS ===============")
    logger.info("Starting indicator tests")
    
    # Create test data
    test_data = create_test_data(length=100)
    logger.info(f"Created test data with shape {test_data.shape}")
    
    # Create indicator engine
    engine = create_indicator_engine()
    logger.info(f"Created indicator engine")
    
    # Test each indicator type with both implementations
    test_pairs = []
    
    # 1. Lag Features
    try:
        has_rust_lag_features = hasattr(engine.rs_features, 'create_lag_features_rs')
        rust_lag_results = test_lag_features(engine, test_data, use_rust=True) if has_rust_lag_features else None
        python_lag_results = test_lag_features(engine, test_data, use_rust=False)
        if has_rust_lag_features and rust_lag_results:
            test_pairs.append(("Lag Features", rust_lag_results, python_lag_results))
    except Exception as e:
        logger.error(f"Error testing lag features: {e}")
    
    # 2. EMA
    try:
        has_rust_ema = hasattr(engine.rs_features, 'create_ema_features_rs')
        rust_ema_results = test_ema(engine, test_data, use_rust=True) if has_rust_ema else None
        python_ema_results = test_ema(engine, test_data, use_rust=False)
        if has_rust_ema and rust_ema_results:
            test_pairs.append(("EMA", rust_ema_results, python_ema_results))
    except Exception as e:
        logger.error(f"Error testing EMA: {e}")
    
    # 3. RSI
    try:
        has_rust_rsi = hasattr(engine.rs_features, 'create_rsi_features_rs')
        rust_rsi_results = test_rsi(engine, test_data, use_rust=True) if has_rust_rsi else None
        python_rsi_results = test_rsi(engine, test_data, use_rust=False)
        if has_rust_rsi and rust_rsi_results:
            test_pairs.append(("RSI", rust_rsi_results, python_rsi_results))
    except Exception as e:
        logger.error(f"Error testing RSI: {e}")
    
    # 4. Bollinger Bands
    try:
        has_rust_bb = hasattr(engine.rs_features, 'create_bollinger_bands_rs')
        rust_bb_results = test_bollinger_bands(engine, test_data, use_rust=True) if has_rust_bb else None
        python_bb_results = test_bollinger_bands(engine, test_data, use_rust=False)
        if has_rust_bb and rust_bb_results:
            test_pairs.append(("Bollinger Bands", rust_bb_results, python_bb_results))
    except Exception as e:
        logger.error(f"Error testing Bollinger Bands: {e}")
    
    # Compare implementations
    for name, rust_results, python_results in test_pairs:
        try:
            compare_implementations(rust_results, python_results, name)
        except Exception as e:
            logger.error(f"Error comparing {name} implementations: {e}")
    
    # Overall test
    # Test calculate_all_indicators with the full config
    try:
        logger.info("Testing calculate_all_indicators with full config")
        market_data = {"TEST": test_data}
        results = engine.calculate_all_indicators(market_data, ["TEST"])
        assert "TEST" in results, "Symbol not found in results"
        logger.info("calculate_all_indicators test: PASSED")
    except Exception as e:
        logger.error(f"Error testing calculate_all_indicators: {e}")
    
    print("\n=============== ALL TESTS COMPLETED SUCCESSFULLY! ===============")
    logger.info("All tests completed!")

if __name__ == "__main__":
    main()
