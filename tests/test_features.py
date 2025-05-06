"""
Test script for feature engineering functions.

This script tests both the Rust and Python implementations of feature engineering functions.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytest

# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Requires missing or unbuilt Rust integration modules (ai_trading_agent_rs)")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the feature engineering functions
from ai_trading_agent.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix,
    create_lag_features_df,
    RUST_AVAILABLE
)

def test_lag_features():
    """Test lag features implementation."""
    print("\n=== Testing Lag Features ===")
    
    # Create a simple time series
    n_samples = 100
    x = np.linspace(0, 4*np.pi, n_samples)
    series = np.sin(x)
    
    # Define lag periods
    lags = [1, 5, 10]
    
    # Create lag features
    lag_features = create_lag_features(series, lags)
    
    # Print results
    print(f"Series shape: {series.shape}")
    print(f"Lag features shape: {lag_features.shape}")
    print(f"First 15 rows of lag features:")
    for i in range(min(15, lag_features.shape[0])):
        row = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in lag_features[i]]
        print(f"Row {i}: {row}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Series')
    for i, lag in enumerate(lags):
        plt.plot(lag_features[:, i], label=f'Lag {lag}')
    plt.legend()
    plt.title('Lag Features')
    plt.savefig('lag_features.png')
    print(f"Plot saved to lag_features.png")
    
    return lag_features

def test_diff_features():
    """Test difference features implementation."""
    print("\n=== Testing Difference Features ===")
    
    # Create a simple time series
    n_samples = 100
    x = np.linspace(0, 4*np.pi, n_samples)
    series = np.sin(x)
    
    # Define periods
    periods = [1, 5, 10]
    
    # Create difference features
    diff_features = create_diff_features(series, periods)
    
    # Print results
    print(f"Series shape: {series.shape}")
    print(f"Difference features shape: {diff_features.shape}")
    print(f"First 15 rows of difference features:")
    for i in range(min(15, diff_features.shape[0])):
        row = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in diff_features[i]]
        print(f"Row {i}: {row}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Series')
    for i, period in enumerate(periods):
        plt.plot(diff_features[:, i], label=f'Diff {period}')
    plt.legend()
    plt.title('Difference Features')
    plt.savefig('diff_features.png')
    print(f"Plot saved to diff_features.png")
    
    return diff_features

def test_pct_change_features():
    """Test percentage change features implementation."""
    print("\n=== Testing Percentage Change Features ===")
    
    # Create a simple time series
    n_samples = 100
    x = np.linspace(0, 4*np.pi, n_samples)
    series = np.sin(x) + 2  # Add offset to avoid zero values
    
    # Define periods
    periods = [1, 5, 10]
    
    # Create percentage change features
    pct_change_features = create_pct_change_features(series, periods)
    
    # Print results
    print(f"Series shape: {series.shape}")
    print(f"Percentage change features shape: {pct_change_features.shape}")
    print(f"First 15 rows of percentage change features:")
    for i in range(min(15, pct_change_features.shape[0])):
        row = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in pct_change_features[i]]
        print(f"Row {i}: {row}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Series')
    for i, period in enumerate(periods):
        plt.plot(pct_change_features[:, i], label=f'Pct Change {period}')
    plt.legend()
    plt.title('Percentage Change Features')
    plt.savefig('pct_change_features.png')
    print(f"Plot saved to pct_change_features.png")
    
    return pct_change_features

def test_rolling_window_features():
    """Test rolling window features implementation."""
    print("\n=== Testing Rolling Window Features ===")
    
    # Create a simple time series
    n_samples = 100
    x = np.linspace(0, 4*np.pi, n_samples)
    series = np.sin(x)
    
    # Define window sizes and feature types
    window_sizes = [5, 10, 20]
    feature_types = ['mean', 'std', 'min', 'max']
    
    # Test each feature type
    for feature_type in feature_types:
        print(f"\nTesting {feature_type} rolling window features:")
        
        # Create rolling window features
        rolling_features = create_rolling_window_features(series, window_sizes, feature_type)
        
        # Print results
        print(f"Series shape: {series.shape}")
        print(f"Rolling {feature_type} features shape: {rolling_features.shape}")
        print(f"First 15 rows of rolling {feature_type} features:")
        for i in range(min(15, rolling_features.shape[0])):
            row = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in rolling_features[i]]
            print(f"Row {i}: {row}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Original Series')
        for i, window in enumerate(window_sizes):
            plt.plot(rolling_features[:, i], label=f'{feature_type} {window}')
        plt.legend()
        plt.title(f'Rolling {feature_type.capitalize()} Features')
        plt.savefig(f'rolling_{feature_type}_features.png')
        print(f"Plot saved to rolling_{feature_type}_features.png")
    
    return rolling_features

def test_feature_matrix():
    """Test feature matrix creation."""
    print("\n=== Testing Feature Matrix ===")
    
    # Create a simple time series
    n_samples = 100
    x = np.linspace(0, 4*np.pi, n_samples)
    series = np.sin(x) + 2  # Add offset to avoid zero values
    
    # Define feature parameters
    lag_periods = [1, 5, 10]
    diff_periods = [1, 5]
    pct_change_periods = [1, 5]
    rolling_windows = {
        'mean': [5, 10],
        'std': [5, 10]
    }
    
    # Create feature matrix
    feature_matrix = create_feature_matrix(
        series,
        lag_periods=lag_periods,
        diff_periods=diff_periods,
        pct_change_periods=pct_change_periods,
        rolling_windows=rolling_windows
    )
    
    # Print results
    print(f"Series shape: {series.shape}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Expected number of features: {len(lag_periods) + len(diff_periods) + len(pct_change_periods) + sum(len(windows) for windows in rolling_windows.values())}")
    print(f"First 5 rows of feature matrix:")
    for i in range(min(5, feature_matrix.shape[0])):
        row = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in feature_matrix[i]]
        print(f"Row {i}: {row[:10] + ['...'] if len(row) > 10 else row}")
    
    return feature_matrix

def test_with_real_data():
    """Test feature engineering with real-like data."""
    print("\n=== Testing with Real-like Data ===")
    
    # Create a DataFrame with date and price columns
    n_samples = 100
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate price data with trend, seasonality, and noise
    trend = np.linspace(100, 150, n_samples)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    noise = np.random.normal(0, 5, n_samples)
    prices = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head:\n{df.head()}")
    
    # Create lag features for price
    lag_df = create_lag_features_df(df, columns=['price'], lags=[1, 5, 10])
    
    print(f"DataFrame with lag features shape: {lag_df.shape}")
    print(f"DataFrame with lag features head:\n{lag_df.head()}")
    
    # Plot original price and lag features
    plt.figure(figsize=(12, 6))
    plt.plot(lag_df['date'], lag_df['price'], label='Original Price')
    for lag in [1, 5, 10]:
        plt.plot(lag_df['date'], lag_df[f'price_lag_{lag}'], label=f'Price Lag {lag}')
    plt.legend()
    plt.title('Price and Lag Features')
    plt.savefig('price_lag_features.png')
    print(f"Plot saved to price_lag_features.png")
    
    return lag_df

def main():
    """Run all tests."""
    print(f"Rust extensions available: {RUST_AVAILABLE}")
    
    # Run tests
    test_lag_features()
    test_diff_features()
    test_pct_change_features()
    test_rolling_window_features()
    test_feature_matrix()
    test_with_real_data()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
