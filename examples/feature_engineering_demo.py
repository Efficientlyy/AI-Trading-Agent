"""
Feature Engineering Demo

This script demonstrates how to use the Rust-accelerated feature engineering functions
to create lag features for trading strategy development.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix
)


def generate_sample_data(n_samples=1000, volatility=0.01, trend=0.0001):
    """Generate sample price data with a slight trend and volatility."""
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate random returns with a slight trend
    returns = np.random.normal(trend, volatility, n_samples)
    
    # Convert returns to prices
    prices = 100 * np.cumprod(1 + returns)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Add some additional columns
    df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.002, n_samples))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.003, n_samples)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.003, n_samples)))
    df['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df


def create_all_features(price_data):
    """Create a comprehensive set of features from price data."""
    # Extract price series
    close = price_data['close'].values
    high = price_data['high'].values
    low = price_data['low'].values
    volume = price_data['volume'].values
    
    # Create lag features
    print("Creating lag features...")
    lag_periods = [1, 2, 3, 5, 10, 21]
    close_lags = create_lag_features(close, lag_periods)
    
    # Create difference features
    print("Creating difference features...")
    diff_periods = [1, 2, 5, 10]
    close_diffs = create_diff_features(close, diff_periods)
    
    # Create percentage change features
    print("Creating percentage change features...")
    pct_periods = [1, 2, 5, 10, 21]
    close_pct_changes = create_pct_change_features(close, pct_periods)
    
    # Create rolling window features
    print("Creating rolling window features...")
    window_sizes = [5, 10, 21, 63]
    
    close_rolling_means = create_rolling_window_features(close, window_sizes, 'mean')
    close_rolling_stds = create_rolling_window_features(close, window_sizes, 'std')
    
    # Calculate price volatility (high-low range)
    print("Calculating price volatility...")
    price_range = high - low
    range_rolling_means = create_rolling_window_features(price_range, window_sizes, 'mean')
    
    # Create volume features
    print("Creating volume features...")
    volume_pct_changes = create_pct_change_features(volume, [1, 5, 10])
    volume_rolling_means = create_rolling_window_features(volume, window_sizes, 'mean')
    
    # Combine all features into a DataFrame
    print("Combining all features...")
    feature_names = []
    
    # Lag feature names
    feature_names.extend([f'close_lag_{lag}' for lag in lag_periods])
    
    # Difference feature names
    feature_names.extend([f'close_diff_{period}' for period in diff_periods])
    
    # Percentage change feature names
    feature_names.extend([f'close_pct_{period}' for period in pct_periods])
    
    # Rolling mean feature names
    feature_names.extend([f'close_mean_{window}' for window in window_sizes])
    
    # Rolling std feature names
    feature_names.extend([f'close_std_{window}' for window in window_sizes])
    
    # Price range feature names
    feature_names.extend([f'range_mean_{window}' for window in window_sizes])
    
    # Volume feature names
    feature_names.extend([f'volume_pct_{period}' for period in [1, 5, 10]])
    feature_names.extend([f'volume_mean_{window}' for window in window_sizes])
    
    # Combine all features
    all_features = np.hstack([
        close_lags,
        close_diffs,
        close_pct_changes,
        close_rolling_means,
        close_rolling_stds,
        range_rolling_means,
        volume_pct_changes,
        volume_rolling_means
    ])
    
    # Create DataFrame with features
    feature_df = pd.DataFrame(
        all_features,
        index=price_data.index,
        columns=feature_names
    )
    
    # Add original price data
    result_df = pd.concat([price_data, feature_df], axis=1)
    
    return result_df


def simple_trading_strategy(data, lookback=10, volatility_window=21):
    """
    Simple trading strategy using lag features.
    
    Strategy:
    1. Buy when price is above the lookback period moving average
       AND volatility is below its recent average
    2. Sell when price is below the lookback period moving average
       OR volatility is above 1.5x its recent average
    """
    # Create necessary features using the feature matrix function
    close = data['close'].values
    
    # Create a feature matrix with the required features
    features = create_feature_matrix(
        close,
        lag_periods=[1],
        rolling_windows={
            'mean': [lookback],
            'std': [volatility_window]
        }
    )
    
    # Extract features
    lag_1 = features[:, 0]
    ma = features[:, 1]
    volatility = features[:, 2]
    
    # Calculate volatility ratio (current volatility / average volatility)
    volatility_ratio = np.zeros_like(volatility)
    for i in range(volatility_window*2, len(volatility)):
        recent_vol_avg = np.nanmean(volatility[i-volatility_window:i])
        if not np.isnan(recent_vol_avg) and recent_vol_avg != 0:
            volatility_ratio[i] = volatility[i] / recent_vol_avg
    
    # Generate signals
    signals = np.zeros(len(close))
    
    for i in range(max(lookback, volatility_window*2), len(close)):
        # Buy condition: price > MA and volatility is low
        if close[i] > ma[i] and volatility_ratio[i] < 1.0:
            signals[i] = 1
        # Sell condition: price < MA or volatility is high
        elif close[i] < ma[i] or volatility_ratio[i] > 1.5:
            signals[i] = -1
    
    # Convert to positions (holding periods)
    position = 0
    positions = np.zeros(len(signals))
    
    for i in range(len(signals)):
        if signals[i] == 1:
            position = 1
        elif signals[i] == -1:
            position = 0
        positions[i] = position
    
    # Calculate returns
    returns = np.zeros(len(positions))
    for i in range(1, len(positions)):
        if positions[i-1] == 1:
            returns[i] = (close[i] / close[i-1]) - 1
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    # Buy and hold returns
    buy_hold_returns = (close / close[0]) - 1
    
    return {
        'signals': signals,
        'positions': positions,
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'buy_hold_returns': buy_hold_returns
    }


def plot_strategy_results(data, results):
    """Plot the strategy results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price and positions
    ax1.plot(data.index, data['close'], label='Price')
    
    # Highlight holding periods
    buy_signals = np.where(results['signals'] == 1)[0]
    sell_signals = np.where(results['signals'] == -1)[0]
    
    for i in buy_signals:
        ax1.scatter(data.index[i], data['close'].iloc[i], color='green', marker='^', s=100)
    
    for i in sell_signals:
        ax1.scatter(data.index[i], data['close'].iloc[i], color='red', marker='v', s=100)
    
    # Highlight holding periods
    holding_periods = []
    start_idx = None
    
    for i in range(len(results['positions'])):
        if results['positions'][i] == 1 and (i == 0 or results['positions'][i-1] == 0):
            start_idx = i
        elif results['positions'][i] == 0 and i > 0 and results['positions'][i-1] == 1:
            if start_idx is not None:
                holding_periods.append((start_idx, i))
                start_idx = None
    
    # If still holding at the end
    if start_idx is not None:
        holding_periods.append((start_idx, len(results['positions'])-1))
    
    for start, end in holding_periods:
        ax1.axvspan(data.index[start], data.index[end], alpha=0.2, color='green')
    
    ax1.set_title('Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot returns
    ax2.plot(data.index, results['cumulative_returns'], label='Strategy Returns')
    ax2.plot(data.index, results['buy_hold_returns'], label='Buy & Hold Returns')
    ax2.set_title('Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Returns')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_results.png')
    plt.close()


def main():
    """Main function to run the demo."""
    print("Generating sample data...")
    data = generate_sample_data(n_samples=1000, volatility=0.015, trend=0.0001)
    
    # Set date as index
    data = data.set_index('date')
    
    print("Creating features...")
    # Uncomment to create all features (can be slow)
    # data_with_features = create_all_features(data)
    
    print("Running trading strategy...")
    results = simple_trading_strategy(data, lookback=20, volatility_window=30)
    
    print("Plotting results...")
    plot_strategy_results(data, results)
    
    # Calculate performance metrics
    total_return = results['cumulative_returns'][-1]
    buy_hold_return = results['buy_hold_returns'][-1]
    
    # Calculate annualized return
    days = (data.index[-1] - data.index[0]).days
    ann_factor = 365 / days
    ann_return = ((1 + total_return) ** ann_factor) - 1
    ann_buy_hold = ((1 + buy_hold_return) ** ann_factor) - 1
    
    # Calculate Sharpe ratio (simplified)
    returns_std = np.std(results['returns']) * np.sqrt(252)
    sharpe = ann_return / returns_std if returns_std != 0 else 0
    
    print("\nStrategy Performance:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Annualized Return: {ann_return:.2%}")
    print(f"Annualized Buy & Hold: {ann_buy_hold:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    print("\nDemo completed! Check strategy_results.png for visualization.")


if __name__ == "__main__":
    main()
