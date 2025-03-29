"""Example script for market regime detection."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.figure
import matplotlib.cm as cm
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, cast, Union
from datetime import datetime, timedelta

from .factory import RegimeDetectorFactory
from .base_detector import BaseRegimeDetector


def download_market_data(symbol: str = 'SPY', period: str = '2y') -> Dict[str, Any]:
    """
    Download market data for a given symbol.
    
    Args:
        symbol: Ticker symbol (default: 'SPY')
        period: Time period (default: '2y')
        
    Returns:
        Dictionary containing market data
    """
    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    # Calculate returns
    df["Return"] = df['Close'].pct_change()
    
    # Convert to dictionary
    data = {
        'symbol': symbol,
        'dates': df.index.tolist(),
        'prices': df['Close'].values.tolist(),
        'returns': df['Return'].fillna(0).values.tolist(),
        'volumes': df['Volume'].values.tolist(),
        'highs': df['High'].values.tolist(),
        'lows': df['Low'].values.tolist()
    }
    
    return data


def plot_regimes(data: Dict[str, Any], labels: List[int], title: str) -> matplotlib.figure.Figure:
    """
    Plot market data with regime labels.
    
    Args:
        data: Dictionary containing market data
        labels: List of regime labels
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.gcf()
    
    # Get data
    dates = data['dates']
    prices = data['prices']
    
    # Plot prices
    plt.plot(dates, prices, color='black', alpha=0.6)
    
    # Color background by regime
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    y_min = min_price - 0.1 * price_range
    y_max = max_price + 0.1 * price_range
    
    # Get unique regimes
    unique_regimes = sorted(set(labels))
    
    # Create colors for regimes
    cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(unique_regimes))]
    
    # Plot regimes
    for i, regime in enumerate(unique_regimes):
        regime_indices = [j for j, label in enumerate(labels) if label == regime]
        for idx in regime_indices:
            if idx < len(dates) - 1:
                plt.axvspan(dates[idx], dates[idx + 1], alpha=0.3, color=colors[i])
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    handles = [mpatches.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.3) for i in range(len(unique_regimes))]
    regime_labels = [f"Regime {regime}" for regime in unique_regimes]
    plt.legend(handles, regime_labels, loc='upper left')
    
    return fig


def main():
    """Run the example."""
    # Get parameters from environment variables or use defaults
    symbol = os.environ.get("EXAMPLE_SYMBOL", "SPY")
    period = os.environ.get("EXAMPLE_PERIOD", "2y")
    output_dir = os.environ.get("EXAMPLE_OUTPUT_DIR", ".")
    n_regimes = int(os.environ.get("EXAMPLE_N_REGIMES", "3"))
    trend_method = os.environ.get("EXAMPLE_TREND_METHOD", "ma_crossover")
    
    # Download market data
    print(f"Downloading market data for {symbol} over {period}...")
    data = download_market_data(symbol=symbol, period=period)
    print(f"Downloaded data for {data['symbol']} with {len(data['prices'])} data points")
    
    # Create detectors
    factory = RegimeDetectorFactory()
    
    # Determine which methods to use
    methods_str = os.environ.get("EXAMPLE_METHODS", "all")
    if methods_str == "all":
        available_methods = factory.get_available_methods()
    else:
        available_methods = methods_str.split(",")
    
    print(f"Using methods: {', '.join(available_methods)}")
    
    # Create all detectors with explicit typing
    detectors: Dict[str, BaseRegimeDetector] = {}
    
    for method in available_methods:
        if method == "volatility":
            detectors[method] = factory.create(method, n_regimes=n_regimes)
        elif method == "momentum":
            detectors[method] = factory.create(method, n_regimes=n_regimes, momentum_type='roc')
        elif method == "hmm":
            detectors[method] = factory.create(method, n_regimes=n_regimes)
        elif method == "trend":
            detectors[method] = factory.create(method, n_regimes=n_regimes, trend_method=trend_method)
        elif method == "ensemble":
            # Create ensemble detector with all other methods
            ensemble_methods = [m for m in available_methods if m != "ensemble"]
            if ensemble_methods:
                detectors[method] = factory.create(
                    method,
                    n_regimes=n_regimes,
                    methods=ensemble_methods,
                    voting='soft',
                    ensemble_type='bagging'
                )
            else:
                # If no other methods specified, use default methods
                detectors[method] = factory.create(
                    method,
                    n_regimes=n_regimes
                )
    
    # Detect regimes and plot
    plt.figure(figsize=(15, 10))
    n_detectors = len(detectors)
    
    for i, (name, detector) in enumerate(detectors.items()):
        # Detect regimes
        print(f"Detecting regimes using {name} method...")
        labels = detector.fit_predict(data)
        
        # Plot
        plt.subplot(n_detectors, 1, i + 1)
        fig = plot_regimes(data, labels, f"{name.capitalize()} Regimes")
        
        # Print statistics
        print(f"\n{name.upper()} REGIME STATISTICS:")
        for regime, stats in detector.regime_stats.items():
            if regime not in ['transitions', 'transitions_per_period']:
                # Convert regime to int if it's a string that can be converted
                regime_int = int(regime) if isinstance(regime, (int, str)) and str(regime).isdigit() else 0
                regime_name = detector.get_regime_name(regime_int)
                print(f"  {regime_name}:")
                for stat_name, stat_value in stats.items():
                    print(f"    {stat_name}: {stat_value:.4f}")
        
        # Print additional info for ensemble detector
        if name == "ensemble" and hasattr(detector, "get_detector_weights"):
            weights = detector.get_detector_weights()
            detector_names = detector.get_detector_names()
            print("\n  ENSEMBLE WEIGHTS:")
            for detector_name, weight in zip(detector_names, weights):
                print(f"    {detector_name}: {weight:.4f}")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{symbol}_regimes_comparison.png")
    plt.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")
    plt.show()
    
    # Additional plot for ensemble detector
    ensemble_detector = detectors.get('ensemble')
    if ensemble_detector and hasattr(ensemble_detector, 'get_ensemble_probas'):
        ensemble_probas = ensemble_detector.get_ensemble_probas()
        
        if ensemble_probas is not None:
            plt.figure(figsize=(12, 6))
            
            # Plot ensemble probabilities
            for i in range(n_regimes):
                regime_name = ensemble_detector.get_regime_name(i)
                plt.plot(data['dates'], ensemble_probas[:, i], label=regime_name)
            
            plt.title("Ensemble Regime Probabilities")
            plt.xlabel("Date")
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            ensemble_plot_path = os.path.join(output_dir, f"{symbol}_ensemble_probas.png")
            plt.savefig(ensemble_plot_path)
            print(f"Saved ensemble probabilities plot to {ensemble_plot_path}")
            plt.show()
    
    # Additional plots for trend detector
    trend_detector = detectors.get('trend')
    if trend_detector and hasattr(trend_detector, 'get_trend_series'):
        trend_series = trend_detector.get_trend_series()
        
        if trend_series is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(data['dates'], trend_series)
            
            # Get trend method if available
            trend_method = getattr(trend_detector, 'trend_method', 'unknown')
            plt.title(f"Trend Indicator ({trend_method})")
            plt.xlabel('Date')
            plt.ylabel('Trend Strength')
            
            # Add horizontal lines for thresholds if available
            if hasattr(trend_detector, 'get_regime_thresholds'):
                thresholds = trend_detector.get_regime_thresholds()
                if thresholds:
                    for threshold in thresholds:
                        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
            
            plt.grid(True, alpha=0.3)
            trend_output_path = os.path.join(output_dir, f"{symbol}_trend_indicator.png")
            plt.savefig(trend_output_path)
            print(f"Saved trend indicator plot to {trend_output_path}")
            plt.show()


if __name__ == "__main__":
    main() 