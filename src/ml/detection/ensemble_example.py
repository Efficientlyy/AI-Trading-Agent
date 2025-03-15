"""Example script demonstrating the ensemble regime detector."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    df['Return'] = df['Close'].pct_change()
    
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


def plot_regimes_comparison(data: Dict[str, Any], detectors_results: Dict[str, Dict[str, Any]], title: str = ""):
    """
    Plot regime comparison for multiple detectors.
    
    Args:
        data: Dictionary containing market data
        detectors_results: Dictionary of detector results
        title: Plot title (default: empty string)
    """
    # Get number of detectors
    n_detectors = len(detectors_results)
    
    # Create figure
    fig, axes = plt.subplots(n_detectors, 1, figsize=(15, 4 * n_detectors), sharex=True)
    
    if n_detectors == 1:
        axes = [axes]
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Plot each detector
    for i, (name, result) in enumerate(detectors_results.items()):
        ax = axes[i]
        
        # Get data
        dates = data['dates']
        prices = data['prices']
        labels = result['labels']
        detector = result['detector']
        
        # Plot prices
        ax.plot(dates, prices, color='black', alpha=0.6, label='Price')
        
        # Get unique regimes
        unique_regimes = sorted(set(labels))
        
        # Get colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis', len(unique_regimes))
        
        # Create background for each regime
        for j, regime in enumerate(unique_regimes):
            # Get regime name
            regime_name = detector.get_regime_name(regime)
            
            # Create mask for current regime
            mask = np.array(labels) == regime
            
            # Fill background
            ax.fill_between(dates, ax.get_ylim()[0], ax.get_ylim()[1],
                           where=mask, color=cmap(j), alpha=0.3,
                           label=regime_name)
        
        # Set title and labels
        ax.set_title(f"{name.capitalize()} Detector")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    # Set x label on bottom plot
    axes[-1].set_xlabel("Date")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_ensemble_probabilities(data: Dict[str, Any], ensemble_detector: BaseRegimeDetector):
    """
    Plot ensemble probabilities.
    
    Args:
        data: Dictionary containing market data
        ensemble_detector: Ensemble detector instance
    """
    # Get ensemble probabilities
    ensemble_probas = ensemble_detector.get_ensemble_probas()
    
    if ensemble_probas is None:
        print("No ensemble probabilities available")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot probabilities for each regime
    for i in range(ensemble_detector.n_regimes):
        regime_name = ensemble_detector.get_regime_name(i)
        ax.plot(data['dates'], ensemble_probas[:, i], label=regime_name)
    
    # Set title and labels
    ax.set_title("Ensemble Regime Probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_detector_weights(weights: List[float], detector_names: List[str]):
    """
    Plot detector weights.
    
    Args:
        weights: List of detector weights
        detector_names: List of detector names
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart of weights
    bars = ax.bar(detector_names, weights, alpha=0.7)
    
    # Set title and labels
    ax.set_title("Ensemble Detector Weights")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Weight")
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f"{height:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def run_ensemble_example():
    """Run the ensemble detector example."""
    print("Ensemble Regime Detector Example")
    print("================================")
    
    # Download market data
    symbol = 'SPY'
    period = '3y'
    print(f"\nDownloading market data for {symbol} over {period}...")
    data = download_market_data(symbol=symbol, period=period)
    print(f"Downloaded {len(data['prices'])} data points")
    
    # Create factory
    factory = RegimeDetectorFactory()
    
    # Create individual detectors
    print("\nCreating individual detectors...")
    individual_detectors = {
        'volatility': factory.create('volatility', n_regimes=3, vol_window=21),
        'momentum': factory.create('momentum', n_regimes=3, momentum_type='roc'),
        'trend': factory.create('trend', n_regimes=3, trend_method='ma_crossover'),
        'hmm': factory.create('hmm', n_regimes=3, hmm_type='gaussian')
    }
    
    # Detect regimes with individual detectors
    print("\nDetecting regimes with individual detectors...")
    individual_results = {}
    
    for name, detector in individual_detectors.items():
        print(f"  - {name}...")
        labels = detector.fit_predict(data)
        individual_results[name] = {
            'detector': detector,
            'labels': labels
        }
    
    # Create ensemble detectors with different techniques
    print("\nCreating ensemble detectors with different techniques...")
    ensemble_detectors = {
        'ensemble_bagging': factory.create(
            'ensemble',
            n_regimes=3,
            methods=list(individual_detectors.keys()),
            voting='soft',
            ensemble_type='bagging'
        ),
        'ensemble_boosting': factory.create(
            'ensemble',
            n_regimes=3,
            methods=list(individual_detectors.keys()),
            voting='soft',
            ensemble_type='boosting'
        ),
        'ensemble_stacking': factory.create(
            'ensemble',
            n_regimes=3,
            methods=list(individual_detectors.keys()),
            voting='soft',
            ensemble_type='stacking'
        )
    }
    
    # Detect regimes with ensemble detectors
    print("\nDetecting regimes with ensemble detectors...")
    ensemble_results = {}
    
    for name, detector in ensemble_detectors.items():
        print(f"  - {name}...")
        labels = detector.fit_predict(data)
        ensemble_results[name] = {
            'detector': detector,
            'labels': labels
        }
        
        # Print ensemble weights
        weights = detector.get_detector_weights()
        detector_names = detector.get_detector_names()
        print(f"    Weights: ", end="")
        for detector_name, weight in zip(detector_names, weights):
            print(f"{detector_name}={weight:.2f} ", end="")
        print()
    
    # Plot regime comparison for individual detectors
    print("\nPlotting regime comparison for individual detectors...")
    plot_regimes_comparison(data, individual_results, "Individual Detector Regimes")
    
    # Plot regime comparison for ensemble detectors
    print("\nPlotting regime comparison for ensemble detectors...")
    plot_regimes_comparison(data, ensemble_results, "Ensemble Detector Regimes")
    
    # Plot ensemble probabilities for bagging
    print("\nPlotting ensemble probabilities for bagging...")
    plot_ensemble_probabilities(data, ensemble_detectors['ensemble_bagging'])
    
    # Plot detector weights for boosting
    print("\nPlotting detector weights for boosting...")
    boosting_weights = ensemble_detectors['ensemble_boosting'].get_detector_weights()
    boosting_names = ensemble_detectors['ensemble_boosting'].get_detector_names()
    plot_detector_weights(boosting_weights, boosting_names)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_ensemble_example() 