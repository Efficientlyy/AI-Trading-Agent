"""
Integration Validation for Adaptive Response System

This script demonstrates and validates the full integration of the
Market Regime Classification and Adaptive Response System.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our components
from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
from ai_trading_agent.market_regime import MarketRegimeConfig, MarketRegimeType, VolatilityRegimeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """Create synthetic test data for different market regimes."""
    # Create date range for last 2 years of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic price data
    n = len(dates)
    
    # Base trend
    base_trend = np.linspace(300, 450, n)
    
    # Add cycles and noise
    cycle1 = 20 * np.sin(np.linspace(0, 15 * np.pi, n))  # Long cycle
    cycle2 = 10 * np.sin(np.linspace(0, 50 * np.pi, n))  # Short cycle
    noise = np.random.normal(0, 5, n)
    
    # COVID crash effect (sharp drop and recovery)
    covid_effect = np.zeros(n)
    crash_start = int(n * 0.25)  # 25% into the series
    crash_end = crash_start + 40
    recovery_end = crash_end + 80
    
    # Generate crash and recovery
    for i in range(crash_start, crash_end):
        covid_effect[i] = -120 * (i - crash_start) / (crash_end - crash_start)
    
    for i in range(crash_end, recovery_end):
        covid_effect[i] = -120 * (1 - (i - crash_end) / (recovery_end - crash_end))
    
    # Create final price series
    prices = base_trend + cycle1 + cycle2 + noise + covid_effect
    
    # Create synthetic volume data (higher in volatile periods)
    base_volume = 5000000 * np.ones(n)
    volume_cycle = 2000000 * np.sin(np.linspace(0, 25 * np.pi, n)) ** 2
    volume_noise = np.random.normal(0, 500000, n)
    
    # Volume spikes during crash
    volume_spikes = np.zeros(n)
    for i in range(crash_start, crash_end + 20):
        spike_factor = 1 + 3 * np.exp(-0.1 * abs(i - crash_start - 10))
        volume_spikes[i] = 5000000 * spike_factor
    
    volumes = base_volume + volume_cycle + volume_noise + volume_spikes
    volumes = np.maximum(volumes, 1000000)  # Ensure minimum volume
    
    # Calculate highs and lows (simple approximation)
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
    
    # Store in a dictionary
    data = {
        'SPY': {
            'prices': pd.Series(prices, index=dates),
            'volume': pd.Series(volumes, index=dates),
            'high': pd.Series(highs, index=dates),
            'low': pd.Series(lows, index=dates)
        }
    }
    
    # Add a few more assets with correlations to SPY
    tickers = ['QQQ', 'IWM', 'TLT', 'GLD']
    correlations = [0.9, 0.85, -0.4, -0.2]  # Correlation with SPY
    
    for ticker, corr in zip(tickers, correlations):
        # Create correlated prices
        correlated_noise = np.random.normal(0, 0.015 * prices, n)
        ticker_prices = corr * prices + np.sqrt(1 - corr**2) * correlated_noise
        
        # Add to data dictionary
        data[ticker] = {
            'prices': pd.Series(ticker_prices, index=dates)
        }
    
    return data


class SimpleMarketDataProvider:
    """Simple market data provider for testing."""
    
    def __init__(self, use_test_data=True):
        self.use_test_data = use_test_data
        self.test_data = create_test_data()
    
    def get_market_data(self):
        """Return market data for testing."""
        return self.test_data


def test_adaptive_orchestrator():
    """Test the AdaptiveHealthOrchestrator with synthetic data."""
    logger.info("Starting Adaptive Health Orchestrator validation test")
    
    # Create the orchestrator with default settings
    orchestrator = AdaptiveHealthOrchestrator(
        regime_config=MarketRegimeConfig(),
        temporal_pattern_enabled=True,
        adaptation_interval_minutes=0.01  # Set low for testing
    )
    
    # Create and register the market data provider
    data_provider = SimpleMarketDataProvider()
    orchestrator.register_market_data_source('market_data', data_provider)
    
    # Run a cycle to trigger market regime detection and adaptation
    logger.info("Running orchestrator cycle...")
    results = orchestrator.run_cycle()
    
    # Get the detected market regime
    global_regime = orchestrator.get_current_regime('global')
    
    # Log the results
    logger.info(f"Detected market regime: {global_regime.get('regime_type')}")
    logger.info(f"Volatility regime: {global_regime.get('volatility_type')}")
    logger.info(f"Confidence: {global_regime.get('confidence'):.2f}")
    
    # Get regime statistics
    stats = orchestrator.get_regime_statistics()
    
    if stats and 'regime_counts' in stats:
        logger.info("Regime statistics:")
        for regime, count in stats['regime_counts'].items():
            logger.info(f"  {regime}: {count}")
    
    return {
        'orchestrator': orchestrator,
        'results': results,
        'global_regime': global_regime
    }


def plot_market_data_with_regimes(data, orchestrator):
    """Plot market data with detected regimes."""
    spy_data = data['SPY']
    prices = spy_data['prices']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot prices
    prices[-180:].plot(ax=ax, label='SPY Price')
    
    # Get current regime
    global_regime = orchestrator.get_current_regime('global')
    
    # Add title with regime information
    ax.set_title(f"Last 180 Days with Detected Regime: {global_regime.get('regime_type')}\n"
                f"Volatility: {global_regime.get('volatility_type')}, "
                f"Confidence: {global_regime.get('confidence'):.2f}")
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add background color based on regime
    regime_colors = {
        'BULL': 'lightgreen',
        'BEAR': 'lightcoral',
        'VOLATILE': 'lightyellow',
        'SIDEWAYS': 'lightblue',
        'TRENDING': 'lightcyan',
        'CHOPPY': 'lavender',
        'BREAKDOWN': 'mistyrose',
        'RECOVERY': 'palegreen'
    }
    
    # Set background color
    regime_type = global_regime.get('regime_type')
    ax.set_facecolor(regime_colors.get(regime_type, 'white'))
    
    # Save the figure
    save_path = os.path.join(os.path.dirname(__file__), 'market_regime_integration_test.png')
    plt.savefig(save_path)
    logger.info(f"Plot saved to: {save_path}")
    
    # Show the figure
    plt.show()


if __name__ == "__main__":
    # Run the integration test
    validation_results = test_adaptive_orchestrator()
    
    # Plot the results
    plot_market_data_with_regimes(
        create_test_data(),
        validation_results['orchestrator']
    )
    
    logger.info("Integration validation complete!")
