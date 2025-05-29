"""
Market Regime Classification Demo

This script demonstrates how to use the Market Regime Classification system
to detect market regimes using historical price and volume data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Ensure the ai_trading_agent package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_trading_agent.market_regime import (
    MarketRegimeClassifier,
    MarketRegimeConfig,
    MarketRegimeType,
    VolatilityRegimeType
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_historical_data(tickers, period="1y", interval="1d"):
    """
    Fetch historical data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        period: Data period (e.g., '1y' for 1 year)
        interval: Data interval (e.g., '1d' for daily)
        
    Returns:
        Dictionary of DataFrames with price and volume data
    """
    data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if not hist.empty:
                data[ticker] = {
                    'prices': hist['Close'],
                    'high': hist['High'],
                    'low': hist['Low'],
                    'volume': hist['Volume'],
                    'returns': hist['Close'].pct_change().fillna(0)
                }
                logger.info(f"Downloaded {len(hist)} data points for {ticker}")
            else:
                logger.warning(f"No data found for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    return data


def plot_regime_classification(prices, regimes, volatility, asset_id):
    """
    Plot price chart with regime classifications.
    
    Args:
        prices: Price series
        regimes: Series of regime classifications
        volatility: Series of volatility values
        asset_id: Asset identifier for the chart title
    """
    plt.figure(figsize=(14, 10))
    
    # Create 3 subplots
    ax1 = plt.subplot(3, 1, 1)  # Price chart
    ax2 = plt.subplot(3, 1, 2)  # Regime chart
    ax3 = plt.subplot(3, 1, 3)  # Volatility chart
    
    # Plot prices
    ax1.plot(prices.index, prices.values)
    ax1.set_title(f'{asset_id} Price')
    ax1.grid(True)
    
    # Plot regimes as background colors
    unique_regimes = regimes.unique()
    colors = {
        MarketRegimeType.BULL.value: 'green',
        MarketRegimeType.BEAR.value: 'red',
        MarketRegimeType.SIDEWAYS.value: 'gray',
        MarketRegimeType.VOLATILE.value: 'orange',
        MarketRegimeType.TRENDING.value: 'blue',
        MarketRegimeType.CHOPPY.value: 'purple',
        MarketRegimeType.BREAKDOWN.value: 'darkred',
        MarketRegimeType.RECOVERY.value: 'lightgreen',
        MarketRegimeType.UNKNOWN.value: 'lightgray'
    }
    
    # Find regime change points
    changes = []
    current_regime = regimes.iloc[0]
    changes.append((regimes.index[0], current_regime))
    
    for i, regime in enumerate(regimes):
        if regime != current_regime:
            changes.append((regimes.index[i], regime))
            current_regime = regime
    
    # Plot colored background for each regime
    for i in range(len(changes) - 1):
        start_date = changes[i][0]
        end_date = changes[i + 1][0]
        regime = changes[i][1]
        ax1.axvspan(start_date, end_date, color=colors.get(regime, 'gray'), alpha=0.2)
    
    # Plot last regime to the end
    if changes:
        ax1.axvspan(changes[-1][0], prices.index[-1], 
                   color=colors.get(changes[-1][1], 'gray'), alpha=0.2)
    
    # Plot regimes as a categorical series
    regime_codes = [list(MarketRegimeType.__members__.values()).index(MarketRegimeType(r)) 
                    for r in regimes]
    ax2.plot(regimes.index, regime_codes, 'ko-', markersize=4)
    ax2.set_yticks(range(len(MarketRegimeType.__members__)))
    ax2.set_yticklabels([r.value for r in MarketRegimeType])
    ax2.set_title('Market Regime Classification')
    ax2.grid(True)
    
    # Plot volatility
    ax3.plot(volatility.index, volatility.values, 'r-')
    ax3.set_title('Rolling Volatility (annualized)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demo function."""
    logger.info("Starting Market Regime Classification Demo")
    
    # Configuration
    config = MarketRegimeConfig(
        lookback_period=60,
        short_lookback=20,
        medium_lookback=60,
        long_lookback=120,
        volatility_window=20
    )
    
    # Initialize classifier
    classifier = MarketRegimeClassifier(config)
    
    # Define asset groups for correlation analysis
    classifier.correlation_analyzer.register_asset_group(
        "tech", ["AAPL", "MSFT", "GOOG", "AMZN"]
    )
    classifier.correlation_analyzer.register_asset_group(
        "finance", ["JPM", "BAC", "GS", "WFC"]
    )
    classifier.correlation_analyzer.register_asset_group(
        "energy", ["XOM", "CVX", "COP", "SLB"]
    )
    classifier.correlation_analyzer.register_asset_group(
        "safe_assets", ["TLT", "GLD", "IEF"]
    )
    classifier.correlation_analyzer.register_asset_group(
        "risk_assets", ["SPY", "QQQ", "IWM"]
    )
    
    # Fetch historical data
    tickers = [
        "SPY",  # S&P 500 ETF
        "QQQ",  # Nasdaq 100 ETF
        "IWM",  # Russell 2000 ETF
        "TLT",  # 20+ Year Treasury ETF
        "GLD",  # Gold ETF
        "AAPL", "MSFT", "GOOG", "AMZN",  # Tech
        "JPM", "BAC", "GS", "WFC",       # Finance
        "XOM", "CVX", "COP", "SLB"       # Energy
    ]
    
    market_data = fetch_historical_data(tickers, period="2y", interval="1d")
    
    if not market_data:
        logger.error("Failed to fetch market data")
        return
    
    # Focus on SPY for primary analysis
    primary_asset = "SPY"
    if primary_asset not in market_data:
        logger.error(f"{primary_asset} data not available")
        return
    
    # Classify regimes for each day in the last 6 months
    prices = market_data[primary_asset]['prices']
    volumes = market_data[primary_asset]['volume']
    high_prices = market_data[primary_asset]['high']
    low_prices = market_data[primary_asset]['low']
    
    # Use the most recent 6 months
    start_date = prices.index[-1] - pd.Timedelta(days=180)
    mask = prices.index >= start_date
    recent_prices = prices[mask]
    
    # Dictionary to store regimes and volatility
    regimes = {}
    volatilities = {}
    
    # Create related_assets dictionary for correlation analysis
    related_assets = {ticker: data for ticker, data in market_data.items() if ticker != primary_asset}
    
    # Classify regime for each day using a rolling window
    for i in range(30, len(recent_prices)):
        date = recent_prices.index[i]
        
        # Use data up to this date
        price_window = prices[:date]
        volume_window = volumes[:date] if volumes is not None else None
        high_window = high_prices[:date] if high_prices is not None else None
        low_window = low_prices[:date] if low_prices is not None else None
        
        # Prepare related assets data up to this date
        related_window = {}
        for ticker, data in related_assets.items():
            related_window[ticker] = {
                'prices': data['prices'][:date] if 'prices' in data else None,
                'returns': data['returns'][:date] if 'returns' in data else None
            }
        
        # Classify regime
        regime_info = classifier.classify_regime(
            prices=price_window,
            volumes=volume_window,
            high_prices=high_window,
            low_prices=low_window,
            asset_id=primary_asset,
            related_assets=related_window
        )
        
        # Store results
        regimes[date] = regime_info.regime_type.value
        volatilities[date] = regime_info.metrics.get('volatility', {}).get('current_volatility')
    
    # Convert to series for plotting
    regime_series = pd.Series(regimes)
    volatility_series = pd.Series(volatilities)
    
    # Plot results
    plot_regime_classification(recent_prices, regime_series, volatility_series, primary_asset)
    
    # Print regime statistics
    stats = classifier.get_regime_statistics(primary_asset)
    print("\nRegime Statistics:")
    print(f"Regime Counts: {stats['regime_counts']}")
    if stats.get('average_duration'):
        print("\nAverage Duration (days):")
        for regime, duration in stats['average_duration'].items():
            print(f"  {regime}: {duration:.1f}")
    
    # Get and print regime transitions
    transitions = classifier.get_regime_transitions(primary_asset)
    if transitions:
        print("\nRegime Transitions:")
        for t in transitions[-5:]:  # Show last 5 transitions
            print(f"  {t['timestamp']}: {t['from_regime']} -> {t['to_regime']} ({t['significance']})")
    
    logger.info("Market Regime Classification Demo completed")


if __name__ == "__main__":
    main()
