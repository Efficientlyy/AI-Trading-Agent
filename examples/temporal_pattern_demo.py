"""
Temporal Pattern Recognition Demo

This script demonstrates the capabilities of the Temporal Pattern Recognition System
for market regimes. It showcases:
1. Seasonality detection in market data
2. Regime transition probability modeling
3. Multi-timeframe confirmation logic
4. Integrated temporal pattern recognition

The demo uses historical market data to analyze and visualize these features.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.market_regime import (
    MarketRegimeType,
    VolatilityRegimeType,
    SeasonalityDetector,
    TransitionProbabilityModel,
    MultiTimeframeConfirmation,
    TemporalPatternRecognition,
    MarketRegimeInfo
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_market_data(symbols, start_date, end_date, interval='1d'):
    """
    Fetch historical market data for the specified symbols.
    """
    data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Skip if data is empty
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            continue
            
        data[symbol] = df
        logger.info(f"Retrieved {len(df)} periods of data for {symbol}")
    
    return data


def demo_seasonality_detection(price_data, asset_id='SPY'):
    """
    Demonstrate seasonality detection capabilities.
    """
    logger.info("\n=== Seasonality Detection Demo ===")
    
    # Create detector
    detector = SeasonalityDetector()
    
    # Check for seasonality
    result = detector.detect_seasonality(price_data['Close'], asset_id)
    
    # Display results
    print(f"Has seasonality: {result['has_seasonality']}")
    if result['has_seasonality']:
        print("\nDetected seasonal periods:")
        for period in result['acf_results']['seasonal_periods']:
            print(f"  - Period: {period['period']} with ACF value: {period['acf_value']:.4f}")
    
    # Generate a forecast if seasonality is detected
    if result['has_seasonality']:
        forecast = detector.get_seasonal_forecast(
            price_data['Close'], 
            asset_id,
            forecast_periods=30
        )
        
        # Plot original data and forecast
        plt.figure(figsize=(12, 6))
        plt.plot(price_data.index[-100:], price_data['Close'][-100:], label='Historical price')
        
        # Create forecast dates
        last_date = price_data.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(forecast['forecast']) + 1)]
        
        plt.plot(forecast_dates, forecast['forecast'], label='Seasonal forecast', linestyle='--')
        
        plt.title(f'Seasonality Analysis for {asset_id}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    # Check for calendar patterns
    if result['calendar_patterns']:
        print("\nCalendar patterns detected:")
        for pattern, details in result['calendar_patterns'].items():
            print(f"  - {pattern}: Mean effect: {details['mean_effect']:.4f}%, Consistency: {details['consistency']:.2f}")


def demo_transition_probability(price_data, asset_id='SPY'):
    """
    Demonstrate regime transition probability modeling.
    """
    logger.info("\n=== Regime Transition Probability Demo ===")
    
    # Create regime info objects for historical data
    # This is a simplified example - in practice, you'd use the actual regime classifier
    
    # Create transition model
    model = TransitionProbabilityModel()
    
    # Generate synthetic regime history for demonstration
    logger.info("Generating synthetic regime history for demonstration...")
    regimes = [MarketRegimeType.BULL, MarketRegimeType.TRENDING, MarketRegimeType.BEAR, 
               MarketRegimeType.VOLATILE, MarketRegimeType.SIDEWAYS]
    
    # Create synthetic regime changes
    for i, date in enumerate(price_data.index):
        # Simple algorithm to assign regimes based on price changes
        pct_change = price_data['Close'].pct_change().iloc[i] if i > 0 else 0
        volatility = price_data['Close'].pct_change().rolling(20).std().iloc[i] if i >= 20 else 0
        
        # Determine regime based on price action (simplified)
        if pct_change > 0.01:
            regime = MarketRegimeType.BULL
        elif pct_change < -0.01:
            regime = MarketRegimeType.BEAR
        elif volatility > 0.02:
            regime = MarketRegimeType.VOLATILE
        elif abs(pct_change) < 0.003:
            regime = MarketRegimeType.SIDEWAYS
        else:
            regime = MarketRegimeType.TRENDING
            
        # Create regime info
        regime_info = MarketRegimeInfo(
            regime_type=regime,
            confidence=0.8,
            volatility_regime=VolatilityRegimeType.HIGH if volatility > 0.02 else VolatilityRegimeType.LOW,
            liquidity_regime=None,
            correlation_regime=None,
            metrics={},
            timestamp=date
        )
        
        # Add to model
        model.add_regime_observation(regime_info, asset_id)
    
    # Build transition matrix
    logger.info("Building transition probability matrix...")
    transition_matrix = model.build_transition_matrix(asset_id)
    
    # Display matrix
    print("\nRegime Transition Probability Matrix:")
    for start_regime, transitions in transition_matrix['transition_matrix'].items():
        print(f"\nFrom {start_regime}:")
        for end_regime, prob in transitions.items():
            print(f"  → To {end_regime}: {prob:.3f}")
    
    # Display regime counts
    print("\nRegime Distribution:")
    total_count = sum(transition_matrix['regime_counts'].values())
    for regime, count in transition_matrix['regime_counts'].items():
        print(f"  {regime}: {count} observations ({count/total_count*100:.1f}%)")
    
    # Get stability of each regime
    stability = model.get_regime_stability(asset_id)
    print("\nRegime Stability (probability of remaining in the same regime):")
    for regime, prob in stability.items():
        print(f"  {regime}: {prob:.3f}")
    
    # Current regime for prediction
    current_regime = MarketRegimeType.TRENDING
    print(f"\nPredicting next regime from current regime: {current_regime.value}")
    
    # Predict next regime
    next_regime_probs = model.predict_next_regime(current_regime, asset_id)
    print("\nNext Regime Probabilities:")
    for regime, prob in sorted(next_regime_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {prob:.3f}")
    
    # Most likely transition
    most_likely, prob = model.get_most_likely_transition(current_regime, asset_id)
    if most_likely:
        print(f"\nMost likely transition: {current_regime.value} → {most_likely} (probability: {prob:.3f})")
    else:
        print(f"\nNo significant regime transition likely from {current_regime.value}")


def demo_multi_timeframe_confirmation(data, symbol='SPY'):
    """
    Demonstrate multi-timeframe confirmation logic.
    """
    logger.info("\n=== Multi-Timeframe Confirmation Demo ===")
    
    # Create multi-timeframe confirmation object
    mtf = MultiTimeframeConfirmation(
        timeframes=['1D', '1W', '1M'],
        weights={'1D': 1.0, '1W': 2.0, '1M': 3.0}
    )
    
    # Analyze with multi-timeframe confirmation
    result = mtf.analyze_multi_timeframe(
        data=data,
        asset_id=symbol,
        volumes=data['Volume']
    )
    
    # Display results
    print(f"Confirmed regime: {result['confirmed_regime']}")
    print(f"Agreement score: {result['agreement_score']:.3f}")
    
    print("\nTimeframe Analysis:")
    for tf, details in result['timeframe_regimes'].items():
        print(f"  {tf}: {details['regime']} (confidence: {details['confidence']:.3f})")
    
    # Check for divergence
    divergence = mtf.detect_divergence(symbol)
    if divergence['has_divergence']:
        print("\nDivergence detected:")
        for tf_info in divergence['divergent_timeframes']:
            print(f"  {tf_info['timeframe']} shows {tf_info['regime']} (different from confirmed regime)")
        print(f"Divergence score: {divergence['divergence_score']:.3f}")
    else:
        print("\nNo significant divergence between timeframes")


def demo_integrated_temporal_patterns(data, symbol='SPY'):
    """
    Demonstrate the integrated temporal pattern recognition system.
    """
    logger.info("\n=== Integrated Temporal Pattern Recognition Demo ===")
    
    # Create temporal pattern recognition object
    tpr = TemporalPatternRecognition()
    
    # Analyze temporal patterns
    result = tpr.analyze_temporal_patterns(
        prices=data['Close'],
        asset_id=symbol,
        volumes=data['Volume'],
        ohlcv_data=data
    )
    
    # Display comprehensive results
    print(f"Current regime: {result['current_regime']['regime_type']}")
    print(f"Volatility regime: {result['current_regime']['volatility_regime']}")
    print(f"Confidence: {result['current_regime']['confidence']:.3f}")
    
    print("\nSeasonality:")
    print(f"  Has seasonality: {result['seasonality']['has_seasonality']}")
    if result['seasonality']['seasonal_periods']:
        print("  Detected periods:", [p for p in result['seasonality']['seasonal_periods'] if 'period' in p])
    
    print("\nTransition Probabilities:")
    if result['transition_probabilities']['next_regime_probabilities']:
        for regime, prob in sorted(result['transition_probabilities']['next_regime_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {regime}: {prob:.3f}")
    else:
        print("  Insufficient history for transition probabilities")
    
    print("\nMulti-Timeframe Analysis:")
    print(f"  Confirmed regime: {result['multi_timeframe']['confirmed_regime']}")
    print(f"  Agreement score: {result['multi_timeframe']['agreement_score']:.3f}")
    
    # Check for alignment signal
    alignment = tpr.get_timeframe_alignment_signal(symbol)
    if alignment['has_alignment']:
        print(f"\nTimeframe Alignment Signal: {alignment['aligned_regime']} with score {alignment['agreement_score']:.3f}")
        print(f"  Aligned timeframes: {alignment['aligned_timeframes']}")
    
    # Check for transition opportunity
    opportunity = tpr.detect_regime_transition_opportunity(symbol)
    if opportunity['transition_opportunity']:
        print(f"\nRegime Transition Opportunity Detected!")
        print(f"  Current: {opportunity['current_regime']} → Potential: {opportunity['potential_next_regime']}")
        print(f"  Confidence: {opportunity['confidence']:.3f}")
        print(f"  Timeframe confirmation: {opportunity['has_timeframe_confirmation']}")
        print(f"  Seasonal alignment: {opportunity['has_seasonal_alignment']}")
    else:
        print("\nNo significant regime transition opportunity detected")


def main():
    # Set up
    sns.set(style="whitegrid")
    
    # Define assets and date range
    symbols = ['SPY', 'QQQ', 'TLT', 'GLD']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    try:
        market_data = fetch_market_data(symbols, start_date, end_date)
        if not market_data:
            logger.error("Failed to fetch market data")
            return
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return
    
    # Run demos
    symbol = 'SPY'  # Use S&P 500 ETF for demos
    if symbol in market_data:
        print(f"\nRunning demos using {symbol} data from {start_date} to {end_date}")
        
        # 1. Seasonality Detection Demo
        demo_seasonality_detection(market_data[symbol], symbol)
        
        # 2. Transition Probability Demo
        demo_transition_probability(market_data[symbol], symbol)
        
        # 3. Multi-Timeframe Confirmation Demo
        demo_multi_timeframe_confirmation(market_data[symbol], symbol)
        
        # 4. Integrated Temporal Patterns Demo
        demo_integrated_temporal_patterns(market_data[symbol], symbol)
        
    else:
        logger.error(f"Data for {symbol} is not available")


if __name__ == "__main__":
    main()
