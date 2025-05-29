"""
Adaptive System Integration Test

This script tests the integration between the Market Regime Classification system 
and the Adaptive Response System, including temporal pattern recognition, regime-adaptive
parameter adjustment, and strategy switching.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules
from ai_trading_agent.market_regime import (
    MarketRegimeType,
    TemporalPatternRecognition,
    TemporalPatternOptimizer
)
from ai_trading_agent.agent.adaptive_manager import AdaptiveStrategyManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_data(tickers=['SPY', 'QQQ', 'IWM', 'TLT'], start_date='2020-01-01'):
    """
    Fetch historical market data for testing.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        
    Returns:
        Dictionary of DataFrames with historical data
    """
    data = {}
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if not df.empty:
                data[ticker] = df
                logger.info(f"Retrieved {len(df)} data points for {ticker}")
            else:
                logger.warning(f"No data found for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    return data


def test_temporal_pattern_recognition(data, symbol='SPY'):
    """
    Test the Temporal Pattern Recognition system with real data.
    
    Args:
        data: Dictionary of DataFrames with historical data
        symbol: Symbol to test
        
    Returns:
        Analysis results
    """
    logger.info("\n==== Testing Temporal Pattern Recognition ====")
    
    if symbol not in data:
        logger.error(f"No data available for {symbol}")
        return None
    
    # Create the temporal pattern recognition system
    tpr = TemporalPatternRecognition()
    
    # Analyze temporal patterns
    df = data[symbol]
    results = tpr.analyze_temporal_patterns(
        prices=df['Close'],
        volumes=df['Volume'],
        asset_id=symbol,
        ohlcv_data=df
    )
    
    # Display results
    logger.info(f"Current regime: {results['current_regime']['regime_type']}")
    logger.info(f"Volatility regime: {results['current_regime']['volatility_regime']}")
    
    if results['seasonality']['has_seasonality']:
        logger.info("Seasonality detected with periods: " + 
                   str([p for p in results['seasonality']['seasonal_periods'] 
                       if isinstance(p, dict) and 'period' in p]))
    
    # Check for regime transitions
    transition_opp = tpr.detect_regime_transition_opportunity(symbol)
    if transition_opp['transition_opportunity']:
        logger.info(f"Transition opportunity detected from {transition_opp['current_regime']} " +
                   f"to {transition_opp['potential_next_regime']} " +
                   f"with confidence {transition_opp['confidence']:.2f}")
    
    # Check for timeframe alignment
    alignment = tpr.get_timeframe_alignment_signal(symbol)
    if alignment['has_alignment']:
        logger.info(f"Timeframe alignment detected for {alignment['aligned_regime']} " +
                  f"with score {alignment['agreement_score']:.2f}")
    
    return results


def test_adaptive_strategy_manager(data, symbol='SPY'):
    """
    Test the Adaptive Strategy Manager with real data.
    
    Args:
        data: Dictionary of DataFrames with historical data
        symbol: Symbol to test
        
    Returns:
        Adaptation results
    """
    logger.info("\n==== Testing Adaptive Strategy Manager ====")
    
    if symbol not in data:
        logger.error(f"No data available for {symbol}")
        return None
    
    # First run the temporal pattern recognition to get market regimes
    tpr = TemporalPatternRecognition()
    
    df = data[symbol]
    temporal_results = tpr.analyze_temporal_patterns(
        prices=df['Close'],
        volumes=df['Volume'],
        asset_id=symbol,
        ohlcv_data=df
    )
    
    # Create a mock strategy manager for testing
    class MockStrategyManager:
        def __init__(self):
            self.current_strategy = "test_strategy"
            
        def switch_strategy(self, new_strategy):
            logger.info(f"Mock switching from {self.current_strategy} to {new_strategy}")
            self.current_strategy = new_strategy
            return True
    
    # Initialize adaptive manager
    strategy_manager = MockStrategyManager()
    
    adaptive_manager = AdaptiveStrategyManager(
        strategy_manager=strategy_manager,
        performance_history=[
            {"sharpe_ratio": 0.8, "max_drawdown": 0.05, "win_rate": 0.52},
            {"sharpe_ratio": 0.7, "max_drawdown": 0.07, "win_rate": 0.48},
        ],
        available_strategies=["momentum", "mean_reversion", "trend_following", "test_strategy"],
        enable_temporal_adaptation=True
    )
    
    # Test adaptive position sizing
    market_regime = temporal_results['current_regime']['regime_type']
    volatility_regime = temporal_results['current_regime']['volatility_regime']
    
    position_size = adaptive_manager.get_adaptive_position_size(
        market_regime, 
        volatility_regime,
        {"sharpe_ratio": 0.5, "max_drawdown": 0.1, "win_rate": 0.51}
    )
    
    logger.info(f"Adaptive position size for {market_regime} regime: {position_size:.2f}")
    
    # Test timeframe selection
    optimal_timeframe = adaptive_manager.select_optimal_timeframe(volatility_regime, market_regime)
    logger.info(f"Optimal timeframe for {volatility_regime} volatility: {optimal_timeframe}")
    
    # Test overall adaptation
    adaptation_results = adaptive_manager.evaluate_and_adapt(
        metrics={
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.1,
            "win_rate": 0.51,
            "profit_factor": 1.1
        },
        market_regime=market_regime,
        volatility_regime=volatility_regime,
        price_data={
            "close": df['Close'].values,
            "volume": df['Volume'].values
        }
    )
    
    logger.info("Adaptation actions taken:")
    for action in adaptation_results["actions_taken"]:
        logger.info(f"  - {action}")
    
    # Get adapted parameters for current regime
    adapted_params = adaptive_manager.get_adapted_parameters(market_regime)
    
    logger.info(f"Adapted parameters for {market_regime} regime:")
    for param, value in adapted_params.items():
        logger.info(f"  - {param}: {value}")
    
    return adaptation_results


def main():
    """Main test function"""
    logger.info("Starting Adaptive System Integration Test")
    
    # Fetch market data
    data = fetch_data(start_date='2020-01-01')
    
    if not data:
        logger.error("No data available for testing.")
        return
    
    # Test 1: Temporal Pattern Recognition
    test_temporal_pattern_recognition(data)
    
    # Test 2: Adaptive Strategy Manager
    test_adaptive_strategy_manager(data)
    
    logger.info("Integration tests completed.")


if __name__ == "__main__":
    main()
