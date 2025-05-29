"""
Market Regime Classification and Adaptive Response System Test

This script tests the integration between our market regime classification system
and adaptive response system using real historical market data to evaluate performance
across different market regimes.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegimeValidator:
    """
    A lightweight market regime classifier that doesn't rely on complex dependencies.
    
    This class implements basic technical indicators to classify market regimes
    for testing the adaptive response system.
    """
    
    def __init__(self):
        self.lookback_window = 20
        
    def classify_regime(self, data):
        """
        Classify the market regime based on price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with regime classification
        """
        # Ensure we have enough data
        if len(data) < 2 * self.lookback_window:
            return {
                "regime": "unknown",
                "volatility": "unknown",
                "confidence": 0.0
            }
        
        # Calculate returns and technical indicators
        data_copy = data.copy()  # Work with a copy to avoid SettingWithCopyWarning
        
        # Calculate returns
        data_copy['returns'] = data_copy['Close'].pct_change()
        
        # Calculate rolling indicators
        data_copy['volatility'] = data_copy['returns'].rolling(window=self.lookback_window).std()
        data_copy['ma50'] = data_copy['Close'].rolling(window=50).mean()
        data_copy['ma200'] = data_copy['Close'].rolling(window=200).mean()
        
        # Get the most recent values - convert to Python scalars to avoid pandas comparison issues
        latest_idx = -1
        volatility = float(data_copy['volatility'].iloc[latest_idx])
        price = float(data_copy['Close'].iloc[latest_idx])
        ma50 = float(data_copy['ma50'].iloc[latest_idx])
        ma200 = float(data_copy['ma200'].iloc[latest_idx])
        
        # Calculate trend strength
        trend_ratio = price / ma200 - 1 if not np.isnan(ma200) and ma200 != 0 else 0
        
        # Determine volatility regime
        if volatility < 0.01:
            volatility_regime = "low"
        elif volatility < 0.02:
            volatility_regime = "medium"
        elif volatility < 0.03:
            volatility_regime = "high"
        else:
            volatility_regime = "extreme"
        
        # Determine market regime
        # Use simple logic based on MAs and volatility
        if trend_ratio > 0.05 and ma50 > ma200:
            regime = "bull"
            confidence = min(0.9, 0.5 + trend_ratio)
        elif trend_ratio < -0.05 and ma50 < ma200:
            regime = "bear"
            confidence = min(0.9, 0.5 - trend_ratio)
        elif volatility > 0.025:
            regime = "volatile"
            confidence = min(0.9, 0.4 + volatility * 10)
        elif abs(trend_ratio) < 0.03:
            regime = "sideways"
            confidence = min(0.9, 0.5 + (0.03 - abs(trend_ratio)) * 10)
        else:
            regime = "trending"
            confidence = min(0.9, 0.5 + abs(trend_ratio) * 5)
        
        return {
            "regime": regime,
            "volatility": volatility_regime,
            "trend_ratio": trend_ratio,
            "volatility_value": volatility,
            "confidence": confidence,
            "ma50_above_ma200": ma50 > ma200
        }


class AdaptiveStrategyTester:
    """
    Test implementation of adaptive strategy manager to validate our approach.
    
    This simulates the behavior of our actual adaptive response system but
    with simplified logic to avoid dependency issues.
    """
    
    def __init__(self):
        # Default parameters for different regimes
        self.regime_parameters = {
            "bull": {
                "position_size": 1.0,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "timeframe": "1D"
            },
            "bear": {
                "position_size": 0.5,
                "stop_loss": 0.03,
                "take_profit": 0.07,
                "timeframe": "4H"
            },
            "sideways": {
                "position_size": 0.5,
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "timeframe": "1D"
            },
            "volatile": {
                "position_size": 0.3,
                "stop_loss": 0.07,
                "take_profit": 0.10,
                "timeframe": "1W"
            },
            "trending": {
                "position_size": 0.7,
                "stop_loss": 0.04,
                "take_profit": 0.08,
                "timeframe": "1D"
            },
            "unknown": {
                "position_size": 0.3,
                "stop_loss": 0.03,
                "take_profit": 0.05,
                "timeframe": "1D"
            }
        }
        
        # Available strategies
        self.strategies = ["momentum", "mean_reversion", "trend_following", "volatility_breakout"]
        self.current_strategy = "trend_following"
        self.strategy_history = []
    
    def adapt_to_regime(self, regime_info, performance_metrics=None):
        """
        Adapt strategy parameters based on market regime.
        
        Args:
            regime_info: Dictionary with regime classification
            performance_metrics: Optional performance metrics
            
        Returns:
            Dictionary with adapted parameters and actions
        """
        regime = regime_info["regime"]
        volatility = regime_info["volatility"]
        
        # Get base parameters for this regime
        params = self.regime_parameters.get(regime, self.regime_parameters["unknown"]).copy()
        
        # Adjust based on volatility
        if volatility == "high":
            params["position_size"] *= 0.8
            params["stop_loss"] *= 1.2
        elif volatility == "extreme":
            params["position_size"] *= 0.6
            params["stop_loss"] *= 1.5
            params["timeframe"] = "1W"  # Use weekly timeframe in extreme volatility
        
        # Adjust based on performance metrics (if available)
        if performance_metrics:
            sharpe = performance_metrics.get("sharpe_ratio", 0)
            drawdown = performance_metrics.get("max_drawdown", 0)
            
            # Reduce position size if recent performance is poor
            if sharpe < 0.5 or drawdown > 0.1:
                params["position_size"] *= 0.8
                logger.info(f"Reducing position size due to poor performance (Sharpe: {sharpe:.2f}, Drawdown: {drawdown:.2f})")
        
        # Select appropriate strategy for the regime
        old_strategy = self.current_strategy
        
        if regime == "bull":
            new_strategy = "momentum"
        elif regime == "bear":
            new_strategy = "trend_following"
        elif regime == "sideways":
            new_strategy = "mean_reversion"
        elif regime == "volatile":
            new_strategy = "volatility_breakout"
        else:
            new_strategy = "trend_following"
        
        # Only switch if different
        actions = []
        if new_strategy != old_strategy:
            self.current_strategy = new_strategy
            self.strategy_history.append({
                "timestamp": datetime.now(),
                "from": old_strategy,
                "to": new_strategy,
                "reason": f"Regime change to {regime}"
            })
            actions.append(f"Switched strategy from {old_strategy} to {new_strategy}")
        
        actions.append(f"Adjusted position size to {params['position_size']:.2f}")
        actions.append(f"Set timeframe to {params['timeframe']}")
        
        return {
            "parameters": params,
            "strategy": new_strategy,
            "actions": actions
        }


def fetch_market_data(symbols=['SPY', 'QQQ'], start_date='2019-01-01', end_date=None):
    """
    Fetch historical market data for testing.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary with DataFrames of market data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    data = {}
    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}...")
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                logger.info(f"Retrieved {len(df)} data points for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return data


def test_market_periods(data, symbol):
    """
    Test the system with different market periods to see how it adapts.
    
    Args:
        data: Dictionary with market data
        symbol: Symbol to test
        
    Returns:
        Test results
    """
    if symbol not in data:
        logger.error(f"No data for {symbol}")
        return []
    
    # Define interesting historical periods to test
    periods = [
        {"name": "Bull Market", "start": "2019-01-01", "end": "2019-12-31"},
        {"name": "COVID Crash", "start": "2020-02-15", "end": "2020-04-15"},
        {"name": "Recovery", "start": "2020-04-16", "end": "2020-12-31"},
        {"name": "2021 Bull Run", "start": "2021-01-01", "end": "2021-11-30"},
        {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-10-31"}
    ]
    
    # Initialize test components
    regime_classifier = MarketRegimeValidator()
    adaptive_manager = AdaptiveStrategyTester()
    
    all_results = []
    
    # Create a figure for plotting
    num_periods = len(periods)
    fig, axs = plt.subplots(num_periods, 1, figsize=(12, 5 * num_periods), sharex=False)
    
    # Process each period
    for i, period in enumerate(periods):
        logger.info(f"\n=== Testing {period['name']} ({period['start']} to {period['end']}) ===")
        
        # Filter data for this period
        start_date = pd.Timestamp(period['start'])
        end_date = pd.Timestamp(period['end'])
        
        period_data = data[symbol].loc[
            (data[symbol].index >= start_date) & 
            (data[symbol].index <= end_date)
        ]
        
        if len(period_data) < 50:
            logger.warning(f"Insufficient data for {period['name']}")
            continue
        
        # Classify the regime
        regime_info = regime_classifier.classify_regime(period_data)
        
        logger.info(f"Classified as: {regime_info['regime']} regime (confidence: {regime_info['confidence']:.2f})")
        logger.info(f"Volatility: {regime_info['volatility']} ({regime_info['volatility_value']:.4f})")
        logger.info(f"Trend ratio: {regime_info['trend_ratio']:.4f}")
        
        # Create mock performance metrics
        perf_metrics = {
            "sharpe_ratio": 0.8 if regime_info['regime'] in ['bull', 'trending'] else 0.3,
            "max_drawdown": 0.05 if regime_info['regime'] in ['bull', 'trending'] else 0.15
        }
        
        # Adapt to the regime
        adaptation = adaptive_manager.adapt_to_regime(regime_info, perf_metrics)
        
        logger.info("Adaptation actions:")
        for action in adaptation["actions"]:
            logger.info(f"  - {action}")
        
        logger.info("Adapted parameters:")
        for param, value in adaptation["parameters"].items():
            logger.info(f"  - {param}: {value}")
            
        # Store the results
        all_results.append({
            "period": period["name"],
            "regime": regime_info,
            "adaptation": adaptation
        })
        
        # Plot this period
        ax = axs[i]
        period_data['Close'].plot(ax=ax, label=f"{symbol} Price")
        
        # Add moving averages
        period_data['MA50'] = period_data['Close'].rolling(window=50).mean()
        period_data['MA200'] = period_data['Close'].rolling(window=200).mean()
        period_data['MA50'].plot(ax=ax, label='50-day MA', linestyle='--')
        period_data['MA200'].plot(ax=ax, label='200-day MA', linestyle='-.')
        
        # Highlight regime with background color
        bg_color = 'lightgreen'
        if regime_info['regime'] == 'bear':
            bg_color = 'lightcoral'
        elif regime_info['regime'] == 'volatile':
            bg_color = 'lightyellow'
        elif regime_info['regime'] == 'sideways':
            bg_color = 'lightblue'
            
        ax.set_facecolor(bg_color)
        
        # Set title and legend
        title = f"{period['name']}: {regime_info['regime'].upper()} regime, {regime_info['volatility']} volatility"
        subtitle = f"Strategy: {adaptation['strategy']}, Position Size: {adaptation['parameters']['position_size']:.2f}"
        ax.set_title(f"{title}\n{subtitle}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Label axes
        ax.set_ylabel('Price')
        if i == num_periods - 1:
            ax.set_xlabel('Date')
    
    # Adjust layout and save figure
    plt.tight_layout()
    
    # Save the figure before showing it
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_regime_test_results.png"))
    
    # Show the figure
    plt.show()
    
    return all_results


def main():
    """Main test function"""
    logger.info("Starting Market Regime and Adaptive Response System Test with Real Data")
    
    # Fetch historical market data
    data = fetch_market_data(
        symbols=['SPY', 'QQQ', 'IWM'], 
        start_date='2019-01-01'
    )
    
    if not data:
        logger.error("No data fetched, exiting.")
        return
    
    # Test with SPY data
    test_results = test_market_periods(data, 'SPY')
    
    # Display summary
    if test_results:
        logger.info("\n=== Test Summary ===")
        for result in test_results:
            logger.info(f"{result['period']}: {result['regime']['regime']} regime, "
                      f"Strategy: {result['adaptation']['strategy']}")
    
    logger.info("Test completed successfully.")
    logger.info(f"Results saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_regime_test_results.png')}")


if __name__ == "__main__":
    main()
