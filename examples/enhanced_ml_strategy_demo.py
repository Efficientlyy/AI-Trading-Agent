"""Demo script for the enhanced ML-based price prediction strategy."""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Optional

from src.models.market_data import CandleData, TimeFrame
from src.ml.models.enhanced_price_prediction import EnhancedPricePredictionStrategy


async def main() -> None:
    """Run the enhanced ML strategy demo."""
    # Initialize strategy
    strategy = EnhancedPricePredictionStrategy(
        strategy_id="enhanced_ml_demo",
        trading_symbols=["BTC/USD"],
        lookback_window=100,
        prediction_horizon=12,
        confidence_threshold=0.6,
        correlation_threshold=0.7
    )
    
    # Define trading parameters
    symbol = "BTC/USD"
    timeframe = TimeFrame.HOUR_1
    lookback_days = 30
    
    # Generate sample data for testing
    print(f"\nGenerating sample data for {symbol}...")
    candles: List[CandleData] = []
    
    # Start from 30 days ago
    end_time = datetime.utcnow()
    current_time = end_time - timedelta(days=lookback_days)
    base_price = 50000.0  # Starting price
    
    # Generate more realistic price movements
    price_trend = 0.0  # Overall trend
    volatility = 0.02  # Base volatility
    regime_changes = []
    
    # Generate candles with regime changes
    while current_time < end_time:
        # Switch between trending and ranging regimes every 24 hours
        if len(candles) % 24 == 0:
            price_trend = np.random.normal(0, 0.02)  # New trend direction
            volatility = np.random.uniform(0.01, 0.03)  # New volatility level
            regime_changes.append(len(candles))  # Mark regime change
        
        # Calculate price movement
        price_change = np.random.normal(price_trend, volatility)
        base_price *= (1 + price_change)
        
        # Generate realistic OHLCV data
        high_low_spread = base_price * np.random.uniform(0.005, 0.015)
        open_price = base_price * (1 + np.random.normal(0, 0.002))
        close_price = base_price * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, close_price) + high_low_spread/2
        low_price = min(open_price, close_price) - high_low_spread/2
        
        # Volume increases with volatility and price changes
        volume = np.random.gamma(
            shape=2.0,
            scale=abs(price_change) * 1000 + volatility * 5000
        )
        
        candle = CandleData(
            symbol=symbol,
            timestamp=current_time,
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(volume),
            exchange="demo",
            timeframe=timeframe
        )
        candles.append(candle)
        current_time += timedelta(hours=1)
    
    # Initialize and start the strategy
    await strategy._strategy_initialize()
    await strategy._strategy_start()
    
    # Train strategy on historical data
    print("\nTraining strategy on historical data...")
    for candle in candles[:-24]:  # Leave last 24 hours for testing
        await strategy.process_candle(candle)
    
    # Create results dataframe
    results = []
    
    # Test on recent data
    print("\nTesting strategy on recent data...")
    for candle in candles[-24:]:
        # Get prediction before updating with new candle
        features = strategy._extract_features(candle.symbol)
        if not features.empty:
            prediction = strategy._generate_predictions(candle.symbol)
            
            if prediction:
                results.append({
                    'timestamp': candle.timestamp,
                    'price': float(candle.close),
                    'direction': int(prediction['direction']),
                    'confidence': float(prediction['confidence']),
                    'position_size': float(prediction['position_size']),
                    'stop_loss': float(prediction['stop_loss']),
                    'take_profit': float(prediction['take_profit'])
                })
        
        # Update strategy with new candle
        await strategy.process_candle(candle)
    
    # Stop the strategy
    await strategy._strategy_stop()
    
    # Convert results to dataframe
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No predictions generated")
        return
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Price subplot
    plt.subplot(311)
    plt.plot(df['timestamp'], df['price'], label='Price')
    plt.title(f'{symbol} Price and Predictions')
    plt.legend()
    plt.grid(True)
    
    # Prediction subplot
    plt.subplot(312)
    plt.plot(df['timestamp'], df['confidence'] * df['direction'], 
             label='Prediction Signal')
    plt.fill_between(df['timestamp'], 
                    df['take_profit'] - df['price'],
                    df['stop_loss'] - df['price'],
                    alpha=0.2, label='Risk/Reward Zone')
    plt.title('ML Strategy Predictions')
    plt.legend()
    plt.grid(True)
    
    # Market regime subplot
    plt.subplot(313)
    plt.plot(df['timestamp'], df['position_size'], label='Position Size')
    plt.title('Dynamic Position Sizing')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\nStrategy Performance:")
    print(f"Total predictions: {len(df)}")
    print(f"Average confidence: {df['confidence'].mean():.2%}")
    print(f"Average position size: {df['position_size'].mean():.2f}")
    print(f"Average risk/reward ratio: {((df['take_profit'] - df['price'])/(df['price'] - df['stop_loss'])).mean():.2f}")


if __name__ == "__main__":
    asyncio.run(main()) 