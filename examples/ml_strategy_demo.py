"""Demo script for the ML-based price prediction strategy."""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Optional

from src.models.market_data import CandleData, TimeFrame
from src.ml.models.price_prediction_strategy import PricePredictionStrategy


async def main() -> None:
    """Run the ML strategy demo."""
    # Initialize strategy
    strategy = PricePredictionStrategy()
    
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
    
    while current_time <= end_time:
        # Generate random price movement
        price_change = np.random.normal(0, 0.02)  # 2% standard deviation
        base_price *= (1 + price_change)
        
        # Create candle data
        candle = CandleData(
            symbol=symbol,
            exchange="sample",
            timestamp=current_time,
            timeframe=timeframe,
            open=base_price * (1 + np.random.normal(0, 0.001)),
            high=base_price * (1 + abs(np.random.normal(0, 0.002))),
            low=base_price * (1 - abs(np.random.normal(0, 0.002))),
            close=base_price,
            volume=np.random.uniform(100, 1000),
            complete=True
        )
        candles.append(candle)
        
        # Move to next time period
        current_time += timedelta(hours=1)
    
    print(f"Generated {len(candles)} sample candles")
    
    # Start the strategy
    await strategy._strategy_start()
    
    # Train the strategy
    print("\nTraining ML strategy...")
    for candle in candles[:-24]:  # Leave last 24 hours for testing
        await strategy._strategy_on_candle(candle)
    
    # Create results dataframe
    results = []
    
    # Test on recent data
    print("\nTesting strategy on recent data...")
    for candle in candles[-24:]:
        # Get prediction before updating with new candle
        features = strategy._prepare_features(symbol)
        if features.size > 0:
            prediction = strategy._generate_prediction(symbol, features)
            
            results.append({
                'timestamp': candle.timestamp,
                'price': float(candle.close),
                'direction': int(prediction['direction']),
                'confidence': float(prediction['confidence']),
                'target_return': float(prediction['target_return']),
                'max_loss': float(prediction['max_loss'])
            })
        
        # Update strategy with new candle
        await strategy._strategy_on_candle(candle)
    
    # Stop the strategy
    await strategy._strategy_stop()
    
    # Convert results to dataframe
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No predictions generated")
        return
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Price subplot
    plt.subplot(211)
    plt.plot(df['timestamp'], df['price'], label='Price')
    plt.title(f'{symbol} Price and Predictions')
    plt.legend()
    plt.grid(True)
    
    # Prediction subplot
    plt.subplot(212)
    plt.plot(df['timestamp'], df['confidence'] * df['direction'], 
             label='Prediction Signal')
    plt.fill_between(df['timestamp'], 
                    df['target_return'] * df['direction'],
                    -df['max_loss'] * df['direction'],
                    alpha=0.2, label='Risk/Reward Zone')
    plt.title('ML Strategy Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    next_returns = df['price'].diff().shift(-1).fillna(0).astype(float)
    correct_direction = np.sum(
        (next_returns > 0.0).astype(np.int32) == 
        (df['direction'] > 0).astype(np.int32)
    )
    accuracy = float(correct_direction) / len(df)
    
    print("\nStrategy Performance:")
    print(f"Directional Accuracy: {accuracy:.1%}")
    print("\nAverage Metrics:")
    print(f"Confidence: {df['confidence'].mean():.1%}")
    print(f"Target Return: {df['target_return'].mean():.1%}")
    print(f"Max Loss: {df['max_loss'].mean():.1%}")
    
    # Feature importance
    if symbol in strategy.models:
        importance = dict(zip(strategy.feature_names,
                            strategy.models[symbol].feature_importances_))
        print("\nFeature Importance:")
        for feat, imp in sorted(importance.items(), 
                              key=lambda x: x[1], reverse=True):
            print(f"{feat}: {imp:.3f}")


if __name__ == "__main__":
    asyncio.run(main()) 