"""
Example of a simple RSI-based trading strategy using the AI Trading Agent framework.

This example demonstrates how to use the Rust-accelerated RSI indicator
in a trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_acquisition.mock_provider import MockDataProvider
from src.data_processing.indicators import calculate_rsi
from src.trading_engine.models import Order, OrderType, OrderSide

# Create a mock data provider for testing
data_provider = MockDataProvider()

# Fetch some historical data
start_date = datetime.now() - timedelta(days=60)
end_date = datetime.now()
symbol = "BTC/USDT"
timeframe = "1h"

# Get historical data
import asyncio

# Create an async function to fetch data
async def fetch_data():
    data_dict = await data_provider.fetch_historical_data(
        symbols=[symbol],
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    return data_dict[symbol]  # Return the dataframe for our symbol

# Run the async function to get our dataframe
df = asyncio.run(fetch_data())

# Calculate RSI using our Rust-accelerated implementation
df['rsi'] = calculate_rsi(df['close'], window=14)

# Create a simple RSI strategy
# Buy when RSI < 30 (oversold)
# Sell when RSI > 70 (overbought)
df['signal'] = 0
df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy signal
df.loc[df['rsi'] > 70, 'signal'] = -1  # Sell signal

# Backtest the strategy
initial_balance = 10000.0
position = 0
balance = initial_balance
portfolio_value = [initial_balance]
trades = []

for i in range(1, len(df)):
    if df['signal'].iloc[i-1] == 1 and position == 0:  # Buy signal and no position
        price = df['close'].iloc[i]
        position = balance / price
        balance = 0
        trades.append({
            'type': 'buy',
            'price': price,
            'timestamp': df.index[i],
            'rsi': df['rsi'].iloc[i-1]
        })
    elif df['signal'].iloc[i-1] == -1 and position > 0:  # Sell signal and has position
        price = df['close'].iloc[i]
        balance = position * price
        position = 0
        trades.append({
            'type': 'sell',
            'price': price,
            'timestamp': df.index[i],
            'rsi': df['rsi'].iloc[i-1]
        })
    
    # Calculate portfolio value
    if position > 0:
        portfolio_value.append(balance + position * df['close'].iloc[i])
    else:
        portfolio_value.append(balance)

# Print results
print(f"Initial balance: ${initial_balance:.2f}")
print(f"Final balance: ${portfolio_value[-1]:.2f}")
print(f"Return: {(portfolio_value[-1] / initial_balance - 1) * 100:.2f}%")
print(f"Number of trades: {len(trades)}")

# Plot the results
plt.figure(figsize=(14, 10))

# Plot 1: Price and RSI signals
plt.subplot(3, 1, 1)
plt.plot(df.index, df['close'], label='Price')
plt.scatter(
    [t['timestamp'] for t in trades if t['type'] == 'buy'],
    [t['price'] for t in trades if t['type'] == 'buy'],
    marker='^', color='g', s=100, label='Buy'
)
plt.scatter(
    [t['timestamp'] for t in trades if t['type'] == 'sell'],
    [t['price'] for t in trades if t['type'] == 'sell'],
    marker='v', color='r', s=100, label='Sell'
)
plt.title('Price and Trading Signals')
plt.legend()

# Plot 2: RSI
plt.subplot(3, 1, 2)
plt.plot(df.index, df['rsi'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
plt.title('Relative Strength Index (RSI)')
plt.legend()

# Plot 3: Portfolio Value
plt.subplot(3, 1, 3)
# Make sure portfolio_value has the same length as df.index
portfolio_value_plot = portfolio_value[:len(df.index)]
plt.plot(df.index, portfolio_value_plot, label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend()

plt.tight_layout()
plt.savefig('rsi_strategy_results.png')
plt.show()
