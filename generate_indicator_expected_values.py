import pandas as pd
from ai_trading_agent.signal_processing.indicators import simple_moving_average, relative_strength_index, moving_average_convergence_divergence, bollinger_bands

# Sample data for Simple Moving Average
sma_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
sma_series = pd.Series(sma_data)
sma_window_3 = simple_moving_average(sma_series, window=3)
sma_window_5 = simple_moving_average(sma_series, window=5)

print("--- SMA Window 3 ---")
print(sma_window_3.tolist())
print("--- SMA Window 5 ---")
print(sma_window_5.tolist())

# Sample data for Relative Strength Index
rsi_data = [
    44.33, 44.09, 43.73, 43.07, 43.17, 44.83, 46.19, 45.72, 46.28, 46.07,
    45.55, 46.46, 45.87, 45.75, 45.13, 43.69, 42.59, 43.19, 44.79, 44.05
]
rsi_series = pd.Series(rsi_data)
rsi_window_14 = relative_strength_index(rsi_series, window=14)

print("--- RSI Window 14 ---")
print(rsi_window_14.tolist())

# Sample data for Moving Average Convergence Divergence
macd_data = [
    22.27, 22.15, 22.41, 22.41, 22.43, 22.12, 22.26, 22.17, 22.32, 22.18,
    22.57, 22.37, 22.42, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36,
    22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36
]
macd_series = pd.Series(macd_data)
macd_df = moving_average_convergence_divergence(macd_series)

print("--- MACD ---")
print("MACD:", macd_df['MACD'].tolist())
print("Signal_Line:", macd_df['Signal_Line'].tolist())

# Sample data for Bollinger Bands
bb_data = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39
]
bb_series = pd.Series(bb_data)
bb_df = bollinger_bands(bb_series)

print("--- Bollinger Bands ---")
print("Middle_Band:", bb_df['Middle_Band'].tolist())
print("Upper_Band:", bb_df['Upper_Band'].tolist())
print("Lower_Band:", bb_df['Lower_Band'].tolist())
