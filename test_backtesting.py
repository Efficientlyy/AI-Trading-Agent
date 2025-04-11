"""
Test script for the backtesting engine.
This script tests the backtesting engine with sample data to ensure it works correctly.
"""
import logging
import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import traceback
import sys
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Add Sentiment Analysis Imports ---
try:
    from ai_trading_agent.sentiment_analysis.analyzer import SentimentAnalyzer
    from ai_trading_agent.sentiment_analysis.signal_generator import SentimentSignalGenerator
    from ai_trading_agent.sentiment_analysis.connectors import load_sentiment_data_from_csv
    logger.info("Successfully imported sentiment analysis modules")
except ImportError as e:
    logger.error(f"Failed to import sentiment analysis modules: {e}")
    # Decide if we should raise or continue with dummy signals
    raise # For now, fail if imports don't work

# Import the backtesting engine
try:
    from ai_trading_agent.backtesting_engine import run_backtest
    logger.info("Successfully imported backtesting engine")
except ImportError as e:
    logger.error(f"Failed to import backtesting engine: {e}")
    raise

def generate_sample_data(symbol, days=100, start_price=100.0, volatility=0.02):
    """Generate sample OHLCV data for testing."""
    data = []
    price = start_price
    base_date = datetime.datetime(2023, 1, 1)
    
    for i in range(days):
        date = base_date + datetime.timedelta(days=i)
        timestamp = int(date.timestamp())
        
        # Generate random price movement
        change_percent = random.normalvariate(0, volatility)
        price = price * (1 + change_percent)
        
        # Generate OHLCV data
        open_price = price
        high_price = price * (1 + random.uniform(0, volatility))
        low_price = price * (1 - random.uniform(0, volatility))
        close_price = price * (1 + random.normalvariate(0, volatility/2))
        volume = random.uniform(1000, 10000)
        
        # Create bar
        bar = {
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
        data.append(bar)
    
    return data

# --- Add New Order Generation Function (using DataFrame) ---
def generate_orders_from_ohlcv_signals(ohlcv_df: pd.DataFrame, symbol: str) -> list:
    """Generate orders based on signals in an OHLCV DataFrame."""
    orders = []
    position = 0  # 0 = no position, 1 = long, -1 = short
    logger.info(f"Generating orders for {symbol} based on aligned signals...")

    if 'signal' not in ohlcv_df.columns:
        logger.error(f"Missing 'signal' column in DataFrame for {symbol}. Cannot generate orders.")
        return []
    if ohlcv_df.empty:
        logger.warning(f"Empty DataFrame for {symbol}. No orders generated.")
        return []

    # Iterate through the DataFrame rows
    for timestamp, row in ohlcv_df.iterrows():
        signal = row['signal']
        close_price = row['close']
        timestamp_unix = int(timestamp.timestamp()) # Convert back for order dict

        # Basic check for valid price
        if pd.isna(close_price) or close_price <= 0:
            logger.warning(f"Skipping order generation at {timestamp} for {symbol} due to invalid price: {close_price}")
            continue

        # Generate order based on signal and current position
        if signal > 0 and position <= 0: # Buy signal and not already long
            # Close short position first if exists
            if position < 0:
                 orders.append({
                    'id': f"order_{symbol}_{len(orders)}_close_short",
                    'timestamp': timestamp_unix,
                    'symbol': symbol,
                    'side': 'buy', # Buy to close short
                    'quantity': 1.0, # Assuming quantity=1 for simplicity
                    'price': close_price,
                    'type': 'market'
                 })
            # Open long position
            orders.append({
                'id': f"order_{symbol}_{len(orders)}_open_long",
                'timestamp': timestamp_unix,
                'symbol': symbol,
                'side': 'buy',
                'quantity': 1.0,
                'price': close_price,
                'type': 'market'
            })
            position = 1
            logger.debug(f"Generated BUY order for {symbol} at {timestamp}, price {close_price:.2f}")
        elif signal < 0 and position >= 0: # Sell signal and not already short
            # Close long position first if exists
            if position > 0:
                orders.append({
                    'id': f"order_{symbol}_{len(orders)}_close_long",
                    'timestamp': timestamp_unix,
                    'symbol': symbol,
                    'side': 'sell', # Sell to close long
                    'quantity': 1.0, # Assuming quantity=1
                    'price': close_price,
                    'type': 'market'
                })
            # Open short position (if short selling is allowed/intended)
            # For now, let's assume we just close longs on sell signal if already long
            # If you want to open short positions, uncomment the block below
            # orders.append({
            #     'id': f"order_{symbol}_{len(orders)}_open_short",
            #     'timestamp': timestamp_unix,
            #     'symbol': symbol,
            #     'side': 'sell',
            #     'quantity': 1.0,
            #     'price': close_price,
            #     'type': 'market'
            # })
            # position = -1 # Uncomment if opening short positions
            if position > 0: # Only log if we actually closed a long
               logger.debug(f"Generated SELL order (close long) for {symbol} at {timestamp}, price {close_price:.2f}")
            position = 0 # Set position to neutral after closing long

        # Optional: Logic to close position on neutral signal
        elif signal == 0 and position != 0:
            side_to_close = 'sell' if position > 0 else 'buy'
            close_type = "close_long" if position > 0 else "close_short"
            orders.append({
                'id': f"order_{symbol}_{len(orders)}_{close_type}_neutral",
                'timestamp': timestamp_unix,
                'symbol': symbol,
                'side': side_to_close,
                'quantity': 1.0,
                'price': close_price,
                'type': 'market'
            })
            logger.debug(f"Generated CLOSE order ({side_to_close}) for {symbol} at {timestamp} due to neutral signal, price {close_price:.2f}")
            position = 0

    logger.info(f"Finished generating {len(orders)} orders for {symbol}.")
    return orders

def plot_backtest_results(data_list, orders, result):
    """Plot the backtest results."""
    # --- Minor modification to handle data_list input ---
    if not data_list:
        logger.warning("No data provided for plotting.")
        return
    symbol = orders[0]['symbol'] if orders else 'Unknown Symbol' # Get symbol from orders if possible
    df = pd.DataFrame(data_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    try:
        # Convert data to DataFrame for easier plotting
        # df = pd.DataFrame([
        #     {
        #         'timestamp': bar['timestamp'],
        #         'close': bar['close'],
        #         'datetime': datetime.datetime.fromtimestamp(bar['timestamp'])
        #     }
        #     for bar in data
        # ])
        
        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price data
        ax1.plot(df.index, df['close'], label='Price')
        
        # Plot buy and sell signals
        buy_orders = [order for order in orders if order['side'] == 'buy']
        sell_orders = [order for order in orders if order['side'] == 'sell']
        
        buy_times = [datetime.datetime.fromtimestamp(order['timestamp']) for order in buy_orders]
        sell_times = [datetime.datetime.fromtimestamp(order['timestamp']) for order in sell_orders]
        
        buy_prices = [order['price'] for order in buy_orders]
        sell_prices = [order['price'] for order in sell_orders]
        
        ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        ax1.set_title('Price and Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Extract metrics
        final_capital = result.get('final_capital', 0)
        metrics = result.get('metrics', {})
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # Plot a simple equity curve (just a line from initial to final capital)
        initial_capital = 10000  # Assuming this is our initial capital
        ax2.plot([df.index[0], df.index[-1]], 
                [initial_capital, final_capital], 'b-')
        
        ax2.set_title(f'Performance Metrics: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        logger.info(f"Backtest results saved to backtest_results.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting backtest results: {e}")
        logger.error(traceback.format_exc())

def print_separator(char='=', length=80):
    """Print a separator line to the console."""
    print(char * length)
    logger.info(char * length)

def main():
    """Main function to run the backtest."""
    try:
        # --- Setup: Load sentiment data and instantiate analyzers ONCE ---
        print_separator()
        logger.info("Starting backtesting test script with Sentiment Analysis")
        print_separator()

        sentiment_csv_path = os.path.join('data', 'sample_sentiment_data.csv')
        logger.info(f"Loading sentiment data from {sentiment_csv_path}")
        try:
            sentiment_df_all = load_sentiment_data_from_csv(sentiment_csv_path)
            sentiment_df_all['timestamp'] = pd.to_datetime(sentiment_df_all['timestamp'])
            sentiment_df_all.set_index('timestamp', inplace=True)
            logger.info(f"Loaded {len(sentiment_df_all)} total sentiment records.")
        except Exception as e:
            logger.error(f"Failed to load or process sentiment data from {sentiment_csv_path}: {e}", exc_info=True)
            return # Stop execution if sentiment data fails

        try:
            analyzer = SentimentAnalyzer() # Instantiate once
            signal_gen = SentimentSignalGenerator(buy_threshold=0.1, sell_threshold=-0.1) # Instantiate once, adjust thresholds if needed
            logger.info("Sentiment Analyzer and Signal Generator initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment components: {e}", exc_info=True)
            return # Stop execution

        # Generate sample data for multiple assets
        symbols = ['AAPL', 'GOOG', 'MSFT', 'TSLA'] # Use symbols from sample CSV
        all_data = {} # Keep original list data for plotting
        all_orders = []

        print("Processing symbols...")
        logger.info("Processing symbols...")
        for symbol in symbols:
            print_separator('-')
            logger.info(f"Processing symbol: {symbol}")

            # Generate price data (remains the same)
            data = generate_sample_data(
                symbol=symbol,
                days=100, # Adjust days if needed to match sentiment data time range
                start_price=random.uniform(50, 200), # Random start price
                volatility=random.uniform(0.01, 0.03) # Random volatility
            )
            all_data[symbol] = data
            logger.info(f"Generated {len(data)} OHLCV data points for {symbol}")

            # --- Convert OHLCV to DataFrame ---
            ohlcv_df = pd.DataFrame(data)
            if ohlcv_df.empty:
                logger.warning(f"Generated empty OHLCV data for {symbol}. Skipping.")
                continue
            ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='s')
            ohlcv_df.set_index('timestamp', inplace=True)
            logger.debug(f"Created OHLCV DataFrame for {symbol} with index from {ohlcv_df.index.min()} to {ohlcv_df.index.max()}")

            # --- Filter, Analyze, Generate Signals, Align ---
            sentiment_df = sentiment_df_all[sentiment_df_all['symbol'] == symbol].copy()

            if sentiment_df.empty:
                logger.warning(f"No sentiment data found for symbol {symbol} in the loaded CSV. Generating neutral signals (0).")
                ohlcv_df['signal'] = 0 # Assign neutral signals
            else:
                logger.info(f"Found {len(sentiment_df)} sentiment records for {symbol}. Analyzing...")
                # Calculate sentiment scores
                sentiment_df['score'] = sentiment_df['text'].apply(analyzer.analyze)
                logger.info(f"Sentiment analysis complete for {symbol}.")

                # Generate signals from scores
                raw_signals = signal_gen.generate_signals_from_scores(sentiment_df['score'])
                logger.info(f"Generated {len(raw_signals[raw_signals != 0])} non-neutral raw signals for {symbol}.")

                # Align signals to OHLCV data using forward fill
                logger.info(f"Aligning {len(raw_signals)} signals to {len(ohlcv_df)} OHLCV data points for {symbol}...")
                aligned_signals = raw_signals.reindex(ohlcv_df.index, method='ffill').fillna(0).astype(int)
                ohlcv_df['signal'] = aligned_signals
                logger.info(f"Alignment complete. {len(aligned_signals[aligned_signals != 0])} non-neutral aligned signals.")
                logger.debug(f"First few aligned signals for {symbol}:\n{ohlcv_df['signal'].head()}")

            # --- Generate orders using the new function ---
            orders = generate_orders_from_ohlcv_signals(ohlcv_df, symbol)
            all_orders.extend(orders)
            print(f"Generated {len(orders)} orders for {symbol}")
            logger.info(f"Generated {len(orders)} orders for {symbol}")

        # Sort orders by timestamp (remains the same)
        all_orders.sort(key=lambda x: x['timestamp'])

        # Create backtest configuration
        config = {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'slippage': 0.001
        }

        print_separator('-')
        print(f"Running backtest with {len(all_orders)} orders across {len(symbols)} symbols...")
        logger.info(f"Running backtest with {len(all_orders)} orders across {len(symbols)} symbols...")
        print_separator('-')

        # Run the backtest
        start_time = time.time()
        result = run_backtest(all_data, all_orders, config)
        end_time = time.time()

        # Pretty print the results
        print_separator()
        print("Backtest completed. Results:")
        logger.info("Backtest completed. Results:")
        print_separator('-')

        print(f"Execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Final capital: ${result.get('final_capital', 0):.2f}")
        logger.info(f"Final capital: ${result.get('final_capital', 0):.2f}")

        metrics = result.get('metrics', {})
        print(f"Total return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Total return: {metrics.get('total_return', 0):.2%}")
        print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print_separator()

        # Save results to a JSON file
        with open('backtest_results.json', 'w') as f:
            json.dump(result, f, indent=4)
        print("Results saved to backtest_results.json")
        logger.info("Results saved to backtest_results.json")

        # Plot the results for the first symbol found in the data
        plot_symbol = next((s for s in symbols if s in all_data), None)
        if plot_symbol:
            logger.info(f"Generating performance charts for {plot_symbol}...")
            plot_backtest_results(all_data[plot_symbol],
                                 [o for o in all_orders if o['symbol'] == plot_symbol],
                                 result)
        else:
            logger.warning("No data available for any symbol to plot.")

        print("Test completed successfully!")
        logger.info("Test completed successfully!")

    except Exception as e:
        print(f"Error in main function: {e}")
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
