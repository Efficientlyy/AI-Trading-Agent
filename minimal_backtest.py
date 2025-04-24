import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading_agent.agent.factory import (
    create_data_manager,
    create_strategy,
    create_risk_manager,
    create_orchestrator,
    create_strategy_manager
)
from ai_trading_agent.agent.strategy import SimpleStrategyManager
from ai_trading_agent.agent.portfolio import SimplePortfolioManager
from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler

# Dummy config compatibility checker
def check_config_compatibility(config):
    return []

# Set up more detailed logging to diagnose the "No history" error
logging.basicConfig(
    filename="minimal_backtest.log",
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger()
logger.propagate = True

# Add a console handler to see logs in real-time
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("TEST LOG OUTPUT: Script started with enhanced logging")

def generate_sample_data(symbols, start_date, end_date, data_dir):
    logger.info(f"Generating dummy price data from {start_date} to {end_date} for {symbols}")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    for symbol in symbols:
        df = pd.DataFrame(index=dates)
        df['timestamp'] = df.index.strftime('%Y-%m-%d')
        df['open'] = np.random.uniform(100, 200, size=len(dates))
        df['high'] = df['open'] + np.random.uniform(0, 10, size=len(dates))
        df['low'] = df['open'] - np.random.uniform(0, 10, size=len(dates))
        df['close'] = df['open'] + np.random.uniform(-5, 5, size=len(dates))
        df['volume'] = np.random.randint(1000, 10000, size=len(dates))
        df = df.reset_index(drop=True)
        filename = os.path.join(data_dir, f"{symbol}_ohlcv_1d.csv")
        df.to_csv(filename, index=False)
        logger.info(f"Saved OHLCV data for {symbol} to {filename} with {len(df)} rows")
        # Log the date range to verify
        if not df.empty:
            logger.info(f"{symbol} data date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return True

def generate_sample_sentiment(symbols, start_date, end_date, data_dir):
    logger.info(f"Generating dummy sentiment data from {start_date} to {end_date} for {symbols}")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiment_rows = []
    for symbol in symbols:
        # Generate more extreme sentiment values to ensure signals cross thresholds
        sentiments = np.random.uniform(-0.5, 0.5, size=len(dates))
        for dt, sentiment in zip(dates, sentiments):
            sentiment_rows.append({
                'timestamp': dt.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'sentiment_score': sentiment  # Using sentiment_score to match what SentimentStrategy expects
            })
    df = pd.DataFrame(sentiment_rows)
    sentiment_file = os.path.join(data_dir, 'synthetic_sentiment.csv')
    df.to_csv(sentiment_file, index=False)
    
    # Log first few rows to verify sentiment data format
    logger.info(f"Saved sentiment data to {sentiment_file} with {len(df)} rows")
    logger.info(f"First few rows of sentiment data:\n{df.head(3).to_string()}")
    
    # Log sample statistics for each symbol to verify distribution
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol]
        logger.info(f"Sentiment stats for {symbol}: count={len(symbol_data)}, "
                   f"min={symbol_data['sentiment_score'].min():.4f}, "
                   f"max={symbol_data['sentiment_score'].max():.4f}, "
                   f"mean={symbol_data['sentiment_score'].mean():.4f}")
    return True

def run_backtest_with_factory():
    logger.info("Starting minimal backtest with factory system")
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    symbols = ["AAPL", "GOOG", "MSFT"]
    initial_capital = 100000.0
    data_dir = os.path.join('temp_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    else:
        logger.info(f"Data directory already exists: {data_dir}")
    generate_sample_data(symbols, start_date, end_date, data_dir)
    generate_sample_sentiment(symbols, start_date, end_date, data_dir)
    logger.info("Sample data generated and saved.")
    agent_config = {
        "data_manager": {
            "type": "SimpleDataManager",
            "config": {
                "data_dir": data_dir,
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": "1d",
                "data_types": ["ohlcv", "sentiment"]
            }
        },
        "strategy": {
            "type": "SentimentStrategy",
            "config": {
                "name": "SentimentStrategy",
                "symbols": symbols,
                "buy_threshold": 0.1,  # Buy when sentiment > 0.1
                "sell_threshold": -0.1, # Sell when sentiment < -0.1
                "position_size_pct": 0.1,
                "lookback_period": 5  # Ensure we specify a lookback period
            }
        },
        "risk_manager": {
            "type": "SimpleRiskManager",
            "config": {},
        },
        "backtest": {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signal_processing": {
                "sentiment_filter": "ema",
                "sentiment_filter_window": 10,
                "price_filter": "savgol",
                "price_filter_window": 15,
                "regime_detection": "volatility",
                "regime_window": 20,
                "regime_vol_threshold": 0.025
            },
            "available_strategies": [
                'SentimentStrategy',
                'MovingAverageCrossover',
                'RSIStrategy',
                'BollingerBandsStrategy',
                'ConservativeStrategy',
                'AggressiveStrategy',
                'MultiStrategyEnsemble',
                'OptimizedMAC'
            ],
            "optimizer": {
                "type": "GAOptimizer",
                "config": {
                    "fitness_fn": "dummy_fitness",
                    "param_space": {
                        'sentiment_threshold': [0.1, 0.2, 0.3, 0.4],
                        'position_size_pct': [0.05, 0.1, 0.15, 0.2]
                    }
                }
            }
        }
    }
    logger.info("Agent config created.")
    warnings = check_config_compatibility(agent_config)

    # --- Merge advanced config into strategy config ---
    for k in ["signal_processing", "symbols", "available_strategies", "optimizer"]:
        if k in agent_config["backtest"]:
            agent_config["strategy"]["config"][k] = agent_config["backtest"][k]
    # --- Ensure essential fields are present ---
    for k in ["symbols", "start_date", "end_date"]:
        if k in agent_config["data_manager"]["config"]:
            agent_config["strategy"]["config"][k] = agent_config["data_manager"]["config"][k]

    for warning in warnings:
        logger.warning(warning)
    data_manager = create_data_manager(agent_config["data_manager"])
    logger.info("Data manager created.")
    
    # Verify data loaded correctly
    for symbol in symbols:
        timestamp = pd.Timestamp(end_date)  # Use end_date as our reference point
        df = data_manager.get_historical_data([symbol], lookback=1, timestamp=timestamp)
        if df and symbol in df:
            df = df[symbol]  # Extract the dataframe for this symbol from the dict
            logger.info(f"Sample data for {symbol}: {df.shape}, columns: {df.columns.tolist()}")
            logger.debug(f"First few rows of {symbol} data:\n{df.head(3).to_string()}")
            # Check for required columns in the data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for {symbol}: {missing_columns}")
        else:
            logger.error(f"Failed to load data for {symbol}")
    
    # Explicitly test get_historical_data to diagnose "No history" error
    test_date = pd.Timestamp("2020-02-01")
    logger.info(f"Testing historical data retrieval for date: {test_date}")
    lookback = 5
    historical_window = data_manager.get_historical_data(symbols=symbols, lookback=lookback, timestamp=test_date)
    
    if historical_window:
        logger.info(f"Successfully retrieved historical data window with lookback={lookback}")
        for symbol, df in historical_window.items():
            logger.info(f"Historical data for {symbol}: {df.shape}, date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    else:
        logger.error(f"Failed to retrieve historical data window with lookback={lookback}")
    
    # Check for sentiment data availability
    logger.info("Verifying sentiment data availability")
    sentiment_data = data_manager.get_sentiment_data(symbols=symbols)
    if sentiment_data:
        logger.info(f"Sentiment data retrieved: {len(sentiment_data)} rows")
        for symbol in symbols:
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol]
            logger.info(f"Sentiment data for {symbol}: {len(symbol_sentiment)} rows")
            if not symbol_sentiment.empty:
                logger.info(f"Sample sentiment for {symbol}:\n{symbol_sentiment.head(3).to_string()}")
    else:
        logger.error("Failed to retrieve sentiment data")
    
    strategy = create_strategy(agent_config["strategy"])
    logger.info(f"Strategy created: {strategy}")
    # Ensure all required fields are present in manager_config (build directly from agent_config)
    manager_config = dict(agent_config["strategy"]["config"])
    for k in ["symbols", "start_date", "end_date"]:
        if k in agent_config["data_manager"]["config"]:
            manager_config[k] = agent_config["data_manager"]["config"][k]
    strategy_manager = create_strategy_manager(strategy, manager_type="SentimentStrategyManager", data_manager=data_manager, config=manager_config)
    logger.info("Strategy manager created via factory and configured.")
    portfolio_manager = SimplePortfolioManager(initial_cash=initial_capital, config={})
    logger.info("Portfolio manager created.")
    risk_manager = create_risk_manager(agent_config["risk_manager"])
    logger.info("Risk manager created.")
    execution_handler = SimulatedExecutionHandler(portfolio_manager=portfolio_manager, config={})
    logger.info("Execution handler created.")
    from ai_trading_agent.optimization.ga_optimizer import GAOptimizer
    def dummy_fitness(params):
        return 1.0
    param_space = {
        'sentiment_threshold': [0.1, 0.2, 0.3, 0.4],
        'position_size_pct': [0.05, 0.1, 0.15, 0.2]
    }
    optimizer = GAOptimizer(fitness_fn=dummy_fitness, param_space=param_space)
    available_strategies = [
        'SentimentStrategy',
        'MovingAverageCrossover',
        'RSIStrategy',
        'BollingerBandsStrategy',
        'ConservativeStrategy',
        'AggressiveStrategy',
        'MultiStrategyEnsemble',
        'OptimizedMAC'
    ]
    # --- Advanced signal processing config example ---
    agent_config["backtest"]["signal_processing"] = {
        "sentiment_filter": "ema",
        "sentiment_filter_window": 10,
        "price_filter": "savgol",
        "price_filter_window": 15,
        "regime_detection": "volatility",
        "regime_window": 20,
        "regime_vol_threshold": 0.025
    }
    agent_config["backtest"]["available_strategies"] = available_strategies
    agent_config["backtest"]["optimizer"] = optimizer
    
    # Ensure the required keys are in the orchestrator config
    agent_config["backtest"]["symbols"] = symbols
    agent_config["backtest"]["start_date"] = start_date
    agent_config["backtest"]["end_date"] = end_date
    
    logger.info("Adaptive intelligence configured.")
    orchestrator = create_orchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=agent_config["backtest"]
    )
    logger.info("Orchestrator created.")
    
    # Add a monkey patch to the orchestrator to debug signal generation
    original_run = orchestrator.run
    
    def run_with_debug(*args, **kwargs):
        logger.info("Running orchestrator with debug logging")
        try:
            # Manually check if signals are generated
            logger.info(f"Testing signal generation")
            
            # Get data for test - using get_historical_data correctly without timestamp parameter
            current_data = {}
            for symbol in symbols:
                # Use get_historical_data with lookback=1 to get current data
                data = data_manager.get_historical_data([symbol], lookback=1)
                if data and symbol in data:
                    data = data[symbol]  # Extract the dataframe for this symbol
                    if not data.empty:
                        logger.info(f"Found data for {symbol}")
                        # Use the most recent row as current data
                        if 'timestamp' in data.columns and not data['timestamp'].empty:
                            if isinstance(data['timestamp'].iloc[0], str):
                                data['timestamp'] = pd.to_datetime(data['timestamp'])
                            # Sort to get the most recent data point
                            data = data.sort_values('timestamp')
                            current_data[symbol] = data.iloc[-1]
                            logger.info(f"Using most recent data point for {symbol}: {data.iloc[-1]['timestamp']}")
                        else:
                            current_data[symbol] = data.iloc[-1]
                            logger.info(f"Using last row for {symbol} (no timestamp column)")
                    else:
                        logger.error(f"Empty dataframe for {symbol}")
                else:
                    logger.error(f"No data returned for {symbol}")
            
            # Get historical data window with explicit debugging
            lookback = strategy.lookback_period if hasattr(strategy, 'lookback_period') else 5
            logger.info(f"Retrieving historical data with lookback={lookback}")
            
            # CRITICAL FIX: Add robust error handling for historical data retrieval
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    historical_data = data_manager.get_historical_data(
                        symbols=symbols, 
                        lookback=lookback
                    )
                    
                    # Check if we have valid data
                    if historical_data and all(not df.empty for sym, df in historical_data.items()):
                        logger.info(f"Successfully retrieved historical data on attempt {attempt}")
                        break
                    else:
                        # Handle empty data issue by increasing the lookback
                        logger.warning(f"Attempt {attempt}: Empty or missing historical data. Increasing lookback.")
                        lookback = lookback * 2  # Double the lookback as fallback
                except Exception as e:
                    logger.warning(f"Attempt {attempt}: Error retrieving historical data: {e}")
                    if attempt < max_attempts:
                        logger.info(f"Retrying with increased lookback...")
                        lookback = lookback * 2  # Double the lookback as fallback
                    else:
                        logger.error("All attempts failed to retrieve historical data")
                        historical_data = None
            
            # Add detailed debugging for the "No history" error
            if historical_data:
                for sym, hist_df in historical_data.items():
                    logger.info(f"Historical data for {sym}: {hist_df.shape}")
                    logger.debug(f"First few rows for {sym}:\n{hist_df.head(3).to_string()}")
                    
                    # CRITICAL FIX: Check for empty dataframes and attempt recovery
                    if hist_df.empty:
                        logger.error(f"CRITICAL: Empty historical dataframe for {sym}! Attempting emergency recovery.")
                        # Attempt to fetch data with a larger lookback window as emergency fallback
                        emergency_data = data_manager.get_historical_data(
                            symbols=[sym], 
                            lookback=lookback * 2  # Double the lookback period
                        )
                        
                        if emergency_data and sym in emergency_data and not emergency_data[sym].empty:
                            logger.info(f"Emergency recovery successful for {sym}. Using alternate data source.")
                            historical_data[sym] = emergency_data[sym]
                            hist_df = historical_data[sym]  # Update the reference
                        else:
                            logger.critical(f"Emergency recovery failed for {sym}. This will cause 'No history' error!")
                    
                    elif len(hist_df) < lookback:
                        logger.warning(f"Historical data for {sym} has fewer rows ({len(hist_df)}) than required lookback ({lookback})")
                        
                    # CRITICAL FIX: Verify timestamp column format
                    if 'timestamp' in hist_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(hist_df['timestamp']):
                            logger.info(f"Converting timestamp to datetime for {sym}")
                            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                            
                    # Verify sentiment columns are present
                    sentiment_cols = [col for col in hist_df.columns if 'sentiment' in col.lower()]
                    if sentiment_cols:
                        logger.info(f"Found sentiment columns for {sym}: {sentiment_cols}")
                    else:
                        logger.warning(f"No sentiment columns found for {sym}. Available columns: {hist_df.columns.tolist()}")
                        
                        # CRITICAL FIX: More robust sentiment data joining
                        sentiment_file = os.path.join(data_dir, 'synthetic_sentiment.csv')
                        if os.path.exists(sentiment_file):
                            logger.info(f"Adding sentiment data from {sentiment_file}")
                            sent_df = pd.read_csv(sentiment_file)
                            sent_df = sent_df[sent_df['symbol'] == sym]
                            if not sent_df.empty:
                                # Convert timestamp to datetime for proper merging
                                sent_df['timestamp'] = pd.to_datetime(sent_df['timestamp'])
                                if not pd.api.types.is_datetime64_any_dtype(hist_df['timestamp']):
                                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                                    
                                # CRITICAL FIX: Use outer merge to ensure we don't lose data and fill missing values
                                hist_df = pd.merge(
                                    hist_df, 
                                    sent_df[['timestamp', 'sentiment_score']], 
                                    on='timestamp', 
                                    how='left'
                                )
                                
                                # Fill any NaN sentiment values with neutral (0.0)
                                if 'sentiment_score' in hist_df.columns and hist_df['sentiment_score'].isna().any():
                                    logger.warning(f"Filling {hist_df['sentiment_score'].isna().sum()} missing sentiment values")
                                    hist_df['sentiment_score'] = hist_df['sentiment_score'].fillna(0.0)
                                    
                                logger.info(f"Added sentiment data to {sym} historical data")
                                historical_data[sym] = hist_df
            else:
                logger.error(f"No historical data window returned")
                # Try to diagnose why no historical data is returned
                logger.info("Attempting to diagnose missing historical data issue:")
                
                # CRITICAL FIX: Create synthetic historical data as last resort fallback if all else fails
                logger.warning("Creating synthetic historical data as emergency fallback")
                historical_data = {}
                for symbol in symbols:
                    # Create synthetic dates without assuming a specific test_date
                    end_date = pd.Timestamp.now()
                    dates = pd.date_range(end=end_date, periods=lookback+1)
                    df = pd.DataFrame(index=dates)
                    df['timestamp'] = df.index
                    df['open'] = np.random.uniform(100, 200, size=len(dates))
                    df['high'] = df['open'] + np.random.uniform(0, 10, size=len(dates))
                    df['low'] = df['open'] - np.random.uniform(0, 10, size=len(dates))
                    df['close'] = df['open'] + np.random.uniform(-5, 5, size=len(dates))
                    df['volume'] = np.random.randint(1000, 10000, size=len(dates))
                    df['sentiment_score'] = np.random.uniform(-0.5, 0.5, size=len(dates))
                    df = df.reset_index(drop=True)
                    historical_data[symbol] = df
                    logger.warning(f"Created emergency synthetic data for {symbol} with {len(df)} rows")
                
                # Log a clear marker that synthetic data was used
                logger.warning("USING SYNTHETIC FALLBACK DATA - This is not real market data!")
                
            # Get current portfolio state
            portfolio_state = portfolio_manager.get_state()
            
            # Generate signals
            if strategy_manager:
                logger.info(f"Manually generating signals with strategy_manager...")
                
                # CRITICAL FIX: Ensure we have data to pass to the strategy
                if not current_data and historical_data:
                    logger.warning("Missing current_data but have historical_data. Using most recent historical data point.")
                    for symbol, hist_df in historical_data.items():
                        if not hist_df.empty:
                            # Sort by timestamp to get most recent
                            if isinstance(hist_df['timestamp'].iloc[0], str):
                                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                            hist_df = hist_df.sort_values('timestamp')
                            current_data[symbol] = hist_df.iloc[-1]
                            logger.info(f"Using most recent historical datapoint for {symbol} as current data")
                
                # Find current timestamp for debugging (use the most recent timestamp from data)
                current_timestamp = None
                if current_data:
                    for symbol, data_point in current_data.items():
                        if 'timestamp' in data_point:
                            if current_timestamp is None or data_point['timestamp'] > current_timestamp:
                                current_timestamp = data_point['timestamp']
                                logger.info(f"Using timestamp {current_timestamp} from {symbol} for signal generation")
                
                # Verify we have data to pass to the strategy
                if current_data and historical_data:
                    try:
                        signals = strategy_manager.generate_signals(
                            current_data=current_data, 
                            portfolio_state=portfolio_state,
                            timestamp=current_timestamp,  # Pass current timestamp from data
                            historical_data=historical_data  # Pass the historical data explicitly
                        )
                        logger.info(f"Generated signals: {signals}")
                        
                        # Generate orders from signals
                        if signals:
                            logger.info(f"Generating orders from signals: {signals}")
                            orders = portfolio_manager.generate_orders(
                                signals=signals, 
                                market_data=current_data
                            )
                            logger.info(f"Generated orders: {orders}")
                        else:
                            logger.warning("No signals generated during debug test")
                    except Exception as e:
                        logger.exception(f"Error during signal generation: {e}")
                        logger.error("This could be the source of the 'No history' error!")
                else:
                    logger.error("Missing current_data or historical_data required for signal generation")
            else:
                logger.error("Strategy manager not available for signal generation")
        
        except Exception as e:
            logger.exception(f"Error during debug signal generation: {e}")
        
        # Call the original run method
        logger.info("Proceeding with original orchestrator.run() method")
        return original_run(*args, **kwargs)
    
    orchestrator.run = run_with_debug
    
    logger.info("Running backtest...")
    results = orchestrator.run()
    logger.info("Backtest finished.")
    if results:
        logger.info(f"Backtest completed successfully")
        print("\n=== Backtest Performance Metrics ===")
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            logger.info(f"Performance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                    print(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
                    print(f"  {key}: {value}")
        if 'adaptive_reason' in results:
            logger.info(f"Adaptive Reason: {results['adaptive_reason']}")
            print(f"Adaptive Reason: {results['adaptive_reason']}")
        print("====================================\n")
    else:
        logger.warning("Backtest did not return results")
        print("Backtest did not return results.")
    return results

if __name__ == "__main__":
    run_backtest_with_factory()
