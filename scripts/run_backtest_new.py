#!/usr/bin/env python
# scripts/run_backtest.py

"""
Main script for running backtests with the AI Trading Agent.

This script configures and executes a backtest for the sentiment-based trading strategy
across multiple assets.
"""

import os
import sys
import traceback
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Set up error logging
error_log_path = "backtest_error_detailed.log"
with open(error_log_path, "w") as error_log:
    error_log.write(f"=== BACKTEST ERROR LOG ===\n")
    error_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        error_log.write(f"Added project root to sys.path: {project_root}\n\n")
        
        # Import components with detailed error logging
        error_log.write("Importing common modules...\n")
        from ai_trading_agent.common.logging_config import setup_logging
        error_log.write("✓ Successfully imported logging_config\n\n")
        
        error_log.write("Importing trading engine components...\n")
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        error_log.write("✓ Successfully imported enums\n")
        
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        error_log.write("✓ Successfully imported models\n")
        
        from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
        error_log.write("✓ Successfully imported portfolio_manager\n\n")
        
        error_log.write("Importing backtesting components...\n")
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        error_log.write("✓ Successfully imported performance_metrics\n")
        
        from ai_trading_agent.backtesting.backtester import Backtester
        error_log.write("✓ Successfully imported Backtester\n\n")
        
        # Only import RustBacktester if needed
        try:
            from ai_trading_agent.backtesting.rust_backtester import RustBacktester
            error_log.write("✓ Successfully imported RustBacktester\n")
            RUST_AVAILABLE = True
        except ImportError as e:
            error_log.write(f"Note: RustBacktester not available: {e}\n")
            error_log.write("This is expected if Rust extensions are not installed.\n")
            RUST_AVAILABLE = False
        
        error_log.write("Importing agent components...\n")
        from ai_trading_agent.agent.data_manager import SimpleDataManager
        error_log.write("✓ Successfully imported data_manager\n")
        
        from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
        error_log.write("✓ Successfully imported strategy\n")
        
        from ai_trading_agent.agent.risk_manager import SimpleRiskManager
        error_log.write("✓ Successfully imported risk_manager\n")
        
        from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
        error_log.write("✓ Successfully imported execution_handler\n")
        
        from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
        error_log.write("✓ Successfully imported orchestrator\n\n")
        
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Continue with the rest of the script...
        
        def load_data(symbols, start_date, end_date):
            """
            Load historical price data for the specified symbols.
            
            Args:
                symbols: List of ticker symbols
                start_date: Start date for historical data
                end_date: End date for historical data
                
            Returns:
                Dictionary mapping symbols to DataFrames with OHLCV data
            """
            logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # In a real implementation, this would load data from a database or API
            # For this example, we'll generate synthetic data
            data = {}
            
            for symbol in symbols:
                # Generate synthetic price data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                n = len(dates)
                
                # Start with a random walk
                close = 100 + np.random.normal(0, 1, n).cumsum()
                
                # Ensure price doesn't go negative
                close = np.maximum(close, 1)
                
                # Generate other OHLCV data
                high = close * (1 + np.random.uniform(0, 0.02, n))
                low = close * (1 - np.random.uniform(0, 0.02, n))
                open_price = low + np.random.uniform(0, 1, n) * (high - low)
                volume = np.random.uniform(100000, 1000000, n)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                }, index=dates)
                
                data[symbol] = df
            
            logger.info(f"Data loaded successfully with {len(data[symbols[0]])} bars per symbol")
            return data
        
        def load_sentiment_data(symbols, start_date, end_date):
            """
            Load sentiment data for the specified symbols.
            
            Args:
                symbols: List of ticker symbols
                start_date: Start date for sentiment data
                end_date: End date for sentiment data
                
            Returns:
                Dictionary mapping symbols to DataFrames with sentiment scores
            """
            logger.info(f"Loading sentiment data for {len(symbols)} symbols")
            
            # In a real implementation, this would load data from a sentiment analysis source
            # For this example, we'll generate synthetic sentiment data
            sentiment_data = {}
            
            for symbol in symbols:
                # Generate synthetic sentiment data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                n = len(dates)
                
                # Generate sentiment scores between -1 and 1
                sentiment = np.random.normal(0, 0.5, n)
                sentiment = np.clip(sentiment, -1, 1)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'sentiment': sentiment
                }, index=dates)
                
                sentiment_data[symbol] = df
            
            logger.info(f"Sentiment data loaded successfully")
            return sentiment_data
        
        def run_backtest(symbols, start_date, end_date, initial_capital=100000.0):
            """
            Run a backtest for the specified symbols and date range.
            
            Args:
                symbols: List of ticker symbols
                start_date: Start date for the backtest
                end_date: End date for the backtest
                initial_capital: Initial capital for the portfolio
                
            Returns:
                Dictionary containing backtest results
            """
            logger.info(f"Starting backtest with {len(symbols)} symbols and {initial_capital} initial capital")
            
            # Load price data
            price_data = load_data(symbols, start_date, end_date)
            
            # Load sentiment data
            sentiment_data = load_sentiment_data(symbols, start_date, end_date)
            
            # Create data manager
            data_manager = SimpleDataManager(price_data, sentiment_data)
            
            # Create strategy manager
            strategy = SentimentStrategy(
                symbols=symbols,
                sentiment_threshold=0.3,
                position_size_pct=0.1
            )
            strategy_manager = SimpleStrategyManager(strategy)
            
            # Create risk manager
            risk_manager = SimpleRiskManager(
                max_position_size_pct=0.2,
                max_portfolio_risk_pct=0.05,
                stop_loss_pct=0.05
            )
            
            # Create execution handler
            execution_handler = SimulatedExecutionHandler(
                config={
                    'commission_rate': 0.001,
                    'slippage_pct': 0.001
                }
            )
            
            # Create orchestrator
            orchestrator = BacktestOrchestrator(
                data_manager=data_manager,
                strategy_manager=strategy_manager,
                risk_manager=risk_manager,
                execution_handler=execution_handler,
                initial_capital=initial_capital
            )
            
            # Run backtest
            try:
                # Use RustBacktester if available, otherwise use Python Backtester
                if RUST_AVAILABLE:
                    logger.info("Using Rust-accelerated backtester")
                    backtester = RustBacktester(
                        data=price_data,
                        initial_capital=initial_capital,
                        commission_rate=0.001,
                        slippage=0.001
                    )
                    results = backtester.run(orchestrator.process_bar)
                else:
                    logger.info("Using Python backtester")
                    backtester = Backtester(
                        data=price_data,
                        initial_capital=initial_capital,
                        commission_rate=0.001,
                        slippage=0.001
                    )
                    results = backtester.run(orchestrator.process_bar)
                
                logger.info(f"Backtest completed successfully")
                return results
            except Exception as e:
                logger.error(f"Error running backtest: {e}")
                raise
        
        def analyze_results(results):
            """
            Analyze backtest results and print performance metrics.
            
            Args:
                results: Dictionary containing backtest results
            """
            logger.info("Analyzing backtest results")
            
            # Extract results
            portfolio_history = results['portfolio_history']
            trade_history = results['trade_history']
            
            # Calculate performance metrics
            metrics = calculate_metrics(
                portfolio_history=portfolio_history,
                trade_history=trade_history,
                initial_capital=portfolio_history[0]['total_value'],
                risk_free_rate=0.0
            )
            
            # Print performance metrics
            logger.info(f"Total Return: {metrics.total_return:.2%}")
            logger.info(f"Annualized Return: {metrics.annualized_return:.2%}")
            logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
            logger.info(f"Win Rate: {metrics.win_rate:.2%}")
            logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
            
            return metrics
        
        if __name__ == '__main__':
            try:
                # Define backtest parameters
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
                start_date = '2022-01-01'
                end_date = '2022-12-31'
                initial_capital = 100000.0
                
                # Run backtest
                results = run_backtest(symbols, start_date, end_date, initial_capital)
                
                # Analyze results
                metrics = analyze_results(results)
                
                logger.info("Backtest completed successfully")
                
            except Exception as e:
                logger.error(f"Error in backtest execution: {e}")
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"\n\nERROR in backtest execution: {e}\n")
                    error_log.write("Traceback:\n")
                    error_log.write(traceback.format_exc())
                print(f"Error occurred. See {error_log_path} for details.")
            finally:
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"\n=== END OF LOG ===\n")
    
    except Exception as e:
        error_log.write(f"\n❌ ERROR during import phase: {e}\n\n")
        error_log.write("Traceback:\n")
        error_log.write(traceback.format_exc())
        error_log.write("\n")
        print(f"Error during import phase. See {error_log_path} for details.")
