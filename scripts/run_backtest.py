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
from decimal import Decimal
import pandas as pd
import numpy as np

# Set up error logging
error_log_path = "backtest_error_detailed.log"
with open(error_log_path, "w", encoding='utf-8') as error_log:
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
        # Define global variable for Rust availability
        global RUST_AVAILABLE
        try:
            from ai_trading_agent.backtesting.rust_backtester import RustBacktester
            error_log.write("✓ Successfully imported RustBacktester\n")
            RUST_AVAILABLE = True
        except ImportError as e:
            error_log.write(f"Note: RustBacktester not available: {e}\n")
            error_log.write("This is expected if Rust extensions are not installed.\n")
            RUST_AVAILABLE = False
            # Define a placeholder RustBacktester class to avoid NameError
            class RustBacktester:
                def __init__(self, *args, **kwargs):
                    raise ImportError("Rust extensions not available. Cannot use RustBacktester.")
        
        error_log.write("Importing agent components...\n")
        from ai_trading_agent.agent.data_manager import SimpleDataManager
        error_log.write("✓ Successfully imported data_manager\n")
        
                # Removed SimpleStrategyManager, added IntegratedStrategyManager and specific strategies
        from ai_trading_agent.agent.integrated_manager import IntegratedStrategyManager
        from ai_trading_agent.strategies.ma_crossover_strategy import MACrossoverStrategy
        from ai_trading_agent.strategies.sentiment_strategy import SentimentStrategy # Ensure this is the correct one if multiple exist
        error_log.write("✓ Successfully imported strategy\n")
        
        from ai_trading_agent.agent.risk_manager import SimpleRiskManager
        error_log.write("✓ Successfully imported risk_manager\n")
        
        from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
        error_log.write("✓ Successfully imported execution_handler\n")
        
        from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
        error_log.write("✓ Successfully imported portfolio_manager\n")
        
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
        
        def run_backtest(symbols, start_date, end_date, initial_capital=100000.0,
                      aggregation_method='weighted_average', strategy_weights=None,
                      market_regime='normal', volatility='medium', trend_strength='medium'):
            """
            Run a backtest for the specified symbols and date range.
            
            Args:
                symbols: List of ticker symbols
                start_date: Start date for the backtest
                end_date: End date for the backtest
                initial_capital: Initial capital for the portfolio
                aggregation_method: Method to combine signals ('weighted_average', 'dynamic_contextual', 
                                   'rule_based', or 'majority_vote')
                strategy_weights: Dictionary mapping strategy names to weights
                market_regime: Current market regime ('normal', 'trending', 'volatile', 'crisis')
                volatility: Current market volatility ('low', 'medium', 'high')
                trend_strength: Current market trend strength ('weak', 'medium', 'strong')
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
            # Construct the config dictionary for SimpleDataManager
            data_manager_config = {
                'data_dir': os.path.join(project_root, 'data'), # Assuming data is in project_root/data
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': '1d', # Assuming 1d timeframe, adjust if needed
                'data_types': ['ohlcv', 'sentiment'] # Load both types
            }
            data_manager = SimpleDataManager(config=data_manager_config)
            
            # Manually set the data in the data manager
            logger.info("Setting data directly in the data manager")
            data_manager.data = price_data
            
            # Create a combined index from the price data
            all_indices = set()
            for df in price_data.values():
                all_indices.update(df.index)
            data_manager.combined_index = pd.DatetimeIndex(sorted(list(all_indices)))
            data_manager.current_index = 0
            
            logger.info(f"Combined index created with {len(data_manager.combined_index)} timestamps")
            
            # Create strategy manager (NEW SETUP)
            # Instantiate individual strategies
            ma_crossover_strategy = MACrossoverStrategy(
                symbols=symbols,
                fast_period=20,   # Corrected argument name
                slow_period=50,   # Corrected argument name
                risk_pct=0.02,      # Use expected risk parameters
                max_position_pct=0.1 # Use expected risk parameters (adjust value if needed)
            )
            # Create the config dictionary for SentimentStrategy
            sentiment_config = {
                'assets': symbols, # Use 'assets' key as expected by SentimentStrategy
                'sentiment_threshold': 0.2, # Example threshold
                'max_position_size': 0.05, # Use 'max_position_size' key
                # Add other necessary config keys if needed, e.g.:
                # 'risk_per_trade': 0.01,
                # 'stop_loss_pct': 0.03,
                # 'take_profit_pct': 0.06
            }
            sentiment_strategy = SentimentStrategy(name='Sentiment', config=sentiment_config)
            
            # Instantiate the IntegratedStrategyManager with the specified aggregation method
            # Use default strategy weights if none provided
            default_strategy_weights = {
                'MACrossover': 0.6,  # Give more weight to technical analysis
                'Sentiment': 0.4     # Give less weight to sentiment
            }
            
            # Use provided strategy weights or default
            strategy_weights_to_use = strategy_weights if strategy_weights else default_strategy_weights
            
            # Configure the IntegratedStrategyManager
            integrated_manager_config = {
                'name': f'IntegratedManager_{aggregation_method}',
                'aggregation_method': aggregation_method,
                'strategy_weights': strategy_weights_to_use,
                # Market regime information for dynamic contextual combination
                'market_regime': market_regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                # Add priority rules for rule-based combination
                'priority_rules': [
                    {'condition': 'confidence_score > 0.9', 'action': 'use_highest_confidence'},
                    {'condition': 'signal_disagreement > 0.8', 'action': 'reduce_confidence'},
                    {'condition': 'default', 'action': 'weighted_average'}
                ]
            }
            
            logger.info(f"Using signal aggregation method: {aggregation_method}")
            strategy_manager = IntegratedStrategyManager(
                config=integrated_manager_config,
                data_manager=data_manager # Pass the data manager instance
            )
            
            # Add the individual strategies to the manager (method expects only the strategy object)
            strategy_manager.add_strategy(ma_crossover_strategy)
            strategy_manager.add_strategy(sentiment_strategy)
            
            # Create risk manager with config dictionary
            risk_config = {
                'max_position_size': 0.2,  # Changed from max_position_size_pct to match expected parameter
                'max_portfolio_risk_pct': 0.05,
                'stop_loss_pct': 0.05
            }
            risk_manager = SimpleRiskManager(config=risk_config)
            
            # Create portfolio manager
            portfolio_manager = PortfolioManager(
                initial_capital=Decimal('100000.0'),  # Starting with 100k
                risk_per_trade=Decimal('0.02'),       # 2% risk per trade
                max_position_size=Decimal('0.1'),     # Max 10% in any position
                max_correlation=0.7,                  # Max correlation between positions
                rebalance_frequency="daily"           # Rebalance daily
            )
            
            # Create execution handler with portfolio manager
            execution_handler = SimulatedExecutionHandler(
                portfolio_manager=portfolio_manager,  # Pass the portfolio manager
                config={
                    'commission_rate': 0.001,
                    'slippage_pct': 0.001
                }
            )
            
            # Create orchestrator
            orchestrator = BacktestOrchestrator(
                data_manager=data_manager,
                strategy_manager=strategy_manager,
                portfolio_manager=portfolio_manager,  # Add the portfolio manager
                risk_manager=risk_manager,
                execution_handler=execution_handler,
                config={
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': symbols,
                    'data_types': ['ohlcv', 'sentiment'],
                    'timeframe': '1d'
                }
            )
            
            # Run backtest
            try:
                # Use RustBacktester if available, otherwise use Python Backtester
                logger.info(f"RUST_AVAILABLE = {RUST_AVAILABLE}")
                
                # Always use Python Backtester for now to avoid issues
                logger.info("Using Python backtester")
                # Use portfolio_manager's starting balance
                backtester = Backtester(
                    data=price_data,
                    initial_capital=float(portfolio_manager.portfolio.starting_balance),  # Convert Decimal to float
                    commission_rate=0.001,
                    slippage=0.001
                )
                
                # Instead of passing a callback, let the orchestrator run itself
                logger.info("Running the backtest orchestrator directly")
                results = orchestrator.run()
                
                # Check if results is None or empty
                if results is None or not results:
                    logger.warning("Backtest completed but returned no results")
                    # Return a minimal results structure to avoid errors
                    return {
                        'portfolio_history': [],
                        'trades': [],
                        'orders_generated': [],
                        'signals': [],
                        'performance_metrics': {}
                    }
                
                # Log the results structure
                logger.info(f"Backtest completed with results keys: {list(results.keys())}")
                
                logger.info(f"Backtest completed successfully")
                return results
            except Exception as e:
                logger.error(f"Error running backtest: {e}", exc_info=True)
                # Log detailed error information to the error log file
                with open(error_log_path, "a", encoding='utf-8') as error_log:
                    error_log.write("\nDetailed error during backtest execution:\n")
                    error_log.write(f"Error type: {type(e).__name__}\n")
                    error_log.write(f"Error message: {str(e)}\n")
                    error_log.write("Traceback:\n")
                    error_log.write(traceback.format_exc())
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
