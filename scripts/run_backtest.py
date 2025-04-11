import logging
import sys
import os
from datetime import datetime
import pandas as pd

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root) # Add the directory *containing* src

# Import agent components (using try-except for flexibility)
# Use absolute imports from the perspective of project_root
try:
    from src.agent.data_manager import SimpleDataManager
    from src.agent.strategy import SimpleStrategyManager, SentimentStrategy
    from src.agent.portfolio import SimplePortfolioManager
    from src.agent.risk_manager import SimpleRiskManager
    from src.agent.execution_handler import SimulatedExecutionHandler
    from src.agent.orchestrator import BacktestOrchestrator
except ImportError as e:
    print(f"Error importing agent components: {e}")
    print("Ensure the script is run from the project root or src is in the PYTHONPATH.")
    sys.exit(1)

# --- Configuration ---
LOG_LEVEL = logging.INFO # INFO, DEBUG
INITIAL_CASH = 100000.0
SYMBOLS = ['AAPL', 'MSFT'] # Example symbols
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
DATA_DIR = os.path.join(project_root, 'data') # Directory for generated/loaded data
SENTIMENT_FILE = os.path.join(DATA_DIR, 'synthetic_sentiment.csv')

# --- Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Main Backtest Function ---
def run_sentiment_backtest():
    logger.info("--- Starting Sentiment Strategy Backtest ---")

    # 1. Data Manager Configuration and Initialization
    #    Generate synthetic data for this example
    data_manager_config = {
        'data_dir': DATA_DIR,
        'symbols': SYMBOLS,
        'start_date': START_DATE,
        'end_date': END_DATE
    }
    data_manager = SimpleDataManager(config=data_manager_config)

    # Generate synthetic data if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, f"{SYMBOLS[0]}_ohlcv.csv")):
        logger.info("Generating synthetic OHLCV data...")
        data_manager.generate_synthetic_ohlcv(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE, timeframe='1d')
    if not os.path.exists(SENTIMENT_FILE):
        logger.info("Generating synthetic sentiment data...")
        data_manager.generate_synthetic_sentiment(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE)

    # 2. Strategy Manager Configuration and Initialization
    sentiment_strategy_config = {
        'buy_threshold': 0.1,
        'sell_threshold': -0.1,
        'sentiment_source': 'sentiment_score' # Column name in sentiment data
    }
    sentiment_strategy = SentimentStrategy(name="SentimentVader", config=sentiment_strategy_config)

    strategy_manager = SimpleStrategyManager(config=sentiment_strategy_config, data_manager=data_manager)
    strategy_manager.add_strategy(sentiment_strategy)
    logger.info("Strategy Manager initialized and Sentiment Strategy added.")

    # 3. Portfolio Manager Configuration and Initialization
    portfolio_specific_config = {
        'allocation_fraction_per_trade': 0.10 # Example: Allocate 10% of portfolio value per trade
    }
    portfolio_manager = SimplePortfolioManager(initial_cash=INITIAL_CASH, config=portfolio_specific_config)
    logger.info(f"Portfolio Manager initialized with initial cash: {INITIAL_CASH:.2f}")

    # 4. Risk Manager Configuration and Initialization
    risk_manager_config = {
         'max_position_size': 100,
         'stop_loss_pct': 0.05
    }
    risk_manager = SimpleRiskManager(config=risk_manager_config)
    logger.info("Risk Manager initialized.")

    # 5. Execution Handler Configuration and Initialization
    # !! Must be initialized AFTER portfolio_manager !!
    execution_handler_config = {
        'commission_per_trade': 0.0, # Example: No commission
        'slippage_model': 'percentage', # Example: Percentage-based slippage
        'percentage_slippage': 0.0005 # Example: 0.05%
    }
    execution_handler = SimulatedExecutionHandler(
        portfolio_manager=portfolio_manager, # Pass the created portfolio_manager
        config=execution_handler_config
    )
    logger.info("Execution Handler initialized.")

    # 6. Orchestrator Configuration and Initialization
    orchestrator_config = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'symbols': SYMBOLS,
        'data_types': ['ohlcv', 'sentiment'], # Specify data types needed
        'timeframe': '1d'
    }
    orchestrator = BacktestOrchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=orchestrator_config
    )

    # 7. Run the backtest
    logger.info("Running backtest orchestrator...")
    try:
        results = orchestrator.run()
        logger.info("Backtest finished successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the backtest run: {e}", exc_info=True)
        results = None

    # 8. Display Results
    if results:
        logger.info("--- Backtest Results ---")
        logger.info(f"Initial Cash: {INITIAL_CASH:.2f}")

        perf_metrics = results.get('performance_metrics')

        if perf_metrics:
            if 'error' in perf_metrics:
                logger.error(f"Performance metrics calculation failed: {perf_metrics['error']}")
            else:
                logger.info("Performance Metrics:")
                # Calculate final value from history if not directly in metrics (though it should be)
                final_value = results['portfolio_history'][-1]['value'] if results.get('portfolio_history') else INITIAL_CASH
                logger.info(f"  Final Portfolio Value: {final_value:.2f}")
                for key, value in perf_metrics.items():
                    if isinstance(value, (int, float)):
                         # Format percentages and ratios nicely
                         if 'pct' in key:
                             logger.info(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
                         elif key == 'sharpe_ratio':
                             logger.info(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                         else:
                              logger.info(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        logger.info(f"  {key.replace('_', ' ').title()}: {value}")

                # Log total orders/fills separately as they are not strictly 'performance'
                logger.info(f"Operational Metrics:")
                logger.info(f"  Total Orders Generated: {len(results.get('orders_generated', []))}")
                logger.info(f"  Total Fills Executed: {len(results.get('fills_executed', []))}")

        else:
            logger.warning("Performance metrics not found in results.")
            # Fallback to basic info if metrics are missing but history exists
            if results.get('portfolio_history'):
                 final_value = results['portfolio_history'][-1]['value']
                 logger.info(f"Final Portfolio Value (from history): {final_value:.2f}")
                 logger.info(f"Total Orders Generated: {len(results.get('orders_generated', []))}")
                 logger.info(f"Total Fills Executed: {len(results.get('fills_executed', []))}")

        # Optionally, save portfolio history to CSV
        if results.get('portfolio_history'):
            history_df = pd.DataFrame(results['portfolio_history'])
            history_df.set_index('timestamp', inplace=True)
            history_file = os.path.join(DATA_DIR, 'backtest_portfolio_history.csv')
            history_df.to_csv(history_file)
            logger.info(f"Portfolio history saved to {history_file}")
        else:
             logger.warning("Portfolio history not found in results.")
    else:
        logger.error("Backtest did not produce results.")

    logger.info("--- Backtest Script Finished ---")

# --- Script Execution ---
if __name__ == "__main__":
    run_sentiment_backtest()
