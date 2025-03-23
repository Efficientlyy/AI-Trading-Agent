"""
Sentiment Backtesting Demo.

This example demonstrates how to use the sentiment data collector to gather historical
sentiment data and then run a backtest using the sentiment backtester framework.
"""

import asyncio
import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data.sentiment_collector import SentimentCollector
from src.backtesting.sentiment_backtester import SentimentBacktester, SentimentStrategy, AdvancedSentimentStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure reports directory exists
reports_dir = Path('reports')
reports_dir.mkdir(exist_ok=True)

async def collect_historical_sentiment():
    """Collect historical sentiment data for backtesting."""
    logger.info("Collecting historical sentiment data...")
    
    # Initialize sentiment collector
    collector = SentimentCollector()
    
    # Define collection parameters
    symbol = "BTC"
    start_date = datetime.datetime.now() - datetime.timedelta(days=90)
    end_date = datetime.datetime.now()
    
    # Collect Fear & Greed Index data
    try:
        logger.info("Collecting Fear & Greed Index data...")
        fear_greed_data = await collector.collect_historical_data(
            source="fear_greed",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            save=True
        )
        logger.info(f"Collected {len(fear_greed_data['timestamp'])} Fear & Greed data points")
        
        # Visualize the collected data
        plt.figure(figsize=(12, 6))
        plt.plot(fear_greed_data['timestamp'], fear_greed_data['sentiment_value'])
        plt.title(f"Fear & Greed Index for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Sentiment (0-1)")
        plt.grid(True)
        plt.savefig(reports_dir / "fear_greed_historical.png")
        
    except Exception as e:
        logger.error(f"Error collecting Fear & Greed data: {e}")
    
    # Add collection for other sentiment sources here when implemented
    # (news, social media, on-chain)
    
    logger.info("Historical sentiment data collection complete")

def run_basic_sentiment_backtest():
    """Run a basic sentiment strategy backtest."""
    logger.info("Running basic sentiment strategy backtest...")
    
    # Define backtest configuration
    config = {
        'symbol': 'BTC-USD',
        'start_date': datetime.datetime.now() - datetime.timedelta(days=90),
        'end_date': datetime.datetime.now(),
        'sources': ['fear_greed'],  # Only using Fear & Greed for demonstration
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'SentimentStrategy',
        'strategy_config': {
            'symbol': 'BTC-USD',
            'sentiment_threshold_buy': 0.7,
            'sentiment_threshold_sell': 0.3,
            'contrarian': False,  # Contrarian mode would invert the sentiment signals
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_positions': 1,
            'source_weights': {'fear_greed': 1.0}
        },
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Initialize backtester
    backtester = SentimentBacktester(config)
    
    # Run backtest
    try:
        results = backtester.run_backtest()
        
        # Generate report
        report = backtester.generate_report(
            results, 
            output_path=reports_dir / "basic_sentiment_backtest_report.txt"
        )
        print("\nBASIC SENTIMENT STRATEGY BACKTEST REPORT:")
        print(report)
        
        # Visualize results
        backtester.visualize_results(
            results, 
            output_path=reports_dir / "basic_sentiment_backtest_plot.png"
        )
        
    except Exception as e:
        logger.error(f"Error in basic sentiment backtest: {e}")

def run_advanced_sentiment_backtest():
    """Run an advanced sentiment strategy backtest."""
    logger.info("Running advanced sentiment strategy backtest...")
    
    # Define backtest configuration
    config = {
        'symbol': 'BTC-USD',
        'start_date': datetime.datetime.now() - datetime.timedelta(days=90),
        'end_date': datetime.datetime.now(),
        'sources': ['fear_greed'],  # Only using Fear & Greed for demonstration
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'AdvancedSentimentStrategy',
        'strategy_config': {
            'symbol': 'BTC-USD',
            'sentiment_threshold_buy': 0.7,
            'sentiment_threshold_sell': 0.3,
            'contrarian': False,
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_positions': 1,
            'source_weights': {'fear_greed': 1.0},
            # Advanced parameters
            'trend_window': 14,
            'trend_strength_threshold': 0.3,
            'use_adaptive_thresholds': True,
            'technical_confirmation': True,
            'regime_thresholds': {
                'low_volatility': {
                    'sentiment_threshold_buy': 0.65,
                    'sentiment_threshold_sell': 0.35,
                    'position_size': 1.0
                },
                'high_volatility': {
                    'sentiment_threshold_buy': 0.8,
                    'sentiment_threshold_sell': 0.2,
                    'position_size': 0.5
                }
            }
        },
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Initialize backtester
    backtester = SentimentBacktester(config)
    
    # Run backtest
    try:
        results = backtester.run_backtest()
        
        # Generate report
        report = backtester.generate_report(
            results,
            output_path=reports_dir / "advanced_sentiment_backtest_report.txt"
        )
        print("\nADVANCED SENTIMENT STRATEGY BACKTEST REPORT:")
        print(report)
        
        # Visualize results
        backtester.visualize_results(
            results,
            output_path=reports_dir / "advanced_sentiment_backtest_plot.png"
        )
        
    except Exception as e:
        logger.error(f"Error in advanced sentiment backtest: {e}")

def run_parameter_optimization():
    """Run parameter optimization for the sentiment strategy."""
    logger.info("Running parameter optimization...")
    
    # Define backtest configuration
    config = {
        'symbol': 'BTC-USD',
        'start_date': datetime.datetime.now() - datetime.timedelta(days=90),
        'end_date': datetime.datetime.now(),
        'sources': ['fear_greed'],
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'AdvancedSentimentStrategy',
        'strategy_config': {
            'symbol': 'BTC-USD',
            'contrarian': False,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'source_weights': {'fear_greed': 1.0},
            'trend_window': 14,
            'technical_confirmation': True
        },
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Initialize backtester
    backtester = SentimentBacktester(config)
    
    # Define parameter grid
    param_grid = {
        'sentiment_threshold_buy': [0.6, 0.7, 0.8],
        'sentiment_threshold_sell': [0.2, 0.3, 0.4],
        'trend_strength_threshold': [0.2, 0.3, 0.4],
        'use_adaptive_thresholds': [True, False]
    }
    
    # Run optimization
    try:
        logger.info("Starting parameter optimization...")
        optimization_results = backtester.run_parameter_optimization(
            param_grid=param_grid,
            metric='sharpe_ratio',
            report=True
        )
        
        # Log best parameters
        best_params = optimization_results['best_params']
        best_score = optimization_results['best_score']
        
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        logger.info(f"Best Sharpe ratio: {best_score:.4f}")
        
        # Save optimization results
        with open(reports_dir / "sentiment_optimization_results.txt", 'w') as f:
            f.write("SENTIMENT STRATEGY PARAMETER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best Sharpe ratio: {best_score:.4f}\n\n")
            
            f.write("Top 10 Parameter Combinations:\n")
            for i, result in enumerate(optimization_results['all_results'][:10]):
                f.write(f"\nRank {i+1}:\n")
                f.write(f"Parameters: {result['params']}\n")
                f.write(f"Sharpe ratio: {result['score']:.4f}\n")
        
        # Run backtest with optimized parameters
        logger.info("Running backtest with optimized parameters...")
        
        # Update strategy config with best parameters
        optimized_config = config.copy()
        optimized_config['strategy_config'].update(best_params)
        
        # Create new backtester with optimized config
        optimized_backtester = SentimentBacktester(optimized_config)
        
        # Run backtest
        optimized_results = optimized_backtester.run_backtest()
        
        # Generate report
        optimized_report = optimized_backtester.generate_report(
            optimized_results,
            output_path=reports_dir / "optimized_sentiment_backtest_report.txt"
        )
        print("\nOPTIMIZED SENTIMENT STRATEGY BACKTEST REPORT:")
        print(optimized_report)
        
        # Visualize results
        optimized_backtester.visualize_results(
            optimized_results,
            output_path=reports_dir / "optimized_sentiment_backtest_plot.png"
        )
        
    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}")

async def main():
    """Main function to run the demo."""
    print("\n=== SENTIMENT BACKTESTING DEMO ===\n")
    
    # Step 1: Collect historical sentiment data
    await collect_historical_sentiment()
    
    # Step 2: Run basic sentiment strategy backtest
    run_basic_sentiment_backtest()
    
    # Step 3: Run advanced sentiment strategy backtest
    run_advanced_sentiment_backtest()
    
    # Step 4: Run parameter optimization
    run_parameter_optimization()
    
    print("\n=== DEMO COMPLETE ===\n")
    print(f"Reports and visualizations saved to {reports_dir} directory")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())