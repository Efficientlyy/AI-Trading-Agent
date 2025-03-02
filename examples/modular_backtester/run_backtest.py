"""
Backtesting runner script.

This script serves as the entry point for running backtest simulations,
providing command-line options and example usage of the modular backtesting system.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.modular_backtester.models import TimeFrame, CandleData
from examples.modular_backtester.strategies import (
    MovingAverageCrossoverStrategy, 
    EnhancedMAStrategy,
    RSIStrategy, 
    MACDStrategy, 
    MultiStrategySystem
)
from examples.modular_backtester.backtester import StrategyBacktester
from examples.modular_backtester.data_utils import (
    generate_sample_data, 
    load_csv_data,
    save_to_csv,
    resample_timeframe
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)

logger = logging.getLogger("backtest_runner")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtesting Framework Runner')
    
    # Data source options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-file', type=str, help='Path to historical data CSV file')
    data_group.add_argument('--symbol', type=str, default='BTC-USD', help='Trading symbol')
    data_group.add_argument('--timeframe', type=str, default='1h', 
                           choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
                           help='Candle timeframe')
    data_group.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    data_group.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    data_group.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    
    # Strategy options
    strategy_group = parser.add_argument_group('Strategy Options')
    strategy_group.add_argument('--strategy', type=str, default='ma_crossover',
                              choices=['ma_crossover', 'enhanced_ma', 'rsi', 'macd', 'multi'],
                              help='Trading strategy to use')
    strategy_group.add_argument('--fast-period', type=int, default=8, help='Fast period for MA/MACD')
    strategy_group.add_argument('--slow-period', type=int, default=21, help='Slow period for MA/MACD')
    strategy_group.add_argument('--signal-period', type=int, default=9, help='Signal period for MACD')
    strategy_group.add_argument('--rsi-period', type=int, default=14, help='Period for RSI')
    strategy_group.add_argument('--overbought', type=float, default=70.0, help='Overbought threshold')
    strategy_group.add_argument('--oversold', type=float, default=30.0, help='Oversold threshold')
    
    # Backtester options
    backtest_group = parser.add_argument_group('Backtest Options')
    backtest_group.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital')
    backtest_group.add_argument('--position-size', type=float, default=0.1, 
                              help='Position size as fraction of capital (0.0-1.0)')
    backtest_group.add_argument('--use-stop-loss', action='store_true', help='Enable stop loss')
    backtest_group.add_argument('--stop-loss-pct', type=float, default=0.05, 
                              help='Stop loss percentage')
    backtest_group.add_argument('--use-take-profit', action='store_true', help='Enable take profit')
    backtest_group.add_argument('--take-profit-pct', type=float, default=0.1, 
                              help='Take profit percentage')
    backtest_group.add_argument('--use-trailing-stop', action='store_true', help='Enable trailing stop')
    backtest_group.add_argument('--trailing-stop-pct', type=float, default=0.03, 
                              help='Trailing stop percentage')
    backtest_group.add_argument('--commission', type=float, default=0.001, 
                              help='Commission percentage')
    
    # Optimization options
    optimize_group = parser.add_argument_group('Optimization Options')
    optimize_group.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    optimize_group.add_argument('--optimize-metric', type=str, default='sharpe_ratio',
                              choices=['sharpe_ratio', 'total_pnl', 'profit_factor', 'win_rate'],
                              help='Metric to optimize')
    optimize_group.add_argument('--max-iterations', type=int, default=20, 
                              help='Maximum iterations for optimization')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', type=str, default='reports', 
                             help='Directory for output reports')
    output_group.add_argument('--save-results', action='store_true', 
                             help='Save detailed results to JSON')
    
    return parser.parse_args()


def string_to_timeframe(timeframe_str: str) -> TimeFrame:
    """Convert string timeframe to TimeFrame enum."""
    if timeframe_str == '1m':
        return TimeFrame.MINUTE_1
    elif timeframe_str == '5m':
        return TimeFrame.MINUTE_5
    elif timeframe_str == '15m':
        return TimeFrame.MINUTE_15
    elif timeframe_str == '30m':
        return TimeFrame.MINUTE_30
    elif timeframe_str == '1h':
        return TimeFrame.HOUR_1
    elif timeframe_str == '4h':
        return TimeFrame.HOUR_4
    elif timeframe_str == '1d':
        return TimeFrame.DAY_1
    elif timeframe_str == '1w':
        return TimeFrame.WEEK_1
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe_str}")


def get_strategy(args):
    """Create and configure a strategy based on arguments."""
    if args.strategy == 'ma_crossover':
        return MovingAverageCrossoverStrategy(
            fast_ma_period=args.fast_period,
            slow_ma_period=args.slow_period,
            fast_ma_type="EMA",
            slow_ma_type="EMA",
            min_confidence=0.4
        )
    elif args.strategy == 'enhanced_ma':
        return EnhancedMAStrategy(
            fast_ma_period=args.fast_period,
            slow_ma_period=args.slow_period,
            fast_ma_type="EMA",
            slow_ma_type="EMA",
            min_confidence=0.4,
            volatility_period=20,
            trend_strength_period=100,
            use_regime_filter=True
        )
    elif args.strategy == 'rsi':
        return RSIStrategy(
            period=args.rsi_period,
            overbought_threshold=args.overbought,
            oversold_threshold=args.oversold,
            min_confidence=0.5
        )
    elif args.strategy == 'macd':
        return MACDStrategy(
            fast_period=args.fast_period,
            slow_period=args.slow_period,
            signal_period=args.signal_period,
            min_confidence=0.5
        )
    elif args.strategy == 'multi':
        # Create a multi-strategy system with multiple sub-strategies
        rsi_strategy = RSIStrategy(
            period=args.rsi_period,
            overbought_threshold=args.overbought,
            oversold_threshold=args.oversold
        )
        
        macd_strategy = MACDStrategy(
            fast_period=args.fast_period,
            slow_period=args.slow_period,
            signal_period=args.signal_period
        )
        
        ma_strategy = EnhancedMAStrategy(
            fast_ma_period=args.fast_period,
            slow_ma_period=args.slow_period
        )
        
        # Create multi-strategy with weights
        return MultiStrategySystem([
            (rsi_strategy, 0.3),
            (macd_strategy, 0.3),
            (ma_strategy, 0.4)
        ], min_consensus=0.6)
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")


def get_historical_data(args) -> Dict[str, List[CandleData]]:
    """Get historical data based on arguments."""
    timeframe = string_to_timeframe(args.timeframe)
    symbol = args.symbol
    data = {}
    
    if args.data_file and os.path.exists(args.data_file):
        # Load data from file
        candles = load_csv_data(args.data_file, symbol, timeframe)
        data[symbol] = candles
    elif args.generate_data:
        # Generate synthetic data
        start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
        end_date = datetime.now()
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Generate data
        candles = generate_sample_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date,
            base_price=10000.0,
            volatility=0.015,
            trend_strength=0.0001,
            with_cycles=True
        )
        
        # Save generated data
        os.makedirs('data/historical', exist_ok=True)
        file_path = f'data/historical/{symbol}_{args.timeframe}.csv'
        save_to_csv(candles, file_path)
        logger.info(f"Sample data saved to {file_path}")
        
        data[symbol] = candles
    else:
        raise ValueError("No data source specified. Use --data-file or --generate-data.")
    
    return data


def setup_backtester(args, data: Dict[str, List[CandleData]], strategy) -> StrategyBacktester:
    """Set up and configure the backtester."""
    # Create backtester
    backtester = StrategyBacktester(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        use_stop_loss=args.use_stop_loss,
        stop_loss_pct=args.stop_loss_pct,
        use_take_profit=args.use_take_profit,
        take_profit_pct=args.take_profit_pct,
        enable_trailing_stop=args.use_trailing_stop,
        trailing_stop_pct=args.trailing_stop_pct,
        commission_pct=args.commission
    )
    
    # Set strategy
    backtester.set_strategy(strategy)
    
    # Add historical data
    for symbol, candles in data.items():
        backtester.add_historical_data(symbol, candles)
    
    return backtester


def run_optimization(backtester: StrategyBacktester, args) -> Dict[str, Any]:
    """Run parameter optimization."""
    logger.info("Starting parameter optimization...")
    
    # Define parameter ranges to optimize
    parameter_ranges = {}
    
    if args.strategy == 'ma_crossover' or args.strategy == 'enhanced_ma':
        parameter_ranges = {
            'fast_ma_period': [5, 8, 13, 21],
            'slow_ma_period': [21, 34, 55, 89],
            'min_confidence': [0.3, 0.4, 0.5, 0.6]
        }
    elif args.strategy == 'rsi':
        parameter_ranges = {
            'period': [7, 9, 14, 21],
            'overbought_threshold': [65, 70, 75, 80],
            'oversold_threshold': [20, 25, 30, 35],
            'min_confidence': [0.4, 0.5, 0.6]
        }
    elif args.strategy == 'macd':
        parameter_ranges = {
            'fast_period': [8, 12, 16],
            'slow_period': [21, 26, 34],
            'signal_period': [7, 9, 12],
            'min_confidence': [0.4, 0.5, 0.6]
        }
    
    # Run optimization
    results = backtester.optimize_parameters(
        parameter_ranges=parameter_ranges,
        optimization_metric=args.optimize_metric,
        max_iterations=args.max_iterations
    )
    
    logger.info("Optimization completed.")
    logger.info(f"Best parameters: {results['parameters']}")
    logger.info(f"Best {args.optimize_metric}: {results['metrics'].get(args.optimize_metric)}")
    
    return results


def run_backtest(args):
    """Run the backtest based on command line arguments."""
    # Get historical data
    data = get_historical_data(args)
    
    # Get strategy
    strategy = get_strategy(args)
    
    # Set up backtester
    backtester = setup_backtester(args, data, strategy)
    
    # Run optimization if requested
    if args.optimize:
        optimization_results = run_optimization(backtester, args)
        
        # Apply best parameters to strategy
        for param, value in optimization_results['parameters'].items():
            if hasattr(strategy, param):
                setattr(strategy, param, value)
        
        # Re-run backtest with optimized parameters
        logger.info("Running backtest with optimized parameters...")
    
    # Run backtest
    metrics = backtester.run_backtest()
    
    # Generate report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = backtester.generate_report(args.output_dir)
    
    # Save detailed results if requested
    if args.save_results:
        results_file = os.path.join(args.output_dir, 
                                    f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        backtester.save_results(results_file)
    
    # Print summary
    print("\n=== BACKTEST SUMMARY ===")
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    
    metrics_dict = metrics.to_dict()
    print(f"Total P&L: ${metrics_dict['total_pnl']:.2f}")
    print(f"Total Trades: {metrics_dict['total_trades']}")
    print(f"Win Rate: {metrics_dict['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics_dict['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics_dict['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics_dict['sharpe_ratio']:.2f}")
    
    print(f"\nDetailed report saved to: {report_path}")
    

def main():
    """Main entry point."""
    args = parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main() 