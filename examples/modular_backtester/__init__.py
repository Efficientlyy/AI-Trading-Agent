"""
Modular Backtesting Framework

A comprehensive framework for backtesting trading strategies with features like:
- Multiple strategy types (MA, RSI, MACD, multi-strategy)
- Risk management (stop loss, take profit, trailing stop)
- Position sizing based on volatility
- Market regime detection
- Parameter optimization
- Detailed performance metrics and reporting
"""

from .models import Signal, SignalType, CandleData, Position, TimeFrame, MarketRegime
from .strategies import (
    Strategy, 
    MovingAverageCrossoverStrategy, 
    EnhancedMAStrategy,
    RSIStrategy, 
    MACDStrategy, 
    MultiStrategySystem
)
from .backtester import BacktestMetrics, StrategyBacktester
from .data_utils import (
    generate_sample_data,
    load_csv_data,
    save_to_csv,
    resample_timeframe,
    add_noise_to_data,
    prepare_data_for_indicators,
    calculate_returns,
    add_gap_data
)

__version__ = '0.1.0'
__author__ = 'AI Trading Agent Team'

# Export all modules
__all__ = [
    # Models
    'Signal', 'SignalType', 'CandleData', 'Position', 'TimeFrame', 'MarketRegime',
    
    # Strategies
    'Strategy', 'MovingAverageCrossoverStrategy', 'EnhancedMAStrategy',
    'RSIStrategy', 'MACDStrategy', 'MultiStrategySystem',
    
    # Backtester
    'BacktestMetrics', 'StrategyBacktester',
    
    # Data Utils
    'generate_sample_data', 'load_csv_data', 'save_to_csv', 'resample_timeframe',
    'add_noise_to_data', 'prepare_data_for_indicators', 'calculate_returns', 'add_gap_data'
] 