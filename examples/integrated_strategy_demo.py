"""Integrated strategy demo showcasing risk management, performance tracking, and event detection.

This script demonstrates how the enhanced ML strategy works with risk management,
performance analytics, and market event detection.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import matplotlib.dates as mdates

from src.ml.models.enhanced_price_prediction import EnhancedPricePredictionStrategy
from src.risk.risk_manager import RiskManager, PositionRisk
from src.analytics.performance_tracker import PerformanceTracker, TradeStats
from src.events.market_events import EventDetector, EventType
from src.models.market_data import CandleData
from src.common.timeframe import TimeFrame

async def main():
    """Run the integrated strategy demo."""
    # Initialize components
    initial_capital = 100000.0
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        max_portfolio_risk=0.02,    # 2% max portfolio risk
        max_position_risk=0.01,     # 1% max position risk
        max_correlation_risk=0.7,   # 70% max correlation
        target_risk_reward=2.0      # 2:1 minimum RR ratio
    )
    
    performance_tracker = PerformanceTracker(initial_capital)
    event_detector = EventDetector(
        volatility_threshold=2.0,
        volume_threshold=3.0,
        trend_threshold=0.1,
        correlation_threshold=0.3,
        lookback_window=100
    )
    
    # Initialize strategy with required parameters
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    strategy = EnhancedPricePredictionStrategy(
        strategy_id="ML_STRATEGY_001",
        trading_symbols=symbols,
        exchange="binance",
        timeframe=TimeFrame.HOUR_1
    )
    await strategy.start()
    
    # Generate sample market data
    print("Generating sample market data...")
    num_days = 30
    periods_per_day = 24  # Hourly data
    total_periods = num_days * periods_per_day
    
    # Initialize price series with realistic crypto behavior
    base_prices = {
        "BTC/USD": 50000.0,
        "ETH/USD": 3000.0,
        "SOL/USD": 100.0
    }
    
    price_data: Dict[str, List[CandleData]] = {symbol: [] for symbol in symbols}
    correlations: Dict[str, Dict[str, float]] = {}
    
    # Generate correlated price movements
    correlation_matrix = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    # Generate correlated returns
    np.random.seed(42)
    returns = np.random.multivariate_normal(
        mean=[0.0001, 0.0002, 0.0003],
        cov=correlation_matrix * 0.0002,
        size=total_periods
    )
    
    # Convert returns to prices and generate candles
    for t in range(total_periods):
        timestamp = datetime.now() - timedelta(days=num_days) + timedelta(hours=t)
        
        # Update correlations periodically
        if t % 24 == 0:
            correlations = {
                symbol: {
                    other: float(np.random.normal(
                        correlation_matrix[i][j],
                        0.1
                    ))
                    for j, other in enumerate(symbols)
                    if other != symbol
                }
                for i, symbol in enumerate(symbols)
            }
        
        for i, symbol in enumerate(symbols):
            # Calculate price with momentum and mean reversion
            if not price_data[symbol]:
                price = base_prices[symbol]
            else:
                price = price_data[symbol][-1].close * (1 + returns[t, i])
            
            # Add some volatility clusters
            if t % 100 < 20:  # Higher volatility periods
                price *= 1 + np.random.normal(0, 0.02)
            
            # Generate realistic candle data
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price_data[symbol][-1].close if price_data[symbol] else price
            volume = base_prices[symbol] * np.random.lognormal(0, 1) * 10
            
            candle = CandleData(
                symbol=symbol,
                timestamp=timestamp,
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(price),
                volume=float(volume)
            )
            price_data[symbol].append(candle)
            
            # Process market events
            events = event_detector.add_data_point(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=volume,
                correlations=correlations.get(symbol)
            )
            
            # Log significant events
            for event in events:
                if event.significance > 0.7:  # Only log high significance events
                    print(f"\nHigh significance event detected: {event.description}")
                    print(f"Symbol: {event.symbol}, Time: {event.timestamp}")
                    print(f"Significance: {event.significance:.2%}")
    
    print("\nTraining strategy...")
    # Train strategy on historical data
    for symbol in symbols:
        candles = price_data[symbol][:-24]  # Leave last day for testing
        for candle in candles:
            strategy.on_candle(candle)
    
    print("\nRunning strategy with risk management...")
    # Test strategy with risk management
    test_candles = [
        price_data[symbol][-24:] for symbol in symbols
    ]
    
    active_positions: Dict[str, PositionRisk] = {}
    
    for i in range(len(test_candles[0])):
        timestamp = test_candles[0][i].timestamp
        
        # Update positions and check for exits
        for symbol in list(active_positions.keys()):
            candle = next(
                c[i] for c in test_candles
                if c[i].symbol == symbol
            )
            
            # Update position risk metrics
            risk_manager.update_position_risk(
                symbol=symbol,
                current_price=candle.close,
                timestamp=timestamp
            )
            
            # Check if position should be closed
            if risk_manager.should_close_position(symbol, candle.close):
                pos = active_positions[symbol]
                
                # Record trade
                trade_stats = TradeStats(
                    symbol=symbol,
                    entry_time=datetime.fromtimestamp(pos.time_in_trade),
                    exit_time=timestamp,
                    entry_price=pos.entry_price,
                    exit_price=candle.close,
                    position_size=pos.position_size,
                    pnl=pos.unrealized_pnl,
                    return_pct=(candle.close / pos.entry_price - 1) * 100,
                    holding_period=timestamp - datetime.fromtimestamp(pos.time_in_trade),
                    max_drawdown=pos.unrealized_pnl / (pos.position_size * pos.entry_price),
                    max_runup=max(0, pos.unrealized_pnl / (pos.position_size * pos.entry_price)),
                    risk_reward_ratio=pos.risk_reward_ratio,
                    strategy_name=strategy.strategy_id
                )
                performance_tracker.add_trade(trade_stats)
                
                print(f"\nClosed position: {symbol}")
                print(f"PnL: ${pos.unrealized_pnl:.2f}")
                print(f"Return: {trade_stats.return_pct:.2%}")
                
                del active_positions[symbol]
        
        # Process new candles
        for candle in (c[i] for c in test_candles):
            # Get strategy prediction
            strategy.on_candle(candle)
            prediction = strategy.predict(candle.symbol)
            
            if prediction and prediction.confidence > 0.6:
                if candle.symbol not in active_positions:
                    # Calculate position size
                    recent_prices = [
                        c.close for c in price_data[candle.symbol][-20:]
                    ]
                    volatility = float(
                        np.std(recent_prices) / np.mean(recent_prices)
                    )
                    
                    position_size = risk_manager.calculate_position_size(
                        symbol=candle.symbol,
                        entry_price=candle.close,
                        stop_loss=candle.close * (1 - 0.02),  # 2% stop loss
                        take_profit=candle.close * (1 + 0.04),  # 4% take profit
                        volatility=volatility,
                        correlation_score=max(
                            correlations.get(candle.symbol, {}).values(),
                            default=0
                        ),
                        confidence_score=prediction.confidence
                    )
                    
                    # Check if we can open position
                    if risk_manager.can_open_position(
                        candle.symbol,
                        position_size,
                        candle.close
                    ):
                        # Open new position
                        active_positions[candle.symbol] = PositionRisk(
                            symbol=candle.symbol,
                            position_size=position_size,
                            entry_price=candle.close,
                            current_price=candle.close,
                            stop_loss=candle.close * (1 - 0.02),
                            take_profit=candle.close * (1 + 0.04),
                            unrealized_pnl=0.0,
                            risk_reward_ratio=2.0,
                            time_in_trade=timestamp.timestamp()
                        )
                        
                        print(f"\nOpened position: {candle.symbol}")
                        print(f"Size: {position_size:.4f}")
                        print(f"Entry: ${candle.close:.2f}")
    
    # Print final performance metrics
    print("\nFinal Performance Metrics:")
    metrics = performance_tracker.get_metrics()
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Total Return: {metrics.return_pct:.2%}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Equity Curve
    plt.subplot(311)
    equity_curve = performance_tracker.get_equity_curve()
    plt.plot(equity_curve.index, equity_curve.equity)
    plt.title("Equity Curve")
    plt.grid(True)
    
    # Plot 2: Drawdown
    plt.subplot(312)
    drawdown_curve = performance_tracker.get_drawdown_curve()
    plt.fill_between(drawdown_curve.index, 0, drawdown_curve.drawdown * 100)
    plt.title("Drawdown (%)")
    plt.grid(True)
    
    # Plot 3: Event Timeline
    plt.subplot(313)
    events = event_detector.get_recent_events(limit=1000)
    event_times = [mdates.date2num(e.timestamp) for e in events]
    event_significance = [e.significance for e in events]
    plt.scatter(event_times, event_significance, alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.title("Market Events (Significance)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main()) 