"""
Enhanced Sentiment Strategy Demo

This example demonstrates the enhanced sentiment strategy in action,
showing how it combines sentiment analysis with technical indicators
and market regime detection to generate more accurate trading signals.
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

from src.common.config import config
from src.common.events import event_bus
from src.common.datetime_utils import utc_now
from src.models.events import SentimentEvent, MarketRegimeEvent, TechnicalIndicatorEvent
from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy


# Simulate market regimes
class MarketRegimeSimulator:
    """Simulates market regime detection for testing."""
    
    def __init__(self):
        """Initialize the market regime simulator."""
        self.logger = logging.getLogger("regime_simulator")
        self.regimes = {}  # symbol -> regime
    
    async def start_simulation(self, symbols=None, update_interval=3600):
        """Start simulating market regimes.
        
        Args:
            symbols: Symbols to simulate regimes for (default: BTC/USDT, ETH/USDT)
            update_interval: How often to update regimes in seconds
        """
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT"]
            
        self.logger.info("Starting market regime simulation")
        
        # Initialize with random regimes
        for symbol in symbols:
            self.regimes[symbol] = random.choice(["bullish", "bearish", "neutral", "volatile"])
        
        # Publish initial regimes
        for symbol, regime in self.regimes.items():
            await self._publish_regime_event(symbol, regime)
        
        # Start update task
        while True:
            # Wait for next update
            await asyncio.sleep(update_interval)
            
            # Update regimes with some probability
            for symbol in symbols:
                # 30% chance of regime change
                if random.random() < 0.3:
                    new_regime = random.choice(["bullish", "bearish", "neutral", "volatile"])
                    if new_regime != self.regimes[symbol]:
                        self.regimes[symbol] = new_regime
                        await self._publish_regime_event(symbol, new_regime)
    
    async def _publish_regime_event(self, symbol, regime):
        """Publish a market regime event.
        
        Args:
            symbol: The trading pair symbol
            regime: The market regime
        """
        event = MarketRegimeEvent(
            source="regime_detector",
            payload={
                "symbol": symbol,
                "regime": regime,
                "confidence": random.uniform(0.7, 0.95),
                "timestamp": utc_now().isoformat()
            }
        )
        
        await event_bus.publish("market_regime_event", event)
        self.logger.info(f"Published {regime} regime for {symbol}")


# Simulate sentiment events
class SentimentSimulator:
    """Simulates sentiment events for testing."""
    
    def __init__(self):
        """Initialize the sentiment simulator."""
        self.logger = logging.getLogger("sentiment_simulator")
        self.base_sentiment = {}  # symbol -> base sentiment
    
    async def start_simulation(self, symbols=None, update_interval=900):
        """Start simulating sentiment events.
        
        Args:
            symbols: Symbols to simulate sentiment for
            update_interval: How often to update sentiment in seconds
        """
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
            
        self.logger.info("Starting sentiment event simulation")
        
        # Initialize with random sentiment
        for symbol in symbols:
            self.base_sentiment[symbol] = random.uniform(0.3, 0.7)
        
        # Simulate various sources
        sources = ["social_media", "news", "market", "onchain", "aggregator"]
        
        # Start update loop
        while True:
            # For each symbol, publish sentiment from different sources
            for symbol in symbols:
                # Base sentiment shifts gradually
                shift = random.uniform(-0.05, 0.05)
                self.base_sentiment[symbol] += shift
                self.base_sentiment[symbol] = max(0.1, min(0.9, self.base_sentiment[symbol]))
                
                # Publish sentiment from different sources with some variation
                for source in sources:
                    # Add some random variation to the base sentiment
                    variation = random.uniform(-0.1, 0.1)
                    sentiment_value = max(0.1, min(0.9, self.base_sentiment[symbol] + variation))
                    
                    # Determine direction
                    if sentiment_value > 0.55:
                        direction = "bullish"
                    elif sentiment_value < 0.45:
                        direction = "bearish"
                    else:
                        direction = "neutral"
                    
                    # Random confidence level
                    confidence = random.uniform(0.6, 0.9)
                    
                    # Sometimes skip to simulate irregular updates
                    if random.random() < 0.2:
                        continue
                    
                    # Create and publish event
                    await self._publish_sentiment_event(
                        symbol=symbol,
                        source=f"{source}_sentiment",
                        value=sentiment_value,
                        direction=direction,
                        confidence=confidence
                    )
            
            # Wait for next update
            await asyncio.sleep(update_interval)
    
    async def _publish_sentiment_event(self, symbol, source, value, direction, confidence):
        """Publish a sentiment event.
        
        Args:
            symbol: The trading pair symbol
            source: The sentiment source
            value: The sentiment value
            direction: The sentiment direction
            confidence: The confidence level
        """
        is_extreme = value > 0.8 or value < 0.2
        
        event = SentimentEvent(
            source=source,
            payload={
                "symbol": symbol,
                "sentiment_value": value,
                "sentiment_direction": direction,
                "confidence": confidence,
                "is_extreme": is_extreme,
                "timestamp": utc_now().isoformat(),
                "tags": ["extreme"] if is_extreme else []
            }
        )
        
        await event_bus.publish("sentiment_event", event)
        
        if is_extreme:
            self.logger.info(f"Published extreme {direction} sentiment ({value:.2f}) for {symbol} from {source}")


# Generate synthetic market data
class MarketDataSimulator:
    """Simulates market data for testing."""
    
    def __init__(self):
        """Initialize the market data simulator."""
        self.logger = logging.getLogger("market_data_simulator")
        self.prices = {}  # symbol -> price
    
    async def start_simulation(self, symbols=None, timeframes=None, update_interval=60):
        """Start simulating market data.
        
        Args:
            symbols: Symbols to simulate data for
            timeframes: Timeframes to generate candles for
            update_interval: How often to update data in seconds
        """
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
            
        if timeframes is None:
            timeframes = [TimeFrame("1m"), TimeFrame("5m"), TimeFrame("1h")]
            
        self.logger.info("Starting market data simulation")
        
        # Initialize prices
        self.prices = {
            "BTC/USDT": 30000.0,
            "ETH/USDT": 1800.0,
            "SOL/USDT": 100.0,
            "XRP/USDT": 0.5
        }
        
        # Initialize price history (for RSI calculation)
        self.price_history = {symbol: [] for symbol in symbols}
        
        # Start update loop
        last_timeframe_update = {tf: utc_now() for tf in timeframes}
        
        while True:
            now = utc_now()
            
            # Update prices
            for symbol in symbols:
                # Random price change
                change_pct = random.uniform(-0.005, 0.005)  # -0.5% to +0.5%
                self.prices[symbol] *= (1 + change_pct)
                
                # Add to price history
                self.price_history[symbol].append(self.prices[symbol])
                
                # Keep only last 100 prices
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
                
                # Generate RSI if we have enough history
                if len(self.price_history[symbol]) > 14:
                    rsi = self._calculate_rsi(self.price_history[symbol], 14)
                    await self._publish_technical_event(symbol, "RSI", rsi)
            
            # Generate candles for each timeframe
            for timeframe in timeframes:
                # Check if it's time to update this timeframe
                seconds_per_candle = timeframe.to_seconds()
                if (now - last_timeframe_update[timeframe]).total_seconds() >= seconds_per_candle:
                    # Update timestamp
                    last_timeframe_update[timeframe] = now
                    
                    # Generate candles for each symbol
                    for symbol in symbols:
                        # Create candle
                        price = self.prices[symbol]
                        open_price = price * (1 - random.uniform(0, 0.005))
                        high_price = price * (1 + random.uniform(0, 0.01))
                        low_price = price * (1 - random.uniform(0, 0.01))
                        volume = random.uniform(10, 100) * price
                        
                        candle = CandleData(
                            symbol=symbol,
                            exchange="simulation",
                            timeframe=timeframe,
                            timestamp=now,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=price,
                            volume=volume
                        )
                        
                        # Publish candle event
                        await event_bus.publish("candle_data_event", candle)
            
            # Wait for next update
            await asyncio.sleep(update_interval)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI from price history.
        
        Args:
            prices: List of prices
            period: RSI period
            
        Returns:
            RSI value
        """
        if len(prices) <= period:
            return 50  # Default if not enough data
            
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Get last 'period' deltas
        deltas = deltas[-period:]
        
        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [abs(d) if d < 0 else 0 for d in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100  # No losses, RSI is 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _publish_technical_event(self, symbol, indicator, value):
        """Publish a technical indicator event.
        
        Args:
            symbol: The trading pair symbol
            indicator: The indicator name
            value: The indicator value
        """
        event = TechnicalIndicatorEvent(
            source="technical_analyzer",
            payload={
                "symbol": symbol,
                "indicator": indicator,
                "value": value,
                "timestamp": utc_now().isoformat()
            }
        )
        
        await event_bus.publish("technical_indicator_event", event)


# Signal handler
async def signal_handler(signal):
    """Handle signals from the strategy.
    
    Args:
        signal: The signal event
    """
    logger = logging.getLogger("signal_handler")
    logger.info(f"Received signal: {signal.signal_type.name} {signal.direction} for {signal.symbol}")
    logger.info(f"  Price: {signal.price:.2f}")
    logger.info(f"  Confidence: {signal.confidence:.2f}")
    logger.info(f"  Reason: {signal.reason}")
    
    if signal.metadata:
        logger.info("  Metadata:")
        for key, value in signal.metadata.items():
            logger.info(f"    {key}: {value}")


async def run_enhanced_sentiment_demo():
    """Run the enhanced sentiment strategy demo."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("enhanced_sentiment_demo")
    logger.info("Starting enhanced sentiment strategy demo")
    
    # Create strategy
    strategy = EnhancedSentimentStrategy()
    
    # Initialize and start strategy
    await strategy.initialize()
    await strategy.start()
    
    # Subscribe to signals
    event_bus.subscribe("signal", signal_handler)
    
    # Create simulators
    market_data_sim = MarketDataSimulator()
    sentiment_sim = SentimentSimulator()
    regime_sim = MarketRegimeSimulator()
    
    # Start simulators in separate tasks
    market_data_task = asyncio.create_task(market_data_sim.start_simulation())
    sentiment_task = asyncio.create_task(sentiment_sim.start_simulation())
    regime_task = asyncio.create_task(regime_sim.start_simulation())
    
    try:
        # Run for specified time
        runtime = 600  # 10 minutes
        logger.info(f"Demo will run for {runtime} seconds")
        logger.info("Monitor the logs to see signals being generated")
        
        await asyncio.sleep(runtime)
        
        logger.info("Enhanced sentiment strategy demo completed")
    finally:
        # Cancel simulation tasks
        market_data_task.cancel()
        sentiment_task.cancel()
        regime_task.cancel()
        
        # Stop strategy
        await strategy.stop()
        
        try:
            await market_data_task
        except asyncio.CancelledError:
            pass
            
        try:
            await sentiment_task
        except asyncio.CancelledError:
            pass
            
        try:
            await regime_task
        except asyncio.CancelledError:
            pass


async def generate_and_visualize_strategy_performance():
    """Generate and visualize sample strategy performance data."""
    # Create price data
    dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
    
    # Create price series with some trends and volatility
    prices = [30000]
    for i in range(1, 90):
        if i < 30:
            trend = 0.003  # Uptrend
        elif i < 60:
            trend = -0.004  # Downtrend
        else:
            trend = 0.002  # Uptrend again
            
        # Add some random noise
        noise = random.uniform(-0.015, 0.015)
        price_change = trend + noise
        prices.append(prices[-1] * (1 + price_change))
    
    # Create sentiment data with some relationship to price
    # Base sentiment follows price changes with some lag and noise
    base_sentiment = [0.5]
    for i in range(1, 90):
        # Sentiment partly follows price changes with some lag
        if i > 5:
            price_change = (prices[i-5] - prices[i-6]) / prices[i-6]
            sentiment_change = price_change * 2 + random.uniform(-0.05, 0.05)
        else:
            sentiment_change = random.uniform(-0.05, 0.05)
            
        new_sentiment = base_sentiment[-1] + sentiment_change
        # Keep sentiment between 0 and 1
        new_sentiment = max(0.1, min(0.9, new_sentiment))
        base_sentiment.append(new_sentiment)
    
    # Create signals based on sentiment thresholds
    basic_signals = []
    for i, sentiment in enumerate(base_sentiment):
        if sentiment > 0.7:
            basic_signals.append(('buy', dates[i], prices[i]))
        elif sentiment < 0.3:
            basic_signals.append(('sell', dates[i], prices[i]))
    
    # Create enhanced signals with more advanced rules
    # Add some market regime data
    regimes = []
    current_regime = "neutral"
    for i in range(90):
        # Regime changes every ~20-30 days
        if i % 25 == 0:
            current_regime = random.choice(["bullish", "bearish", "neutral"])
        regimes.append(current_regime)
    
    # Add technical indicators (like RSI)
    rsi_values = []
    for i in range(90):
        if i < 14:
            rsi_values.append(50)  # Default for not enough data
        else:
            # Calculate price changes for last 14 days
            price_changes = [(prices[j] - prices[j-1]) for j in range(i-13, i+1)]
            
            # Separate gains and losses
            gains = [change if change > 0 else 0 for change in price_changes]
            losses = [abs(change) if change < 0 else 0 for change in price_changes]
            
            # Calculate RSI
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
    
    # Generate enhanced signals
    enhanced_signals = []
    for i, sentiment in enumerate(base_sentiment):
        # Skip first 14 days for RSI calculation
        if i < 14:
            continue
            
        # Basic sentiment check
        if sentiment > 0.7 or sentiment < 0.3:
            signal_direction = 'buy' if sentiment > 0.7 else 'sell'
            
            # Check regime alignment
            regime_aligned = (signal_direction == 'buy' and regimes[i] != "bearish") or \
                            (signal_direction == 'sell' and regimes[i] != "bullish")
            
            # Check technical confirmation
            rsi = rsi_values[i]
            
            # For buy signals, RSI should not be overbought (>70)
            # For sell signals, RSI should not be oversold (<30)
            technical_aligned = (signal_direction == 'buy' and rsi < 70) or \
                               (signal_direction == 'sell' and rsi > 30)
            
            # Only generate signal if both conditions are met
            if regime_aligned and technical_aligned:
                enhanced_signals.append((signal_direction, dates[i], prices[i]))
    
    # Calculate hypothetical performance
    # Simple approach: calculate returns assuming equal position sizes
    basic_returns = [1.0]  # Start with 1.0 (no gain/loss)
    enhanced_returns = [1.0]
    
    basic_position = None
    enhanced_position = None
    
    for i in range(1, 90):
        # Check for basic strategy signals
        basic_signal = next((s for s in basic_signals if s[1] == dates[i]), None)
        if basic_signal:
            signal_type, _, price = basic_signal
            
            # Close existing position
            if basic_position:
                if basic_position[0] == 'buy':
                    # Calculate return for long position
                    pct_change = price / basic_position[1] - 1
                    basic_returns.append(basic_returns[-1] * (1 + pct_change))
                else:
                    # Calculate return for short position
                    pct_change = 1 - price / basic_position[1]
                    basic_returns.append(basic_returns[-1] * (1 + pct_change))
            
            # Open new position
            basic_position = (signal_type, price)
        else:
            # No signal, just copy previous return
            basic_returns.append(basic_returns[-1])
        
        # Check for enhanced strategy signals
        enhanced_signal = next((s for s in enhanced_signals if s[1] == dates[i]), None)
        if enhanced_signal:
            signal_type, _, price = enhanced_signal
            
            # Close existing position
            if enhanced_position:
                if enhanced_position[0] == 'buy':
                    # Calculate return for long position
                    pct_change = price / enhanced_position[1] - 1
                    enhanced_returns.append(enhanced_returns[-1] * (1 + pct_change))
                else:
                    # Calculate return for short position
                    pct_change = 1 - price / enhanced_position[1]
                    enhanced_returns.append(enhanced_returns[-1] * (1 + pct_change))
            
            # Open new position
            enhanced_position = (signal_type, price)
        else:
            # No signal, just copy previous return
            enhanced_returns.append(enhanced_returns[-1])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Sentiment': base_sentiment,
        'RSI': rsi_values,
        'Regime': regimes,
        'Basic_Returns': basic_returns,
        'Enhanced_Returns': enhanced_returns
    })
    
    # Plot the data
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Price and signals
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['Date'], df['Price'], 'b-', label='Price')
    
    # Plot basic signals
    for signal_type, date, price in basic_signals:
        if signal_type == 'buy':
            ax1.scatter(date, price, marker='^', color='green', s=100)
        else:
            ax1.scatter(date, price, marker='v', color='red', s=100)
    
    # Plot enhanced signals
    for signal_type, date, price in enhanced_signals:
        if signal_type == 'buy':
            ax1.scatter(date, price, marker='^', color='green', s=100, edgecolors='black')
        else:
            ax1.scatter(date, price, marker='v', color='red', s=100, edgecolors='black')
    
    ax1.set_title('BTC/USDT Price with Trading Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Sentiment and RSI
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df['Date'], df['Sentiment'], 'g-', label='Sentiment')
    ax2.plot(df['Date'], [0.7] * len(df), 'g--', alpha=0.5)
    ax2.plot(df['Date'], [0.3] * len(df), 'r--', alpha=0.5)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Date'], df['RSI'], 'r-', label='RSI')
    ax2_twin.plot(df['Date'], [70] * len(df), 'r--', alpha=0.5)
    ax2_twin.plot(df['Date'], [30] * len(df), 'g--', alpha=0.5)
    
    # Add regime as background color
    for i, date in enumerate(df['Date']):
        if i < len(df)-1:
            if df['Regime'][i] == 'bullish':
                ax2.axvspan(date, df['Date'][i+1], alpha=0.2, color='green')
            elif df['Regime'][i] == 'bearish':
                ax2.axvspan(date, df['Date'][i+1], alpha=0.2, color='red')
    
    ax2.set_title('Sentiment, RSI, and Market Regime')
    ax2.set_ylabel('Sentiment (0-1)')
    ax2_twin.set_ylabel('RSI (0-100)')
    ax2.grid(True, alpha=0.3)
    
    # Create combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 3: Strategy Returns
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df['Date'], df['Basic_Returns'], 'b-', label='Basic Sentiment Strategy')
    ax3.plot(df['Date'], df['Enhanced_Returns'], 'g-', label='Enhanced Sentiment Strategy')
    
    ax3.set_title('Strategy Performance Comparison')
    ax3.set_ylabel('Returns (starting at 1.0)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'examples/output/sentiment'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/strategy_performance.png')
    
    logging.getLogger("enhanced_sentiment_demo").info(f"Strategy performance visualization saved to {output_dir}/strategy_performance.png")
    
    # Calculate and return performance metrics
    basic_final_return = (df['Basic_Returns'].iloc[-1] / df['Basic_Returns'].iloc[0]) - 1
    enhanced_final_return = (df['Enhanced_Returns'].iloc[-1] / df['Enhanced_Returns'].iloc[0]) - 1
    
    # Calculate drawdowns
    basic_peak = df['Basic_Returns'].cummax()
    basic_drawdown = (df['Basic_Returns'] / basic_peak) - 1
    basic_max_drawdown = basic_drawdown.min()
    
    enhanced_peak = df['Enhanced_Returns'].cummax()
    enhanced_drawdown = (df['Enhanced_Returns'] / enhanced_peak) - 1
    enhanced_max_drawdown = enhanced_drawdown.min()
    
    return {
        'basic_signals': len(basic_signals),
        'enhanced_signals': len(enhanced_signals),
        'basic_return': basic_final_return * 100,
        'enhanced_return': enhanced_final_return * 100,
        'basic_max_drawdown': basic_max_drawdown * 100,
        'enhanced_max_drawdown': enhanced_max_drawdown * 100
    }


if __name__ == "__main__":
    # First visualize strategy performance
    metrics = asyncio.run(generate_and_visualize_strategy_performance())
    
    print("\nStrategy Performance Metrics:")
    print(f"Basic Sentiment Strategy:")
    print(f"  - Signals generated: {metrics['basic_signals']}")
    print(f"  - Total return: {metrics['basic_return']:.2f}%")
    print(f"  - Maximum drawdown: {metrics['basic_max_drawdown']:.2f}%")
    
    print(f"Enhanced Sentiment Strategy:")
    print(f"  - Signals generated: {metrics['enhanced_signals']}")
    print(f"  - Total return: {metrics['enhanced_return']:.2f}%")
    print(f"  - Maximum drawdown: {metrics['enhanced_max_drawdown']:.2f}%")
    
    # Then run the live demo
    print("\nStarting live strategy demo...")
    asyncio.run(run_enhanced_sentiment_demo())