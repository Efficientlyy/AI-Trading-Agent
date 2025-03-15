"""
AI Trading Agent Demo

This example demonstrates the complete AI Trading Agent system with the
multi-agent architecture and decision engine integration.

The demo shows how:
1. Technical indicator signals are generated
2. Pattern recognition identifies chart patterns 
3. Sentiment analysis evaluates market sentiment
4. Decision engine aggregates these predictions
5. High-confidence trading signals are generated

This produces a comprehensive trading strategy with 75%+ win rate targeting.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np

# Add project root to path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import configure_logging
from src.models.market_data import CandleData, TimeFrame
from src.models.analysis_events import TechnicalIndicatorEvent, PatternEvent, CandleDataEvent
from src.models.events import SentimentEvent
from src.analysis_agents.technical_indicators import TechnicalIndicatorAgent
from src.analysis_agents.pattern_recognition import PatternRecognitionAgent
from src.analysis_agents.sentiment_analysis import SentimentAnalysisAgent
from src.decision_engine.engine import DecisionEngine
from src.decision_engine.models import Direction


# Configure logging
configure_logging()
logger = logging.getLogger("ai_trading_agent_demo")


class SignalCollector:
    """Collects signals from the decision engine for demo purposes."""
    
    def __init__(self):
        """Initialize signal collector."""
        self.signals = []
        
    async def handle_signal_event(self, event):
        """Handle signal events from the decision engine."""
        logger.info(f"🎯 TRADING SIGNAL: {event.payload.get('symbol')} {event.payload.get('signal_type')} {event.payload.get('direction')}")
        logger.info(f"   Price: {event.payload.get('price')}, Confidence: {event.payload.get('confidence'):.2f}")
        logger.info(f"   Stop Loss: {event.payload.get('stop_loss')}, Take Profit: {event.payload.get('take_profit')}")
        logger.info(f"   Reason: {event.payload.get('reason')}")
        logger.info("-" * 80)
        
        self.signals.append({
            "timestamp": event.timestamp,
            "symbol": event.payload.get("symbol"),
            "signal_type": event.payload.get("signal_type"),
            "direction": event.payload.get("direction"),
            "price": event.payload.get("price"),
            "confidence": event.payload.get("confidence"),
            "stop_loss": event.payload.get("stop_loss"),
            "take_profit": event.payload.get("take_profit"),
            "reason": event.payload.get("reason")
        })


async def load_sample_data(symbol="BTC/USDT"):
    """Load sample candle data for demonstration purposes."""
    try:
        # Try to load actual data from historical directory
        df = pd.read_csv(f"data/historical/{symbol.replace('/', '-')}_1h.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info(f"Loaded {len(df)} historical candles for {symbol}")
    except:
        # Generate synthetic data if file doesn't exist
        logger.info(f"Generating synthetic data for {symbol}")
        
        # Start date for synthetic data
        start_date = datetime.utcnow() - timedelta(days=30)
        
        # Generate timestamps for hourly data over 30 days
        timestamps = [start_date + timedelta(hours=i) for i in range(30*24)]
        
        # Generate OHLCV data with realistic patterns
        base_price = 50000 if symbol == "BTC/USDT" else 3000  # Base price for BTC or ETH
        price_volatility = 0.01  # 1% hourly volatility
        
        # Generate with some patterns built in
        prices = []
        price = base_price
        
        # Add a trend component
        trend = np.linspace(0, 0.15, len(timestamps))  # 15% trend over the period
        
        # Add cyclical component
        cycles = 0.08 * np.sin(np.linspace(0, 6*np.pi, len(timestamps)))  # 8% cycle 
        
        # Generate price series
        for i in range(len(timestamps)):
            random_walk = price_volatility * np.random.randn()
            daily_change = trend[i] + cycles[i] + random_walk
            price = price * (1 + daily_change)
            
            # Generate candle
            daily_volatility = price * 0.005  # 0.5% typical range
            open_price = price * (1 + 0.001 * np.random.randn())
            high_price = max(price, open_price) + daily_volatility * abs(np.random.randn())
            low_price = min(price, open_price) - daily_volatility * abs(np.random.randn())
            close_price = price
            volume = abs(np.random.randn()) * price * 10 + price * 5  # Volume correlated with volatility
            
            prices.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
        
        # Create DataFrame
        df = pd.DataFrame(prices, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
    # Return as list of CandleData objects
    candles = []
    for _, row in df.iterrows():
        candle = CandleData(
            timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else pd.to_datetime(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            symbol=symbol,
            exchange="demo",
            timeframe=TimeFrame.HOUR_1
        )
        candles.append(candle)
    
    return candles


async def publish_candle_events(candles, delay=0.1):
    """Publish candle data events with a delay to simulate real-time data."""
    for candle in candles:
        # Create candle event
        event = CandleDataEvent(
            source="data_feed",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            candle_timestamp=candle.timestamp
        )
        
        # Publish event
        await event_bus.publish(event)
        
        # Simulate realtime by waiting
        await asyncio.sleep(delay)


async def publish_sentiment_events(symbols, count=10, delay=1.0):
    """Publish synthetic sentiment events."""
    for i in range(count):
        for symbol in symbols:
            # Randomly generate sentiment
            sentiment_value = random.uniform(0.3, 0.7)  # Somewhat neutral range
            if random.random() < 0.1:  # 10% chance of extreme sentiment
                sentiment_value = random.uniform(0.8, 0.95) if random.random() < 0.5 else random.uniform(0.05, 0.2)
                
            direction = "bullish" if sentiment_value > 0.5 else "bearish"
            confidence = 0.6 + (abs(sentiment_value - 0.5) * 0.8)  # Higher confidence for extreme values
            
            # Determine if this is extreme sentiment (for contrarian signals)
            is_extreme = sentiment_value >= 0.8 or sentiment_value <= 0.2
            signal_type = "contrarian" if is_extreme else "sentiment_shift"
            
            # Create sentiment payload
            details = {
                "symbol": symbol,
                "sentiment_value": sentiment_value,
                "sentiment_direction": direction,
                "confidence": confidence,
                "is_extreme": is_extreme,
                "signal_type": signal_type
            }
            
            # Create and publish sentiment event
            event = SentimentEvent(
                source="sentiment_feed",
                symbol=symbol,
                sentiment_value=sentiment_value,
                sentiment_direction=direction,
                confidence=confidence,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            await event_bus.publish(event)
        
        # Wait between batches of events
        await asyncio.sleep(delay)


async def publish_pattern_events(symbols, count=5, delay=2.0):
    """Publish synthetic pattern events."""
    pattern_templates = [
        {
            "name": "HeadAndShoulders",
            "confidence": 0.85,
            "timeframe": TimeFrame.HOUR_4,
            "direction": Direction.BEARISH
        },
        {
            "name": "InverseHeadAndShoulders",
            "confidence": 0.88,
            "timeframe": TimeFrame.HOUR_4,
            "direction": Direction.BULLISH
        },
        {
            "name": "DoubleBottom",
            "confidence": 0.82,
            "timeframe": TimeFrame.HOUR_1,
            "direction": Direction.BULLISH
        },
        {
            "name": "BullishEngulfing",
            "confidence": 0.76,
            "timeframe": TimeFrame.HOUR_1,
            "direction": Direction.BULLISH
        },
        {
            "name": "BearishEngulfing",
            "confidence": 0.74,
            "timeframe": TimeFrame.HOUR_1, 
            "direction": Direction.BEARISH
        }
    ]
    
    for i in range(count):
        for symbol in symbols:
            # Get a random recent price for the symbol
            base_price = 50000 if symbol == "BTC/USDT" else 3000
            current_price = base_price * (1 + 0.1 * np.random.randn())
            
            # Select a random pattern template
            pattern = random.choice(pattern_templates)
            
            # Calculate target and invalidation based on pattern direction
            pattern_size = current_price * 0.03  # 3% pattern size
            
            if pattern["direction"] == Direction.BULLISH:
                target_price = current_price * (1 + random.uniform(0.02, 0.05))
                invalidation_price = current_price * (1 - random.uniform(0.01, 0.02))
            else:
                target_price = current_price * (1 - random.uniform(0.02, 0.05))
                invalidation_price = current_price * (1 + random.uniform(0.01, 0.02))
            
            # Create and publish pattern event
            event = PatternEvent(
                source="pattern_recognition",
                symbol=symbol,
                pattern_name=pattern["name"],
                timeframe=pattern["timeframe"],
                confidence=pattern["confidence"] * random.uniform(0.95, 1.05),  # Some variation
                target_price=target_price,
                invalidation_price=invalidation_price
            )
            
            await event_bus.publish(event)
        
        # Wait between batches of events
        await asyncio.sleep(delay)


async def publish_indicator_events(symbols, count=10, delay=1.0):
    """Publish synthetic technical indicator events."""
    for i in range(count):
        for symbol in symbols:
            # Randomly choose indicator type
            indicator_type = random.choice(["RSI", "MACD"])
            timeframe = random.choice([TimeFrame.HOUR_1, TimeFrame.HOUR_4])
            
            # Generate indicator-specific values
            values = {}
            now = datetime.utcnow()
            
            if indicator_type == "RSI":
                # Sometimes generate oversold or overbought signals
                if random.random() < 0.2:  # 20% chance of significant RSI
                    rsi_value = random.uniform(10, 30) if random.random() < 0.5 else random.uniform(70, 90)
                else:
                    rsi_value = random.uniform(40, 60)  # Neutral territory
                    
                values[now] = rsi_value
                
            elif indicator_type == "MACD":
                # Determine if this is a crossover event with 25% probability
                is_crossover = random.random() < 0.25
                
                if is_crossover:
                    # Generate crossover
                    if random.random() < 0.5:  # Bullish crossover
                        macd = random.uniform(0.01, 0.1)
                        signal = macd - random.uniform(0.01, 0.05)
                        histogram = macd - signal
                    else:  # Bearish crossover
                        macd = -random.uniform(0.01, 0.1)
                        signal = macd + random.uniform(0.01, 0.05)
                        histogram = macd - signal
                else:
                    # No crossover
                    macd = random.uniform(-0.2, 0.2)
                    signal = random.uniform(-0.2, 0.2)
                    histogram = macd - signal
                
                values[now] = {
                    "macd": macd,
                    "signal": signal,
                    "histogram": histogram
                }
            
            # Create and publish indicator event
            event = TechnicalIndicatorEvent(
                source="technical_indicators",
                symbol=symbol,
                indicator_name=indicator_type,
                values=values,
                timeframe=timeframe
            )
            
            await event_bus.publish(event)
        
        # Wait between batches
        await asyncio.sleep(delay)


async def main():
    """Run the AI Trading Agent demo."""
    logger.info("=" * 80)
    logger.info("Starting AI Trading Agent Demo")
    logger.info("=" * 80)
    
    # Symbols to analyze
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    try:
        # Start event bus
        await event_bus.start()
        
        # Create signal collector
        signal_collector = SignalCollector()
        event_bus.subscribe("SignalEvent", signal_collector.handle_signal_event)
        
        # Initialize components
        logger.info("Initializing analysis agents and decision engine...")
        
        # Technical indicators agent
        technical_agent = TechnicalIndicatorAgent("technical_indicators")
        await technical_agent.initialize()
        
        # Pattern recognition agent
        pattern_agent = PatternRecognitionAgent("pattern_recognition")
        await pattern_agent.initialize()
        
        # Sentiment analysis agent
        sentiment_agent = SentimentAnalysisAgent("sentiment_analysis")
        await sentiment_agent.initialize()
        
        # Decision engine
        decision_engine = DecisionEngine()
        await decision_engine.initialize()
        
        # Start components
        logger.info("Starting components...")
        await technical_agent.start()
        await pattern_agent.start()
        await sentiment_agent.start()
        await decision_engine.start()
        
        # Load sample data
        logger.info("Loading sample data...")
        candles_btc = await load_sample_data("BTC/USDT")
        candles_eth = await load_sample_data("ETH/USDT")
        
        # Process historical data
        logger.info("Publishing historical data...")
        await publish_candle_events(candles_btc[-50:], delay=0.02)  # Last 50 candles
        await publish_candle_events(candles_eth[-50:], delay=0.02)  # Last 50 candles
        
        # Allow some time for processing
        logger.info("Processing historical data...")
        await asyncio.sleep(2)
        
        # Start live simulation
        logger.info("\n" + "=" * 80)
        logger.info("Starting live simulation...")
        logger.info("=" * 80 + "\n")
        
        # Create publishing tasks
        candle_task = asyncio.create_task(
            publish_candle_events(candles_btc[-20:] + candles_eth[-20:], delay=0.5)
        )
        sentiment_task = asyncio.create_task(
            publish_sentiment_events(symbols, count=15, delay=1.0)
        )
        pattern_task = asyncio.create_task(
            publish_pattern_events(symbols, count=8, delay=2.0)
        )
        indicator_task = asyncio.create_task(
            publish_indicator_events(symbols, count=20, delay=0.8)
        )
        
        # Wait for all tasks to complete
        await asyncio.gather(candle_task, sentiment_task, pattern_task, indicator_task)
        
        # Allow some time for final processing
        logger.info("Finalizing signal generation...")
        await asyncio.sleep(5)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Demo Results Summary")
        logger.info("=" * 80)
        logger.info(f"Total generated signals: {len(signal_collector.signals)}")
        
        if signal_collector.signals:
            logger.info("\nSignal Breakdown:")
            signal_types = {}
            symbols_count = {}
            directions = {}
            
            for signal in signal_collector.signals:
                signal_types[signal["signal_type"]] = signal_types.get(signal["signal_type"], 0) + 1
                symbols_count[signal["symbol"]] = symbols_count.get(signal["symbol"], 0) + 1
                directions[signal["direction"]] = directions.get(signal["direction"], 0) + 1
            
            logger.info(f"By signal type: {signal_types}")
            logger.info(f"By symbol: {symbols_count}")
            logger.info(f"By direction: {directions}")
            
            logger.info("\nLatest Signals:")
            for signal in signal_collector.signals[-3:]:
                logger.info(f"- {signal['symbol']} {signal['signal_type']} {signal['direction']} @ {signal['price']:.2f}")
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error in demo: {str(e)}")
    finally:
        # Cleanup
        logger.info("Shutting down components...")
        
        try:
            await decision_engine.stop()
            await sentiment_agent.stop()
            await pattern_agent.stop()
            await technical_agent.stop()
            await event_bus.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        
        logger.info("Demo shutdown complete.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
