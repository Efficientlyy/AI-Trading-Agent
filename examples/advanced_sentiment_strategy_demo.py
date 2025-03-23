"""
Advanced Sentiment Strategy Demo

This example demonstrates the advanced sentiment strategy with:
1. Market impact assessment
2. Adaptive parameters based on market regime
3. Multi-timeframe sentiment analysis
4. Sentiment trend identification
"""

import asyncio
import logging
from datetime import datetime, timedelta
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src.common.events import event_bus
from src.common.logging import setup_logging
from src.common.datetime_utils import utc_now
from src.models.events import SentimentEvent, MarketRegimeEvent, MarketImpactEvent
from src.models.market_data import CandleData, OrderBookData, TimeFrame
from src.models.signals import Signal
from src.strategy.advanced_sentiment_strategy import AdvancedSentimentStrategy


# Set up logging
logger = logging.getLogger("advanced_sentiment_demo")


# Event handlers
async def signal_handler(signal: Signal) -> None:
    """Handle strategy signals."""
    logger.info(f"Received {signal.direction} signal for {signal.symbol}:")
    logger.info(f"  Type: {signal.signal_type}")
    logger.info(f"  Price: {signal.price}")
    logger.info(f"  Confidence: {signal.confidence:.2f}")
    logger.info(f"  Reason: {signal.reason}")
    
    meta = signal.metadata
    if meta:
        logger.info("  Metadata:")
        logger.info(f"    Sentiment value: {meta.get('sentiment_value', 'N/A'):.2f}")
        logger.info(f"    Original sentiment: {meta.get('original_sentiment', 'N/A')}")
        logger.info(f"    Sentiment direction: {meta.get('sentiment_direction', 'N/A')}")
        logger.info(f"    Signal score: {meta.get('signal_score', 'N/A'):.2f}")
        logger.info(f"    Market regime: {meta.get('regime', 'N/A')}")
        logger.info(f"    Sentiment trend: {meta.get('sentiment_trend', 'N/A')}")
        logger.info(f"    Trend strength: {meta.get('trend_strength', 'N/A')}")
        logger.info(f"    Market impact: {meta.get('market_impact_factor', 'N/A')}")


# Utility functions for the demo
def generate_sample_candles(symbol: str, count: int, start_price: float = 50000.0, trend: str = "neutral") -> list[CandleData]:
    """Generate sample candle data for testing.
    
    Args:
        symbol: The trading pair symbol
        count: The number of candles to generate
        start_price: The starting price
        trend: Price trend ('bullish', 'bearish', 'volatile', 'ranging', 'neutral')
        
    Returns:
        List of sample candle data
    """
    candles = []
    now = datetime.utcnow()
    price = start_price
    
    # Set trend parameters
    if trend == "bullish":
        drift = 0.002  # positive drift
        volatility = 0.015
    elif trend == "bearish":
        drift = -0.002  # negative drift
        volatility = 0.015
    elif trend == "volatile":
        drift = 0.0
        volatility = 0.03  # higher volatility
    elif trend == "ranging":
        drift = 0.0
        volatility = 0.01  # lower volatility
    else:  # neutral
        drift = 0.0
        volatility = 0.015
    
    for i in range(count):
        # Generate random price movement with drift
        change_pct = random.normalvariate(drift, volatility)
        close_price = price * (1 + change_pct)
        
        # Generate candle with some randomness
        high_price = max(price, close_price) * random.uniform(1.0, 1.01)
        low_price = min(price, close_price) * random.uniform(0.99, 1.0)
        
        # Create candle
        candle = CandleData(
            symbol=symbol,
            exchange="Binance",
            timeframe=TimeFrame("1h"),
            timestamp=now - timedelta(hours=count-i),
            open=price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=random.uniform(100, 1000)
        )
        
        candles.append(candle)
        
        # Update price for next candle
        price = close_price
        
    return candles


def generate_sample_order_book(symbol: str, mid_price: float, depth: int = 10) -> OrderBookData:
    """Generate sample order book data.
    
    Args:
        symbol: The trading pair symbol
        mid_price: The current mid price
        depth: The depth of the order book
        
    Returns:
        Sample order book data
    """
    bids = []
    asks = []
    
    # Generate bids (below mid price)
    for i in range(depth):
        price = mid_price * (1 - 0.0001 * (i + 1))
        size = random.uniform(0.5, 5.0)
        bids.append([price, size])
    
    # Generate asks (above mid price)
    for i in range(depth):
        price = mid_price * (1 + 0.0001 * (i + 1))
        size = random.uniform(0.5, 5.0)
        asks.append([price, size])
    
    # Create order book
    order_book = OrderBookData(
        symbol=symbol,
        exchange="Binance",
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks
    )
    
    return order_book


def generate_regime_changes(symbol: str, candles: list[CandleData]) -> list[MarketRegimeEvent]:
    """Generate sample market regime changes based on candle data.
    
    Args:
        symbol: The trading pair symbol
        candles: List of candle data
        
    Returns:
        List of market regime events
    """
    if len(candles) < 20:
        return []
        
    events = []
    
    # Split candles into segments
    segment_length = len(candles) // 4
    
    # Generate different regimes for each segment
    regimes = ["bullish", "volatile", "ranging", "bearish"]
    
    for i, regime in enumerate(regimes):
        start_idx = i * segment_length
        if start_idx < len(candles):
            # Create regime event
            event = MarketRegimeEvent(
                source="regime_detector",
                payload={
                    "symbol": symbol,
                    "regime": regime,
                    "confidence": random.uniform(0.7, 0.9),
                    "period_start": candles[start_idx].timestamp.isoformat(),
                    "timestamp": candles[start_idx].timestamp.isoformat()
                }
            )
            events.append(event)
    
    return events


def generate_market_impact_events(symbol: str, candles: list[CandleData]) -> list[MarketImpactEvent]:
    """Generate sample market impact events based on candle data.
    
    Args:
        symbol: The trading pair symbol
        candles: List of candle data
        
    Returns:
        List of market impact events
    """
    if len(candles) < 10:
        return []
        
    events = []
    
    # Generate 3-5 market impact events spread throughout the candle data
    num_events = random.randint(3, 5)
    indices = sorted(random.sample(range(len(candles)), num_events))
    
    impact_types = ["news", "order_flow", "whale_activity", "regulatory_news"]
    
    for i, idx in enumerate(indices):
        impact_type = random.choice(impact_types)
        impact_value = random.normalvariate(0, 0.3)  # -1 to 1 scale
        
        # Create impact event
        event = MarketImpactEvent(
            source="market_impact_analyzer",
            payload={
                "symbol": symbol,
                "impact_type": impact_type,
                "impact_value": impact_value,
                "confidence": random.uniform(0.6, 0.9),
                "timestamp": candles[idx].timestamp.isoformat(),
                "details": {
                    "estimated_price_impact": impact_value,
                    "liquidity_adjustment": random.uniform(0.8, 1.2),
                }
            }
        )
        events.append(event)
    
    return events


def generate_sentiment_events(symbol: str, candles: list[CandleData]) -> list[SentimentEvent]:
    """Generate sample sentiment events with different patterns.
    
    Args:
        symbol: The trading pair symbol
        candles: List of candle data
        
    Returns:
        List of sentiment events
    """
    if len(candles) < 20:
        return []
        
    events = []
    
    # Generate sentiment events for different sources
    sources = ["social_media", "news", "market", "onchain"]
    
    for source in sources:
        # Create base sentiment pattern (start neutral, then trend)
        base_pattern = [0.5] * 5  # Start neutral
        
        # Add a trend based on source
        if source == "social_media":
            # Rising trend
            for i in range(15):
                base_pattern.append(min(0.9, 0.5 + i * 0.025))
        elif source == "news":
            # Volatile pattern
            for i in range(15):
                if i % 5 == 0:
                    base_pattern.append(random.uniform(0.7, 0.9))  # Spike
                else:
                    base_pattern.append(random.uniform(0.4, 0.6))  # Normal
        elif source == "market":
            # Falling trend
            for i in range(15):
                base_pattern.append(max(0.1, 0.5 - i * 0.025))
        else:  # onchain
            # Neutral with slight upward bias
            for i in range(15):
                base_pattern.append(random.uniform(0.45, 0.65))
        
        # Generate one event per 5 candles
        for i in range(0, len(candles), 5):
            if i // 5 < len(base_pattern):
                sentiment_value = base_pattern[i // 5]
                
                # Add some noise
                sentiment_value = max(0.1, min(0.9, sentiment_value + random.normalvariate(0, 0.05)))
                
                # Determine direction
                if sentiment_value > 0.6:
                    direction = "bullish"
                elif sentiment_value < 0.4:
                    direction = "bearish"
                else:
                    direction = "neutral"
                
                # Create event
                event = SentimentEvent(
                    source=f"{source}_analyzer",
                    payload={
                        "symbol": symbol,
                        "sentiment_value": sentiment_value,
                        "sentiment_direction": direction,
                        "confidence": random.uniform(0.7, 0.9),
                        "timestamp": candles[i].timestamp.isoformat()
                    }
                )
                events.append(event)
    
    return events


def plot_strategy_results(
    symbol: str,
    candles: list[CandleData],
    sentiment_events: list[dict],
    signals: list[dict],
    regimes: list[dict]
) -> None:
    """Plot strategy results with price, sentiment, and signals.
    
    Args:
        symbol: The trading pair symbol
        candles: List of candle data
        sentiment_events: List of sentiment events
        signals: List of signals
        regimes: List of market regimes
    """
    # Convert candles to DataFrame
    candle_data = []
    for candle in candles:
        candle_data.append({
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume
        })
    
    df_candles = pd.DataFrame(candle_data)
    
    # Convert sentiment events to DataFrame
    if sentiment_events:
        df_sentiment = pd.DataFrame(sentiment_events)
        df_sentiment["timestamp"] = pd.to_datetime(df_sentiment["timestamp"])
    else:
        df_sentiment = pd.DataFrame(columns=["timestamp", "source", "value"])
    
    # Convert signals to DataFrame
    if signals:
        df_signals = pd.DataFrame(signals)
        df_signals["timestamp"] = pd.to_datetime(df_signals["timestamp"])
    else:
        df_signals = pd.DataFrame(columns=["timestamp", "direction", "price"])
    
    # Convert regimes to DataFrame
    if regimes:
        df_regimes = pd.DataFrame(regimes)
        df_regimes["timestamp"] = pd.to_datetime(df_regimes["timestamp"])
    else:
        df_regimes = pd.DataFrame(columns=["timestamp", "regime"])
    
    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Advanced Sentiment Strategy - {symbol}", fontsize=16)
    
    # Plot price
    axs[0].plot(df_candles["timestamp"], df_candles["close"], label="Price", color="black")
    
    # Plot signals
    for _, signal in df_signals.iterrows():
        if signal["direction"] == "long":
            axs[0].scatter(signal["timestamp"], signal["price"], marker="^", color="green", s=100, label="_Long")
        elif signal["direction"] == "short":
            axs[0].scatter(signal["timestamp"], signal["price"], marker="v", color="red", s=100, label="_Short")
    
    # Plot regime changes
    for _, regime in df_regimes.iterrows():
        color = {
            "bullish": "green",
            "bearish": "red",
            "volatile": "orange",
            "ranging": "blue"
        }.get(regime["regime"], "gray")
        
        axs[0].axvline(x=regime["timestamp"], color=color, linestyle="--", alpha=0.5)
        axs[0].text(regime["timestamp"], df_candles["close"].max(), regime["regime"], 
                  color=color, ha="left", va="bottom", rotation=90)
    
    # Plot sentiment
    for source in df_sentiment["source"].unique():
        source_data = df_sentiment[df_sentiment["source"] == source]
        axs[1].plot(source_data["timestamp"], source_data["value"], 
                  label=source, marker="o", linestyle="-", alpha=0.7)
    
    # Add neutral line
    axs[1].axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    
    # Add thresholds
    axs[1].axhline(y=0.7, color="green", linestyle=":", alpha=0.5, label="Bullish threshold")
    axs[1].axhline(y=0.3, color="red", linestyle=":", alpha=0.5, label="Bearish threshold")
    
    # Format axes
    axs[0].set_ylabel("Price")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper left")
    
    axs[1].set_ylabel("Sentiment Value")
    axs[1].set_ylim(0, 1)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_advanced_sentiment_strategy.png")
    
    logger.info(f"Plot saved to {symbol}_advanced_sentiment_strategy.png")


async def run_demo():
    """Run the advanced sentiment strategy demo."""
    logger.info("Starting advanced sentiment strategy demo")
    
    # Create strategy
    strategy = AdvancedSentimentStrategy("advanced_sentiment_demo")
    
    # Set up signal tracking
    signals_received = []
    
    async def signal_tracker(signal: Signal):
        signal_data = {
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "price": signal.price,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "metadata": signal.metadata
        }
        signals_received.append(signal_data)
        await signal_handler(signal)
    
    # Subscribe to signals
    event_bus.subscribe("signal", signal_tracker)
    
    try:
        # Initialize and start strategy
        await strategy.initialize()
        await strategy.start()
        
        # Generate sample data
        symbol = "BTC/USDT"
        candles = generate_sample_candles(symbol, 100, start_price=50000, trend="volatile")
        
        # Generate regime events
        regime_events = generate_regime_changes(symbol, candles)
        
        # Track regimes for plotting
        regimes_data = []
        
        # Generate market impact events
        impact_events = generate_market_impact_events(symbol, candles)
        
        # Generate sentiment events
        sentiment_events = generate_sentiment_events(symbol, candles)
        
        # Track sentiment for plotting
        sentiment_data = []
        
        # Process data in chronological order
        all_events = []
        
        # Add candles to events
        for candle in candles:
            all_events.append(("candle", candle.timestamp, candle))
            
            # Add order book every 5 candles
            if candles.index(candle) % 5 == 0:
                order_book = generate_sample_order_book(symbol, candle.close)
                all_events.append(("order_book", candle.timestamp, order_book))
        
        # Add regime events
        for event in regime_events:
            timestamp = datetime.fromisoformat(event.payload["timestamp"])
            all_events.append(("regime", timestamp, event))
            
            # Track for plotting
            regimes_data.append({
                "timestamp": timestamp,
                "regime": event.payload["regime"],
                "confidence": event.payload["confidence"]
            })
            
        # Add market impact events
        for event in impact_events:
            timestamp = datetime.fromisoformat(event.payload["timestamp"])
            all_events.append(("impact", timestamp, event))
        
        # Add sentiment events
        for event in sentiment_events:
            timestamp = datetime.fromisoformat(event.payload["timestamp"])
            all_events.append(("sentiment", timestamp, event))
            
            # Track for plotting
            sentiment_data.append({
                "timestamp": timestamp,
                "source": event.source.split("_")[0],
                "value": event.payload["sentiment_value"],
                "direction": event.payload["sentiment_direction"]
            })
        
        # Sort events by timestamp
        all_events.sort(key=lambda x: x[1])
        
        # Process events
        logger.info(f"Processing {len(all_events)} events")
        for event_type, _, event in all_events:
            if event_type == "candle":
                await strategy.process_candle(event)
            elif event_type == "order_book":
                await strategy.process_order_book(event)
            elif event_type == "regime":
                await event_bus.publish("market_regime_event", event)
            elif event_type == "impact":
                await event_bus.publish("market_impact_event", event)
            elif event_type == "sentiment":
                await event_bus.publish("sentiment_event", event)
            
            # Add a small delay to allow events to be processed
            await asyncio.sleep(0.01)
        
        # Allow time for final processing
        await asyncio.sleep(1)
        
        # Display summary
        logger.info(f"Demo completed with {len(candles)} candles processed")
        logger.info(f"Generated {len(signals_received)} trading signals")
        
        # Plot results
        plot_strategy_results(
            symbol=symbol,
            candles=candles,
            sentiment_events=sentiment_data,
            signals=signals_received,
            regimes=regimes_data
        )
        
    finally:
        # Stop strategy
        await strategy.stop()
        logger.info("Advanced sentiment strategy demo completed")


if __name__ == "__main__":
    try:
        # Set up logging
        setup_logging(level=logging.INFO)
        
        # Run demo
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)