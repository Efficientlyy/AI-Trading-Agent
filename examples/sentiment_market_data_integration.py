#!/usr/bin/env python3
"""
Sentiment Analysis with Real Market Data Integration

This script demonstrates how to integrate real market data from cryptocurrency
exchanges with the sentiment analysis system. It:

1. Fetches real-time market data from Binance
2. Simulates sentiment analysis from various sources
3. Combines market data and sentiment signals to generate trading decisions
4. Visualizes the relationship between price movements and sentiment

For demonstration purposes, the sentiment data is simulated, but the price
data comes from the real Binance API.
"""

import asyncio
import os
import json
import logging
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import time
from typing import Dict, List, Optional, Any

# Add parent directory to path to allow imports from src
parent_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(parent_dir))

# Create output directory
output_dir = Path("examples/output/sentiment_integration")
output_dir.mkdir(parents=True, exist_ok=True)

# Import exchange connector
from src.execution.exchange.binance import BinanceExchangeConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_integration")


# Mock sentiment components
class SentimentEvent:
    """Mock sentiment event class."""
    
    def __init__(self, source, symbol, value, direction, confidence):
        self.source = source
        self.symbol = symbol
        self.sentiment_value = value
        self.sentiment_direction = direction
        self.confidence = confidence
        self.timestamp = datetime.now()
        self.details = {}


class SentimentGenerator:
    """Generator for simulated sentiment data."""
    
    def __init__(self):
        """Initialize the sentiment generator."""
        self.sources = ["social_media", "news", "market_indicators", "onchain"]
        self.sentiment_history = {}  # symbol -> [values]
        
    async def generate_sentiment(self, symbol: str, current_price: float, price_history: List[float]) -> SentimentEvent:
        """Generate a sentiment event influenced by price data.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            price_history: Recent price history
            
        Returns:
            A sentiment event
        """
        # Calculate price change
        if len(price_history) >= 2:
            price_change = (current_price - price_history[0]) / price_history[0]
        else:
            price_change = 0
        
        # Base sentiment on price change with some randomness
        base_sentiment = 0.5 + (price_change * 5)  # Scale factor of 5 makes small changes more visible
        
        # Add noise to make it realistic
        noise = random.uniform(-0.15, 0.15)
        sentiment_value = max(0.1, min(0.9, base_sentiment + noise))
        
        # Initialize sentiment history for this symbol if needed
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = [0.5]  # Start neutral
        
        # Add some persistence/trend from previous sentiment
        prev_sentiment = self.sentiment_history[symbol][-1]
        sentiment_value = (sentiment_value + prev_sentiment) / 2
        
        # Add to history
        self.sentiment_history[symbol].append(sentiment_value)
        if len(self.sentiment_history[symbol]) > 20:
            self.sentiment_history[symbol].pop(0)
        
        # Determine direction
        if sentiment_value > 0.6:
            direction = "bullish"
        elif sentiment_value < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Randomize confidence
        confidence = random.uniform(0.6, 0.95)
        
        # Determine source
        source = random.choice(self.sources)
        
        # Create sentiment event
        return SentimentEvent(
            source=f"{source}_sentiment",
            symbol=symbol,
            value=sentiment_value,
            direction=direction,
            confidence=confidence
        )


class EnhancedSentimentStrategy:
    """Enhanced sentiment strategy that incorporates real market data."""
    
    def __init__(self):
        """Initialize the strategy."""
        self.name = "enhanced_sentiment_strategy"
        self.logger = logger.getChild("strategy")
        
        # Strategy parameters
        self.sentiment_threshold_bullish = 0.65
        self.sentiment_threshold_bearish = 0.35
        self.min_confidence = 0.7
        self.contrarian_mode = False
        
        # Technical parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Data storage
        self.prices = {}  # symbol -> [prices]
        self.sentiment_data = {}  # symbol -> {source -> data}
        
        # Signal tracking
        self.signals = []
        
    def add_price(self, symbol: str, price: float, timestamp: datetime):
        """Add a price data point.
        
        Args:
            symbol: Trading pair symbol
            price: Market price
            timestamp: Timestamp of the price data
        """
        if symbol not in self.prices:
            self.prices[symbol] = []
            
        self.prices[symbol].append({
            "price": price,
            "timestamp": timestamp
        })
        
        # Keep only last 100 price points
        if len(self.prices[symbol]) > 100:
            self.prices[symbol].pop(0)
            
    def process_sentiment(self, event: SentimentEvent):
        """Process a sentiment event.
        
        Args:
            event: The sentiment event to process
        """
        symbol = event.symbol
        source = event.source
        
        # Initialize data structures if needed
        if symbol not in self.sentiment_data:
            self.sentiment_data[symbol] = {}
            
        # Store sentiment data
        self.sentiment_data[symbol][source] = {
            "value": event.sentiment_value,
            "direction": event.sentiment_direction,
            "confidence": event.confidence,
            "timestamp": event.timestamp
        }
        
        # Check if we should generate a signal
        self.check_for_signal(symbol)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate the RSI technical indicator.
        
        Args:
            prices: List of price values
            period: RSI period
            
        Returns:
            RSI value or None if not enough data
        """
        if len(prices) <= period:
            return None
            
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Get the most recent 'period' changes
        recent_changes = changes[-period:]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in recent_changes]
        losses = [abs(change) if change < 0 else 0 for change in recent_changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100  # No losses, RSI is 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_for_signal(self, symbol: str):
        """Check if we should generate a trading signal.
        
        Args:
            symbol: Trading pair symbol
        """
        # Skip if we don't have price data
        if symbol not in self.prices or len(self.prices[symbol]) < 15:
            return
            
        # Skip if we don't have sentiment data
        if symbol not in self.sentiment_data or len(self.sentiment_data[symbol]) < 2:
            return
            
        # Get the latest price
        latest_price = self.prices[symbol][-1]["price"]
        price_history = [p["price"] for p in self.prices[symbol]]
        
        # Calculate RSI
        rsi = self.calculate_rsi(price_history, self.rsi_period)
        
        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        for source, data in self.sentiment_data[symbol].items():
            # Skip old data (older than 1 hour)
            if (datetime.now() - data["timestamp"]).total_seconds() > 3600:
                continue
                
            # Skip sources with low confidence
            if data["confidence"] < self.min_confidence:
                continue
                
            # Determine source weight
            if "social_media" in source:
                weight = 1.0
            elif "news" in source:
                weight = 1.2
            elif "market_indicators" in source:
                weight = 1.5
            elif "onchain" in source:
                weight = 1.3
            else:
                weight = 1.0
                
            # Add to weighted calculation
            weighted_sentiment += data["value"] * weight
            total_weight += weight
        
        # Skip if we don't have enough valid sentiment data
        if total_weight == 0:
            return
            
        # Calculate final sentiment value
        final_sentiment = weighted_sentiment / total_weight
        
        # Determine signal direction
        signal_direction = None
        signal_type = None
        
        if not self.contrarian_mode:
            # Normal mode - follow sentiment
            if final_sentiment >= self.sentiment_threshold_bullish:
                signal_direction = "long"
                signal_type = "entry"
            elif final_sentiment <= self.sentiment_threshold_bearish:
                signal_direction = "short"
                signal_type = "entry"
        else:
            # Contrarian mode - go against extreme sentiment
            if final_sentiment >= 0.8:
                signal_direction = "short"
                signal_type = "entry"
            elif final_sentiment <= 0.2:
                signal_direction = "long"
                signal_type = "entry"
                
        # Skip if no signal direction determined
        if not signal_direction or not signal_type:
            return
            
        # Apply technical confirmation if RSI is available
        if rsi is not None:
            # Skip buy signals when overbought
            if signal_direction == "long" and rsi > self.rsi_overbought:
                self.logger.info(f"Skipping {signal_direction} signal due to overbought RSI: {rsi:.1f}")
                return
                
            # Skip sell signals when oversold
            if signal_direction == "short" and rsi < self.rsi_oversold:
                self.logger.info(f"Skipping {signal_direction} signal due to oversold RSI: {rsi:.1f}")
                return
        
        # Generate signal
        signal = {
            "symbol": symbol,
            "direction": signal_direction,
            "type": signal_type,
            "price": latest_price,
            "sentiment": final_sentiment,
            "rsi": rsi,
            "timestamp": datetime.now()
        }
        
        self.signals.append(signal)
        
        self.logger.info(f"Generated {signal_direction} signal for {symbol} at ${latest_price:.2f}")
        self.logger.info(f"  Sentiment: {final_sentiment:.2f}, RSI: {rsi:.1f}")
        

async def run_data_integration_demo():
    """Main function to run the demo."""
    logger.info("Starting Sentiment Analysis with Real Market Data Integration Demo")
    
    # Create the Binance connector
    connector = BinanceExchangeConnector(
        exchange_id="binance",
        testnet=True  # Use the testnet for safety
    )
    
    # Initialize connector
    initialized = await connector.initialize()
    if not initialized:
        logger.error("Failed to initialize Binance connector")
        return
        
    logger.info("Binance connector initialized successfully")
    
    # Create sentiment generator and strategy
    sentiment_gen = SentimentGenerator()
    strategy = EnhancedSentimentStrategy()
    
    # Define trading pairs to monitor
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    # Price data storage
    price_history = {symbol: [] for symbol in symbols}
    
    # Run the integration loop
    try:
        logger.info("Starting market data and sentiment integration")
        logger.info(f"Monitoring symbols: {', '.join(symbols)}")
        
        for iteration in range(30):  # Run for 30 iterations
            logger.info(f"\n--- Iteration {iteration + 1} ---")
            
            # Get market data from Binance
            for symbol in symbols:
                try:
                    # Get current ticker data
                    ticker = await connector.get_ticker(symbol)
                    price = float(ticker["last"])
                    
                    # Log price data
                    logger.info(f"{symbol} - Current price: ${price:.2f}")
                    
                    # Store price history
                    price_history[symbol].append(price)
                    if len(price_history[symbol]) > 50:
                        price_history[symbol].pop(0)
                    
                    # Update strategy with price data
                    strategy.add_price(symbol, price, datetime.now())
                    
                    # Generate sentiment event based on real price data
                    sentiment_event = await sentiment_gen.generate_sentiment(
                        symbol=symbol,
                        current_price=price,
                        price_history=price_history[symbol]
                    )
                    
                    # Log sentiment data
                    logger.info(f"{symbol} - {sentiment_event.source}: {sentiment_event.sentiment_direction} "
                              f"({sentiment_event.sentiment_value:.2f}) with {sentiment_event.confidence:.2f} confidence")
                    
                    # Process sentiment event
                    strategy.process_sentiment(sentiment_event)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                
                # Short delay between symbols
                await asyncio.sleep(0.5)
            
            # Sleep between iterations
            await asyncio.sleep(2)
        
        # Display results
        logger.info("\n=== Integration Results ===")
        logger.info(f"Total trading signals generated: {len(strategy.signals)}")
        
        for signal in strategy.signals:
            logger.info(f"{signal['timestamp'].strftime('%H:%M:%S')} - {signal['symbol']}: "
                      f"{signal['direction']} signal at ${signal['price']:.2f} "
                      f"(Sentiment: {signal['sentiment']:.2f}, RSI: {signal['rsi']:.1f if signal['rsi'] else 'N/A'})")
                      
        # Save results to file
        results = {
            "signals": [
                {
                    "symbol": s["symbol"],
                    "direction": s["direction"],
                    "price": s["price"],
                    "sentiment": s["sentiment"],
                    "rsi": s["rsi"],
                    "timestamp": s["timestamp"].isoformat()
                }
                for s in strategy.signals
            ],
            "price_history": {
                symbol: prices for symbol, prices in price_history.items()
            }
        }
        
        with open(output_dir / "integration_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Results saved to {output_dir}/integration_results.json")
        
    except Exception as e:
        logger.exception(f"Error in demo: {str(e)}")
    finally:
        # Shutdown connector
        await connector.shutdown()
        logger.info("Binance connector shutdown")


async def create_visualization_script():
    """Create a Python script to visualize the results."""
    visualization_script = """#!/usr/bin/env python3
\"\"\"
Sentiment and Price Visualization

This script visualizes the relationship between price, sentiment, and trading signals
from the sentiment market data integration demo.
\"\"\"

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from pathlib import Path

# Load results file
results_file = Path("examples/output/sentiment_integration/integration_results.json")
if not results_file.exists():
    print(f"Results file not found: {results_file}")
    exit(1)

with open(results_file, "r") as f:
    data = json.load(f)

signals = data["signals"]
price_history = data["price_history"]

# Convert timestamps to datetime objects
for signal in signals:
    signal["timestamp"] = datetime.fromisoformat(signal["timestamp"])

# Choose a symbol to visualize (use the one with most signals)
symbol_count = {}
for signal in signals:
    symbol = signal["symbol"]
    symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
    
target_symbol = max(symbol_count.items(), key=lambda x: x[1])[0] if symbol_count else "BTC/USDT"

# Filter signals for the target symbol
symbol_signals = [s for s in signals if s["symbol"] == target_symbol]

# Create price points with timestamps (synthesize from the signal timestamps)
# In a real implementation, you would use actual timestamped price data
price_times = sorted([s["timestamp"] for s in symbol_signals])
if not price_times:
    print("No signals found for visualization")
    exit(0)
    
prices = price_history.get(target_symbol, [])
if not prices:
    print("No price data found for visualization")
    exit(0)

# Trim prices to match number of timestamps
if len(prices) > len(price_times):
    prices = prices[-len(price_times):]
elif len(prices) < len(price_times):
    price_times = price_times[-len(prices):]

# Create sentiment values (from signals)
sentiments = [s["sentiment"] for s in symbol_signals]
rsi_values = [s["rsi"] for s in symbol_signals]

# Create the visualization
plt.figure(figsize=(12, 10))

# Plot 1: Price chart
ax1 = plt.subplot(3, 1, 1)
ax1.plot(price_times, prices, 'b-', label=f"{target_symbol} Price")
ax1.set_title(f"{target_symbol} Price Chart")
ax1.set_ylabel("Price (USD)")
ax1.grid(True, alpha=0.3)

# Add buy/sell signals
for signal in symbol_signals:
    if signal["direction"] == "long":
        ax1.scatter(signal["timestamp"], signal["price"], marker='^', color='green', s=100, label='Buy Signal')
    elif signal["direction"] == "short":
        ax1.scatter(signal["timestamp"], signal["price"], marker='v', color='red', s=100, label='Sell Signal')

handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())

# Plot 2: Sentiment chart
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(price_times, sentiments, 'g-', label="Sentiment")
ax2.axhline(y=0.65, color='g', linestyle='--', alpha=0.5, label="Bullish Threshold")
ax2.axhline(y=0.35, color='r', linestyle='--', alpha=0.5, label="Bearish Threshold")
ax2.set_title("Sentiment Analysis")
ax2.set_ylabel("Sentiment Value (0-1)")
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: RSI chart
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(price_times, rsi_values, 'purple', label="RSI")
ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label="Overbought")
ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label="Oversold")
ax3.set_title("Relative Strength Index (RSI)")
ax3.set_ylabel("RSI Value")
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Format the x axis
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.tight_layout()
plt.savefig("examples/output/sentiment_integration/visualization.png")
print("Visualization saved to examples/output/sentiment_integration/visualization.png")
"""

    # Save the script
    script_path = output_dir / "visualize_results.py"
    with open(script_path, "w") as f:
        f.write(visualization_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Visualization script created: {script_path}")


if __name__ == "__main__":
    asyncio.run(run_data_integration_demo())
    asyncio.run(create_visualization_script())