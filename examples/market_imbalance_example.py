"""Example script demonstrating how to use the Market Imbalance Strategy.

This script shows how to set up and run the Market Imbalance Strategy, which
monitors order book imbalances to generate trading signals.
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import Config, config
from src.common.events import EventBus
from src.common.logging import get_logger
from src.models.events import EventType
from src.models.market_data import TimeFrame, TradeData
from src.models.signals import Signal, SignalType
from src.rust_bridge import OrderBookProcessor, create_order_book_processor
from src.strategy.market_imbalance import MarketImbalanceStrategy
from examples.orderbook_example import generate_sample_orderbook, generate_order_book_update


logger = get_logger("examples", "market_imbalance")


async def run_example() -> None:
    """Run the market imbalance strategy example."""
    # Configure logging
    # Just use basic Python logging since we can't check the signature 
    # of configure_logging() in this example
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Market Imbalance Strategy example")
    
    # Configure the EventBus
    event_bus = EventBus()
    
    # Sample configuration
    config_data = {
        "strategies": {
            "market_imbalance": {
                "symbols": ["BTC/USD", "ETH/USD"],
                "imbalance_threshold": 1.3,  # Trigger signals at 30% imbalance
                "depth_shallow": 5,
                "depth_medium": 10,
                "depth_deep": 20,
                "min_trade_size": 0.01,
                "max_trade_size": 0.5,
                "take_profit_pct": 0.5,
                "stop_loss_pct": 0.3,
                "signal_cooldown": 60  # 1 minute (shorter for demo)
            }
        }
    }
    
    # Update the global config
    # Since we couldn't find an update method, let's directly merge the config
    for key, value in config_data.items():
        if key in config._config and isinstance(config._config[key], dict) and isinstance(value, dict):
            config._config[key].update(value)
        else:
            config._config[key] = value
    
    # Initialize the strategy
    # We need to implement the required methods to make this non-abstract
    class SimulationMarketImbalanceStrategy(MarketImbalanceStrategy):
        """Extension of MarketImbalanceStrategy for simulation purposes."""
        
        async def process_candle(self, candle_data: Dict[str, Any]) -> None:
            """Process candle data (required implementation)."""
            pass
        
        async def process_trade(self, trade_data: TradeData) -> None:
            """Process trade data (required implementation)."""
            pass
            
        async def process_indicator(self, indicator_data: Dict[str, Any]) -> None:
            """Process indicator data (required implementation)."""
            pass
            
        async def process_pattern(self, pattern_data: Dict[str, Any]) -> None:
            """Process pattern data (required implementation)."""
            pass
        
        async def on_orderbook_update(self, symbol: str, processor: OrderBookProcessor) -> None:
            """Process order book updates."""
            await self.analyze_orderbook(symbol, processor)
    
    strategy = SimulationMarketImbalanceStrategy()
    
    # Subscribe to signals
    signal_count = 0
    
    async def signal_handler(signal_event: Dict) -> None:
        nonlocal signal_count
        signal_count += 1
        signal = Signal.from_dict(signal_event)
        
        # Construct a dictionary of signal properties to log
        # Only include properties that exist on the Signal class
        signal_info = {
            "count": signal_count,
            "symbol": signal.symbol,
            "type": signal.signal_type.name,
            "direction": signal.direction,
            "price": signal.price,
            "confidence": round(signal.confidence, 2),
            "reason": signal.reason
        }
        
        logger.info("⚡ Received trading signal", **signal_info)
        
        if signal.metadata:
            logger.info("Signal metadata", 
                     imbalance=round(signal.metadata.get("imbalance", 0), 3),
                     imbalance_pct=round(signal.metadata.get("imbalance_pct", 0), 1),
                     take_profit=signal.take_profit,
                     stop_loss=signal.stop_loss)
    
    event_bus.subscribe(EventType.SIGNAL, signal_handler)
    
    # Initialize the strategy
    await strategy.initialize()
    
    # Create order book processors
    processors = {}
    for symbol in ["BTC/USD", "ETH/USD"]:
        # Create the processor
        processors[symbol] = create_order_book_processor(
            symbol=symbol,
            exchange="example",
            max_depth=50
        )
        
        # Generate initial order books
        btc_base_price = 50000.0
        eth_base_price = 3000.0
        
        base_price = btc_base_price if symbol == "BTC/USD" else eth_base_price
        
        # Generate initial book
        # Modify parameters to match what's expected in generate_sample_orderbook
        initial_book = generate_sample_orderbook(
            base_price=base_price,
            depth=40,
            spread_pct=0.05
        )
        
        # Process initial updates
        updates = []
        for price, size in initial_book["bids"]:
            updates.append({"type": "bid", "price": price, "size": size})
        for price, size in initial_book["asks"]:
            updates.append({"type": "ask", "price": price, "size": size})
        
        processors[symbol].process_updates(updates)
        
        logger.info(f"Initialized order book for {symbol}", 
                 best_bid=processors[symbol].best_bid_price(),
                 best_ask=processors[symbol].best_ask_price(),
                 mid_price=processors[symbol].mid_price(),
                 imbalance=processors[symbol].book_imbalance(10))
    
    # Start the strategy
    await strategy.start()
    
    # Set up a termination handler
    should_exit = False
    
    def handle_exit_signal(sig, frame):
        nonlocal should_exit
        should_exit = True
        logger.info("Received exit signal, shutting down...")
    
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    
    # Simulate order book updates at different intervals
    logger.info("Starting order book simulation...")
    
    intervals = {
        "BTC/USD": 0.5,  # updates every 0.5 seconds
        "ETH/USD": 0.8,  # updates every 0.8 seconds
    }
    
    last_update = {symbol: time.time() for symbol in processors.keys()}
    
    # Simulation parameters
    imbalance_cycles = {
        "BTC/USD": {
            "period": 60,  # period of imbalance cycle in seconds
            "phase": 0,    # starting phase (0 to 1)
            "amplitude": 0.4,  # max deviation from neutral
        },
        "ETH/USD": {
            "period": 90,  # period of imbalance cycle in seconds
            "phase": 0.5,  # starting with opposite phase
            "amplitude": 0.3,  # max deviation from neutral
        }
    }
    
    # Run for specified time
    start_time = time.time()
    run_time = 300  # run for 5 minutes
    
    try:
        while not should_exit and (time.time() - start_time) < run_time:
            current_time = time.time()
            
            for symbol, processor in processors.items():
                if current_time - last_update[symbol] >= intervals[symbol]:
                    # Get current imbalance phase
                    elapsed = current_time - start_time
                    cycle = imbalance_cycles[symbol]
                    cycle_position = (elapsed / cycle["period"] + cycle["phase"]) % 1.0
                    
                    # Sine wave oscillation between buy and sell imbalance
                    # Values near 0 = sell-biased, values near 1 = buy-biased
                    imbalance_factor = 0.5 + cycle["amplitude"] * (
                        0.5 * (1 + (2 * (cycle_position) - 1))
                    )
                    
                    # Add some randomness
                    imbalance_factor += random.uniform(-0.1, 0.1)
                    imbalance_factor = max(0.1, min(0.9, imbalance_factor))
                    
                    # Generate updates
                    # Modify parameters to match what's expected in generate_order_book_update
                    updates = generate_order_book_update(
                        processor=processor,
                        num_updates=random.randint(3, 10)
                    )
                    
                    # Process updates
                    processor.process_updates(updates)
                    
                    # Process order book in strategy
                    await strategy.on_orderbook_update(symbol, processor)
                    
                    # Log current state periodically
                    if int(elapsed) % 10 == 0 and int(elapsed) != int(elapsed - intervals[symbol]):
                        imbalance = processor.book_imbalance(10)
                        logger.info(f"{symbol} update", 
                                 elapsed=int(elapsed),
                                 best_bid=processor.best_bid_price(),
                                 best_ask=processor.best_ask_price(),
                                 mid_price=processor.mid_price(),
                                 spread=processor.spread(),
                                 imbalance=imbalance,
                                 imbalance_factor=round(imbalance_factor, 2))
                    
                    last_update[symbol] = current_time
            
            # Short sleep to prevent CPU overuse
            await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error("Error in example simulation", error=str(e))
    
    finally:
        # Stop the strategy
        await strategy.stop()
        logger.info("Market Imbalance Strategy example completed", signal_count=signal_count)


def main():
    """Main entry point for the example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main() 