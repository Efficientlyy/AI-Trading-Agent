"""Example script demonstrating how to use the Volume Absorption Strategy.

This script shows how to set up and run the Volume Absorption Strategy, which
detects when large orders are absorbed by the market without significant
price impact, suggesting strong market conviction in a direction.
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import config
from src.common.events import EventBus
from src.common.logging import get_logger
from src.models.events import EventType
from src.models.market_data import TimeFrame, TradeData
from src.models.signals import Signal, SignalType
from src.rust_bridge import OrderBookProcessor, create_order_book_processor
from src.strategy.volume_absorption import VolumeAbsorptionStrategy
from examples.orderbook_example import generate_sample_orderbook, generate_order_book_update


logger = get_logger("examples", "volume_absorption")


def generate_large_order(base_price: float, side: str, size: float = None) -> Dict[str, Any]:
    """Generate a large order for testing.
    
    Args:
        base_price: The base price to use for the order
        side: The side of the order ('buy' or 'sell')
        size: Optional size override, otherwise random size is generated
    
    Returns:
        A dictionary with order details
    """
    # Generate a price near the base price
    price_offset = random.uniform(-0.1, 0.1) * base_price / 100.0  # Within 0.1%
    price = base_price + price_offset
    
    # Generate a random size if not provided
    if size is None:
        size = random.uniform(5.0, 15.0)  # Between 5 and 15 units
    
    return {
        "price": price,
        "size": size,
        "side": side,
        "timestamp": datetime.now()
    }


def simulate_large_order_absorption(
    processor: OrderBookProcessor,
    large_order: Dict[str, Any],
    absorption_pct: float,
    price_impact_pct: float
) -> List[Dict[str, Any]]:
    """Simulate the absorption of a large order.
    
    Args:
        processor: The order book processor
        large_order: The large order to absorb
        absorption_pct: The percentage of the order to absorb
        price_impact_pct: The percentage price impact
    
    Returns:
        A list of order book updates to simulate the absorption
    """
    updates = []
    side = large_order["side"]
    price = large_order["price"]
    size = large_order["size"]
    
    # Add the large order first
    updates.append({
        "type": side,
        "price": price,
        "size": size
    })
    
    # Calculate how much to absorb
    size_to_absorb = size * (absorption_pct / 100.0)
    remaining_size = size - size_to_absorb
    
    # Calculate the new price with impact
    price_impact = price * (price_impact_pct / 100.0)
    if side == "bid" or side == "buy":
        new_price = price + price_impact
    else:
        new_price = price - price_impact
    
    # Update the order with reduced size
    if remaining_size > 0:
        updates.append({
            "type": side,
            "price": price,
            "size": remaining_size
        })
    else:
        # Remove the order completely
        updates.append({
            "type": side,
            "price": price,
            "size": 0.0
        })
    
    # Add some additional market noise
    for _ in range(3):
        noise_side = "bid" if random.random() > 0.5 else "ask"
        noise_price = processor.mid_price() * (1 + random.uniform(-0.2, 0.2) / 100.0)
        noise_size = random.uniform(0.1, 1.0)
        updates.append({
            "type": noise_side,
            "price": noise_price,
            "size": noise_size
        })
    
    return updates


async def run_example() -> None:
    """Run the volume absorption strategy example."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Volume Absorption Strategy example")
    
    # Configure the EventBus
    event_bus = EventBus()
    
    # Sample configuration
    config_data = {
        "strategies": {
            "volume_absorption": {
                "symbols": ["BTC/USD", "ETH/USD"],
                "min_order_size": 5.0,
                "absorption_threshold": 0.8,  # 80% absorption required
                "price_impact_threshold": 0.1,  # Max 0.1% price impact
                "min_trade_size": 0.05,
                "max_trade_size": 0.5,
                "take_profit_pct": 0.7,
                "stop_loss_pct": 0.4,
                "signal_cooldown": 60  # 1 minute (shorter for demo)
            }
        }
    }
    
    # Update the global config
    for key, value in config_data.items():
        if key in config._config and isinstance(config._config[key], dict) and isinstance(value, dict):
            config._config[key].update(value)
        else:
            config._config[key] = value
    
    # Initialize the strategy
    # We need to implement the required methods to make this non-abstract
    class SimulationVolumeAbsorptionStrategy(VolumeAbsorptionStrategy):
        """Extension of VolumeAbsorptionStrategy for simulation purposes."""
        
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
    
    strategy = SimulationVolumeAbsorptionStrategy()
    
    # Subscribe to signals
    signal_count = 0
    
    async def signal_handler(signal_event: Dict) -> None:
        nonlocal signal_count
        signal_count += 1
        signal = Signal.from_dict(signal_event)
        
        # Construct a dictionary of signal properties to log
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
            absorp_ratio = round(signal.metadata.get("absorption_ratio", 0) * 100, 1)
            price_impact = round(signal.metadata.get("price_impact", 0), 2)
            order_size = round(signal.metadata.get("order_size", 0), 2)
            
            logger.info("Signal metadata", 
                     order_size=order_size,
                     absorption_pct=f"{absorp_ratio}%",
                     price_impact=f"{price_impact}%",
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
                 spread=processors[symbol].spread())
    
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
    
    # Regular update intervals
    intervals = {
        "BTC/USD": 0.5,  # updates every 0.5 seconds
        "ETH/USD": 0.8,  # updates every 0.8 seconds
    }
    
    last_update = {symbol: time.time() for symbol in processors.keys()}
    
    # Scenario parameters
    scenarios = [
        # Symbol, time offset, side, size, absorption %, price impact %
        ("BTC/USD", 10, "ask", 10.0, 90, 0.05),  # Strong buy signal
        ("ETH/USD", 20, "bid", 8.0, 85, 0.04),   # Strong sell signal
        ("BTC/USD", 40, "ask", 15.0, 75, 0.12),  # Rejected (price impact too high)
        ("ETH/USD", 60, "bid", 5.0, 60, 0.03),   # Rejected (absorption too low)
        ("BTC/USD", 80, "ask", 12.0, 95, 0.02),  # Strong buy signal
        ("ETH/USD", 100, "ask", 9.0, 88, 0.05),  # Strong buy signal
    ]
    
    # Track which scenarios have been executed
    executed_scenarios = set()
    
    # Run for specified time
    start_time = time.time()
    run_time = 120  # run for 2 minutes
    
    try:
        while not should_exit and (time.time() - start_time) < run_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we need to trigger a scenario
            for i, (symbol, trigger_time, side, size, absorption, impact) in enumerate(scenarios):
                if i not in executed_scenarios and elapsed >= trigger_time:
                    # Create and absorb a large order
                    logger.info(f"--- SCENARIO {i+1}: Large {side} order on {symbol} ---")
                    logger.info(f"Simulating {size} units with {absorption}% absorption and {impact}% price impact")
                    
                    processor = processors[symbol]
                    base_price = processor.best_bid_price() if side == "bid" else processor.best_ask_price()
                    
                    # Create the large order
                    large_order = generate_large_order(base_price, side, size)
                    
                    # Add it to the order book
                    processor.process_updates([{
                        "type": side,
                        "price": large_order["price"],
                        "size": large_order["size"]
                    }])
                    
                    # Give the strategy time to process the new large order
                    await asyncio.sleep(1.5)
                    
                    # Process strategy updates
                    await strategy.analyze_orderbook(symbol, processor)
                    
                    # Now simulate absorption
                    updates = simulate_large_order_absorption(
                        processor=processor,
                        large_order=large_order,
                        absorption_pct=absorption,
                        price_impact_pct=impact
                    )
                    
                    # Process the absorption updates
                    processor.process_updates(updates)
                    
                    # Process strategy updates
                    await strategy.analyze_orderbook(symbol, processor)
                    
                    # Mark scenario as executed
                    executed_scenarios.add(i)
                    
                    # Wait a bit after the scenario
                    await asyncio.sleep(1.0)
            
            # Regular updates between scenarios
            for symbol, processor in processors.items():
                if current_time - last_update[symbol] >= intervals[symbol]:
                    # Generate normal market updates
                    updates = generate_order_book_update(
                        processor=processor,
                        num_updates=random.randint(1, 3)
                    )
                    
                    # Process updates
                    processor.process_updates(updates)
                    
                    # Process order book in strategy
                    await strategy.analyze_orderbook(symbol, processor)
                    
                    last_update[symbol] = current_time
            
            # Short sleep to prevent CPU overuse
            await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error("Error in example simulation", error=str(e))
    
    finally:
        # Stop the strategy
        await strategy.stop()
        logger.info("Volume Absorption Strategy example completed", signal_count=signal_count)


def main():
    """Main entry point for the example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main() 