#!/usr/bin/env python
"""Example script demonstrating the Order Routing System.

This script shows how the Order Routing System works by:
1. Setting up fee schedules for multiple exchanges
2. Creating sample orders
3. Using the OrderRouter to determine the optimal exchange for each order
4. Displaying the routing decisions and estimated savings
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, cast
from pathlib import Path

from src.fees.models import FeeCalculationType, FeeType, FeeTier
from src.fees.service import FeeManager
from src.execution.routing import OrderRouter, RoutingCriteria, ExchangeScore, RoutingDecision
from src.models.order import Order, OrderType, OrderSide
from src.common.component import Component


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("order_routing_example")


# Create a concrete implementation of OrderRouter for our example
class SimpleOrderRouter(OrderRouter):
    """Simple implementation of OrderRouter for the example."""
    
    async def _start(self) -> None:
        """Start the router."""
        pass
    
    async def _stop(self) -> None:
        """Stop the router."""
        pass


# Create an Order subclass for our demo
class DemoOrder:
    """Demo order that adapts to Router's expectations."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with dictionary data."""
        self.order_id = data.get("order_id", "")
        self.exchange_id = data.get("exchange_id")
        self.base_asset = data.get("base_asset", "")
        self.quote_asset = data.get("quote_asset", "")
        self.quantity = data.get("quantity", 0.0)
        self.price = data.get("price")
        self.order_type = data.get("order_type", OrderType.MARKET)
        self.side = data.get("side", OrderSide.BUY)
        self.timestamp = data.get("timestamp", datetime.now())
        
        # Additional properties expected by routing code
        self.exchange = data.get("exchange_id", "")
        self.symbol = f"{self.base_asset}/{self.quote_asset}" if self.base_asset and self.quote_asset else ""


async def setup_fee_schedules(fee_manager: FeeManager) -> None:
    """Set up fee schedules for different exchanges using FeeManager's internal data structure.
    
    Args:
        fee_manager: The fee manager to update
    """
    logger.info("Setting up fee schedules for exchanges...")

    # We'll write the fee structures directly to a JSON file
    # that FeeManager can read, since we can't directly assign dictionaries
    fees_file = fee_manager.data_dir / "fee_schedules.json"
    
    # Define our fee schedules
    fee_schedules = {
        "Binance": {
            "default_maker_fee": 0.0010,
            "default_taker_fee": 0.0010,
            "calculation_type": "percentage",
            "updated_at": datetime.now().isoformat(),
            "tiers": [
                {
                    "min_volume": 0.0,
                    "max_volume": 50000.0,
                    "maker_fee": 0.0010,
                    "taker_fee": 0.0010,
                    "description": "VIP 0"
                },
                {
                    "min_volume": 50000.0,
                    "max_volume": 100000.0,
                    "maker_fee": 0.0009,
                    "taker_fee": 0.0010,
                    "description": "VIP 1"
                },
                {
                    "min_volume": 100000.0,
                    "max_volume": 500000.0,
                    "maker_fee": 0.0008,
                    "taker_fee": 0.0010,
                    "description": "VIP 2"
                }
            ]
        },
        "Coinbase": {
            "default_maker_fee": 0.0040,
            "default_taker_fee": 0.0060,
            "calculation_type": "percentage",
            "updated_at": datetime.now().isoformat(),
            "tiers": [
                {
                    "min_volume": 0.0,
                    "max_volume": None,
                    "maker_fee": 0.0040,
                    "taker_fee": 0.0060,
                    "description": "Standard"
                }
            ]
        },
        "Kraken": {
            "default_maker_fee": 0.0016,
            "default_taker_fee": 0.0026,
            "calculation_type": "percentage",
            "updated_at": datetime.now().isoformat(),
            "tiers": [
                {
                    "min_volume": 0.0,
                    "max_volume": 50000.0,
                    "maker_fee": 0.0016,
                    "taker_fee": 0.0026,
                    "description": "Tier 1"
                },
                {
                    "min_volume": 50000.0, 
                    "max_volume": 100000.0,
                    "maker_fee": 0.0014,
                    "taker_fee": 0.0024,
                    "description": "Tier 2"
                }
            ]
        },
        "OKX": {
            "default_maker_fee": 0.0008,
            "default_taker_fee": 0.0010,
            "calculation_type": "percentage",
            "updated_at": datetime.now().isoformat(),
            "tiers": [
                {
                    "min_volume": 0.0,
                    "max_volume": None,
                    "maker_fee": 0.0008,
                    "taker_fee": 0.0010,
                    "description": "Level 1"
                }
            ]
        },
        "Bybit": {
            "default_maker_fee": 0.0001,
            "default_taker_fee": 0.0006,
            "calculation_type": "percentage",
            "updated_at": datetime.now().isoformat(),
            "tiers": [
                {
                    "min_volume": 0.0,
                    "max_volume": None,
                    "maker_fee": 0.0001,
                    "taker_fee": 0.0006,
                    "description": "VIP 0"
                }
            ]
        }
    }
    
    # Write to JSON file
    with open(fees_file, 'w') as f:
        json.dump(fee_schedules, f, indent=2)
    
    # Force reload of fee schedules
    fee_manager._load_fee_schedules()
    
    logger.info(f"Fee schedules set up for {len(fee_schedules)} exchanges")


def create_sample_orders() -> List[Dict[str, Any]]:
    """Create sample orders for routing.
    
    We're using dictionaries with the expected attributes
    rather than actual Order objects to avoid model restriction issues.
    
    Returns:
        A list of sample order dictionaries
    """
    logger.info("Creating sample orders for routing...")
    
    orders = [
        # Order 1: BTC market buy
        {
            "order_id": "order1",
            "exchange_id": None,  # To be determined by router
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "quantity": 0.5,  # 0.5 BTC
            "price": None,  # Market order
            "order_type": OrderType.MARKET,
            "side": OrderSide.BUY,
            "timestamp": datetime.now()
        },
        
        # Order 2: ETH limit sell
        {
            "order_id": "order2",
            "exchange_id": None,  # To be determined by router
            "base_asset": "ETH",
            "quote_asset": "USDT", 
            "quantity": 10.0,  # 10 ETH
            "price": 2000.0,  # $2,000 per ETH
            "order_type": OrderType.LIMIT,
            "side": OrderSide.SELL,
            "timestamp": datetime.now()
        },
        
        # Order 3: Small SOL buy
        {
            "order_id": "order3",
            "exchange_id": None,  # To be determined by router
            "base_asset": "SOL",
            "quote_asset": "USDT",
            "quantity": 2.0,  # 2 SOL
            "price": 60.0,  # $60 per SOL
            "order_type": OrderType.LIMIT,
            "side": OrderSide.BUY,
            "timestamp": datetime.now()
        },
        
        # Order 4: Large BTC sell
        {
            "order_id": "order4",
            "exchange_id": None,  # To be determined by router
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "quantity": 5.0,  # 5 BTC
            "price": 40000.0,  # $40,000 per BTC
            "order_type": OrderType.LIMIT,
            "side": OrderSide.SELL,
            "timestamp": datetime.now()
        },
    ]
    
    logger.info(f"Created {len(orders)} sample orders")
    return orders


async def setup_exchange_metrics(router: SimpleOrderRouter) -> None:
    """Set up exchange metrics for the router.
    
    Args:
        router: The order router to update
    """
    logger.info("Setting up exchange metrics (latency, reliability, liquidity)...")
    
    # Set up exchange latency metrics (in milliseconds)
    await router.update_exchange_latency("Binance", 45.0)   # 45ms
    await router.update_exchange_latency("Coinbase", 75.0)  # 75ms
    await router.update_exchange_latency("Kraken", 90.0)    # 90ms
    await router.update_exchange_latency("OKX", 60.0)       # 60ms
    await router.update_exchange_latency("Bybit", 55.0)     # 55ms
    
    # Set up exchange reliability scores (0-1)
    await router.update_exchange_reliability("Binance", 0.995)   # 99.5%
    await router.update_exchange_reliability("Coinbase", 0.990)  # 99.0%
    await router.update_exchange_reliability("Kraken", 0.985)    # 98.5%
    await router.update_exchange_reliability("OKX", 0.980)       # 98.0%
    await router.update_exchange_reliability("Bybit", 0.975)     # 97.5%
    
    # Set up liquidity scores for different pairs (0-100)
    # BTC/USDT liquidity
    await router.update_liquidity_scores("Binance", "BTC/USDT", 95.0)
    await router.update_liquidity_scores("Coinbase", "BTC/USDT", 85.0)
    await router.update_liquidity_scores("Kraken", "BTC/USDT", 80.0)
    await router.update_liquidity_scores("OKX", "BTC/USDT", 75.0)
    await router.update_liquidity_scores("Bybit", "BTC/USDT", 70.0)
    
    # ETH/USDT liquidity
    await router.update_liquidity_scores("Binance", "ETH/USDT", 90.0)
    await router.update_liquidity_scores("Coinbase", "ETH/USDT", 85.0)
    await router.update_liquidity_scores("Kraken", "ETH/USDT", 75.0)
    await router.update_liquidity_scores("OKX", "ETH/USDT", 70.0)
    await router.update_liquidity_scores("Bybit", "ETH/USDT", 65.0)
    
    # SOL/USDT liquidity
    await router.update_liquidity_scores("Binance", "SOL/USDT", 80.0)
    await router.update_liquidity_scores("Coinbase", "SOL/USDT", 70.0)
    await router.update_liquidity_scores("Kraken", "SOL/USDT", 60.0)
    await router.update_liquidity_scores("OKX", "SOL/USDT", 65.0)
    await router.update_liquidity_scores("Bybit", "SOL/USDT", 60.0)
    
    logger.info("Exchange metrics set up successfully")


async def route_orders_with_different_criteria(
    router: SimpleOrderRouter, orders: List[Dict[str, Any]]
) -> None:
    """Route orders using different routing criteria.
    
    Args:
        router: The order router to use
        orders: The orders to route (as dictionaries)
    """
    criteria_list = [
        RoutingCriteria.LOWEST_FEE,
        RoutingCriteria.BEST_LIQUIDITY,
        RoutingCriteria.LOWEST_LATENCY,
        RoutingCriteria.BALANCED
    ]
    
    for order_data in orders:
        # Create a demo order object from the dictionary
        order = DemoOrder(order_data)
        
        logger.info(f"\n{'='*80}\nRouting Order {order.order_id}: {order.quantity} {order.base_asset} at {order.price or 'MARKET'} {order.quote_asset}")
        logger.info(f"Order Type: {order.order_type.name}, Side: {order.side.name}")
        
        for criteria in criteria_list:
            # Cast our DemoOrder to Order for type compatibility with the router
            decision = await router.route_order(cast(Order, order), criteria)
            
            logger.info(f"\nRouting Criteria: {criteria.name}")
            logger.info(f"Recommended Exchange: {decision.recommended_exchange}")
            
            if decision.estimated_fee:
                fee_amount = getattr(decision.estimated_fee, 'estimated_amount', 'N/A')
                fee_asset = getattr(decision.estimated_fee, 'asset', 'N/A')
                fee_usd = getattr(decision.estimated_fee, 'usd_value', 0.0)
                logger.info(f"Estimated Fee: {fee_amount} {fee_asset} (${fee_usd:.2f})")
            else:
                logger.info("Estimated Fee: Not available")
                
            logger.info(f"Estimated Savings: ${decision.estimated_savings:.2f}")
            
            if decision.alternative_exchanges:
                logger.info(f"Alternative Exchanges: {', '.join(decision.alternative_exchanges[:3])}")
            else:
                logger.info("Alternative Exchanges: None")


async def main() -> None:
    """Run the order routing example."""
    start_time = datetime.now()
    logger.info(f"Starting order routing example at {start_time}")
    
    # Create fee manager
    data_dir = Path("./data/fees")
    data_dir.mkdir(parents=True, exist_ok=True)
    fee_manager = FeeManager(data_dir=data_dir)
    
    # Set up fee schedules
    await setup_fee_schedules(fee_manager)
    
    # Create order router
    router = SimpleOrderRouter(fee_manager=fee_manager)
    
    # Manually register available exchanges since we're not using config
    router.available_exchanges = {"Binance", "Coinbase", "Kraken", "OKX", "Bybit"}
    
    # Set up exchange metrics
    await setup_exchange_metrics(router)
    
    # Create sample orders
    orders = create_sample_orders()
    
    # Route orders with different criteria
    await route_orders_with_different_criteria(router, orders)
    
    end_time = datetime.now()
    logger.info(f"\nOrder routing example completed at {end_time}")
    logger.info(f"Total time: {(end_time - start_time).total_seconds():.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main()) 