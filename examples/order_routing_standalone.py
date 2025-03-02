#!/usr/bin/env python
"""Standalone order routing example.

This is a self-contained version of the Order Routing System that doesn't
depend on the problematic imports. It demonstrates the core concepts of:
1. Fee management
2. Order routing based on various criteria
3. Exchange selection optimization
"""

import asyncio
import json
import logging
import sys
import os
import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

# Add the parent directory to the Python path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout  # Force output to stdout
)
logger = logging.getLogger("order_routing_standalone")


# --- Enums ---
class OrderType(str, enum.Enum):
    """Type of order."""
    MARKET = "market"  # Market order (executed immediately at market price)
    LIMIT = "limit"    # Limit order (executed at specified price or better)


class OrderSide(str, enum.Enum):
    """Side of an order."""
    BUY = "buy"      # Buy order
    SELL = "sell"    # Sell order


class FeeType(str, enum.Enum):
    """Types of fees charged by exchanges."""
    MAKER = "maker"
    TAKER = "taker"


class RoutingCriteria(enum.Enum):
    """Criteria used for routing decisions."""
    LOWEST_FEE = "lowest_fee"
    BEST_LIQUIDITY = "best_liquidity"
    LOWEST_LATENCY = "lowest_latency"
    BALANCED = "balanced"


# --- Data Models ---
@dataclass
class Order:
    """Simple order model."""
    order_id: str
    exchange_id: Optional[str]
    base_asset: str
    quote_asset: str
    quantity: float
    price: Optional[float]
    order_type: OrderType
    side: OrderSide
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FeeEstimate:
    """Estimate of fee for a transaction."""
    exchange_id: str
    fee_type: FeeType
    asset: str
    rate: float
    estimated_amount: float
    usd_value: float


@dataclass
class ExchangeScore:
    """Score for an exchange based on routing criteria."""
    exchange_id: str
    fee_score: float = 0.0
    liquidity_score: float = 0.0
    latency_score: float = 0.0
    reliability_score: float = 0.0
    total_score: float = 0.0
    estimated_fee: Optional[FeeEstimate] = None


@dataclass
class RoutingDecision:
    """Decision for routing an order."""
    order_id: str
    recommended_exchange: str
    alternative_exchanges: List[str]
    estimated_fee: Optional[FeeEstimate]
    estimated_savings: float
    criteria_used: RoutingCriteria
    exchange_scores: Dict[str, ExchangeScore]


# --- Fee Manager ---
class FeeManager:
    """Simplified fee manager for the standalone example."""
    
    def __init__(self):
        """Initialize the fee manager."""
        self.fee_schedules = {
            "Binance": {
                "maker_fee": 0.0010,  # 0.10%
                "taker_fee": 0.0010,  # 0.10%
            },
            "Coinbase": {
                "maker_fee": 0.0040,  # 0.40%
                "taker_fee": 0.0060,  # 0.60%
            },
            "Kraken": {
                "maker_fee": 0.0016,  # 0.16%
                "taker_fee": 0.0026,  # 0.26%
            },
            "OKX": {
                "maker_fee": 0.0008,  # 0.08%
                "taker_fee": 0.0010,  # 0.10%
            },
            "Bybit": {
                "maker_fee": 0.0001,  # 0.01%
                "taker_fee": 0.0006,  # 0.06%
            }
        }
        logger.info(f"Fee manager initialized with {len(self.fee_schedules)} exchanges")
    
    def estimate_fee(self, exchange_id: str, fee_type: FeeType, asset: str, amount: float, price: float) -> FeeEstimate:
        """Estimate the fee for a transaction.
        
        Args:
            exchange_id: The exchange ID
            fee_type: The type of fee (maker or taker)
            asset: The asset to estimate fee for
            amount: The amount of the transaction
            price: The price of the transaction
            
        Returns:
            A FeeEstimate object
        """
        if exchange_id not in self.fee_schedules:
            raise ValueError(f"Unknown exchange: {exchange_id}")
            
        # Get the rate based on fee type
        rate_key = f"{fee_type.value}_fee"
        rate = self.fee_schedules[exchange_id].get(rate_key, 0.001)  # Default to 0.1%
        
        # Calculate the fee amount
        transaction_value = amount * price if price else 0.0
        fee_amount = transaction_value * rate
        
        # Create the fee estimate
        return FeeEstimate(
            exchange_id=exchange_id,
            fee_type=fee_type,
            asset=asset,
            rate=rate,
            estimated_amount=fee_amount,
            usd_value=fee_amount  # Simplified - assuming USD or stablecoin
        )


# --- Order Router ---
class OrderRouter:
    """Simplified order router for the standalone example."""
    
    def __init__(self, fee_manager: FeeManager):
        """Initialize the order router.
        
        Args:
            fee_manager: FeeManager instance to use for fee estimations
        """
        self.fee_manager = fee_manager
        self.available_exchanges: Set[str] = set()
        self.exchange_latency_ms: Dict[str, float] = {}
        self.exchange_reliability: Dict[str, float] = {}
        self.exchange_liquidity_scores: Dict[str, Dict[str, float]] = {}  # Exchange -> {Asset -> Score}
        logger.info("Order router initialized")
    
    async def route_order(self, order: Order, criteria: RoutingCriteria) -> RoutingDecision:
        """Route an order to the most appropriate exchange.
        
        Args:
            order: The order to route
            criteria: The routing criteria to use
            
        Returns:
            A RoutingDecision with the recommended exchange and alternatives
        """
        logger.info(f"Routing order {order.order_id} using criteria: {criteria.value}")
        
        exchange_scores: Dict[str, ExchangeScore] = {}
        
        # Calculate scores for each available exchange
        for exchange_id in self.available_exchanges:
            score = ExchangeScore(exchange_id=exchange_id)
            
            # Calculate fee score
            fee_type = FeeType.TAKER if order.order_type == OrderType.MARKET else FeeType.MAKER
            
            try:
                # Estimate fee for this exchange
                fee_estimate = self.fee_manager.estimate_fee(
                    exchange_id=exchange_id,
                    fee_type=fee_type,
                    asset=order.quote_asset,  # Assuming fee is in quote currency
                    amount=order.quantity,
                    price=order.price or 0.0,  # Use 0 for market orders without price
                )
                
                score.estimated_fee = fee_estimate
                
                # Lower fee = higher score (inverse relationship)
                fee_usd_value = fee_estimate.usd_value
                
                if fee_usd_value > 0:
                    score.fee_score = 100.0 / fee_usd_value
                else:
                    score.fee_score = 100.0  # Assume zero fee is best
                    
            except Exception as e:
                logger.warning(f"Failed to estimate fee for {exchange_id}: {e}")
                score.fee_score = 0.0
            
            # Calculate liquidity score
            pair = f"{order.base_asset}/{order.quote_asset}"
            score.liquidity_score = self.exchange_liquidity_scores.get(exchange_id, {}).get(pair, 50.0)
            
            # Calculate latency score - lower is better, so invert
            latency = self.exchange_latency_ms.get(exchange_id, 100.0)
            score.latency_score = 100.0 * (1000.0 / (latency + 10.0))  # +10 to avoid division by zero
            
            # Calculate reliability score
            score.reliability_score = 100.0 * self.exchange_reliability.get(exchange_id, 0.5)
            
            # Calculate total score based on criteria
            if criteria == RoutingCriteria.LOWEST_FEE:
                weights = {"fee": 0.7, "liquidity": 0.1, "latency": 0.1, "reliability": 0.1}
            elif criteria == RoutingCriteria.BEST_LIQUIDITY:
                weights = {"fee": 0.1, "liquidity": 0.7, "latency": 0.1, "reliability": 0.1}
            elif criteria == RoutingCriteria.LOWEST_LATENCY:
                weights = {"fee": 0.1, "liquidity": 0.1, "latency": 0.7, "reliability": 0.1}
            else:  # Balanced
                weights = {"fee": 0.25, "liquidity": 0.25, "latency": 0.25, "reliability": 0.25}
                
            score.total_score = (
                score.fee_score * weights["fee"] +
                score.liquidity_score * weights["liquidity"] +
                score.latency_score * weights["latency"] +
                score.reliability_score * weights["reliability"]
            )
            
            exchange_scores[exchange_id] = score
        
        # Sort exchanges by total score
        sorted_exchanges = sorted(
            exchange_scores.items(), 
            key=lambda x: x[1].total_score, 
            reverse=True
        )
        
        if not sorted_exchanges:
            logger.warning(f"No suitable exchange found for order {order.order_id}")
            # Return a default decision if no exchanges are available
            return RoutingDecision(
                order_id=order.order_id,
                recommended_exchange="",
                alternative_exchanges=[],
                estimated_fee=None,
                estimated_savings=0.0,
                criteria_used=criteria,
                exchange_scores={}
            )
        
        # Select the recommended exchange (highest score)
        recommended_exchange_id = sorted_exchanges[0][0]
        recommended_score = sorted_exchanges[0][1]
        
        # Calculate estimated savings (comparing to average of alternatives)
        avg_fee = 0.0
        alt_count = 0
        
        for exchange_id, score in sorted_exchanges[1:]:
            if score.estimated_fee and score.estimated_fee.usd_value > 0:
                avg_fee += score.estimated_fee.usd_value
                alt_count += 1
        
        estimated_savings = 0.0
        if alt_count > 0:
            avg_fee /= alt_count
            if recommended_score.estimated_fee:
                estimated_savings = avg_fee - recommended_score.estimated_fee.usd_value
        
        # Create the routing decision
        decision = RoutingDecision(
            order_id=order.order_id,
            recommended_exchange=recommended_exchange_id,
            alternative_exchanges=[ex_id for ex_id, _ in sorted_exchanges[1:]],
            estimated_fee=recommended_score.estimated_fee,
            estimated_savings=max(0.0, estimated_savings),
            criteria_used=criteria,
            exchange_scores=exchange_scores
        )
        
        return decision
    
    async def update_liquidity_scores(self, exchange_id: str, asset_pair: str, score: float) -> None:
        """Update the liquidity score for an exchange and asset pair."""
        if exchange_id not in self.exchange_liquidity_scores:
            self.exchange_liquidity_scores[exchange_id] = {}
        
        self.exchange_liquidity_scores[exchange_id][asset_pair] = score
        logger.debug(f"Updated liquidity score for {exchange_id} {asset_pair}: {score}")
    
    async def update_exchange_latency(self, exchange_id: str, latency_ms: float) -> None:
        """Update the average latency for an exchange."""
        self.exchange_latency_ms[exchange_id] = latency_ms
        logger.debug(f"Updated latency for {exchange_id}: {latency_ms}ms")
    
    async def update_exchange_reliability(self, exchange_id: str, reliability: float) -> None:
        """Update the reliability score for an exchange."""
        self.exchange_reliability[exchange_id] = reliability
        logger.debug(f"Updated reliability for {exchange_id}: {reliability}")


# --- Example Functions ---
def create_sample_orders() -> List[Order]:
    """Create sample orders for routing."""
    logger.info("Creating sample orders for routing...")
    
    orders = [
        # Order 1: BTC market buy
        Order(
            order_id="order1",
            exchange_id=None,  # To be determined by router
            base_asset="BTC",
            quote_asset="USDT",
            quantity=0.5,  # 0.5 BTC
            price=40000.0,  # Use a price for fee calculation even for market orders
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
        ),
        
        # Order 2: ETH limit sell
        Order(
            order_id="order2",
            exchange_id=None,  # To be determined by router
            base_asset="ETH",
            quote_asset="USDT", 
            quantity=10.0,  # 10 ETH
            price=2000.0,  # $2,000 per ETH
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
        ),
        
        # Order 3: Small SOL buy
        Order(
            order_id="order3",
            exchange_id=None,  # To be determined by router
            base_asset="SOL",
            quote_asset="USDT",
            quantity=2.0,  # 2 SOL
            price=60.0,  # $60 per SOL
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
        ),
        
        # Order 4: Large BTC sell
        Order(
            order_id="order4",
            exchange_id=None,  # To be determined by router
            base_asset="BTC",
            quote_asset="USDT",
            quantity=5.0,  # 5 BTC
            price=40000.0,  # $40,000 per BTC
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
        ),
    ]
    
    logger.info(f"Created {len(orders)} sample orders")
    return orders


async def setup_exchange_metrics(router: OrderRouter) -> None:
    """Set up exchange metrics for the router."""
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
    router: OrderRouter, orders: List[Order]
) -> None:
    """Route orders using different routing criteria."""
    criteria_list = [
        RoutingCriteria.LOWEST_FEE,
        RoutingCriteria.BEST_LIQUIDITY,
        RoutingCriteria.LOWEST_LATENCY,
        RoutingCriteria.BALANCED
    ]
    
    for order in orders:
        logger.info(f"\n{'='*80}\nRouting Order {order.order_id}: {order.quantity} {order.base_asset} at {order.price or 'MARKET'} {order.quote_asset}")
        logger.info(f"Order Type: {order.order_type.value}, Side: {order.side.value}")
        
        for criteria in criteria_list:
            decision = await router.route_order(order, criteria)
            
            logger.info(f"\nRouting Criteria: {criteria.name}")
            logger.info(f"Recommended Exchange: {decision.recommended_exchange}")
            
            if decision.estimated_fee:
                fee_amount = decision.estimated_fee.estimated_amount
                fee_asset = decision.estimated_fee.asset
                fee_usd = decision.estimated_fee.usd_value
                logger.info(f"Estimated Fee: {fee_amount:.6f} {fee_asset} (${fee_usd:.2f})")
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
    logger.info(f"Starting order routing standalone example at {start_time}")
    
    # Create fee manager
    fee_manager = FeeManager()
    
    # Create order router
    router = OrderRouter(fee_manager=fee_manager)
    
    # Manually register available exchanges
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