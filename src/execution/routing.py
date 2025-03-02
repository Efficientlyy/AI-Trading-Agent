"""Order Routing System for the AI Crypto Trading System.

This module provides functionality for intelligently routing orders to the most
cost-effective exchanges based on fee structures, trading volumes, and other factors.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum

from src.common.component import Component
from src.common.logging import get_logger
from src.models.order import Order, OrderType, OrderSide
from src.fees.models import FeeType, FeeEstimate
from src.fees.service import FeeManager


class RoutingCriteria(Enum):
    """Criteria used for routing decisions."""
    LOWEST_FEE = "lowest_fee"
    BEST_LIQUIDITY = "best_liquidity"
    LOWEST_LATENCY = "lowest_latency"
    BALANCED = "balanced"


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


class OrderRouter(Component):
    """Service for routing orders to the most appropriate exchanges.
    
    This component analyzes fee structures, liquidity, latency, and other factors
    to determine the optimal exchange for executing orders.
    """
    
    def __init__(self, fee_manager: Optional[FeeManager] = None):
        """Initialize the order router.
        
        Args:
            fee_manager: Optional FeeManager instance. If None, one will be created.
        """
        super().__init__("order_router")
        self.logger = get_logger("execution", "router")
        self.fee_manager = fee_manager or FeeManager()
        self.available_exchanges: Set[str] = set()
        self.exchange_latency_ms: Dict[str, float] = {}
        self.exchange_reliability: Dict[str, float] = {}
        self.exchange_liquidity_scores: Dict[str, Dict[str, float]] = {}  # Exchange -> {Asset -> Score}
        
    async def _initialize(self) -> None:
        """Initialize the order router."""
        self.logger.info("Initializing order router")
        
        # Load configuration
        self.default_criteria = self.get_config("default_routing_criteria", RoutingCriteria.BALANCED.value)
        
        # Register available exchanges
        exchanges_config = self.get_config("exchanges", {})
        for exchange_id, exchange_config in exchanges_config.items():
            if exchange_config.get("enabled", False):
                self.available_exchanges.add(exchange_id)
                self.exchange_latency_ms[exchange_id] = exchange_config.get("avg_latency_ms", 100.0)
                self.exchange_reliability[exchange_id] = exchange_config.get("reliability", 0.99)
                
        self.logger.info(f"Registered {len(self.available_exchanges)} available exchanges")
        
        # Make sure the fee manager's data directory exists
        # No need to explicitly initialize the FeeManager as it doesn't have an async initialize method
            
        # Subscribe to liquidity updates
        # TODO: Implement subscription to market data for liquidity scoring
        
    async def route_order(self, order: Order, criteria: Optional[RoutingCriteria] = None) -> RoutingDecision:
        """Route an order to the most appropriate exchange.
        
        Args:
            order: The order to route
            criteria: The routing criteria to use, or None to use default
            
        Returns:
            A RoutingDecision with the recommended exchange and alternatives
        """
        actual_criteria = criteria or RoutingCriteria(self.default_criteria)
        
        # Safely access order attributes
        order_id = getattr(order, "order_id", f"unknown-{id(order)}")
        base_asset = getattr(order, "base_asset", "")
        quote_asset = getattr(order, "quote_asset", "")
        quantity = getattr(order, "quantity", 0.0)
        price = getattr(order, "price", 0.0)
        
        self.logger.info(f"Routing order {order_id} using criteria: {actual_criteria.value}")
        
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
                    asset=quote_asset,  # Assuming fee is in quote currency
                    amount=quantity,
                    price=price or 0.0,  # Use 0 for market orders without price
                )
                
                score.estimated_fee = fee_estimate
                
                # Lower fee = higher score (inverse relationship)
                fee_usd_value = 0.0
                if fee_estimate and hasattr(fee_estimate, 'usd_value'):
                    fee_usd_value = float(fee_estimate.usd_value or 0.0)
                
                if fee_usd_value > 0:
                    score.fee_score = 100.0 / fee_usd_value
                else:
                    score.fee_score = 100.0  # Assume zero fee is best
                    
            except Exception as e:
                self.logger.warning(f"Failed to estimate fee for {exchange_id}: {e}")
                score.fee_score = 0.0
            
            # Calculate liquidity score
            pair = f"{base_asset}/{quote_asset}"
            score.liquidity_score = self.exchange_liquidity_scores.get(exchange_id, {}).get(pair, 50.0)
            
            # Calculate latency score - lower is better, so invert
            latency = self.exchange_latency_ms.get(exchange_id, 100.0)
            score.latency_score = 100.0 * (1000.0 / (latency + 10.0))  # +10 to avoid division by zero
            
            # Calculate reliability score
            score.reliability_score = 100.0 * self.exchange_reliability.get(exchange_id, 0.5)
            
            # Calculate total score based on criteria
            if actual_criteria == RoutingCriteria.LOWEST_FEE:
                weights = {"fee": 0.7, "liquidity": 0.1, "latency": 0.1, "reliability": 0.1}
            elif actual_criteria == RoutingCriteria.BEST_LIQUIDITY:
                weights = {"fee": 0.1, "liquidity": 0.7, "latency": 0.1, "reliability": 0.1}
            elif actual_criteria == RoutingCriteria.LOWEST_LATENCY:
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
            self.logger.warning(f"No suitable exchange found for order {order_id}")
            # Return a default decision if no exchanges are available
            return RoutingDecision(
                order_id=order_id,
                recommended_exchange="",
                alternative_exchanges=[],
                estimated_fee=None,
                estimated_savings=0.0,
                criteria_used=actual_criteria,
                exchange_scores={}
            )
        
        # Select the recommended exchange (highest score)
        recommended_exchange_id = sorted_exchanges[0][0]
        recommended_score = sorted_exchanges[0][1]
        
        # Calculate estimated savings (comparing to average of alternatives)
        avg_fee = 0.0
        alt_count = 0
        
        for exchange_id, score in sorted_exchanges[1:]:
            fee_usd_value = 0.0
            if score.estimated_fee and hasattr(score.estimated_fee, 'usd_value'):
                fee_usd_value = float(score.estimated_fee.usd_value or 0.0)
                
            if fee_usd_value > 0:
                avg_fee += fee_usd_value
                alt_count += 1
        
        estimated_savings = 0.0
        if alt_count > 0:
            avg_fee /= alt_count
            if recommended_score.estimated_fee and hasattr(recommended_score.estimated_fee, 'usd_value'):
                recommended_fee = float(recommended_score.estimated_fee.usd_value or 0.0)
                estimated_savings = avg_fee - recommended_fee
        
        # Create the routing decision
        decision = RoutingDecision(
            order_id=order_id,
            recommended_exchange=recommended_exchange_id,
            alternative_exchanges=[ex_id for ex_id, _ in sorted_exchanges[1:]],
            estimated_fee=recommended_score.estimated_fee,
            estimated_savings=max(0.0, estimated_savings),
            criteria_used=actual_criteria,
            exchange_scores=exchange_scores
        )
        
        self.logger.info(
            f"Routing decision for order {order_id}: "
            f"Recommended exchange: {decision.recommended_exchange}, "
            f"Estimated savings: ${decision.estimated_savings:.2f}"
        )
        
        return decision
    
    async def update_liquidity_scores(self, exchange_id: str, asset_pair: str, score: float) -> None:
        """Update the liquidity score for an exchange and asset pair.
        
        Args:
            exchange_id: The exchange ID
            asset_pair: The asset pair (e.g., "BTC/USDT")
            score: The liquidity score (0-100)
        """
        if exchange_id not in self.exchange_liquidity_scores:
            self.exchange_liquidity_scores[exchange_id] = {}
        
        self.exchange_liquidity_scores[exchange_id][asset_pair] = score
        self.logger.debug(f"Updated liquidity score for {exchange_id} {asset_pair}: {score}")
    
    async def update_exchange_latency(self, exchange_id: str, latency_ms: float) -> None:
        """Update the average latency for an exchange.
        
        Args:
            exchange_id: The exchange ID
            latency_ms: The average latency in milliseconds
        """
        self.exchange_latency_ms[exchange_id] = latency_ms
        self.logger.debug(f"Updated latency for {exchange_id}: {latency_ms}ms")
    
    async def update_exchange_reliability(self, exchange_id: str, reliability: float) -> None:
        """Update the reliability score for an exchange.
        
        Args:
            exchange_id: The exchange ID
            reliability: The reliability score (0-1)
        """
        self.exchange_reliability[exchange_id] = reliability
        self.logger.debug(f"Updated reliability for {exchange_id}: {reliability}") 