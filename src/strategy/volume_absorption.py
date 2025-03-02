"""Volume absorption strategy for the AI Crypto Trading System.

This module implements a strategy that detects when large orders are absorbed
by the market without significant price impact, suggesting strong directional
conviction from market participants.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
from collections import deque

from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import TimeFrame, TradeData
from src.models.signals import SignalType
from src.rust_bridge import OrderBookProcessor
from src.strategy.orderbook_strategy import OrderBookStrategy


class VolumeAbsorptionStrategy(OrderBookStrategy):
    """Strategy that detects when large orders are absorbed by the market.
    
    This strategy monitors the order book for significant orders that are filled
    without causing major price movements, which can indicate strong market
    conviction in a direction.
    """
    
    def __init__(self, strategy_id: str = "volume_absorption"):
        """Initialize the volume absorption strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", strategy_id)
        
        # Strategy-specific configuration
        self.min_order_size = config.get(f"strategies.{strategy_id}.min_order_size", 5.0)
        self.absorption_threshold = config.get(f"strategies.{strategy_id}.absorption_threshold", 0.8)
        self.price_impact_threshold = config.get(f"strategies.{strategy_id}.price_impact_threshold", 0.1)
        
        # Position sizing parameters
        self.min_trade_size = config.get(f"strategies.{strategy_id}.min_trade_size", 0.05)
        self.max_trade_size = config.get(f"strategies.{strategy_id}.max_trade_size", 0.5)
        
        # Risk management
        self.take_profit_pct = config.get(f"strategies.{strategy_id}.take_profit_pct", 0.7)
        self.stop_loss_pct = config.get(f"strategies.{strategy_id}.stop_loss_pct", 0.4)
        
        # Signal cooldown period
        self.signal_cooldown = config.get(f"strategies.{strategy_id}.signal_cooldown", 600)  # 10 minutes
        
        # Order book snapshots to track changes
        self.snapshots: Dict[str, Deque[Dict[str, Any]]] = {}
        
        # Signal history
        self.last_signal_time: Dict[str, datetime] = {}
        
        # Track large orders for absorption analysis
        self.large_orders: Dict[str, List[Dict[str, Any]]] = {}
        
        # Window for price stability analysis
        self.price_history: Dict[str, Deque[float]] = {}
        self.price_history_window = 20  # Number of price points to track
    
    async def _orderbook_strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        self.logger.info("Initializing volume absorption strategy",
                     min_order_size=self.min_order_size,
                     absorption_threshold=self.absorption_threshold,
                     price_impact_threshold=self.price_impact_threshold)
        
        # Initialize data structures for each symbol
        for symbol in self.symbols:
            self.snapshots[symbol] = deque(maxlen=5)  # Keep last 5 snapshots
            self.large_orders[symbol] = []
            self.price_history[symbol] = deque(maxlen=self.price_history_window)
    
    async def _orderbook_strategy_start(self) -> None:
        """Strategy-specific startup."""
        self.logger.info("Starting volume absorption strategy")
        
        # Publish status
        await self.publish_status(
            "Volume absorption strategy started",
            {
                "min_order_size": self.min_order_size,
                "absorption_threshold": self.absorption_threshold,
                "price_impact_threshold": self.price_impact_threshold,
                "symbols": list(self.symbols) if self.symbols else "all"
            }
        )
    
    async def _orderbook_strategy_stop(self) -> None:
        """Strategy-specific shutdown."""
        self.logger.info("Stopping volume absorption strategy")
        
        # Clear strategy state
        self.snapshots.clear()
        self.large_orders.clear()
        self.price_history.clear()
        
        # Publish status
        await self.publish_status("Volume absorption strategy stopped")
    
    async def analyze_orderbook(self, symbol: str, processor: OrderBookProcessor) -> None:
        """Analyze an updated order book.
        
        This method detects large orders and monitors their absorption by the market
        without significant price impact.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
        """
        # Skip if we don't have enough data
        if not processor or processor.best_bid_price() <= 0 or processor.best_ask_price() <= 0:
            return
        
        # Get current mid price
        current_mid = processor.mid_price()
        
        # Update price history
        self.price_history[symbol].append(current_mid)
        
        # Take a snapshot of the current order book
        snapshot = self._take_order_book_snapshot(symbol, processor)
        self.snapshots[symbol].append(snapshot)
        
        # We need at least 2 snapshots to detect changes
        if len(self.snapshots[symbol]) < 2:
            return
        
        # Detect large orders by comparing snapshots
        await self._detect_large_orders(symbol, processor)
        
        # Track existing large orders to see if they're being absorbed
        await self._track_order_absorption(symbol, processor)
    
    def _take_order_book_snapshot(self, symbol: str, processor: OrderBookProcessor) -> Dict[str, Any]:
        """Take a snapshot of the current order book state.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
            
        Returns:
            A dictionary with the current order book state
        """
        # Get the current timestamp
        timestamp = datetime.now()
        
        # Get current order book snapshot
        best_bid = processor.best_bid_price()
        best_ask = processor.best_ask_price()
        mid_price = processor.mid_price()
        
        # Get order book depth
        bid_snapshot = []
        ask_snapshot = []
        
        # Get complete order book snapshot
        full_snapshot = processor.snapshot()
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "bids": full_snapshot.get("bids", []),
            "asks": full_snapshot.get("asks", [])
        }
    
    async def _detect_large_orders(self, symbol: str, processor: OrderBookProcessor) -> None:
        """Detect new large orders by comparing snapshots.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
        """
        # Get the current and previous snapshots
        current = self.snapshots[symbol][-1]
        previous = self.snapshots[symbol][-2]
        
        # Check for new large bid orders
        for price, size in current["bids"]:
            # Look for the same price level in previous snapshot
            prev_size = 0.0
            for p, s in previous["bids"]:
                if abs(p - price) < 0.0001:  # Price levels are very close
                    prev_size = s
                    break
            
            # If size increased significantly, it's a new large order
            if size > prev_size and size >= self.min_order_size:
                order_size = size - prev_size
                if order_size >= self.min_order_size:
                    # Record the large order
                    self.large_orders[symbol].append({
                        "timestamp": current["timestamp"],
                        "price": price,
                        "size": order_size,
                        "side": "buy",
                        "initial_mid": current["mid_price"],
                        "absorbed": 0.0,
                        "remaining": order_size,
                        "fully_absorbed": False,
                        "price_impact": 0.0
                    })
                    self.logger.debug("Detected large bid order", 
                                   symbol=symbol, 
                                   price=price, 
                                   size=order_size)
        
        # Check for new large ask orders
        for price, size in current["asks"]:
            # Look for the same price level in previous snapshot
            prev_size = 0.0
            for p, s in previous["asks"]:
                if abs(p - price) < 0.0001:  # Price levels are very close
                    prev_size = s
                    break
            
            # If size increased significantly, it's a new large order
            if size > prev_size and size >= self.min_order_size:
                order_size = size - prev_size
                if order_size >= self.min_order_size:
                    # Record the large order
                    self.large_orders[symbol].append({
                        "timestamp": current["timestamp"],
                        "price": price,
                        "size": order_size,
                        "side": "sell",
                        "initial_mid": current["mid_price"],
                        "absorbed": 0.0,
                        "remaining": order_size,
                        "fully_absorbed": False,
                        "price_impact": 0.0
                    })
                    self.logger.debug("Detected large ask order", 
                                   symbol=symbol, 
                                   price=price, 
                                   size=order_size)
        
        # Limit the number of large orders we track
        if len(self.large_orders[symbol]) > 20:
            # Keep the 10 most recent orders
            self.large_orders[symbol] = self.large_orders[symbol][-10:]
    
    async def _track_order_absorption(self, symbol: str, processor: OrderBookProcessor) -> None:
        """Track the absorption of large orders.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
        """
        if not self.large_orders[symbol]:
            return
        
        # Get the current snapshot
        current = self.snapshots[symbol][-1]
        
        # Track each large order
        remaining_orders = []
        
        for order in self.large_orders[symbol]:
            # Skip orders that are already fully absorbed
            if order["fully_absorbed"]:
                continue
            
            # Check if the order has been removed or reduced
            remaining_size = 0.0
            
            # Check the appropriate side
            if order["side"] == "buy":
                for price, size in current["bids"]:
                    if abs(price - order["price"]) < 0.0001:
                        remaining_size = size
                        break
            else:  # sell
                for price, size in current["asks"]:
                    if abs(price - order["price"]) < 0.0001:
                        remaining_size = size
                        break
            
            # Calculate how much was absorbed
            absorbed_now = max(0.0, order["remaining"] - remaining_size)
            order["absorbed"] += absorbed_now
            order["remaining"] = max(0.0, order["remaining"] - absorbed_now)
            
            # Calculate price impact
            current_mid = current["mid_price"]
            order["price_impact"] = abs(current_mid - order["initial_mid"]) / order["initial_mid"] * 100.0
            
            # Check if the order is fully absorbed
            if order["remaining"] <= 0.0 or remaining_size <= 0.0:
                order["fully_absorbed"] = True
                
                # If a large order was absorbed with minimal price impact, it's a signal
                absorption_ratio = order["absorbed"] / order["size"]
                
                if (absorption_ratio >= self.absorption_threshold and 
                        order["price_impact"] <= self.price_impact_threshold):
                    
                    # Check price stability
                    await self._check_absorption_signal(symbol, processor, order)
            
            # Keep tracking this order if it's not fully absorbed
            if not order["fully_absorbed"]:
                remaining_orders.append(order)
        
        # Update the list with only non-absorbed orders
        self.large_orders[symbol] = remaining_orders
    
    async def _check_absorption_signal(
        self, 
        symbol: str, 
        processor: OrderBookProcessor, 
        order: Dict[str, Any]
    ) -> None:
        """Check if an absorbed order should generate a signal.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
            order: The large order that was absorbed
        """
        # Skip if in cooldown period
        now = datetime.now()
        if symbol in self.last_signal_time:
            cooldown_end = self.last_signal_time[symbol] + timedelta(seconds=self.signal_cooldown)
            if now < cooldown_end:
                return
        
        # Check price stability using the price history
        if len(self.price_history[symbol]) < self.price_history_window:
            return  # Not enough price history
        
        # Calculate price volatility
        prices = list(self.price_history[symbol])
        avg_price = sum(prices) / len(prices)
        volatility = sum(abs(p - avg_price) for p in prices) / avg_price
        
        # Skip if the market is too volatile
        if volatility > 0.005:  # More than 0.5% average deviation
            self.logger.debug("Skipping signal due to high volatility",
                           symbol=symbol,
                           volatility=volatility,
                           order_side=order["side"],
                           absorption_ratio=order["absorbed"] / order["size"])
            return
        
        # Determine the direction of the signal
        # If a large sell order was absorbed, it's bullish (go long)
        # If a large buy order was absorbed, it's bearish (go short)
        direction = "long" if order["side"] == "sell" else "short"
        
        # Calculate the size based on absorption strength
        # More complete absorption gets larger size
        absorption_ratio = order["absorbed"] / order["size"]
        size_multiplier = min(absorption_ratio, 1.0)
        size = self.min_trade_size + (self.max_trade_size - self.min_trade_size) * size_multiplier
        
        # Round to 4 decimal places
        size = round(size, 4)
        
        # Get current price
        current_price = processor.mid_price()
        
        # Calculate take profit and stop loss levels
        take_profit = 0.0
        stop_loss = 0.0
        
        if direction == "long":
            take_profit = current_price * (1 + self.take_profit_pct / 100.0)
            stop_loss = current_price * (1 - self.stop_loss_pct / 100.0)
        else:  # short
            take_profit = current_price * (1 - self.take_profit_pct / 100.0)
            stop_loss = current_price * (1 + self.stop_loss_pct / 100.0)
        
        # Check if market conditions are favorable
        favorable, details = await self.is_favorable_for_trade(symbol, direction, size)
        
        if not favorable:
            self.logger.info("Skipping signal due to unfavorable market conditions", 
                          symbol=symbol, 
                          direction=direction,
                          reason=details.get("reason", "unknown"),
                          order_size=order["size"],
                          absorption_ratio=absorption_ratio)
            return
        
        # Calculate confidence
        # Base confidence from is_favorable_for_trade
        confidence = details["confidence"] * 0.5
        
        # Add absorption factor to confidence (normalized to 0-0.3)
        absorption_confidence = min(absorption_ratio, 1.0) * 0.3
        
        # Add inverse of price impact to confidence (normalized to 0-0.2)
        # Lower price impact = higher confidence
        impact_confidence = (1.0 - min(order["price_impact"] / self.price_impact_threshold, 1.0)) * 0.2
        
        # Combined confidence
        confidence += absorption_confidence + impact_confidence
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        # Generate signal reason
        absorbed_pct = round(absorption_ratio * 100, 1)
        impact_pct = round(order["price_impact"], 2)
        
        if direction == "long":
            reason = f"Large sell order ({order['size']:.2f} units) absorbed ({absorbed_pct}%) with minimal impact ({impact_pct}%)"
        else:
            reason = f"Large buy order ({order['size']:.2f} units) absorbed ({absorbed_pct}%) with minimal impact ({impact_pct}%)"
        
        # Add additional market metrics to metadata
        metadata = {
            "order_size": order["size"],
            "absorption_ratio": absorption_ratio,
            "price_impact": order["price_impact"],
            "order_side": order["side"],
            "order_price": order["price"],
            "order_detected_time": order["timestamp"].isoformat(),
            "market_volatility": volatility,
            "favorable_details": details
        }
        
        # Publish signal
        await self.publish_orderbook_signal(
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            price=current_price,
            size=size,
            confidence=confidence,
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata=metadata
        )
        
        # Update last signal time
        self.last_signal_time[symbol] = now
        
        self.logger.info("Generated volume absorption signal", 
                      symbol=symbol,
                      direction=direction,
                      price=current_price,
                      size=size,
                      confidence=confidence,
                      absorption_ratio=absorption_ratio,
                      price_impact=order["price_impact"],
                      take_profit=take_profit,
                      stop_loss=stop_loss) 