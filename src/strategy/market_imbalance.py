"""Market imbalance strategy for the AI Crypto Trading System.

This module implements a strategy that detects imbalances in the order book
and generates trading signals based on these imbalances.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any

from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import TimeFrame, TradeData
from src.models.signals import SignalType
from src.rust_bridge import OrderBookProcessor
from src.strategy.orderbook_strategy import OrderBookStrategy


class MarketImbalanceStrategy(OrderBookStrategy):
    """Strategy that generates signals based on order book imbalances.
    
    This strategy monitors the order book for significant imbalances between
    buy and sell sides, which can indicate short-term price movements.
    """
    
    def __init__(self, strategy_id: str = "market_imbalance"):
        """Initialize the market imbalance strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "market_imbalance")
        
        # Strategy-specific configuration
        self.imbalance_threshold = config.get(f"strategies.{strategy_id}.imbalance_threshold", 1.5)
        self.depth_shallow = config.get(f"strategies.{strategy_id}.depth_shallow", 5)
        self.depth_medium = config.get(f"strategies.{strategy_id}.depth_medium", 10)
        self.depth_deep = config.get(f"strategies.{strategy_id}.depth_deep", 20)
        
        # Minimum trade size
        self.min_trade_size = config.get(f"strategies.{strategy_id}.min_trade_size", 0.01)
        self.max_trade_size = config.get(f"strategies.{strategy_id}.max_trade_size", 1.0)
        
        # Take profit and stop loss settings
        self.take_profit_pct = config.get(f"strategies.{strategy_id}.take_profit_pct", 0.5)  # 0.5%
        self.stop_loss_pct = config.get(f"strategies.{strategy_id}.stop_loss_pct", 0.3)  # 0.3%
        
        # Signal cooldown period (to avoid signal spamming)
        self.signal_cooldown = config.get(f"strategies.{strategy_id}.signal_cooldown", 300)  # 5 minutes
        
        # Strategy state
        self.last_signal_time: Dict[str, datetime] = {}
        self.historical_imbalances: Dict[str, List[Tuple[datetime, float, float, float]]] = {}
        
        # Imbalance settings for signal generation
        self.shallow_weight = 0.5
        self.medium_weight = 0.3
        self.deep_weight = 0.2
        
        # Strategy indicators
        self.vwap_spreads: Dict[str, List[float]] = {}
        self.imbalance_ema: Dict[str, float] = {}
        self.ema_alpha = 0.1  # EMA smoothing factor
    
    async def _orderbook_strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        self.logger.info("Initializing market imbalance strategy",
                      imbalance_threshold=self.imbalance_threshold,
                      depths=[self.depth_shallow, self.depth_medium, self.depth_deep])
        
        # Initialize historical imbalances for each symbol
        for symbol in self.symbols:
            self.historical_imbalances[symbol] = []
            self.vwap_spreads[symbol] = []
            self.imbalance_ema[symbol] = 1.0  # Start at neutral
    
    async def _orderbook_strategy_start(self) -> None:
        """Strategy-specific startup."""
        self.logger.info("Starting market imbalance strategy")
        
        # Publish status
        await self.publish_status(
            "Market imbalance strategy started",
            {
                "imbalance_threshold": self.imbalance_threshold,
                "depths": [self.depth_shallow, self.depth_medium, self.depth_deep],
                "symbols": list(self.symbols) if self.symbols else "all"
            }
        )
    
    async def _orderbook_strategy_stop(self) -> None:
        """Strategy-specific shutdown."""
        self.logger.info("Stopping market imbalance strategy")
        
        # Clear strategy state
        self.historical_imbalances.clear()
        self.vwap_spreads.clear()
        self.imbalance_ema.clear()
        
        # Publish status
        await self.publish_status("Market imbalance strategy stopped")
    
    async def analyze_orderbook(self, symbol: str, processor: OrderBookProcessor) -> None:
        """Analyze an updated order book.
        
        This method calculates order book imbalances at different depths and
        generates trading signals when significant imbalances are detected.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
        """
        # Skip if we don't have enough data
        if not processor or processor.best_bid_price() <= 0 or processor.best_ask_price() <= 0:
            return
        
        # Get imbalances at different depths
        shallow_imbalance = processor.book_imbalance(self.depth_shallow)
        medium_imbalance = processor.book_imbalance(self.depth_medium)
        deep_imbalance = processor.book_imbalance(self.depth_deep)
        
        # Update EMA of the imbalance
        if symbol not in self.imbalance_ema:
            self.imbalance_ema[symbol] = medium_imbalance
        else:
            self.imbalance_ema[symbol] = (self.ema_alpha * medium_imbalance + 
                                         (1 - self.ema_alpha) * self.imbalance_ema[symbol])
        
        # Calculate weighted imbalance
        weighted_imbalance = (
            (shallow_imbalance * self.shallow_weight) +
            (medium_imbalance * self.medium_weight) +
            (deep_imbalance * self.deep_weight)
        )
        
        # Get current timestamp
        now = datetime.now()
        
        # Store historical data
        self.historical_imbalances[symbol].append((
            now, shallow_imbalance, medium_imbalance, deep_imbalance
        ))
        
        # Keep only the last 100 data points
        if len(self.historical_imbalances[symbol]) > 100:
            self.historical_imbalances[symbol] = self.historical_imbalances[symbol][-100:]
        
        # Calculate VWAP spread
        bid_vwap = processor.vwap("buy", self.depth_medium)
        ask_vwap = processor.vwap("sell", self.depth_medium)
        vwap_spread = (ask_vwap - bid_vwap) / ((ask_vwap + bid_vwap) / 2) * 100  # in percent
        
        # Store VWAP spread history
        self.vwap_spreads[symbol].append(vwap_spread)
        if len(self.vwap_spreads[symbol]) > 50:
            self.vwap_spreads[symbol] = self.vwap_spreads[symbol][-50:]
        
        # Log imbalance data periodically
        if len(self.historical_imbalances[symbol]) % 10 == 0:
            self.logger.debug("Order book imbalance", 
                           symbol=symbol,
                           shallow=shallow_imbalance,
                           medium=medium_imbalance,
                           deep=deep_imbalance,
                           weighted=weighted_imbalance,
                           ema=self.imbalance_ema[symbol],
                           vwap_spread=vwap_spread)
        
        # Check for signal conditions
        await self._check_for_signal(symbol, processor, weighted_imbalance)
    
    async def _check_for_signal(
        self, 
        symbol: str, 
        processor: OrderBookProcessor, 
        imbalance: float
    ) -> None:
        """Check if a signal should be generated based on market imbalance.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
            imbalance: The weighted imbalance value
        """
        # Skip if in cooldown period
        now = datetime.now()
        if symbol in self.last_signal_time:
            cooldown_end = self.last_signal_time[symbol] + timedelta(seconds=self.signal_cooldown)
            if now < cooldown_end:
                return
        
        # Check if imbalance exceeds threshold
        significant_imbalance = False
        direction = ""
        imbalance_pct = 0.0
        
        if imbalance >= self.imbalance_threshold:
            significant_imbalance = True
            direction = "long"
            imbalance_pct = (imbalance - 1.0) * 100  # Convert to percentage above 1.0
        elif imbalance <= (1.0 / self.imbalance_threshold):
            significant_imbalance = True
            direction = "short"
            imbalance_pct = (1.0 - imbalance) * 100  # Convert to percentage below 1.0
        
        if not significant_imbalance:
            return
        
        # Calculate the size based on imbalance strength
        # More extreme imbalances get larger size
        size_multiplier = min(abs(imbalance - 1.0), 1.0)
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
                          imbalance=imbalance)
            return
        
        # Calculate confidence
        # Base confidence from is_favorable_for_trade
        confidence = details["confidence"] * 0.6
        
        # Add imbalance factor to confidence (normalized to 0-0.4)
        imbalance_confidence = min(abs(imbalance - 1.0) / 0.5, 1.0) * 0.4
        confidence += imbalance_confidence
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        # Generate signal reason
        if direction == "long":
            reason = f"Strong buy pressure ({imbalance_pct:.1f}% bid/ask imbalance)"
        else:
            reason = f"Strong sell pressure ({imbalance_pct:.1f}% ask/bid imbalance)"
        
        # Add additional market metrics to metadata
        metadata = {
            "imbalance": imbalance,
            "imbalance_pct": imbalance_pct,
            "shallow_imbalance": self.historical_imbalances[symbol][-1][1],
            "medium_imbalance": self.historical_imbalances[symbol][-1][2],
            "deep_imbalance": self.historical_imbalances[symbol][-1][3],
            "imbalance_ema": self.imbalance_ema[symbol],
            "vwap_spread": self.vwap_spreads[symbol][-1] if self.vwap_spreads[symbol] else 0.0,
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
        
        self.logger.info("Generated market imbalance signal", 
                      symbol=symbol,
                      direction=direction,
                      price=current_price,
                      size=size,
                      confidence=confidence,
                      imbalance=imbalance,
                      take_profit=take_profit,
                      stop_loss=stop_loss) 