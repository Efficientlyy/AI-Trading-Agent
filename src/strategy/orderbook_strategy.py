"""Order book strategy base class for the AI Crypto Trading System.

This module defines a base class for order book-based trading strategies,
which can leverage the high-performance OrderBookProcessor to analyze market
microstructure and generate trading signals.
"""

import asyncio
from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import OrderBookData, TimeFrame, TradeData
from src.models.signals import Signal, SignalType
from src.strategy.base_strategy import Strategy
from src.rust_bridge import create_order_book_processor, OrderBookProcessor


class OrderBookStrategy(Strategy):
    """Base class for order book-based trading strategies.
    
    This strategy type maintains order book processors for each symbol and
    provides methods for analyzing order book data, market impact, and liquidity.
    """
    
    def __init__(self, strategy_id: str):
        """Initialize the order book strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", strategy_id)
        
        # Order book configuration
        self.max_orderbook_depth = config.get(f"strategies.{strategy_id}.max_orderbook_depth", 100)
        self.orderbook_processors: Dict[str, OrderBookProcessor] = {}
        
        # Trading parameters
        self.min_confidence = config.get(f"strategies.{strategy_id}.min_confidence", 0.7)
        self.min_liquidity = config.get(f"strategies.{strategy_id}.min_liquidity", 10.0)  # Base currency units
        self.max_slippage_pct = config.get(f"strategies.{strategy_id}.max_slippage_pct", 0.1)  # 0.1%
        
        # Strategy configuration
        self.default_timeframe = TimeFrame.MINUTE_1
        
        # Market state tracking
        self.last_update_time: Dict[str, datetime] = {}
        
        # Add OrderBookEvent to the subscriptions
        if "OrderBookEvent" not in self.event_subscriptions:
            self.event_subscriptions.append("OrderBookEvent")
    
    async def _strategy_initialize(self) -> None:
        """Initialize the strategy."""
        self.logger.info("Initializing order book strategy", 
                      max_depth=self.max_orderbook_depth)
        
        # Initialize order book processors for each symbol
        for symbol in self.symbols:
            self._create_processor_for_symbol(symbol)
        
        # Additional strategy-specific initialization
        await self._orderbook_strategy_initialize()
    
    async def _strategy_start(self) -> None:
        """Start the strategy."""
        self.logger.info("Starting order book strategy")
        
        # Additional strategy-specific startup
        await self._orderbook_strategy_start()
    
    async def _strategy_stop(self) -> None:
        """Stop the strategy."""
        self.logger.info("Stopping order book strategy")
        
        # Clear order book processors
        self.orderbook_processors.clear()
        
        # Additional strategy-specific shutdown
        await self._orderbook_strategy_stop()
    
    def _create_processor_for_symbol(self, symbol: str) -> OrderBookProcessor:
        """Create an order book processor for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            The created order book processor
        """
        if symbol in self.orderbook_processors:
            return self.orderbook_processors[symbol]
        
        # Extract exchange from symbol if present
        exchange = "unknown"
        if ":" in symbol:
            exchange, symbol_part = symbol.split(":", 1)
        
        processor = create_order_book_processor(
            symbol=symbol,
            exchange=exchange,
            max_depth=self.max_orderbook_depth
        )
        
        self.orderbook_processors[symbol] = processor
        self.logger.info("Created order book processor", symbol=symbol, exchange=exchange)
        
        return processor
    
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process an order book update.
        
        This method is called when a new order book update is received.
        It updates the internal order book processor for the symbol and
        then calls the analyze_orderbook method.
        
        Args:
            orderbook: The order book data to process
        """
        symbol = orderbook.symbol
        
        # Create processor if it doesn't exist
        if symbol not in self.orderbook_processors:
            self._create_processor_for_symbol(symbol)
        
        processor = self.orderbook_processors[symbol]
        
        # Convert order book data to updates
        updates = self._convert_orderbook_to_updates(orderbook)
        
        # Process the updates
        try:
            processing_time = processor.process_updates(updates)
            self.logger.debug("Processed order book update", 
                           symbol=symbol, 
                           processing_time_ms=processing_time)
            
            # Update last update time
            self.last_update_time[symbol] = datetime.now()
            
            # Analyze the updated order book
            await self.analyze_orderbook(symbol, processor)
            
        except Exception as e:
            self.logger.error("Error processing order book update", 
                           symbol=symbol, error=str(e))
    
    def _convert_orderbook_to_updates(self, orderbook: OrderBookData) -> List[Dict[str, Any]]:
        """Convert an OrderBookData object to a list of updates.
        
        Args:
            orderbook: The order book data to convert
            
        Returns:
            A list of order book updates
        """
        updates = []
        timestamp = orderbook.timestamp
        
        # Add bids
        for i, (price, quantity) in enumerate(orderbook.bids):
            updates.append({
                "price": price,
                "side": "buy",
                "quantity": quantity,
                "timestamp": timestamp,
                "sequence": i + 1
            })
        
        # Add asks
        for i, (price, quantity) in enumerate(orderbook.asks):
            updates.append({
                "price": price,
                "side": "sell",
                "quantity": quantity,
                "timestamp": timestamp,
                "sequence": len(orderbook.bids) + i + 1
            })
        
        return updates
    
    def get_market_impact(self, symbol: str, side: str, size: float) -> Dict[str, Any]:
        """Calculate the market impact of an order.
        
        This method determines the average execution price, slippage, and total cost
        of a market order of the given size.
        
        Args:
            symbol: The trading pair symbol
            side: The order side ('buy' or 'sell')
            size: The order size in base currency units
            
        Returns:
            A dictionary with market impact details
        """
        if symbol not in self.orderbook_processors:
            return {
                "avg_price": 0.0,
                "slippage_pct": 0.0,
                "total_value": 0.0,
                "fillable_quantity": 0.0,
                "levels_consumed": 0
            }
        
        processor = self.orderbook_processors[symbol]
        impact = processor.calculate_market_impact(side, size)
        
        return impact
    
    def get_book_imbalance(self, symbol: str, depth: int = 10) -> float:
        """Calculate the order book imbalance ratio.
        
        This method calculates the ratio of bid volume to ask volume within
        the specified depth, providing a measure of buy/sell pressure.
        
        Args:
            symbol: The trading pair symbol
            depth: The number of price levels to include
            
        Returns:
            The imbalance ratio (> 1 means more bids than asks)
        """
        if symbol not in self.orderbook_processors:
            return 1.0
        
        processor = self.orderbook_processors[symbol]
        return processor.book_imbalance(depth)
    
    def get_liquidity(self, symbol: str, side: str, price_depth_pct: float = 0.5) -> float:
        """Calculate available liquidity within a price range.
        
        This method calculates the total quantity available within
        a percentage of the current price.
        
        Args:
            symbol: The trading pair symbol
            side: The order side ('buy' or 'sell')
            price_depth_pct: The price depth as a percentage from the best price
            
        Returns:
            The total quantity available
        """
        if symbol not in self.orderbook_processors:
            return 0.0
        
        processor = self.orderbook_processors[symbol]
        
        # Calculate price depth in absolute terms
        mid_price = processor.mid_price()
        price_depth = mid_price * (price_depth_pct / 100.0)
        
        return processor.liquidity_up_to(side, price_depth)
    
    def get_vwap(self, symbol: str, side: str, depth: int = 10) -> float:
        """Calculate the volume-weighted average price.
        
        This method calculates the VWAP for the specified side and depth.
        
        Args:
            symbol: The trading pair symbol
            side: The order side ('buy' or 'sell')
            depth: The number of price levels to include
            
        Returns:
            The VWAP
        """
        if symbol not in self.orderbook_processors:
            return 0.0
        
        processor = self.orderbook_processors[symbol]
        return processor.vwap(side, depth)
    
    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get the current spread and spread percentage.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            A tuple of (spread, spread_percentage)
        """
        if symbol not in self.orderbook_processors:
            return (0.0, 0.0)
        
        processor = self.orderbook_processors[symbol]
        return (processor.spread(), processor.spread_pct())
    
    def get_orderbook_stats(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive order book statistics.
        
        This method returns a dictionary with various order book metrics.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            A dictionary with order book statistics
        """
        if symbol not in self.orderbook_processors:
            return {}
        
        processor = self.orderbook_processors[symbol]
        
        stats = {
            "best_bid": processor.best_bid_price(),
            "best_ask": processor.best_ask_price(),
            "mid_price": processor.mid_price(),
            "spread": processor.spread(),
            "spread_pct": processor.spread_pct(),
            "book_imbalance": processor.book_imbalance(10),
            "bid_liquidity_100bps": processor.liquidity_up_to("buy", processor.mid_price() * 0.01),
            "ask_liquidity_100bps": processor.liquidity_up_to("sell", processor.mid_price() * 0.01),
            "bid_vwap_10": processor.vwap("buy", 10),
            "ask_vwap_10": processor.vwap("sell", 10),
            "processing_stats": processor.processing_stats()
        }
        
        return stats
    
    async def is_favorable_for_trade(self, symbol: str, side: str, size: float) -> Tuple[bool, Dict[str, Any]]:
        """Determine if market conditions are favorable for a trade.
        
        This method analyzes the current market conditions, including liquidity,
        market impact, and order book imbalance to determine if a trade should
        be executed.
        
        Args:
            symbol: The trading pair symbol
            side: The order side ('buy' or 'sell')
            size: The order size in base currency units
            
        Returns:
            A tuple of (favorable, details)
        """
        if symbol not in self.orderbook_processors:
            return (False, {"reason": "No order book data available"})
        
        # Get market impact
        impact = self.get_market_impact(symbol, side, size)
        
        # Check if the order can be filled
        if impact["fillable_quantity"] < size * 0.95:  # 95% of requested size
            return (False, {
                "reason": "Insufficient liquidity",
                "fillable": impact["fillable_quantity"],
                "requested": size
            })
        
        # Check slippage
        if impact["slippage_pct"] > self.max_slippage_pct:
            return (False, {
                "reason": "Excessive slippage",
                "slippage_pct": impact["slippage_pct"],
                "max_slippage_pct": self.max_slippage_pct
            })
        
        # Check order book imbalance
        imbalance = self.get_book_imbalance(symbol)
        imbalance_favorable = (side == "buy" and imbalance > 1.0) or (side == "sell" and imbalance < 1.0)
        
        # Calculate confidence based on multiple factors
        confidence = 0.0
        
        # Slippage factor: 0 slippage = 1.0 confidence, max_slippage = 0.0 confidence
        slippage_factor = 1.0 - (impact["slippage_pct"] / max(self.max_slippage_pct, 0.0001))
        
        # Imbalance factor: higher confidence if imbalance favors our side
        imbalance_factor = 0.0
        if side == "buy":
            imbalance_factor = min(imbalance, 2.0) / 2.0  # normalize to 0-1
        else:
            imbalance_factor = min(1.0 / max(imbalance, 0.5), 2.0) / 2.0
        
        # Spread factor: tighter spread = higher confidence
        spread_pct = self.get_spread(symbol)[1]
        spread_factor = 1.0 - min(spread_pct / 0.5, 1.0)  # Normalize, 0.5% or more = 0.0
        
        # Weighted confidence calculation
        confidence = (slippage_factor * 0.5) + (imbalance_factor * 0.3) + (spread_factor * 0.2)
        
        details = {
            "confidence": confidence,
            "slippage_pct": impact["slippage_pct"],
            "imbalance": imbalance,
            "imbalance_favorable": imbalance_favorable,
            "spread_pct": spread_pct,
            "avg_price": impact["avg_price"],
            "total_value": impact["total_value"]
        }
        
        return (confidence >= self.min_confidence, details)
    
    async def publish_orderbook_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        direction: str,
        price: float,
        size: float,
        confidence: float,
        reason: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Publish a trading signal based on order book analysis.
        
        Args:
            symbol: The trading pair symbol
            signal_type: The type of signal (entry, exit, etc.)
            direction: The direction of the signal (long, short)
            price: The price at which the signal was generated
            size: The suggested order size
            confidence: The confidence score for the signal (0.0 to 1.0)
            reason: The reason for the signal
            take_profit: Optional price target for take profit
            stop_loss: Optional price level for stop loss
            metadata: Optional additional data for the signal
        """
        # Get market impact details for this size
        impact = self.get_market_impact(symbol, direction.lower(), size)
        
        # Combine with any existing metadata
        combined_metadata = metadata or {}
        combined_metadata.update({
            "size": size,
            "impact": impact,
            "orderbook_stats": self.get_orderbook_stats(symbol)
        })
        
        # Use the default timeframe
        timeframe = self.default_timeframe
        
        # Publish the signal
        await self.publish_signal(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            timeframe=timeframe,
            price=price,
            confidence=confidence,
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata=combined_metadata
        )
    
    @abstractmethod
    async def _orderbook_strategy_initialize(self) -> None:
        """Strategy-specific initialization for the order book strategy.
        
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def _orderbook_strategy_start(self) -> None:
        """Strategy-specific startup for the order book strategy.
        
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def _orderbook_strategy_stop(self) -> None:
        """Strategy-specific shutdown for the order book strategy.
        
        This method should be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def analyze_orderbook(self, symbol: str, processor: OrderBookProcessor) -> None:
        """Analyze an updated order book.
        
        This method is called when an order book update is processed.
        It should implement the strategy-specific order book analysis logic.
        
        Args:
            symbol: The trading pair symbol
            processor: The order book processor for the symbol
        """
        pass 