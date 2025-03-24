"""
Python fallback implementations for market data components.

This module provides pure Python implementations of the market data components
that would normally be provided by the Rust library. It is used when the Rust
components are not available.
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from collections import deque, defaultdict
import bisect

# Python implementation of TimeFrame enum
class TimeFrame:
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    @classmethod
    def from_string(cls, timeframe: str) -> str:
        valid_timeframes = [
            cls.MINUTE_1, cls.MINUTE_3, cls.MINUTE_5, cls.MINUTE_15, cls.MINUTE_30,
            cls.HOUR_1, cls.HOUR_2, cls.HOUR_4, cls.HOUR_6, cls.HOUR_12,
            cls.DAY_1, cls.DAY_3, cls.WEEK_1, cls.MONTH_1
        ]
        if timeframe in valid_timeframes:
            return timeframe
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    @classmethod
    def duration_seconds(cls, timeframe: str) -> int:
        timeframe = cls.from_string(timeframe)
        if timeframe == cls.MINUTE_1:
            return 60
        elif timeframe == cls.MINUTE_3:
            return 60 * 3
        elif timeframe == cls.MINUTE_5:
            return 60 * 5
        elif timeframe == cls.MINUTE_15:
            return 60 * 15
        elif timeframe == cls.MINUTE_30:
            return 60 * 30
        elif timeframe == cls.HOUR_1:
            return 60 * 60
        elif timeframe == cls.HOUR_2:
            return 60 * 60 * 2
        elif timeframe == cls.HOUR_4:
            return 60 * 60 * 4
        elif timeframe == cls.HOUR_6:
            return 60 * 60 * 6
        elif timeframe == cls.HOUR_12:
            return 60 * 60 * 12
        elif timeframe == cls.DAY_1:
            return 60 * 60 * 24
        elif timeframe == cls.DAY_3:
            return 60 * 60 * 24 * 3
        elif timeframe == cls.WEEK_1:
            return 60 * 60 * 24 * 7
        elif timeframe == cls.MONTH_1:
            return 60 * 60 * 24 * 30  # Approximate
        return 0

# Python implementation of CandleData
class CandleData:
    def __init__(self, symbol: str, exchange: str, timestamp: float, 
                 open_price: float, high: float, low: float, close: float, 
                 volume: float, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timeframe = TimeFrame.from_string(timeframe)
    
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    def range(self) -> float:
        return self.high - self.low
    
    def body(self) -> float:
        return abs(self.close - self.open)
    
    def body_percent(self) -> float:
        if self.range() == 0:
            return 0.0
        return (self.body() / self.range()) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }

def create_candle(symbol: str, timestamp: float, open_price: float, high: float, 
                  low: float, close: float, volume: float, timeframe: str) -> CandleData:
    """Python fallback implementation of create_candle."""
    return CandleData(symbol, exchange="unknown", timestamp=timestamp, open_price=open_price,
                     high=high, low=low, close=close, volume=volume, timeframe=timeframe)

# Python implementation of OrderBookData
class OrderBookData:
    def __init__(self, symbol: str, exchange: str, timestamp: Optional[float] = None):
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp if timestamp is not None else time.time()
        self._bids: List[Tuple[float, float]] = []  # Sorted high to low
        self._asks: List[Tuple[float, float]] = []  # Sorted low to high
    
    def add_bid(self, price: float, quantity: float) -> None:
        # Insert maintaining sort order (high to low)
        index = bisect.bisect_left([-p for p, _ in self._bids], -price)
        self._bids.insert(index, (price, quantity))
    
    def add_ask(self, price: float, quantity: float) -> None:
        # Insert maintaining sort order (low to high)
        index = bisect.bisect_left([p for p, _ in self._asks], price)
        self._asks.insert(index, (price, quantity))
    
    def update_bid(self, price: float, quantity: float) -> bool:
        for i, (p, _) in enumerate(self._bids):
            if p == price:
                self._bids[i] = (price, quantity)
                return True
        return False
    
    def update_ask(self, price: float, quantity: float) -> bool:
        for i, (p, _) in enumerate(self._asks):
            if p == price:
                self._asks[i] = (price, quantity)
                return True
        return False
    
    def remove_bid(self, price: float) -> bool:
        for i, (p, _) in enumerate(self._bids):
            if p == price:
                del self._bids[i]
                return True
        return False
    
    def remove_ask(self, price: float) -> bool:
        for i, (p, _) in enumerate(self._asks):
            if p == price:
                del self._asks[i]
                return True
        return False
    
    def has_bid(self, price: float) -> bool:
        return any(p == price for p, _ in self._bids)
    
    def has_ask(self, price: float) -> bool:
        return any(p == price for p, _ in self._asks)
    
    def bids(self) -> List[Tuple[float, float]]:
        return self._bids.copy()
    
    def asks(self) -> List[Tuple[float, float]]:
        return self._asks.copy()
    
    def best_bid(self) -> Optional[Tuple[float, float]]:
        if not self._bids:
            return None
        return self._bids[0]
    
    def best_ask(self) -> Optional[Tuple[float, float]]:
        if not self._asks:
            return None
        return self._asks[0]
    
    def spread(self) -> Optional[float]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask[0] - best_bid[0]
    
    def mid_price(self) -> Optional[float]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid[0] + best_ask[0]) / 2
    
    def truncate_bids(self, max_depth: int) -> None:
        if len(self._bids) > max_depth:
            self._bids = self._bids[:max_depth]
    
    def truncate_asks(self, max_depth: int) -> None:
        if len(self._asks) > max_depth:
            self._asks = self._asks[:max_depth]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'bids': self._bids,
            'asks': self._asks
        }

def create_order_book(symbol: str, exchange: str, timestamp: Optional[float] = None, 
                      bids: Optional[List[Tuple[float, float]]] = None, 
                      asks: Optional[List[Tuple[float, float]]] = None) -> OrderBookData:
    """Python fallback implementation of create_order_book."""
    book = OrderBookData(symbol, exchange, timestamp)
    
    if bids:
        for price, quantity in bids:
            book.add_bid(price, quantity)
    
    if asks:
        for price, quantity in asks:
            book.add_ask(price, quantity)
    
    return book

# Python implementation of OrderBookProcessor
class OrderBookProcessor:
    """Python fallback implementation of OrderBookProcessor."""
    
    def __init__(self, symbol: str, exchange: str, max_depth: int = 100):
        self.symbol = symbol
        self.exchange = exchange
        self.max_depth = max_depth
        self.book = OrderBookData(symbol, exchange)
        self.update_queue: deque = deque()
        self.last_update_time = time.time()
        self.last_sequence = 0
        self._stats: Dict[str, Any] = {
            'updates_processed': 0,
            'levels_added': 0,
            'levels_removed': 0,
            'levels_modified': 0,
            'avg_processing_time_us': 0.0,
            'max_processing_time_us': 0,
            'min_processing_time_us': 0
        }
    
    def process_updates(self, updates: List[Dict[str, Any]]) -> float:
        """Process a batch of order book updates."""
        if not updates:
            return 0.0
        
        start_time = time.time()
        
        # Add all updates to the queue
        for update in updates:
            self.update_queue.append(update)
        
        # Process the queue
        updates_processed = 0
        levels_added = 0
        levels_modified = 0
        levels_removed = 0
        
        while self.update_queue:
            update = self.update_queue.popleft()
            
            # Extract update data
            price = float(update['price'])
            side = update['side'].lower()
            quantity = float(update['quantity'])
            timestamp = update.get('timestamp', time.time())
            sequence = update.get('sequence', 0)
            
            # Check sequence number
            if sequence < self.last_sequence:
                continue
            
            # Update the book
            if side in ('buy', 'bid'):
                if quantity == 0:
                    # Remove price level
                    if self.book.remove_bid(price):
                        levels_removed += 1
                else:
                    # Add or update price level
                    if self.book.has_bid(price):
                        self.book.update_bid(price, quantity)
                        levels_modified += 1
                    else:
                        self.book.add_bid(price, quantity)
                        levels_added += 1
            elif side in ('sell', 'ask'):
                if quantity == 0:
                    # Remove price level
                    if self.book.remove_ask(price):
                        levels_removed += 1
                else:
                    # Add or update price level
                    if self.book.has_ask(price):
                        self.book.update_ask(price, quantity)
                        levels_modified += 1
                    else:
                        self.book.add_ask(price, quantity)
                        levels_added += 1
            
            # Update metadata
            self.last_update_time = timestamp
            self.last_sequence = sequence
            updates_processed += 1
        
        # Enforce max depth
        self.book.truncate_bids(self.max_depth)
        self.book.truncate_asks(self.max_depth)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        processing_time_us = int(processing_time * 1000)  # microseconds
        
        # Update statistics
        self.update_processing_stats(
            updates_processed,
            levels_added,
            levels_modified,
            levels_removed,
            processing_time_us
        )
        
        return processing_time
    
    def calculate_market_impact(self, side: str, size: float) -> Dict[str, Any]:
        """Calculate market impact for a given order size."""
        remaining_size = size
        total_value = 0.0
        levels_consumed = 0
        
        # Price levels to iterate over
        levels = self.book.bids() if side.lower() in ('sell', 'ask') else self.book.asks()
        
        # For sells, we iterate in reverse (highest to lowest)
        if side.lower() in ('sell', 'ask'):
            levels = sorted(levels, key=lambda x: x[0], reverse=True)
        
        for price, quantity in levels:
            levels_consumed += 1
            
            if quantity >= remaining_size:
                # This level can fully fill the remaining size
                total_value += price * remaining_size
                remaining_size = 0
                break
            else:
                # Partial fill from this level
                total_value += price * quantity
                remaining_size -= quantity
        
        # Calculate fillable quantity
        fillable_quantity = size - remaining_size
        
        # Calculate average price
        avg_price = total_value / fillable_quantity if fillable_quantity > 0 else 0.0
        
        # Calculate slippage as a percentage of the best price
        slippage_pct = 0.0
        if fillable_quantity > 0:
            best_price = 0.0
            if side.lower() in ('buy', 'bid'):
                best_ask = self.book.best_ask()
                best_price = best_ask[0] if best_ask else 0.0
                price_diff = avg_price - best_price if best_price > 0 else 0.0
            else:
                best_bid = self.book.best_bid()
                best_price = best_bid[0] if best_bid else 0.0
                price_diff = best_price - avg_price if best_price > 0 else 0.0
            
            slippage_pct = (price_diff * 100.0) / best_price if best_price > 0 else 0.0
        
        return {
            'avg_price': avg_price,
            'slippage_pct': slippage_pct,
            'total_value': total_value,
            'fillable_quantity': fillable_quantity,
            'levels_consumed': levels_consumed
        }
    
    def best_bid_price(self) -> float:
        """Get the best bid price."""
        best_bid = self.book.best_bid()
        return best_bid[0] if best_bid else 0.0
    
    def best_ask_price(self) -> float:
        """Get the best ask price."""
        best_ask = self.book.best_ask()
        return best_ask[0] if best_ask else 0.0
    
    def mid_price(self) -> float:
        """Get the mid price."""
        best_bid = self.best_bid_price()
        best_ask = self.best_ask_price()
        
        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2
        elif best_bid > 0:
            return best_bid
        elif best_ask > 0:
            return best_ask
        else:
            return 0.0
    
    def spread(self) -> float:
        """Get the current bid-ask spread."""
        best_bid = self.best_bid_price()
        best_ask = self.best_ask_price()
        
        if best_bid > 0 and best_ask > 0:
            return best_ask - best_bid
        else:
            return 0.0
    
    def spread_pct(self) -> float:
        """Get the current bid-ask spread as a percentage of the mid price."""
        spread = self.spread()
        mid = self.mid_price()
        
        if mid > 0:
            return (spread * 100.0) / mid
        else:
            return 0.0
    
    def vwap(self, side: str, depth: int) -> float:
        """Calculate the volume-weighted average price (VWAP) for a given side and depth."""
        levels = self.book.bids() if side.lower() in ('buy', 'bid') else self.book.asks()
        
        total_value = 0.0
        total_volume = 0.0
        
        for i, (price, quantity) in enumerate(levels):
            if i >= depth:
                break
            
            total_value += price * quantity
            total_volume += quantity
        
        if total_volume > 0:
            return total_value / total_volume
        else:
            return 0.0
    
    def liquidity_up_to(self, side: str, price_depth: float) -> float:
        """Calculate the total liquidity available up to a given price depth."""
        from_price = 0.0
        levels = []
        
        if side.lower() in ('buy', 'bid'):
            best_bid = self.book.best_bid()
            if best_bid:
                from_price = best_bid[0]
                levels = self.book.bids()
        else:
            best_ask = self.book.best_ask()
            if best_ask:
                from_price = best_ask[0]
                levels = self.book.asks()
        
        total_liquidity = 0.0
        
        for price, quantity in levels:
            price_diff = from_price - price if side.lower() in ('buy', 'bid') else price - from_price
            
            if price_diff <= price_depth:
                total_liquidity += quantity
            else:
                break
        
        return total_liquidity
    
    def book_imbalance(self, depth: int) -> float:
        """Detect order book imbalance (ratio of buy to sell liquidity)."""
        bid_volume = 0.0
        ask_volume = 0.0
        
        for i, (_, quantity) in enumerate(self.book.bids()):
            if i >= depth:
                break
            bid_volume += quantity
        
        for i, (_, quantity) in enumerate(self.book.asks()):
            if i >= depth:
                break
            ask_volume += quantity
        
        if ask_volume > 0:
            return bid_volume / ask_volume
        elif bid_volume > 0:
            return 10.0  # Arbitrary large number indicating strong bid imbalance
        else:
            return 1.0  # No imbalance if both are zero
    
    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current order book."""
        return self.book.to_dict()
    
    def processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def reset(self) -> None:
        """Reset the order book processor."""
        self.book = OrderBookData(self.symbol, self.exchange)
        self.update_queue.clear()
        self.last_update_time = time.time()
        self.last_sequence = 0
        self._stats = {
            'updates_processed': 0,
            'levels_added': 0,
            'levels_removed': 0,
            'levels_modified': 0,
            'avg_processing_time_us': 0.0,
            'max_processing_time_us': 0,
            'min_processing_time_us': 0
        }
    
    def update_processing_stats(self, updates_processed: int, levels_added: int, 
                               levels_modified: int, levels_removed: int, 
                               processing_time_us: int) -> None:
        """Update processing statistics."""
        # Update total counters
        self._stats['updates_processed'] += updates_processed
        self._stats['levels_added'] += levels_added
        self._stats['levels_modified'] += levels_modified
        self._stats['levels_removed'] += levels_removed
        
        # Update timing statistics
        if (self._stats["min_processing_time_us"] = = 0 or 
            processing_time_us < self._stats['min_processing_time_us']):
            self._stats["min_processing_time_us"] = processing_time_us
        
        if processing_time_us > self._stats['max_processing_time_us']:
            self._stats["max_processing_time_us"] = processing_time_us
        
        # Exponential moving average for processing time
        if self._stats["avg_processing_time_us"] = = 0.0:
            self._stats["avg_processing_time_us"] = float(processing_time_us)
        else:
            alpha = 0.05  # Smoothing factor
            self._stats["avg_processing_time_us"] = (
                alpha * float(processing_time_us) + 
                (1.0 - alpha) * self._stats['avg_processing_time_us']
            ) 