# Enhanced Dashboard Integration for Trading-Agent System

"""
Enhanced Dashboard Integration for Trading-Agent System

This module provides an improved dashboard integration that connects
the visualization components with the enhanced market data pipeline.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_market_data_pipeline import EnhancedMarketDataPipeline
from symbol_standardization import SymbolStandardizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("dashboard_integration")

class EnhancedDashboardIntegration:
    """Enhanced dashboard integration for Trading-Agent System"""
    
    def __init__(self, symbols=None, timeframes=None, mock_data_dir=None):
        """Initialize enhanced dashboard integration
        
        Args:
            symbols: List of symbols to track (default: ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
            timeframes: List of timeframes to support (default: ["1m", "5m", "15m", "1h", "4h", "1d"])
            mock_data_dir: Directory for mock data (default: "./test_data")
        """
        # Initialize symbol standardizer
        self.standardizer = SymbolStandardizer()
        
        # Initialize symbols and timeframes
        self.symbols = symbols or ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Initialize market data pipeline
        self.market_data_pipeline = EnhancedMarketDataPipeline(
            symbols=self.symbols,
            timeframes=self.timeframes,
            mock_data_dir=mock_data_dir
        )
        
        # Initialize data stores
        self.chart_data = {}
        self.signal_data = {}
        self.order_data = {}
        self.position_data = {}
        
        # Initialize update timestamps
        self.last_updates = {}
        
        logger.info(f"Enhanced dashboard integration initialized with {len(self.symbols)} symbols")
    
    def get_market_data(self, symbol, timeframe="5m", limit=100, use_cache=True):
        """Get market data for dashboard visualization
        
        Args:
            symbol: Symbol to get data for (any format)
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to return
            use_cache: Whether to use cached data if available
            
        Returns:
            list: Market data as list of candles
        """
        # Use enhanced market data pipeline with fallback to mock data
        data = self.market_data_pipeline.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            use_cache=use_cache,
            fallback_to_mock=True
        )
        
        # Update last update timestamp
        self.last_updates[f"{symbol}_{timeframe}"] = time.time()
        
        # Format data for chart visualization if needed
        chart_data = self._format_for_chart(data)
        
        # Cache chart data
        cache_key = f"{symbol}_{timeframe}"
        self.chart_data[cache_key] = chart_data
        
        return chart_data
    
    def _format_for_chart(self, data):
        """Format data for chart visualization
        
        Args:
            data: Raw market data
            
        Returns:
            list: Formatted data for chart visualization
        """
        # If data is already in the right format, return as is
        if not data:
            return []
        
        # Check if data needs formatting
        if "time" in data[0] and "open" in data[0] and "high" in data[0]:
            # Already in the right format, just ensure all required fields
            return [{
                "time": candle["time"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle.get("volume", 0)
            } for candle in data]
        
        # If data is in a different format, convert it
        formatted_data = []
        for candle in data:
            # Try to extract required fields
            time_val = candle.get("time", candle.get("timestamp", 0))
            open_val = candle.get("open", candle.get("Open", 0))
            high_val = candle.get("high", candle.get("High", 0))
            low_val = candle.get("low", candle.get("Low", 0))
            close_val = candle.get("close", candle.get("Close", 0))
            volume_val = candle.get("volume", candle.get("Volume", 0))
            
            formatted_data.append({
                "time": time_val,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val
            })
        
        return formatted_data
    
    def get_signals(self, symbol, limit=20):
        """Get trading signals for dashboard visualization
        
        Args:
            symbol: Symbol to get signals for (any format)
            limit: Maximum number of signals to return
            
        Returns:
            list: Trading signals
        """
        # Standardize symbol
        internal_symbol = self.standardizer.for_internal(symbol)
        
        # For now, return mock signals
        # In a real implementation, this would fetch from the signal generator
        mock_signals = self._generate_mock_signals(internal_symbol, limit)
        
        # Cache signal data
        self.signal_data[internal_symbol] = mock_signals
        
        return mock_signals
    
    def _generate_mock_signals(self, symbol, limit):
        """Generate mock trading signals
        
        Args:
            symbol: Symbol to generate signals for
            limit: Maximum number of signals to generate
            
        Returns:
            list: Mock trading signals
        """
        # Generate mock signals
        signals = []
        
        # Current time
        current_time = int(time.time() * 1000)
        
        # Signal types
        signal_types = ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "NEUTRAL"]
        
        # Generate signals
        for i in range(limit):
            # Signal time (random in the past)
            signal_time = current_time - (i * 60 * 60 * 1000) - (hash(f"{symbol}_{i}") % 3600000)
            
            # Signal type (weighted towards BUY/SELL)
            signal_type = signal_types[hash(f"{symbol}_type_{i}") % 5]
            
            # Signal strength
            strength = (hash(f"{symbol}_strength_{i}") % 100) / 100
            
            # Signal source
            sources = ["price_action", "pattern_recognition", "momentum", "volume_analysis", "llm_decision"]
            source = sources[hash(f"{symbol}_source_{i}") % len(sources)]
            
            # Create signal
            signal = {
                "id": f"SIG-{hash(f'{symbol}_{i}') % 10000}",
                "symbol": symbol,
                "type": signal_type,
                "strength": strength,
                "time": signal_time,
                "source": source,
                "description": f"{signal_type} signal detected with {strength:.2f} confidence"
            }
            
            signals.append(signal)
        
        return signals
    
    def get_orders(self, symbol, limit=20):
        """Get orders for dashboard visualization
        
        Args:
            symbol: Symbol to get orders for (any format)
            limit: Maximum number of orders to return
            
        Returns:
            list: Orders
        """
        # Standardize symbol
        internal_symbol = self.standardizer.for_internal(symbol)
        
        # For now, return mock orders
        # In a real implementation, this would fetch from the paper trading system
        mock_orders = self._generate_mock_orders(internal_symbol, limit)
        
        # Cache order data
        self.order_data[internal_symbol] = mock_orders
        
        return mock_orders
    
    def _generate_mock_orders(self, symbol, limit):
        """Generate mock orders
        
        Args:
            symbol: Symbol to generate orders for
            limit: Maximum number of orders to generate
            
        Returns:
            list: Mock orders
        """
        # Generate mock orders
        orders = []
        
        # Current time
        current_time = int(time.time() * 1000)
        
        # Order statuses
        statuses = ["NEW", "FILLED", "PARTIALLY_FILLED", "CANCELED", "REJECTED"]
        
        # Generate orders
        for i in range(limit):
            # Order time (random in the past)
            order_time = current_time - (i * 60 * 60 * 1000) - (hash(f"{symbol}_{i}") % 3600000)
            
            # Order side
            side = "BUY" if hash(f"{symbol}_side_{i}") % 2 == 0 else "SELL"
            
            # Order type
            order_type = "MARKET" if hash(f"{symbol}_type_{i}") % 3 == 0 else "LIMIT"
            
            # Order price
            base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 100)
            price = base_price * (0.9 + (hash(f"{symbol}_price_{i}") % 2000) / 10000)
            
            # Order quantity
            quantity = (hash(f"{symbol}_qty_{i}") % 100) / 100
            
            # Order status
            status_idx = min(i % 5, 4)  # Weight towards FILLED for older orders
            status = statuses[status_idx]
            
            # Create order
            order = {
                "orderId": f"ORD-{hash(f'{symbol}_{i}') % 10000}",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "price": price if order_type == "LIMIT" else None,
                "quantity": quantity,
                "status": status,
                "timestamp": order_time
            }
            
            orders.append(order)
        
        return orders
    
    def get_positions(self, symbol=None):
        """Get positions for dashboard visualization
        
        Args:
            symbol: Symbol to get position for (any format, None for all)
            
        Returns:
            list: Positions
        """
        # For now, return mock positions
        # In a real implementation, this would fetch from the paper trading system
        mock_positions = self._generate_mock_positions()
        
        # Filter by symbol if specified
        if symbol:
            internal_symbol = self.standardizer.for_internal(symbol)
            mock_positions = [p for p in mock_positions if p["symbol"] == internal_symbol]
        
        # Cache position data
        self.position_data = {p["symbol"]: p for p in mock_positions}
        
        return mock_positions
    
    def _generate_mock_positions(self):
        """Generate mock positions
        
        Returns:
            list: Mock positions
        """
        # Generate mock positions
        positions = []
        
        # For each symbol
        for symbol in self.symbols:
            # Position size (random)
            size = (hash(symbol) % 100) / 100
            
            # Entry price
            base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 100)
            entry_price = base_price * (0.9 + (hash(f"{symbol}_entry") % 2000) / 10000)
            
            # Current price
            current_price = base_price * (0.9 + (hash(f"{symbol}_current") % 2000) / 10000)
            
            # PnL
            pnl = size * (current_price - entry_price)
            pnl_percent = (current_price - entry_price) / entry_price * 100
            
            # Create position
            position = {
                "symbol": symbol,
                "size": size,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "timestamp": int(time.time() * 1000)
            }
            
            positions.append(position)
        
        return positions
    
    def get_dashboard_data(self, symbol, timeframe="5m", limit=100):
        """Get all dashboard data for a symbol
        
        Args:
            symbol: Symbol to get data for (any format)
            timeframe: Timeframe for market data
            limit: Number of candles to return
            
        Returns:
            dict: Dashboard data
        """
        # Standardize symbol
        internal_symbol = self.standardizer.for_internal(symbol)
        
        # Get all data
        market_data = self.get_market_data(internal_symbol, timeframe, limit)
        signals = self.get_signals(internal_symbol, 10)
        orders = self.get_orders(internal_symbol, 10)
        positions = self.get_positions(internal_symbol)
        
        # Combine into dashboard data
        dashboard_data = {
            "symbol": internal_symbol,
            "timeframe": timeframe,
            "market_data": market_data,
            "signals": signals,
            "orders": orders,
            "positions": positions,
            "timestamp": int(time.time() * 1000)
        }
        
        return dashboard_data

# Example usage
if __name__ == "__main__":
    dashboard = EnhancedDashboardIntegration()
    
    # Test with different symbol formats
    symbols = ["BTC/USDC", "BTCUSDC", "ETH-USDC"]
    
    for symbol in symbols:
        print(f"Getting dashboard data for {symbol}...")
        data = dashboard.get_dashboard_data(symbol)
        
        print(f"Market data: {len(data['market_data'])} candles")
        print(f"Signals: {len(data['signals'])}")
        print(f"Orders: {len(data['orders'])}")
        print(f"Positions: {len(data['positions'])}")
        
        print("---")
