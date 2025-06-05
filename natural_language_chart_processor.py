#!/usr/bin/env python
"""
Natural Language Chart Request Processor

This module processes natural language chart requests and extracts parameters
for the visualization plugin.
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("system_overseer.natural_language_chart_processor")

class NaturalLanguageChartProcessor:
    """Process natural language chart requests."""
    
    def __init__(self):
        """Initialize the processor."""
        # Default values
        self.default_chart_type = "candlestick"
        self.default_interval = "15m"
        self.default_quote_currency = "USDC"
        
        # Supported base currencies
        self.base_currencies = {
            "btc": "BTC",
            "bitcoin": "BTC",
            "eth": "ETH",
            "ethereum": "ETH",
            "sol": "SOL",
            "solana": "SOL"
        }
        
        # Supported chart types
        self.chart_types = {
            "candlestick": ["candlestick", "candle", "candles", "ohlc"],
            "line": ["line", "lines", "price", "prices"],
            "volume": ["volume", "vol", "volumes"]
        }
        
        # Supported intervals
        self.intervals = {
            "1m": ["1m", "1min", "1minute", "1 min", "1 minute", "one minute", "1-minute", "1-min"],
            "5m": ["5m", "5min", "5minute", "5 min", "5 minute", "five minute", "5-minute", "5-min"],
            "15m": ["15m", "15min", "15minute", "15 min", "15 minute", "fifteen minute", "15-minute", "15-min"],
            "30m": ["30m", "30min", "30minute", "30 min", "30 minute", "thirty minute", "30-minute", "30-min"],
            "1h": ["1h", "1hour", "1 hour", "one hour", "1-hour", "hourly"],
            "4h": ["4h", "4hour", "4 hour", "four hour", "4-hour"],
            "1d": ["1d", "1day", "1 day", "one day", "1-day", "daily", "day"],
            "1w": ["1w", "1week", "1 week", "one week", "1-week", "weekly", "week"]
        }
        
        # Supported indicators
        self.indicators = {
            "sma": ["sma", "simple moving average", "moving average"],
            "ema": ["ema", "exponential moving average"],
            "rsi": ["rsi", "relative strength index"],
            "macd": ["macd", "moving average convergence divergence"],
            "bollinger": ["bollinger", "bollinger bands", "bands"]
        }
    
    def process_request(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Process a natural language chart request.
        
        Args:
            text: Natural language request text
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_chart_request, parameters)
                where parameters include symbol, chart_type, interval, indicators
        """
        # Convert to lowercase for easier matching
        text = text.lower()
        
        # Check if this is a chart request
        if not self._is_chart_request(text):
            return False, {}
        
        # Extract parameters
        symbol = self._extract_symbol(text)
        chart_type = self._extract_chart_type(text)
        interval = self._extract_interval(text)
        indicators = self._extract_indicators(text)
        
        # Return parameters
        return True, {
            "symbol": symbol,
            "chart_type": chart_type,
            "interval": interval,
            "indicators": indicators
        }
    
    def _is_chart_request(self, text: str) -> bool:
        """Check if the text is a chart request.
        
        Args:
            text: Natural language request text
            
        Returns:
            bool: True if the text is a chart request
        """
        # Keywords that indicate a chart request
        chart_keywords = [
            "chart", "charts", "graph", "graphs", "plot", "plots",
            "candlestick", "candle", "line", "price", "volume"
        ]
        
        # Check if any base currency is mentioned
        has_currency = any(currency in text for currency in self.base_currencies.keys())
        
        # Check if any chart keyword is mentioned
        has_chart_keyword = any(keyword in text for keyword in chart_keywords)
        
        # Return True if both a currency and a chart keyword are mentioned
        return has_currency and has_chart_keyword
    
    def _extract_symbol(self, text: str) -> str:
        """Extract trading pair symbol from text.
        
        Args:
            text: Natural language request text
            
        Returns:
            str: Trading pair symbol (e.g., BTCUSDC)
        """
        # Check for explicit trading pairs
        for base, symbol in self.base_currencies.items():
            # Check for explicit pairs like BTCUSDC or BTC/USDC
            pair_patterns = [
                f"{symbol}USDC",
                f"{symbol}/USDC",
                f"{symbol} USDC",
                f"{symbol}USDT",
                f"{symbol}/USDT",
                f"{symbol} USDT"
            ]
            
            for pattern in pair_patterns:
                if pattern.lower() in text:
                    # Prefer USDC pairs
                    if "USDT" in pattern:
                        return f"{symbol}USDT"
                    else:
                        return f"{symbol}USDC"
        
        # If no explicit pair is found, check for base currencies
        for base, symbol in self.base_currencies.items():
            if base in text:
                # Default to USDC pair
                return f"{symbol}{self.default_quote_currency}"
        
        # Default to BTCUSDC if no symbol is found
        return f"BTC{self.default_quote_currency}"
    
    def _extract_chart_type(self, text: str) -> str:
        """Extract chart type from text.
        
        Args:
            text: Natural language request text
            
        Returns:
            str: Chart type (candlestick, line, volume)
        """
        for chart_type, keywords in self.chart_types.items():
            for keyword in keywords:
                if keyword in text:
                    return chart_type
        
        # Default to candlestick
        return self.default_chart_type
    
    def _extract_interval(self, text: str) -> str:
        """Extract time interval from text.
        
        Args:
            text: Natural language request text
            
        Returns:
            str: Time interval (e.g., 1m, 15m, 1h)
        """
        for interval, keywords in self.intervals.items():
            for keyword in keywords:
                if keyword in text:
                    return interval
        
        # Default to 15m
        return self.default_interval
    
    def _extract_indicators(self, text: str) -> List[str]:
        """Extract technical indicators from text.
        
        Args:
            text: Natural language request text
            
        Returns:
            List[str]: List of indicators
        """
        indicators = []
        
        for indicator, keywords in self.indicators.items():
            for keyword in keywords:
                if keyword in text:
                    indicators.append(indicator)
                    break
        
        # Default to empty list (no indicators)
        return indicators


# Example usage
if __name__ == "__main__":
    processor = NaturalLanguageChartProcessor()
    
    # Test with various requests
    test_requests = [
        "Hey, can you show me a BTC chart?",
        "I need to see the Bitcoin price",
        "Give me the ETH candlestick chart for 15 minutes",
        "Show me the Solana volume for the last hour",
        "What's the 1 minute chart for BTC looking like?",
        "Can I see the ETH/USDC line chart with SMA?",
        "Show me the daily SOL chart with bollinger bands",
        "How's the weather today?"  # Not a chart request
    ]
    
    for request in test_requests:
        is_chart, params = processor.process_request(request)
        
        if is_chart:
            print(f"Request: {request}")
            print(f"  Symbol: {params['symbol']}")
            print(f"  Chart Type: {params['chart_type']}")
            print(f"  Interval: {params['interval']}")
            print(f"  Indicators: {params['indicators']}")
        else:
            print(f"Not a chart request: {request}")
        
        print()
