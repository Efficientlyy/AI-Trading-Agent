# Symbol Format Standardization Utility

"""
Symbol Format Standardization Utility for Trading-Agent System

This module provides utilities for standardizing symbol formats across
different components of the Trading-Agent system, ensuring consistent
representation regardless of the source or destination component.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("symbol_standardization")

class SymbolStandardizer:
    """Utility class for standardizing symbol formats across the system"""
    
    # Common format mappings
    FORMAT_SLASH = "SLASH"         # BTC/USDC
    FORMAT_DIRECT = "DIRECT"       # BTCUSDC
    FORMAT_DASH = "DASH"           # BTC-USDC
    FORMAT_UNDERSCORE = "UNDER"    # BTC_USDC
    
    def __init__(self):
        """Initialize symbol standardizer"""
        # Common base assets and quote assets for quick reference
        self.common_base_assets = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "AVAX", "MATIC"]
        self.common_quote_assets = ["USDC", "USDT", "USD", "BUSD", "DAI", "TUSD"]
        
        # Exchange-specific format preferences
        self.exchange_formats = {
            "mexc": self.FORMAT_DIRECT,
            "binance": self.FORMAT_DIRECT,
            "kraken": self.FORMAT_DASH,
            "bitvavo": self.FORMAT_DASH,
            "bybit": self.FORMAT_DIRECT,
            "internal": self.FORMAT_SLASH
        }
        
        logger.info("Symbol standardizer initialized")
    
    def standardize(self, symbol: str, target_format: str = FORMAT_SLASH) -> str:
        """Standardize symbol to target format
        
        Args:
            symbol: Symbol to standardize (any common format)
            target_format: Target format (SLASH, DIRECT, DASH, UNDER)
            
        Returns:
            str: Standardized symbol
        """
        # First parse the symbol into base and quote
        base, quote = self._parse_symbol(symbol)
        
        if not base or not quote:
            logger.warning(f"Could not parse symbol: {symbol}")
            return symbol
        
        # Then convert to target format
        if target_format == self.FORMAT_SLASH:
            return f"{base}/{quote}"
        elif target_format == self.FORMAT_DIRECT:
            return f"{base}{quote}"
        elif target_format == self.FORMAT_DASH:
            return f"{base}-{quote}"
        elif target_format == self.FORMAT_UNDERSCORE:
            return f"{base}_{quote}"
        else:
            logger.warning(f"Unknown target format: {target_format}")
            return symbol
    
    def for_exchange(self, symbol: str, exchange: str) -> str:
        """Convert symbol to exchange-specific format
        
        Args:
            symbol: Symbol to convert (any common format)
            exchange: Target exchange (mexc, binance, kraken, bitvavo, bybit)
            
        Returns:
            str: Symbol in exchange-specific format
        """
        exchange = exchange.lower()
        if exchange not in self.exchange_formats:
            logger.warning(f"Unknown exchange: {exchange}, using direct format")
            return self.standardize(symbol, self.FORMAT_DIRECT)
        
        return self.standardize(symbol, self.exchange_formats[exchange])
    
    def for_internal(self, symbol: str) -> str:
        """Convert symbol to internal format (SLASH)
        
        Args:
            symbol: Symbol to convert (any common format)
            
        Returns:
            str: Symbol in internal format (BTC/USDC)
        """
        return self.standardize(symbol, self.FORMAT_SLASH)
    
    def for_api(self, symbol: str) -> str:
        """Convert symbol to API format (DIRECT)
        
        Args:
            symbol: Symbol to convert (any common format)
            
        Returns:
            str: Symbol in API format (BTCUSDC)
        """
        return self.standardize(symbol, self.FORMAT_DIRECT)
    
    def for_mexc(self, symbol: str) -> str:
        """Convert symbol specifically for MEXC API
        
        Args:
            symbol: Symbol to convert (any common format)
            
        Returns:
            str: Symbol in MEXC format (BTCUSDC)
        """
        return self.standardize(symbol, self.FORMAT_DIRECT)
    
    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse symbol into base and quote assets
        
        Args:
            symbol: Symbol to parse
            
        Returns:
            tuple: (base_asset, quote_asset)
        """
        # Check for common separators
        if '/' in symbol:
            return symbol.split('/')
        elif '-' in symbol:
            return symbol.split('-')
        elif '_' in symbol:
            return symbol.split('_')
        
        # For direct format (BTCUSDC), try to identify by common assets
        for quote in sorted(self.common_quote_assets, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        
        # If all else fails, try common base assets
        for base in sorted(self.common_base_assets, key=len, reverse=True):
            if symbol.startswith(base):
                quote = symbol[len(base):]
                return base, quote
        
        logger.warning(f"Could not parse symbol: {symbol}")
        return None, None

# Example usage
if __name__ == "__main__":
    standardizer = SymbolStandardizer()
    
    # Test with different formats
    symbols = ["BTC/USDC", "BTCUSDC", "BTC-USDC", "BTC_USDC"]
    
    for symbol in symbols:
        print(f"Original: {symbol}")
        print(f"Internal: {standardizer.for_internal(symbol)}")
        print(f"API: {standardizer.for_api(symbol)}")
        print(f"MEXC: {standardizer.for_exchange(symbol, 'mexc')}")
        print(f"Bitvavo: {standardizer.for_exchange(symbol, 'bitvavo')}")
        print("---")
