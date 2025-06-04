#!/usr/bin/env python
"""
Symbol Mapping Fix for Trading-Agent System

This module updates the symbol standardization logic to ensure compatibility
with MEXC API and other components of the Trading-Agent system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("symbol_mapping_fix.log")
    ]
)

logger = logging.getLogger("symbol_mapping_fix")

class UpdatedSymbolStandardizer:
    """Updated symbol standardization utility with MEXC compatibility"""
    
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
            "mexc": self.FORMAT_DIRECT,  # MEXC uses direct format (BTCUSDC)
            "binance": self.FORMAT_DIRECT,
            "kraken": self.FORMAT_DASH,
            "bitvavo": self.FORMAT_DASH,
            "bybit": self.FORMAT_DIRECT,
            "internal": self.FORMAT_SLASH
        }
        
        logger.info("Updated symbol standardizer initialized")
    
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

def update_symbol_standardization():
    """Update the symbol_standardization.py file with MEXC-specific improvements"""
    try:
        # Read the original file
        with open('symbol_standardization.py', 'r') as f:
            content = f.read()
        
        # Make a backup
        with open('symbol_standardization.py.bak', 'w') as f:
            f.write(content)
        
        # Add MEXC-specific method if not present
        if 'def for_mexc' not in content:
            # Find the for_api method
            api_method_pos = content.find('def for_api')
            if api_method_pos != -1:
                # Find the end of the for_api method
                next_def_pos = content.find('def', api_method_pos + 1)
                if next_def_pos != -1:
                    # Insert the for_mexc method before the next method
                    mexc_method = """
    def for_mexc(self, symbol: str) -> str:
        \"\"\"Convert symbol specifically for MEXC API
        
        Args:
            symbol: Symbol to convert (any common format)
            
        Returns:
            str: Symbol in MEXC format (BTCUSDC)
        \"\"\"
        return self.standardize(symbol, self.FORMAT_DIRECT)
    
"""
                    updated_content = content[:next_def_pos] + mexc_method + content[next_def_pos:]
                    
                    # Write the updated content
                    with open('symbol_standardization.py', 'w') as f:
                        f.write(updated_content)
                    
                    logger.info("Successfully added for_mexc method to symbol_standardization.py")
                else:
                    logger.error("Could not find the end of the for_api method")
                    return False
            else:
                logger.error("Could not find the for_api method")
                return False
        else:
            logger.info("for_mexc method already exists in symbol_standardization.py")
        
        return True
    except Exception as e:
        logger.error(f"Error updating symbol_standardization.py: {str(e)}")
        return False

def test_symbol_standardization():
    """Test the updated symbol standardization logic"""
    logger.info("Testing updated symbol standardization...")
    
    # Initialize standardizer
    standardizer = UpdatedSymbolStandardizer()
    
    # Test symbols
    test_symbols = [
        "BTC/USDC",
        "BTCUSDC",
        "BTC-USDC",
        "BTC_USDC",
        "ETH/USDT",
        "ETHUSDT",
        "SOL/USDC",
        "SOLUSDC"
    ]
    
    # Test standardization
    results = {}
    for symbol in test_symbols:
        internal = standardizer.for_internal(symbol)
        api = standardizer.for_api(symbol)
        mexc = standardizer.for_mexc(symbol)
        
        results[symbol] = {
            "internal": internal,
            "api": api,
            "mexc": mexc
        }
    
    # Log results
    logger.info(f"Standardization results: {json.dumps(results, indent=2)}")
    
    # Verify MEXC compatibility
    mexc_symbols = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for symbol in test_symbols:
        mexc_format = standardizer.for_mexc(symbol)
        if any(base_quote in mexc_format for base_quote in ["BTC/USDC", "ETH/USDC", "SOL/USDC"]):
            logger.error(f"Invalid MEXC format for {symbol}: {mexc_format}")
        elif mexc_format in mexc_symbols:
            logger.info(f"Valid MEXC format for {symbol}: {mexc_format}")
        else:
            logger.warning(f"Unverified MEXC format for {symbol}: {mexc_format}")
    
    logger.info("Symbol standardization test completed")

if __name__ == "__main__":
    # Test the updated symbol standardization
    test_symbol_standardization()
    
    # Update the symbol_standardization.py file
    update_symbol_standardization()
