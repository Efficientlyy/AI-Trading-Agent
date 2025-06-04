#!/usr/bin/env python
"""
Debug Market Data Pipeline for BTCUSDC

This script isolates and tests the market data pipeline specifically for BTCUSDC
to identify and fix issues with symbol handling and interval mapping.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_market_data")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from fixed_multi_asset_data_service import MultiAssetDataService

def debug_symbol_mapping():
    """Debug symbol mapping between internal and API formats"""
    logger.info("=== Testing Symbol Mapping ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test symbol mapping
    internal_symbol = "BTC/USDC"
    api_symbol = data_service._get_symbol_for_api(internal_symbol)
    logger.info(f"Internal symbol: {internal_symbol} -> API symbol: {api_symbol}")
    
    # Test reverse mapping
    reverse_mapped = data_service._get_asset_from_api_symbol(api_symbol)
    logger.info(f"API symbol: {api_symbol} -> Internal symbol: {reverse_mapped}")
    
    # Test direct API symbol
    direct_api_symbol = "BTCUSDC"
    direct_internal = data_service._get_asset_from_api_symbol(direct_api_symbol)
    logger.info(f"Direct API symbol: {direct_api_symbol} -> Internal symbol: {direct_internal}")
    
    # Print symbol map
    logger.info(f"Symbol map: {data_service.symbol_map}")
    
    return {
        "internal_to_api": {internal_symbol: api_symbol},
        "api_to_internal": {api_symbol: reverse_mapped},
        "direct_api": {direct_api_symbol: direct_internal},
        "symbol_map": data_service.symbol_map
    }

def debug_interval_mapping():
    """Debug interval mapping and normalization"""
    logger.info("=== Testing Interval Mapping ===")
    
    # Initialize data service
    data_service = MultiAssetDataService()
    
    # Test interval normalization
    intervals = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    normalized = {}
    
    for interval in intervals:
        normalized_interval = data_service._normalize_interval(interval)
        logger.info(f"Original interval: {interval} -> Normalized: {normalized_interval}")
        normalized[interval] = normalized_interval
    
    # Print supported intervals
    logger.info(f"Supported intervals: {data_service.supported_intervals}")
    
    return {
        "normalized_intervals": normalized,
        "supported_intervals": data_service.supported_intervals
    }

def debug_ticker_request():
    """Debug ticker request for BTCUSDC"""
    logger.info("=== Testing Ticker Request ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test ticker request
    try:
        ticker = data_service.get_ticker("BTC/USDC")
        logger.info(f"Ticker for BTC/USDC: {ticker}")
        
        # Test direct API symbol
        ticker_api = data_service.get_ticker("BTCUSDC")
        logger.info(f"Ticker for BTCUSDC: {ticker_api}")
        
        return {
            "ticker_internal": ticker,
            "ticker_api": ticker_api
        }
    except Exception as e:
        logger.error(f"Error fetching ticker: {str(e)}")
        return {"error": str(e)}

def debug_orderbook_request():
    """Debug orderbook request for BTCUSDC"""
    logger.info("=== Testing Orderbook Request ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test orderbook request
    try:
        orderbook = data_service.get_orderbook("BTC/USDC")
        logger.info(f"Orderbook for BTC/USDC: {len(orderbook['asks'])} asks, {len(orderbook['bids'])} bids")
        
        # Test direct API symbol
        orderbook_api = data_service.get_orderbook("BTCUSDC")
        logger.info(f"Orderbook for BTCUSDC: {len(orderbook_api['asks'])} asks, {len(orderbook_api['bids'])} bids")
        
        return {
            "orderbook_internal": {
                "asks_count": len(orderbook['asks']),
                "bids_count": len(orderbook['bids'])
            },
            "orderbook_api": {
                "asks_count": len(orderbook_api['asks']),
                "bids_count": len(orderbook_api['bids'])
            }
        }
    except Exception as e:
        logger.error(f"Error fetching orderbook: {str(e)}")
        return {"error": str(e)}

def debug_trades_request():
    """Debug trades request for BTCUSDC"""
    logger.info("=== Testing Trades Request ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test trades request
    try:
        trades = data_service.get_trades("BTC/USDC")
        logger.info(f"Trades for BTC/USDC: {len(trades)} trades")
        
        # Test direct API symbol
        trades_api = data_service.get_trades("BTCUSDC")
        logger.info(f"Trades for BTCUSDC: {len(trades_api)} trades")
        
        return {
            "trades_internal": len(trades),
            "trades_api": len(trades_api)
        }
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        return {"error": str(e)}

def debug_klines_request():
    """Debug klines request for BTCUSDC"""
    logger.info("=== Testing Klines Request ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test klines request for different intervals
    intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
    results = {}
    
    for interval in intervals:
        try:
            logger.info(f"Fetching klines for BTC/USDC with interval {interval}")
            klines = data_service.get_klines("BTC/USDC", interval)
            logger.info(f"Klines for BTC/USDC ({interval}): {len(klines)} candles")
            results[f"internal_{interval}"] = len(klines)
            
            # Test direct API symbol
            logger.info(f"Fetching klines for BTCUSDC with interval {interval}")
            klines_api = data_service.get_klines("BTCUSDC", interval)
            logger.info(f"Klines for BTCUSDC ({interval}): {len(klines_api)} candles")
            results[f"api_{interval}"] = len(klines_api)
        except Exception as e:
            logger.error(f"Error fetching klines for {interval}: {str(e)}")
            results[f"error_{interval}"] = str(e)
    
    return results

def debug_api_request_url():
    """Debug API request URL construction for BTCUSDC"""
    logger.info("=== Testing API Request URL Construction ===")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
    
    # Test URL construction for klines
    symbol = data_service._get_symbol_for_api("BTC/USDC")
    interval = data_service._normalize_interval("1h")
    limit = 100
    
    url = f"{data_service.base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    logger.info(f"Constructed URL for klines: {url}")
    
    # Test direct request to the API
    import requests
    try:
        logger.info(f"Making direct request to: {url}")
        response = requests.get(url, timeout=10)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response content: {response.text[:200]}...")
        
        return {
            "url": url,
            "status_code": response.status_code,
            "response_preview": response.text[:200] if response.status_code == 200 else response.text
        }
    except Exception as e:
        logger.error(f"Error making direct request: {str(e)}")
        return {"error": str(e)}

def fix_multi_asset_data_service():
    """Fix issues in MultiAssetDataService"""
    logger.info("=== Fixing MultiAssetDataService ===")
    
    # Read current implementation
    with open('fixed_multi_asset_data_service.py', 'r') as f:
        current_code = f.read()
    
    # Create fixed implementation
    fixed_code = current_code.replace(
        "def get_klines(self, asset=None, interval=\"1m\", limit=100):",
        """def get_klines(self, asset=None, interval="1m", limit=100):
        \"\"\"Get klines (candlestick data) for specified asset
        
        Args:
            asset: Asset to get klines for (default: current asset)
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines to return
            
        Returns:
            list: Klines data
        \"\"\"
        target_asset = asset or self.current_asset
        
        # Handle direct API symbol format (e.g., BTCUSDC)
        if '/' not in target_asset:
            symbol = target_asset
        else:
            symbol = self._get_symbol_for_api(target_asset)
        
        # Debug logging
        logger.info(f"Processing klines request - Target asset: {target_asset}, API symbol: {symbol}")"""
    )
    
    # Fix interval normalization
    fixed_code = fixed_code.replace(
        "def _normalize_interval(self, interval):",
        """def _normalize_interval(self, interval):
        \"\"\"Normalize interval to MEXC supported format
        
        Args:
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            str: Normalized interval
        \"\"\"
        # MEXC supports these intervals directly
        if interval in ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1w', '1M']:
            return interval
            
        # Map common interval formats to MEXC supported formats"""
    )
    
    # Write fixed implementation
    with open('enhanced_multi_asset_data_service.py', 'w') as f:
        f.write(fixed_code)
    
    logger.info("Fixed implementation written to enhanced_multi_asset_data_service.py")
    return "enhanced_multi_asset_data_service.py"

def create_direct_klines_test():
    """Create a direct test for klines endpoint"""
    logger.info("=== Creating Direct Klines Test ===")
    
    # Create test script
    test_code = """#!/usr/bin/env python
\"\"\"
Direct Test for MEXC Klines Endpoint

This script directly tests the MEXC klines endpoint for BTCUSDC
without going through the MultiAssetDataService.
\"\"\"

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("direct_klines_test")

def test_klines_endpoint():
    \"\"\"Test klines endpoint directly\"\"\"
    # MEXC API base URL
    base_url = "https://api.mexc.com"
    
    # Test parameters
    symbol = "BTCUSDC"
    intervals = ["1m", "5m", "15m", "30m", "60m", "4h", "1d", "1w"]
    limit = 100
    
    results = {}
    
    for interval in intervals:
        url = f"{base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        logger.info(f"Testing URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Received {len(data)} candles for {interval}")
                results[interval] = len(data)
                
                # Print first candle for verification
                if data:
                    logger.info(f"First candle: {data[0]}")
            else:
                logger.error(f"Error response: {response.text}")
                results[interval] = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            results[interval] = f"Exception: {str(e)}"
    
    return results

if __name__ == "__main__":
    logger.info("Starting direct klines test")
    results = test_klines_endpoint()
    
    # Save results
    output_file = "direct_klines_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("=== Test Summary ===")
    for interval, result in results.items():
        logger.info(f"{interval}: {result}")
"""
    
    # Write test script
    with open('direct_klines_test.py', 'w') as f:
        f.write(test_code)
    
    logger.info("Direct klines test written to direct_klines_test.py")
    return "direct_klines_test.py"

def main():
    """Main function to run all debug tests"""
    logger.info("Starting market data pipeline debugging")
    
    # Create results directory
    os.makedirs("debug_results", exist_ok=True)
    
    # Run debug tests
    results = {
        "timestamp": datetime.now().isoformat(),
        "symbol_mapping": debug_symbol_mapping(),
        "interval_mapping": debug_interval_mapping(),
        "ticker_request": debug_ticker_request(),
        "orderbook_request": debug_orderbook_request(),
        "trades_request": debug_trades_request(),
        "klines_request": debug_klines_request(),
        "api_request_url": debug_api_request_url()
    }
    
    # Save results
    output_file = os.path.join("debug_results", f"market_data_debug_{int(time.time())}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Debug results saved to {output_file}")
    
    # Fix MultiAssetDataService
    fixed_file = fix_multi_asset_data_service()
    logger.info(f"Fixed implementation saved to {fixed_file}")
    
    # Create direct klines test
    test_file = create_direct_klines_test()
    logger.info(f"Direct klines test saved to {test_file}")
    
    return output_file, fixed_file, test_file

if __name__ == "__main__":
    main()
