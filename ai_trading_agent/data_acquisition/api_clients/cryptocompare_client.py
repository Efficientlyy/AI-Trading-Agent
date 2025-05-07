"""
CryptoCompare API Client

Client for interacting with the CryptoCompare public API to fetch cryptocurrency data.
"""

import aiohttp
import pandas as pd
import time
from typing import List, Dict, Any, Optional

from ...common import logger

class CryptoCompareClient:
    """Client for the CryptoCompare API."""
    
    def __init__(self):
        """Initialize the CryptoCompare API client."""
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.session = None
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from CryptoCompare."""
        await self._ensure_session()
        
        # Split the pair
        base, quote = symbol.split('/')
        
        # Map timeframe to CryptoCompare format
        timeframe_map = {
            '1m': 'histominute',
            '5m': 'histominute',
            '15m': 'histominute',
            '30m': 'histominute',
            '1h': 'histohour',
            '4h': 'histohour',
            '1d': 'histoday',
            '1w': 'histoday'
        }
        
        # Calculate limit based on timeframe
        limit_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        endpoint = timeframe_map.get(timeframe, 'histoday')
        limit = min(2000, int((end_date - start_date).total_seconds() / 60 / limit_map.get(timeframe, 1)))
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params = {
                'fsym': base.upper(),
                'tsym': quote.upper(),
                'limit': limit,
                'toTs': int(end_date.timestamp())
            }
            
            # Add aggregation for timeframes
            if timeframe in ['5m', '15m', '30m']:
                params['aggregate'] = int(timeframe[:-1])
            elif timeframe == '4h':
                params['aggregate'] = 4
            elif timeframe == '1w':
                params['aggregate'] = 7
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['Response'] == 'Success':
                        # Process the data
                        ohlcv_data = data['Data']
                        
                        # Create a DataFrame
                        df_data = []
                        for item in ohlcv_data:
                            df_data.append({
                                'timestamp': pd.Timestamp(item['time'], unit='s', tz='UTC'),
                                'open': item['open'],
                                'high': item['high'],
                                'low': item['low'],
                                'close': item['close'],
                                'volume': item['volumefrom']
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        
                        return df
                    else:
                        logger.error(f"API error: {data.get('Message', 'Unknown error')}")
                        return None
                else:
                    logger.error(f"Failed to get historical data: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current prices for multiple symbols."""
        await self._ensure_session()
        
        # Extract base and quote currencies
        fsyms = []
        tsyms = []
        symbol_map = {}
        
        for symbol in symbols:
            base, quote = symbol.split('/')
            fsyms.append(base.upper())
            tsyms.append(quote.upper())
            symbol_map[(base.upper(), quote.upper())] = symbol
        
        try:
            url = f"{self.base_url}/pricemultifull"
            params = {
                'fsyms': ','.join(set(fsyms)),
                'tsyms': ','.join(set(tsyms))
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'RAW' in data:
                        result = {}
                        raw_data = data['RAW']
                        
                        for base in raw_data:
                            for quote in raw_data[base]:
                                if (base, quote) in symbol_map:
                                    symbol = symbol_map[(base, quote)]
                                    coin_data = raw_data[base][quote]
                                    result[symbol] = {
                                        'price': coin_data['PRICE'],
                                        'volume_24h': coin_data['VOLUME24HOUR'],
                                        'market_cap': coin_data.get('MKTCAP', 0),
                                        'price_change_24h': coin_data.get('CHANGEPCT24HOUR', 0)
                                    }
                        
                        return result
                    else:
                        logger.error("No RAW data in response")
                        return {}
                else:
                    logger.error(f"Failed to get current prices: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
