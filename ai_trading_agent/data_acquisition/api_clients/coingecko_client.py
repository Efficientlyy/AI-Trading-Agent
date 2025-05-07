"""
CoinGecko API Client

Client for interacting with the CoinGecko public API to fetch cryptocurrency data.
"""

import aiohttp
import pandas as pd
import asyncio
import time
from typing import List, Dict, Any, Optional

from ...common import logger

class CoinGeckoClient:
    """Client for the CoinGecko API."""
    
    def __init__(self):
        """Initialize the CoinGecko API client."""
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.rate_limit_remaining = 50
        self.rate_limit_reset = 0
        self.coin_map = {}  # Maps trading pairs to CoinGecko IDs
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def _get_coin_map(self):
        """Get mapping of trading pairs to CoinGecko IDs."""
        if not self.coin_map:
            await self._ensure_session()
            
            try:
                async with self.session.get(f"{self.base_url}/coins/list") as response:
                    if response.status == 200:
                        coins = await response.json()
                        # Create a map of symbol to ID
                        for coin in coins:
                            self.coin_map[coin['symbol'].upper()] = coin['id']
                    else:
                        logger.error(f"Failed to get coin list: {response.status}")
            except Exception as e:
                logger.error(f"Error getting coin map: {e}")
    
    async def _convert_pair_to_id(self, pair: str) -> Optional[str]:
        """Convert trading pair (e.g., BTC/USDT) to CoinGecko coin ID."""
        await self._get_coin_map()
        
        # Split the pair
        base, quote = pair.split('/')
        
        # Return the ID for the base currency
        return self.coin_map.get(base.upper())
    
    async def _handle_rate_limit(self):
        """Handle API rate limiting."""
        if self.rate_limit_remaining <= 5:
            wait_time = max(0, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from CoinGecko."""
        await self._ensure_session()
        await self._handle_rate_limit()
        
        coin_id = await self._convert_pair_to_id(symbol)
        if not coin_id:
            logger.warning(f"Could not find CoinGecko ID for {symbol}")
            return None
        
        # Convert timeframe to days
        days = 1
        if timeframe == '1d':
            days = 1
        elif timeframe == '1h':
            days = max(1, int((end_date - start_date).total_seconds() / 3600))
        
        # CoinGecko uses Unix timestamps in milliseconds
        from_timestamp = int(start_date.timestamp() * 1000)
        to_timestamp = int(end_date.timestamp() * 1000)
        
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': from_timestamp // 1000,
                'to': to_timestamp // 1000
            }
            
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 50))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Process the data
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', [])
                    
                    if not prices:
                        return None
                    
                    # Create a DataFrame
                    df_data = []
                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0
                        df_data.append({
                            'timestamp': pd.Timestamp(timestamp, unit='ms', tz='UTC'),
                            'open': price,  # CoinGecko doesn't provide OHLC, just prices
                            'high': price,
                            'low': price,
                            'close': price,
                            'volume': volume
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    
                    return df
                else:
                    logger.error(f"Failed to get historical data: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current prices for multiple symbols."""
        await self._ensure_session()
        await self._handle_rate_limit()
        
        # Convert symbols to CoinGecko IDs
        coin_ids = []
        symbol_to_id = {}
        for symbol in symbols:
            coin_id = await self._convert_pair_to_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
                symbol_to_id[coin_id] = symbol
        
        if not coin_ids:
            logger.warning("No valid coin IDs found")
            return {}
        
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': ','.join(coin_ids),
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 50))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                
                if response.status == 200:
                    data = await response.json()
                    
                    result = {}
                    for coin in data:
                        coin_id = coin['id']
                        if coin_id in symbol_to_id:
                            symbol = symbol_to_id[coin_id]
                            result[symbol] = {
                                'price': coin['current_price'],
                                'volume_24h': coin['total_volume'],
                                'market_cap': coin['market_cap'],
                                'price_change_24h': coin.get('price_change_percentage_24h', 0)
                            }
                    
                    return result
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
