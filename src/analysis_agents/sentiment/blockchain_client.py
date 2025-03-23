"""Blockchain data API clients.

This module provides clients for fetching on-chain metrics from blockchain data providers
like Blockchain.com and Glassnode.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import aiohttp

from src.common.config import config
from src.common.logging import get_logger


class BaseBlockchainClient:
    """Base class for blockchain data API clients."""
    
    def __init__(self, api_key: str):
        """Initialize the blockchain client.
        
        Args:
            api_key: API key for blockchain data provider
        """
        self.api_key = api_key
        self.logger = get_logger("clients", "blockchain_base")
        self._session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session.
        
        Returns:
            An aiohttp client session for making API requests
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def get_large_transactions(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get large transaction data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with large transaction data
        """
        raise NotImplementedError("Method must be implemented by subclass")
    
    async def get_active_addresses(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get active address data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with active address data
        """
        raise NotImplementedError("Method must be implemented by subclass")
    
    async def get_hash_rate(self, asset: str, time_period: str = "7d") -> Optional[Dict[str, Any]]:
        """Get hash rate data for a PoW asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with hash rate data, or None if not applicable
        """
        raise NotImplementedError("Method must be implemented by subclass")
    
    async def get_exchange_reserves(self, asset: str, time_period: str = "7d") -> Dict[str, Any]:
        """Get exchange reserves data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with exchange reserves data
        """
        raise NotImplementedError("Method must be implemented by subclass")


class BlockchainComClient(BaseBlockchainClient):
    """Client for Blockchain.com's API."""
    
    def __init__(self, api_key: str = ""):
        """Initialize the Blockchain.com client.
        
        Args:
            api_key: API key for Blockchain.com (optional for some endpoints)
        """
        super().__init__(api_key)
        self.logger = get_logger("clients", "blockchain_com")
        self.base_url = "https://api.blockchain.info"
        self.cache = {}
        self.cache_expiry = {}
        self.default_cache_duration = timedelta(minutes=15)  # Default cache expiry
    
    async def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to Blockchain.com.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            API response as dictionary
        """
        # Create cache key
        cache_key = f"{endpoint}:{json.dumps(params) if params else ''}"
        
        # Check cache
        if cache_key in self.cache and datetime.utcnow() < self.cache_expiry.get(cache_key, datetime.min):
            self.logger.debug("Using cached data", endpoint=endpoint)
            return self.cache[cache_key]
        
        # Build URL
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key if provided
        if params is None:
            params = {}
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            session = await self._get_session()
            
            # Make request
            self.logger.debug("Making API request", url=url)
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error("API request failed", 
                                      endpoint=endpoint, 
                                      status=response.status,
                                      response=await response.text())
                    raise Exception(f"API error: {response.status}")
                
                # Parse JSON response
                data = await response.json()
                
                # Cache result
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = datetime.utcnow() + self.default_cache_duration
                
                return data
                
        except aiohttp.ClientError as e:
            self.logger.error("HTTP error", endpoint=endpoint, error=str(e))
            raise Exception(f"HTTP error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON response", endpoint=endpoint, error=str(e))
            raise Exception(f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.logger.error("Error making API request", endpoint=endpoint, error=str(e))
            raise
    
    async def get_large_transactions(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get large transaction data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with large transaction data
        """
        if asset != "BTC":
            raise ValueError("Blockchain.com API only supports BTC")
        
        try:
            # Blockchain.com doesn't have a dedicated large transactions endpoint,
            # so we'll use the latest transactions and filter for large ones
            data = await self._make_api_request("unconfirmed-transactions")
            
            if "txs" not in data:
                raise Exception("Invalid API response: missing 'txs' field")
            
            transactions = data["txs"]
            
            # Filter for large transactions (>= 1 BTC)
            large_transactions = []
            total_volume = 0
            
            for tx in transactions:
                # Convert value from satoshis to BTC
                value_btc = tx.get("value", 0) / 100000000
                
                if value_btc >= 1.0:  # Consider transactions >= 1 BTC as "large"
                    large_transactions.append(tx)
                    total_volume += value_btc
            
            # Get time window
            if time_period == "24h":
                time_window = 24
            elif time_period == "7d":
                time_window = 24 * 7
            else:
                time_window = 24  # Default to 24h
            
            # Estimate total volume based on the sampling
            # (This is an approximation since we don't have full historical data)
            estimated_full_volume = total_volume * (time_window / 0.5)  # Assuming our sample is ~30 minutes
            
            # Get historical average from cache or use a reasonable estimate
            cache_key = f"avg_volume:{asset}:{time_period}"
            average_volume = self.cache.get(cache_key, estimated_full_volume * 0.8)  # Use 80% of current as fallback
            
            # Update average (simple exponential moving average)
            new_average = (0.8 * average_volume) + (0.2 * estimated_full_volume)
            self.cache[cache_key] = new_average
            
            # Calculate change percentage
            change_percentage = ((estimated_full_volume / average_volume) - 1) * 100
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": len(large_transactions),
                "volume": estimated_full_volume,
                "average_volume": average_volume,
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting large transactions", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "count": 0,
                "volume": 0,
                "average_volume": 0,
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_active_addresses(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get active address data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with active address data
        """
        if asset != "BTC":
            raise ValueError("Blockchain.com API only supports BTC")
        
        try:
            # Calculate number of days
            days = 1
            if time_period == "7d":
                days = 7
            elif time_period == "30d":
                days = 30
                
            # Get number of unique addresses
            data = await self._make_api_request("charts/n-unique-addresses", {
                "timespan": f"{days}days",
                "format": "json"
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("Invalid API response")
            
            # Calculate average active addresses over the period
            active_addresses = sum(entry.get("y", 0) for entry in data) / len(data)
            
            # Get previous period for comparison
            prev_data = await self._make_api_request("charts/n-unique-addresses", {
                "timespan": f"{days}days",
                "format": "json",
                "start": (datetime.utcnow() - timedelta(days=days*2)).timestamp() * 1000,
                "end": (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
            })
            
            if prev_data and isinstance(prev_data, list) and len(prev_data) > 0:
                prev_active_addresses = sum(entry.get("y", 0) for entry in prev_data) / len(prev_data)
                change_percentage = ((active_addresses / prev_active_addresses) - 1) * 100
            else:
                change_percentage = 0
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": int(active_addresses),
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting active addresses", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "count": 0,
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_hash_rate(self, asset: str, time_period: str = "7d") -> Optional[Dict[str, Any]]:
        """Get hash rate data for a PoW asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with hash rate data, or None if not applicable
        """
        if asset != "BTC":
            return None  # Only BTC has hash rate data
        
        try:
            # Calculate number of days
            days = 7
            if time_period == "24h":
                days = 1
            elif time_period == "30d":
                days = 30
                
            # Get hash rate data
            data = await self._make_api_request("charts/hash-rate", {
                "timespan": f"{days}days",
                "format": "json"
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("Invalid API response")
            
            # Get latest hash rate (in TH/s)
            hash_rate = data[-1].get("y", 0)
            
            # Convert to EH/s for consistent reporting
            hash_rate_eh = hash_rate / 1000000  # TH/s to EH/s
            
            # Calculate change percentage
            if len(data) >= 2:
                previous_hash_rate = data[0].get("y", 0) / 1000000  # First entry in the period
                change_percentage = ((hash_rate_eh / previous_hash_rate) - 1) * 100
            else:
                change_percentage = 0
            
            return {
                "asset": asset,
                "time_period": time_period,
                "hash_rate": hash_rate_eh,
                "units": "EH/s",
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting hash rate", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "hash_rate": 0,
                "units": "EH/s",
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_exchange_reserves(self, asset: str, time_period: str = "7d") -> Dict[str, Any]:
        """Get exchange reserves data for an asset.
        
        Note: Blockchain.com doesn't provide direct exchange reserves data.
        This is an approximation using the estimated balance of major exchanges.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with exchange reserves data
        """
        # Blockchain.com doesn't have a direct exchange reserves endpoint,
        # so we'll return estimated data for now
        
        # In a production implementation, you might:
        # 1. Track known exchange addresses manually
        # 2. Subscribe to a service that provides this data
        # 3. Use address tags to identify exchange wallets
        
        if asset != "BTC":
            raise ValueError("Blockchain.com API only supports BTC")
            
        # These are estimated values based on typical industry trends
        reserves = 1500000  # ~1.5M BTC on exchanges
        change_percentage = -2.5  # Typical weekly outflow
            
        return {
            "asset": asset,
            "time_period": time_period,
            "reserves": reserves,
            "change_percentage": change_percentage,
            "timestamp": datetime.utcnow().isoformat(),
            "is_estimated": True
        }


class GlassnodeClient(BaseBlockchainClient):
    """Client for Glassnode's API."""
    
    def __init__(self, api_key: str):
        """Initialize the Glassnode client.
        
        Args:
            api_key: API key for Glassnode
        """
        super().__init__(api_key)
        self.logger = get_logger("clients", "glassnode")
        self.base_url = "https://api.glassnode.com/v1"
        self.cache = {}
        self.cache_expiry = {}
        self.default_cache_duration = timedelta(minutes=15)  # Default cache expiry
        
        # Asset mapping from our symbols to Glassnode's tickers
        self.asset_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "LTC": "litecoin",
            "XRP": "ripple",
            "BCH": "bitcoin-cash",
            "DOT": "polkadot",
            "SOL": "solana",
            "ADA": "cardano",
            "DOGE": "dogecoin",
            # Add more as needed
        }
    
    async def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to Glassnode.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            API response as dictionary
        """
        # Create cache key
        cache_key = f"{endpoint}:{json.dumps(params) if params else ''}"
        
        # Check cache
        if cache_key in self.cache and datetime.utcnow() < self.cache_expiry.get(cache_key, datetime.min):
            self.logger.debug("Using cached data", endpoint=endpoint)
            return self.cache[cache_key]
        
        # Build URL
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key if provided
        if params is None:
            params = {}
        
        params["api_key"] = self.api_key
        
        try:
            session = await self._get_session()
            
            # Make request
            self.logger.debug("Making API request", url=url)
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error("API request failed", 
                                      endpoint=endpoint, 
                                      status=response.status,
                                      response=await response.text())
                    raise Exception(f"API error: {response.status}")
                
                # Parse JSON response
                data = await response.json()
                
                # Cache result
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = datetime.utcnow() + self.default_cache_duration
                
                return data
                
        except aiohttp.ClientError as e:
            self.logger.error("HTTP error", endpoint=endpoint, error=str(e))
            raise Exception(f"HTTP error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON response", endpoint=endpoint, error=str(e))
            raise Exception(f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.logger.error("Error making API request", endpoint=endpoint, error=str(e))
            raise
    
    def _get_time_params(self, time_period: str) -> Dict[str, Any]:
        """Get time parameters based on time period.
        
        Args:
            time_period: Time period (e.g., "24h", "7d")
            
        Returns:
            Dictionary with time parameters
        """
        now = datetime.utcnow()
        
        if time_period == "24h":
            start_time = now - timedelta(days=1)
            interval = "1h"
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
            interval = "24h"
        elif time_period == "30d":
            start_time = now - timedelta(days=30)
            interval = "24h"
        else:
            start_time = now - timedelta(days=1)
            interval = "1h"
        
        # Format timestamps for Glassnode
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(now.timestamp())
        
        return {
            "a": interval,  # interval
            "s": start_timestamp,  # start timestamp
            "u": end_timestamp  # end timestamp
        }
    
    def _get_glassnode_asset(self, asset: str) -> str:
        """Convert our asset symbol to Glassnode's format.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            
        Returns:
            Glassnode's asset identifier
        """
        return self.asset_mapping.get(asset, asset.lower())
    
    async def get_large_transactions(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get large transaction data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with large transaction data
        """
        try:
            glassnode_asset = self._get_glassnode_asset(asset)
            
            # Get time parameters
            time_params = self._get_time_params(time_period)
            
            # Use the large transaction metric
            params = {
                **time_params,
                "i": "24h"  # interval
            }
            
            # Fetch large transactions count (transfers > $100k)
            data = await self._make_api_request(f"metrics/transactions/transfers_volume_more_than_100k_count", {
                **params,
                "c": glassnode_asset
            })
            
            if not data or not isinstance(data, list):
                raise Exception("Invalid API response")
            
            # Calculate count and volume
            count = sum(entry.get("v", 0) for entry in data)
            
            # Fetch large transactions volume (transfers > $100k)
            volume_data = await self._make_api_request(f"metrics/transactions/transfers_volume_more_than_100k_sum", {
                **params,
                "c": glassnode_asset
            })
            
            if not volume_data or not isinstance(volume_data, list):
                raise Exception("Invalid volume API response")
            
            # Calculate volume
            volume = sum(entry.get("v", 0) for entry in volume_data)
            
            # Get historical average
            # Calculate previous period
            now = datetime.utcnow()
            if time_period == "24h":
                prev_start = now - timedelta(days=2)
                prev_end = now - timedelta(days=1)
            elif time_period == "7d":
                prev_start = now - timedelta(days=14)
                prev_end = now - timedelta(days=7)
            else:
                prev_start = now - timedelta(days=2)
                prev_end = now - timedelta(days=1)
                
            prev_params = {
                "a": time_params["a"],
                "s": int(prev_start.timestamp()),
                "u": int(prev_end.timestamp()),
                "i": "24h"
            }
            
            # Fetch previous period volume
            prev_volume_data = await self._make_api_request(f"metrics/transactions/transfers_volume_more_than_100k_sum", {
                **prev_params,
                "c": glassnode_asset
            })
            
            if prev_volume_data and isinstance(prev_volume_data, list) and len(prev_volume_data) > 0:
                prev_volume = sum(entry.get("v", 0) for entry in prev_volume_data)
                average_volume = prev_volume
                
                # Calculate change percentage
                if average_volume > 0:
                    change_percentage = ((volume / average_volume) - 1) * 100
                else:
                    change_percentage = 0
            else:
                average_volume = volume * 0.9  # Fallback
                change_percentage = 10  # Default
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": int(count),
                "volume": volume,
                "average_volume": average_volume,
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting large transactions", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "count": 0,
                "volume": 0,
                "average_volume": 0,
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_active_addresses(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get active address data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with active address data
        """
        try:
            glassnode_asset = self._get_glassnode_asset(asset)
            
            # Get time parameters
            time_params = self._get_time_params(time_period)
            
            # Fetch active addresses count
            data = await self._make_api_request(f"metrics/addresses/active_count", {
                **time_params,
                "c": glassnode_asset
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("Invalid API response")
            
            # Calculate average active addresses over the period
            active_addresses = sum(entry.get("v", 0) for entry in data) / len(data)
            
            # Calculate previous period for comparison
            now = datetime.utcnow()
            if time_period == "24h":
                prev_start = now - timedelta(days=2)
                prev_end = now - timedelta(days=1)
            elif time_period == "7d":
                prev_start = now - timedelta(days=14)
                prev_end = now - timedelta(days=7)
            else:
                prev_start = now - timedelta(days=2)
                prev_end = now - timedelta(days=1)
                
            prev_params = {
                "a": time_params["a"],
                "s": int(prev_start.timestamp()),
                "u": int(prev_end.timestamp())
            }
            
            # Fetch previous period data
            prev_data = await self._make_api_request(f"metrics/addresses/active_count", {
                **prev_params,
                "c": glassnode_asset
            })
            
            if prev_data and isinstance(prev_data, list) and len(prev_data) > 0:
                prev_active_addresses = sum(entry.get("v", 0) for entry in prev_data) / len(prev_data)
                change_percentage = ((active_addresses / prev_active_addresses) - 1) * 100
            else:
                change_percentage = 0
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": int(active_addresses),
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting active addresses", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "count": 0,
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_hash_rate(self, asset: str, time_period: str = "7d") -> Optional[Dict[str, Any]]:
        """Get hash rate data for a PoW asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with hash rate data, or None if not applicable
        """
        # Only certain PoW assets have hash rate
        if asset not in ["BTC", "ETH", "LTC", "BCH"]:
            return None
            
        # ETH switched to PoS, so no longer has hash rate
        if asset == "ETH":
            return None
        
        try:
            glassnode_asset = self._get_glassnode_asset(asset)
            
            # Get time parameters
            time_params = self._get_time_params(time_period)
            
            # Fetch hash rate data
            data = await self._make_api_request(f"metrics/mining/hash_rate_mean", {
                **time_params,
                "c": glassnode_asset
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("Invalid API response")
            
            # Get latest hash rate
            latest_hash_rate = data[-1].get("v", 0)
            
            # Convert to EH/s for consistent reporting
            # Glassnode returns hash rate in different units depending on the asset
            if asset == "BTC":
                hash_rate_eh = latest_hash_rate / 1_000_000_000_000_000_000  # Hash/s to EH/s
            else:
                hash_rate_eh = latest_hash_rate / 1_000_000  # MH/s to EH/s
            
            # Calculate change percentage
            if len(data) >= 2:
                if asset == "BTC":
                    previous_hash_rate = data[0].get("v", 0) / 1_000_000_000_000_000_000
                else:
                    previous_hash_rate = data[0].get("v", 0) / 1_000_000
                
                change_percentage = ((hash_rate_eh / previous_hash_rate) - 1) * 100
            else:
                change_percentage = 0
            
            return {
                "asset": asset,
                "time_period": time_period,
                "hash_rate": hash_rate_eh,
                "units": "EH/s",
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting hash rate", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            return {
                "asset": asset,
                "time_period": time_period,
                "hash_rate": 0,
                "units": "EH/s",
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }
    
    async def get_exchange_reserves(self, asset: str, time_period: str = "7d") -> Dict[str, Any]:
        """Get exchange reserves data for an asset.
        
        Args:
            asset: The cryptocurrency asset (e.g., "BTC")
            time_period: Time period for the data (e.g., "24h", "7d")
            
        Returns:
            Dictionary with exchange reserves data
        """
        try:
            glassnode_asset = self._get_glassnode_asset(asset)
            
            # Get time parameters
            time_params = self._get_time_params(time_period)
            
            # Fetch exchange balance data
            data = await self._make_api_request(f"metrics/distribution/balance_exchanges", {
                **time_params,
                "c": glassnode_asset
            })
            
            if not data or not isinstance(data, list) or len(data) == 0:
                raise Exception("Invalid API response")
            
            # Get current reserves
            latest_reserves = data[-1].get("v", 0)
            
            # Calculate change percentage
            if len(data) >= 2:
                # Get earliest data point in the requested time period
                previous_reserves = data[0].get("v", 0)
                change_percentage = ((latest_reserves / previous_reserves) - 1) * 100
            else:
                change_percentage = 0
            
            return {
                "asset": asset,
                "time_period": time_period,
                "reserves": latest_reserves,
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting exchange reserves", 
                           asset=asset, 
                           time_period=time_period,
                           error=str(e))
            
            # Return fallback data
            # Use realistic fallback values based on asset
            if asset == "BTC":
                reserves = 1500000
            elif asset == "ETH":
                reserves = 15000000
            else:
                reserves = 1000000
                
            return {
                "asset": asset,
                "time_period": time_period,
                "reserves": reserves,
                "change_percentage": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True
            }


class BlockchainClientFactory:
    """Factory for creating blockchain clients."""
    
    @staticmethod
    def create_client(provider: str, api_key: str) -> BaseBlockchainClient:
        """Create a blockchain client for the specified provider.
        
        Args:
            provider: The blockchain data provider ("blockchain_com" or "glassnode")
            api_key: API key for the provider
            
        Returns:
            A blockchain client instance
        """
        logger = get_logger("clients", "blockchain_factory")
        
        if provider == "blockchain_com":
            logger.info("Creating Blockchain.com client")
            return BlockchainComClient(api_key)
        elif provider == "glassnode":
            logger.info("Creating Glassnode client")
            return GlassnodeClient(api_key)
        else:
            logger.error("Unknown blockchain provider", provider=provider)
            raise ValueError(f"Unknown blockchain provider: {provider}")


# Generic client that combines data from multiple sources
class BlockchainClient(BaseBlockchainClient):
    """Client for fetching blockchain data from multiple providers."""
    
    def __init__(self, blockchain_com_api_key: str = "", glassnode_api_key: str = ""):
        """Initialize the blockchain client.
        
        Args:
            blockchain_com_api_key: API key for Blockchain.com
            glassnode_api_key: API key for Glassnode
        """
        super().__init__("")  # Empty API key for base class
        self.logger = get_logger("clients", "blockchain")
        
        # Initialize clients
        self.clients = {}
        
        if blockchain_com_api_key:
            self.clients["blockchain_com"] = BlockchainClientFactory.create_client(
                "blockchain_com", blockchain_com_api_key)
        
        if glassnode_api_key:
            self.clients["glassnode"] = BlockchainClientFactory.create_client(
                "glassnode", glassnode_api_key)
        
        # Default client priority order
        self.client_priority = ["glassnode", "blockchain_com"]
    
    async def close(self):
        """Close all client sessions."""
        for client in self.clients.values():
            await client.close()
    
    async def _call_clients(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Call a method across multiple clients until one succeeds.
        
        Args:
            method_name: The method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            The result from the first successful client
        """
        # Try clients in priority order
        errors = []
        
        for provider in self.client_priority:
            client = self.clients.get(provider)
            if not client:
                continue
                
            try:
                method = getattr(client, method_name)
                result = await method(*args, **kwargs)
                if result and not result.get("is_fallback", False):
                    return result
                else:
                    # If the result is a fallback, try the next client
                    errors.append(f"{provider} returned fallback data")
                    continue
            except Exception as e:
                errors.append(f"{provider} error: {str(e)}")
                continue
        
        # If all clients failed, return mock data
        self.logger.warning(f"All clients failed for {method_name}, using mock data",
                         args=args,
                         kwargs=kwargs,
                         errors=errors)
        
        # Return mock data as a last resort
        return await self._get_mock_data(method_name, *args, **kwargs)
    
    async def _get_mock_data(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Generate mock data for when all API clients fail.
        
        Args:
            method_name: The method name that was called
            *args: Positional arguments from the original call
            **kwargs: Keyword arguments from the original call
            
        Returns:
            Mock data that matches the expected format
        """
        import random
        
        # Extract common parameters
        asset = args[0] if len(args) > 0 else kwargs.get("asset", "BTC")
        time_period = args[1] if len(args) > 1 else kwargs.get("time_period", "24h")
        
        # Set base values based on asset
        base_values = {
            "BTC": {
                "large_tx_count": 150,
                "large_tx_volume": 1000000000,
                "active_addresses": 1000000,
                "hash_rate": 350,
                "exchange_reserves": 1500000
            },
            "ETH": {
                "large_tx_count": 500,
                "large_tx_volume": 500000000,
                "active_addresses": 500000,
                "hash_rate": None,  # ETH is PoS now
                "exchange_reserves": 15000000
            },
            "SOL": {
                "large_tx_count": 300,
                "large_tx_volume": 100000000,
                "active_addresses": 200000,
                "hash_rate": None,
                "exchange_reserves": 50000000
            }
        }.get(asset, {
            "large_tx_count": 100,
            "large_tx_volume": 50000000,
            "active_addresses": 100000,
            "hash_rate": None,
            "exchange_reserves": 1000000
        })
        
        if method_name == "get_large_transactions":
            # Add some randomness
            count = int(base_values["large_tx_count"] * random.uniform(0.7, 1.3))
            volume = base_values["large_tx_volume"] * random.uniform(0.7, 1.3)
            
            # Calculate change percentage from "normal"
            change_percentage = random.uniform(-20, 20)  # -20% to +20%
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": count,
                "volume": volume,
                "average_volume": base_values["large_tx_volume"],
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat(),
                "is_mock": True
            }
            
        elif method_name == "get_active_addresses":
            # Add some randomness
            count = int(base_values["active_addresses"] * random.uniform(0.7, 1.3))
            
            # Calculate change percentage from "normal"
            change_percentage = random.uniform(-15, 15)  # -15% to +15%
            
            return {
                "asset": asset,
                "time_period": time_period,
                "count": count,
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat(),
                "is_mock": True
            }
            
        elif method_name == "get_hash_rate":
            # Only certain assets have hash rate
            if base_values["hash_rate"] is None:
                return None
                
            # Add some randomness
            hash_rate = base_values["hash_rate"] * random.uniform(0.9, 1.1)
            
            # Calculate change percentage from "normal"
            change_percentage = random.uniform(-10, 10)  # -10% to +10%
            
            return {
                "asset": asset,
                "time_period": time_period,
                "hash_rate": hash_rate,
                "units": "EH/s",
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat(),
                "is_mock": True
            }
            
        elif method_name == "get_exchange_reserves":
            # Add some randomness
            reserves = base_values["exchange_reserves"] * random.uniform(0.9, 1.1)
            
            # Calculate change percentage from "normal"
            # Negative means tokens leaving exchanges (generally bullish)
            # Positive means tokens entering exchanges (generally bearish)
            change_percentage = random.uniform(-8, 8)  # -8% to +8%
            
            return {
                "asset": asset,
                "time_period": time_period,
                "reserves": reserves,
                "change_percentage": change_percentage,
                "timestamp": datetime.utcnow().isoformat(),
                "is_mock": True
            }
            
        else:
            # Unknown method, return empty dict
            return {}
    
    async def get_large_transactions(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get large transaction data for an asset."""
        return await self._call_clients("get_large_transactions", asset, time_period)
    
    async def get_active_addresses(self, asset: str, time_period: str = "24h") -> Dict[str, Any]:
        """Get active address data for an asset."""
        return await self._call_clients("get_active_addresses", asset, time_period)
    
    async def get_hash_rate(self, asset: str, time_period: str = "7d") -> Optional[Dict[str, Any]]:
        """Get hash rate data for a PoW asset."""
        return await self._call_clients("get_hash_rate", asset, time_period)
    
    async def get_exchange_reserves(self, asset: str, time_period: str = "7d") -> Dict[str, Any]:
        """Get exchange reserves data for an asset."""
        return await self._call_clients("get_exchange_reserves", asset, time_period)