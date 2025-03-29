"""
Financial data collector for the Early Event Detection System.

This module provides functionality for collecting financial market data
from various sources to detect anomalies and market-moving events.
"""

import asyncio
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import aiohttp
import numpy as np
from scipy import stats

from src.common.config import config
from src.common.logging import get_logger
from src.common.api_client import RetryableAPIClient, CircuitBreaker
from src.analysis_agents.early_detection.models import SourceType, EventSource
from src.analysis_agents.early_detection.optimization import get_cost_optimizer


class FinancialDataCollector:
    """Collector for financial market data."""
    
    def __init__(self):
        """Initialize the financial data collector."""
        self.logger = get_logger("early_detection", "financial_data_collector")
        
        # Configuration
        self.data_types = config.get("early_detection.data_collection.financial_data.types", 
                                    ["price", "volume", "options", "futures"])
        self.interval = config.get("early_detection.data_collection.financial_data.interval", "1h")
        self.assets = config.get("early_detection.assets", ["BTC", "ETH", "SOL", "XRP"])
        
        # API credentials
        self.cryptocompare_key = os.getenv("CRYPTOCOMPARE_KEY") or config.get("apis.cryptocompare.key", "")
        self.binance_key = os.getenv("BINANCE_API_KEY") or config.get("apis.binance.api_key", "")
        self.binance_secret = os.getenv("BINANCE_API_SECRET") or config.get("apis.binance.api_secret", "")
        
        # API client
        self.api_client = RetryableAPIClient(
            max_retries=3,
            backoff_factor=2.0,
            logger=self.logger
        )
        
        # Historical data cache (for anomaly detection)
        self.historical_data = {}
        
        # Last update time
        self.last_update = {}
    
    async def initialize(self):
        """Initialize the financial data collector."""
        self.logger.info("Initializing financial data collector")
        
        try:
            # Load historical data for baseline comparison
            self._load_historical_data()
            
            self.logger.info("Financial data collector initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing financial data collector: {e}")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect financial market data.
        
        Returns:
            List of collected data items
        """
        self.logger.info("Collecting financial market data")
        
        collected_data = []
        
        try:
            # Get cost optimizer for API efficiency
            cost_optimizer = await get_cost_optimizer()
            
            # Check if we should sample financial data based on adaptive sampling
            if not cost_optimizer.adaptive_sampler.should_sample("financial"):
                self.logger.info("Skipping financial data collection due to adaptive sampling")
                return []
            
            # Create collection tasks
            tasks = []
            
            # For each asset, collect different types of data
            for asset in self.assets:
                # Price data
                if "price" in self.data_types:
                    tasks.append(
                        asyncio.create_task(
                            cost_optimizer.api_request(
                                "financial",
                                self._collect_price_data,
                                asset
                            )
                        )
                    )
                
                # Volume data
                if "volume" in self.data_types:
                    tasks.append(
                        asyncio.create_task(
                            cost_optimizer.api_request(
                                "financial",
                                self._collect_volume_data,
                                asset
                            )
                        )
                    )
                
                # Options data (if available)
                if "options" in self.data_types:
                    tasks.append(
                        asyncio.create_task(
                            cost_optimizer.api_request(
                                "financial",
                                self._collect_options_data,
                                asset
                            )
                        )
                    )
                
                # Futures data (if available)
                if "futures" in self.data_types:
                    tasks.append(
                        asyncio.create_task(
                            cost_optimizer.api_request(
                                "financial",
                                self._collect_futures_data,
                                asset
                            )
                        )
                    )
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting financial data: {result}")
                    continue
                
                if result:
                    collected_data.extend(result)
            
            # Calculate market volatility for adaptive sampling
            if collected_data:
                market_volatility = self._calculate_market_volatility(collected_data)
                cost_optimizer.update_market_volatility(market_volatility)
                self.logger.info(f"Updated market volatility: {market_volatility:.4f}")
            
            self.logger.info(f"Collected {len(collected_data)} financial data items")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting financial data: {e}")
            return self._mock_collect()
    
    async def _collect_price_data(self, asset: str) -> List[Dict[str, Any]]:
        """Collect price data for an asset.
        
        Args:
            asset: The asset symbol (e.g., "BTC")
            
        Returns:
            List of collected price data
        """
        self.logger.debug(f"Collecting price data for {asset}")
        
        collected_data = []
        
        try:
            # Form trading pair
            trading_pair = f"{asset}/USD"
            
            # Get current price data from CryptoCompare
            current_data = await self._get_cryptocurrency_data(asset, "USD")
            
            if not current_data:
                self.logger.warning(f"No price data available for {asset}")
                return []
            
            # Extract price data
            price = current_data.get("PRICE", 0)
            open_24h = current_data.get("OPEN24HOUR", 0)
            high_24h = current_data.get("HIGH24HOUR", 0)
            low_24h = current_data.get("LOW24HOUR", 0)
            volume_24h = current_data.get("VOLUME24HOUR", 0)
            change_24h = current_data.get("CHANGE24HOUR", 0)
            change_pct_24h = current_data.get("CHANGEPCT24HOUR", 0)
            
            # Detect anomalies
            is_anomaly, anomaly_score, anomaly_details = self._detect_price_anomalies(
                asset, price, change_pct_24h
            )
            
            # Create source
            source = EventSource(
                id=f"price_{asset}_{int(time.time())}",
                type=SourceType.FINANCIAL_DATA,
                name=f"{asset} Price Data",
                url=None,
                reliability_score=0.95  # Financial data is highly reliable
            )
            
            # Create collected data item
            data_item = {
                "source": source,
                "data_type": "price",
                "asset": asset,
                "trading_pair": trading_pair,
                "timestamp": datetime.now(),
                "data": {
                    "price": price,
                    "open_24h": open_24h,
                    "high_24h": high_24h,
                    "low_24h": low_24h,
                    "volume_24h": volume_24h,
                    "change_24h": change_24h,
                    "change_pct_24h": change_pct_24h
                },
                "metadata": {
                    "interval": self.interval,
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "anomaly_details": anomaly_details
                }
            }
            
            # Update historical data
            self._update_historical_data(asset, "price", price)
            
            collected_data.append(data_item)
            
            self.logger.debug(f"Collected price data for {asset}: ${price:.2f} ({change_pct_24h:.2f}%)")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting price data for {asset}: {e}")
            return []
    
    async def _collect_volume_data(self, asset: str) -> List[Dict[str, Any]]:
        """Collect volume data for an asset.
        
        Args:
            asset: The asset symbol (e.g., "BTC")
            
        Returns:
            List of collected volume data
        """
        self.logger.debug(f"Collecting volume data for {asset}")
        
        collected_data = []
        
        try:
            # Form trading pair
            trading_pair = f"{asset}/USD"
            
            # Get current data from CryptoCompare
            current_data = await self._get_cryptocurrency_data(asset, "USD")
            
            if not current_data:
                self.logger.warning(f"No volume data available for {asset}")
                return []
            
            # Extract volume data
            volume_24h = current_data.get("VOLUME24HOUR", 0)
            volume_24h_to = current_data.get("VOLUME24HOURTO", 0)  # Volume in USD
            total_top_tier_volume_24h = current_data.get("TOTALTOPTIERVOLUME24HTO", 0)
            
            # Detect anomalies
            is_anomaly, anomaly_score, anomaly_details = self._detect_volume_anomalies(
                asset, volume_24h
            )
            
            # Create source
            source = EventSource(
                id=f"volume_{asset}_{int(time.time())}",
                type=SourceType.FINANCIAL_DATA,
                name=f"{asset} Volume Data",
                url=None,
                reliability_score=0.95  # Financial data is highly reliable
            )
            
            # Create collected data item
            data_item = {
                "source": source,
                "data_type": "volume",
                "asset": asset,
                "trading_pair": trading_pair,
                "timestamp": datetime.now(),
                "data": {
                    "volume_24h": volume_24h,
                    "volume_24h_usd": volume_24h_to,
                    "total_top_tier_volume_24h": total_top_tier_volume_24h
                },
                "metadata": {
                    "interval": self.interval,
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "anomaly_details": anomaly_details
                }
            }
            
            # Update historical data
            self._update_historical_data(asset, "volume", volume_24h)
            
            collected_data.append(data_item)
            
            self.logger.debug(f"Collected volume data for {asset}: {volume_24h:.2f} ({volume_24h_to:.2f} USD)")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting volume data for {asset}: {e}")
            return []
    
    async def _collect_options_data(self, asset: str) -> List[Dict[str, Any]]:
        """Collect options data for an asset.
        
        Args:
            asset: The asset symbol (e.g., "BTC")
            
        Returns:
            List of collected options data
        """
        self.logger.debug(f"Collecting options data for {asset}")
        
        # Note: For most cryptocurrencies, options data might not be readily available
        # through free APIs. In a production system, you would integrate with
        # specialized data providers like Deribit API for options data.
        
        # For now, return mock data if this is BTC or ETH (which have liquid options markets)
        if asset not in ["BTC", "ETH"]:
            return []
        
        try:
            # Create source
            source = EventSource(
                id=f"options_{asset}_{int(time.time())}",
                type=SourceType.FINANCIAL_DATA,
                name=f"{asset} Options Data",
                url=None,
                reliability_score=0.9  # Slightly less reliable than direct price data
            )
            
            # Generate mock options data based on asset
            if asset == "BTC":
                put_call_ratio = 0.8 + (0.4 * np.random.random())  # 0.8-1.2
                implied_volatility = 65 + (20 * np.random.random())  # 65-85%
                options_volume = 10000 + (5000 * np.random.random())  # 10k-15k
            else:  # ETH
                put_call_ratio = 0.7 + (0.5 * np.random.random())  # 0.7-1.2
                implied_volatility = 75 + (25 * np.random.random())  # 75-100%
                options_volume = 8000 + (4000 * np.random.random())  # 8k-12k
            
            # Detect anomalies in options data
            is_anomaly = False
            anomaly_score = 0.0
            anomaly_details = {}
            
            # Check for unusual put/call ratio
            if put_call_ratio > 1.5 or put_call_ratio < 0.5:
                is_anomaly = True
                anomaly_score = max(anomaly_score, min(1.0, abs(put_call_ratio - 1.0)))
                anomaly_details["unusual_put_call_ratio"] = put_call_ratio
            
            # Check for unusual implied volatility
            normal_iv = 70  # baseline
            iv_deviation = abs(implied_volatility - normal_iv) / normal_iv
            if iv_deviation > 0.3:  # 30% deviation
                is_anomaly = True
                anomaly_score = max(anomaly_score, min(1.0, iv_deviation))
                anomaly_details["unusual_implied_volatility"] = implied_volatility
            
            # Create collected data item
            data_item = {
                "source": source,
                "data_type": "options",
                "asset": asset,
                "timestamp": datetime.now(),
                "data": {
                    "put_call_ratio": put_call_ratio,
                    "implied_volatility": implied_volatility,
                    "options_volume": options_volume
                },
                "metadata": {
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "anomaly_details": anomaly_details,
                    "is_mock_data": True  # Flag to indicate this is mock data
                }
            }
            
            self.logger.debug(f"Generated options data for {asset}: P/C Ratio: {put_call_ratio:.2f}, IV: {implied_volatility:.2f}%")
            return [data_item]
            
        except Exception as e:
            self.logger.error(f"Error collecting options data for {asset}: {e}")
            return []
    
    async def _collect_futures_data(self, asset: str) -> List[Dict[str, Any]]:
        """Collect futures data for an asset.
        
        Args:
            asset: The asset symbol (e.g., "BTC")
            
        Returns:
            List of collected futures data
        """
        self.logger.debug(f"Collecting futures data for {asset}")
        
        collected_data = []
        
        try:
            # For now, use Binance futures data if available (or mock data)
            if self.binance_key and self.binance_secret:
                # In a full implementation, you would make API calls to Binance futures API
                pass
            
            # Generate standardized data model regardless of source
            
            # Create source
            source = EventSource(
                id=f"futures_{asset}_{int(time.time())}",
                type=SourceType.FINANCIAL_DATA,
                name=f"{asset} Futures Data",
                url=None,
                reliability_score=0.9  # Slightly less reliable than direct price data
            )
            
            # Generate mock futures data based on asset
            # In a real implementation, this would come from the API
            spot_price = await self._get_spot_price(asset)
            
            if not spot_price or spot_price <= 0:
                return []
            
            # Generate a reasonable futures premium (annualized 5-15%)
            annual_premium_pct = 5 + (10 * np.random.random())  # 5-15%
            
            # Calculate 3-month futures price with premium
            quarterly_premium_pct = annual_premium_pct * (3/12)  # 3 months
            quarterly_futures_price = spot_price * (1 + quarterly_premium_pct/100)
            
            # Generate funding rate (-0.1% to 0.1%)
            funding_rate = (np.random.random() * 0.2) - 0.1  # -0.1% to 0.1%
            
            # Generate open interest
            if asset == "BTC":
                open_interest = 500000000 + (100000000 * np.random.random())  # $500M-$600M
                volume_24h = 20000000000 + (5000000000 * np.random.random())  # $20B-$25B
            elif asset == "ETH":
                open_interest = 300000000 + (50000000 * np.random.random())  # $300M-$350M
                volume_24h = 10000000000 + (2000000000 * np.random.random())  # $10B-$12B
            else:
                open_interest = 50000000 + (10000000 * np.random.random())  # $50M-$60M
                volume_24h = 1000000000 + (500000000 * np.random.random())  # $1B-$1.5B
            
            # Detect anomalies
            is_anomaly = False
            anomaly_score = 0.0
            anomaly_details = {}
            
            # Check for unusual basis (difference between futures and spot)
            basis_pct = ((quarterly_futures_price / spot_price) - 1) * 100
            if basis_pct > 20 or basis_pct < -5:  # Unusually high premium or backwardation
                is_anomaly = True
                anomaly_score = max(anomaly_score, min(1.0, abs(basis_pct) / 30))  # Normalize to 0-1
                anomaly_details["unusual_basis"] = basis_pct
            
            # Check for unusual funding rate
            if abs(funding_rate) > 0.05:  # Greater than 0.05% is unusual
                is_anomaly = True
                anomaly_score = max(anomaly_score, min(1.0, abs(funding_rate) / 0.1))  # Normalize to 0-1
                anomaly_details["unusual_funding_rate"] = funding_rate
            
            # Create collected data item
            data_item = {
                "source": source,
                "data_type": "futures",
                "asset": asset,
                "timestamp": datetime.now(),
                "data": {
                    "spot_price": spot_price,
                    "quarterly_futures_price": quarterly_futures_price,
                    "basis_pct": basis_pct,
                    "funding_rate": funding_rate,
                    "open_interest": open_interest,
                    "volume_24h": volume_24h
                },
                "metadata": {
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                    "anomaly_details": anomaly_details,
                    "is_mock_data": True  # Flag to indicate this is mock data
                }
            }
            
            collected_data.append(data_item)
            
            self.logger.debug(f"Collected futures data for {asset}: Basis: {basis_pct:.2f}%, Funding: {funding_rate:.4f}%")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting futures data for {asset}: {e}")
            return []
    
    async def _get_cryptocurrency_data(self, asset: str, currency: str = "USD") -> Dict[str, Any]:
        """Get cryptocurrency data from CryptoCompare.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            currency: Currency to convert to (e.g., "USD")
            
        Returns:
            Dictionary with price and other data
        """
        try:
            # Check if we have an API key
            api_key_header = {}
            if self.cryptocompare_key:
                api_key_header = {"authorization": f"Apikey {self.cryptocompare_key}"}
            
            # Build URL
            url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={asset}&tsyms={currency}"
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=api_key_header) as response:
                    if response.status == 200:
                        data = response.json()
                        
                        # Extract data
                        if "RAW" in data and asset in data["RAW"] and currency in data["RAW"][asset]:
                            return data["RAW"][asset][currency]
                        
                        self.logger.warning(f"Invalid response structure from CryptoCompare for {asset}")
                        return {}
                    else:
                        self.logger.error(f"Failed to fetch CryptoCompare data: {response.status}")
                        return {}
        
        except Exception as e:
            self.logger.error(f"Error fetching CryptoCompare data for {asset}: {e}")
            return {}
    
    async def _get_spot_price(self, asset: str, currency: str = "USD") -> float:
        """Get spot price for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            currency: Currency to convert to (e.g., "USD")
            
        Returns:
            Spot price or 0 if not available
        """
        data = await self._get_cryptocurrency_data(asset, currency)
        return data.get("PRICE", 0)
    
    async def _load_historical_data(self):
        """Load historical data for anomaly detection baseline."""
        self.logger.info("Loading historical data for anomaly detection")
        
        try:
            # For each asset, load price and volume history
            for asset in self.assets:
                # Initialize data structure
                if asset not in self.historical_data:
                    self.historical_data[asset] = {
                        "price": [],
                        "volume": [],
                        "options": [],
                        "futures": []
                    }
                
                # Fetch historical price data
                await self._load_historical_price_data(asset)
                
                # Fetch historical volume data
                await self._load_historical_volume_data(asset)
                
                # Options and futures historical data would be loaded here
                # in a full implementation
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    async def _load_historical_price_data(self, asset: str):
        """Load historical price data for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
        """
        try:
            # Check if we have an API key
            api_key_header = {}
            if self.cryptocompare_key:
                api_key_header = {"authorization": f"Apikey {self.cryptocompare_key}"}
            
            # Build URL for daily historical data (last 30 days)
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={asset}&tsym=USD&limit=30"
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=api_key_header) as response:
                    if response.status == 200:
                        data = response.json()
                        
                        # Extract data
                        if "Data" in data and "Data" in data["Data"]:
                            history = data["Data"]["Data"]
                            
                            # Extract closing prices
                            prices = [item["close"] for item in history if "close" in item]
                            
                            # Store in historical data
                            self.historical_data[asset]["price"] = prices
                            
                            self.logger.debug(f"Loaded {len(prices)} historical price points for {asset}")
                        else:
                            self.logger.warning(f"Invalid response structure from CryptoCompare for {asset} historical data")
                    else:
                        self.logger.error(f"Failed to fetch historical data: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error loading historical price data for {asset}: {e}")
    
    async def _load_historical_volume_data(self, asset: str):
        """Load historical volume data for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
        """
        try:
            # Check if we have an API key
            api_key_header = {}
            if self.cryptocompare_key:
                api_key_header = {"authorization": f"Apikey {self.cryptocompare_key}"}
            
            # Build URL for daily historical data (last 30 days)
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={asset}&tsym=USD&limit=30"
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=api_key_header) as response:
                    if response.status == 200:
                        data = response.json()
                        
                        # Extract data
                        if "Data" in data and "Data" in data["Data"]:
                            history = data["Data"]["Data"]
                            
                            # Extract volumes
                            volumes = [item["volumefrom"] for item in history if "volumefrom" in item]
                            
                            # Store in historical data
                            self.historical_data[asset]["volume"] = volumes
                            
                            self.logger.debug(f"Loaded {len(volumes)} historical volume points for {asset}")
                        else:
                            self.logger.warning(f"Invalid response structure from CryptoCompare for {asset} historical volume data")
                    else:
                        self.logger.error(f"Failed to fetch historical volume data: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error loading historical volume data for {asset}: {e}")
    
    def _update_historical_data(self, asset: str, data_type: str, value: float):
        """Update historical data with new value.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            data_type: Type of data (e.g., "price", "volume")
            value: New value to add
        """
        if asset not in self.historical_data:
            self.historical_data[asset] = {
                "price": [],
                "volume": [],
                "options": [],
                "futures": []
            }
        
        # Add new value
        self.historical_data[asset][data_type].append(value)
        
        # Keep only the last 30 values
        self.historical_data[asset][data_type] = self.historical_data[asset][data_type][-30:]
    
    def _detect_price_anomalies(self, asset: str, price: float, change_pct: float) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in price data.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            price: Current price
            change_pct: 24-hour percent change
            
        Returns:
            Tuple of (is_anomaly, anomaly_score, anomaly_details)
        """
        is_anomaly = False
        anomaly_score = 0.0
        anomaly_details = {}
        
        # Check if we have enough historical data
        if asset in self.historical_data and len(self.historical_data[asset]["price"]) >= 10:
            historical_prices = self.historical_data[asset]["price"]
            
            # Check for large price changes
            if abs(change_pct) > 10:  # More than 10% change in 24h
                is_anomaly = True
                anomaly_score = min(1.0, abs(change_pct) / 30)  # Normalize to 0-1 (max at 30%)
                anomaly_details["large_price_change"] = change_pct
            
            # Check for statistical anomalies using Z-score
            mean_price = np.mean(historical_prices)
            std_price = np.std(historical_prices)
            
            if std_price > 0:
                z_score = (price - mean_price) / std_price
                
                if abs(z_score) > 2.5:  # More than 2.5 standard deviations
                    is_anomaly = True
                    z_score_normalized = min(1.0, abs(z_score) / 5)  # Normalize to 0-1 (max at 5 std dev)
                    anomaly_score = max(anomaly_score, z_score_normalized)
                    anomaly_details["z_score"] = z_score
            
            # Add historical context
            anomaly_details["mean_price_30d"] = mean_price
            anomaly_details["std_price_30d"] = std_price
        else:
            # If we don't have enough history, only check for large changes
            if abs(change_pct) > 15:  # Higher threshold without historical context
                is_anomaly = True
                anomaly_score = min(1.0, abs(change_pct) / 30)
                anomaly_details["large_price_change"] = change_pct
        
        return is_anomaly, anomaly_score, anomaly_details
    
    def _detect_volume_anomalies(self, asset: str, volume: float) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in volume data.
        
        Args:
            asset: Asset symbol (e.g., "BTC")
            volume: Current volume
            
        Returns:
            Tuple of (is_anomaly, anomaly_score, anomaly_details)
        """
        is_anomaly = False
        anomaly_score = 0.0
        anomaly_details = {}
        
        # Check if we have enough historical data
        if asset in self.historical_data and len(self.historical_data[asset]["volume"]) >= 10:
            historical_volumes = self.historical_data[asset]["volume"]
            
            # Calculate statistics
            mean_volume = np.mean(historical_volumes)
            std_volume = np.std(historical_volumes)
            
            # Calculate volume change percentage
            volume_change_pct = ((volume - mean_volume) / mean_volume) * 100 if mean_volume > 0 else 0
            
            # Check for large volume changes
            if abs(volume_change_pct) > 100:  # More than 100% change from average
                is_anomaly = True
                anomaly_score = min(1.0, abs(volume_change_pct) / 300)  # Normalize to 0-1 (max at 300%)
                anomaly_details["large_volume_change"] = volume_change_pct
            
            # Check for statistical anomalies using Z-score
            if std_volume > 0:
                z_score = (volume - mean_volume) / std_volume
                
                if abs(z_score) > 3:  # More than 3 standard deviations
                    is_anomaly = True
                    z_score_normalized = min(1.0, abs(z_score) / 6)  # Normalize to 0-1 (max at 6 std dev)
                    anomaly_score = max(anomaly_score, z_score_normalized)
                    anomaly_details["z_score"] = z_score
            
            # Add historical context
            anomaly_details["mean_volume_30d"] = mean_volume
            anomaly_details["std_volume_30d"] = std_volume
        else:
            # Without historical data, we can't effectively detect volume anomalies
            pass
        
        return is_anomaly, anomaly_score, anomaly_details
    
    def _calculate_market_volatility(self, collected_data: List[Dict[str, Any]]) -> float:
        """Calculate overall market volatility from collected data.
        
        Args:
            collected_data: List of collected data items
            
        Returns:
            Market volatility score (0-1)
        """
        volatility_scores = []
        
        # Extract volatility indicators from price and volume data
        for item in collected_data:
            if item["data_type"] = = "price":
                # Extract price change percentage
                change_pct = item["data"].get("change_pct_24h", 0)
                
                # Normalize to 0-1 (max at 20%)
                volatility_score = min(1.0, abs(change_pct) / 20)
                volatility_scores.append(volatility_score)
                
            elif item["data_type"] = = "volume":
                # Check if this item has anomaly info
                anomaly_score = item["metadata"].get("anomaly_score", 0)
                
                if anomaly_score > 0:
                    volatility_scores.append(anomaly_score)
            
            # Add any other anomaly scores
            anomaly_score = item["metadata"].get("anomaly_score", 0)
            if anomaly_score > 0:
                volatility_scores.append(anomaly_score)
        
        # Calculate overall volatility
        if volatility_scores:
            # Use higher percentiles to be more sensitive to extreme values
            overall_volatility = np.percentile(volatility_scores, 80)
            return overall_volatility
        else:
            # Default to moderate volatility
            return 0.5
    
    async def _mock_collect(self) -> List[Dict[str, Any]]:
        """Collect mock financial data for testing.
        
        Returns:
            List of mock collected data
        """
        self.logger.info("Collecting mock financial data")
        
        # Simulate a delay
        await asyncio.sleep(0.2)
        
        collected_data = []
        
        # Generate mock data for each asset and data type
        for asset in self.assets:
            # Generate price data
            if "price" in self.data_types:
                # Create baseline prices
                if asset == "BTC":
                    base_price = 50000
                    daily_vol = 0.04  # 4% daily volatility
                elif asset == "ETH":
                    base_price = 3000
                    daily_vol = 0.05  # 5% daily volatility
                elif asset == "SOL":
                    base_price = 100
                    daily_vol = 0.07  # 7% daily volatility
                else:  # XRP
                    base_price = 1
                    daily_vol = 0.06  # 6% daily volatility
                
                # Generate price change
                change_pct = np.random.normal(0, daily_vol) * 100
                price = base_price * (1 + change_pct/100)
                high_24h = price * (1 + np.random.random() * 0.02)  # Up to 2% higher
                low_24h = price * (1 - np.random.random() * 0.02)   # Up to 2% lower
                
                # Detect anomalies
                is_anomaly = False
                anomaly_score = 0.0
                anomaly_details = {}
                
                # Check for large price changes
                if abs(change_pct) > 10:
                    is_anomaly = True
                    anomaly_score = min(1.0, abs(change_pct) / 30)
                    anomaly_details["large_price_change"] = change_pct
                
                # Create source
                source = EventSource(
                    id=f"price_{asset}_{int(time.time())}",
                    type=SourceType.FINANCIAL_DATA,
                    name=f"{asset} Price Data",
                    url=None,
                    reliability_score=0.95
                )
                
                # Create collected data item
                data_item = {
                    "source": source,
                    "data_type": "price",
                    "asset": asset,
                    "trading_pair": f"{asset}/USD",
                    "timestamp": datetime.now(),
                    "data": {
                        "price": price,
                        "open_24h": price / (1 + change_pct/100),
                        "high_24h": high_24h,
                        "low_24h": low_24h,
                        "volume_24h": base_price * 10000,  # Volume in asset units
                        "change_24h": price - (price / (1 + change_pct/100)),
                        "change_pct_24h": change_pct
                    },
                    "metadata": {
                        "interval": self.interval,
                        "is_anomaly": is_anomaly,
                        "anomaly_score": anomaly_score,
                        "anomaly_details": anomaly_details
                    }
                }
                
                collected_data.append(data_item)
            
            # Generate volume data
            if "volume" in self.data_types:
                # Create baseline volumes
                if asset == "BTC":
                    base_volume = 100000
                elif asset == "ETH":
                    base_volume = 500000
                elif asset == "SOL":
                    base_volume = 2000000
                else:  # XRP
                    base_volume = 10000000
                
                # Add randomness (±30%)
                volume_multiplier = 0.7 + (np.random.random() * 0.6)
                volume = base_volume * volume_multiplier
                
                # Occasionally generate volume spikes
                if np.random.random() < 0.1:  # 10% chance
                    volume *= 2 + np.random.random()  # 2-3x spike
                    is_anomaly = True
                    anomaly_score = 0.7 + (np.random.random() * 0.3)  # 0.7-1.0
                    anomaly_details = {"volume_spike": True}
                else:
                    is_anomaly = False
                    anomaly_score = 0.0
                    anomaly_details = {}
                
                # Create source
                source = EventSource(
                    id=f"volume_{asset}_{int(time.time())}",
                    type=SourceType.FINANCIAL_DATA,
                    name=f"{asset} Volume Data",
                    url=None,
                    reliability_score=0.95
                )
                
                # Create collected data item
                data_item = {
                    "source": source,
                    "data_type": "volume",
                    "asset": asset,
                    "trading_pair": f"{asset}/USD",
                    "timestamp": datetime.now(),
                    "data": {
                        "volume_24h": volume,
                        "volume_24h_usd": volume * (price if 'price' in locals() else base_price),
                        "total_top_tier_volume_24h": volume * 1.5  # 50% more including all exchanges
                    },
                    "metadata": {
                        "interval": self.interval,
                        "is_anomaly": is_anomaly,
                        "anomaly_score": anomaly_score,
                        "anomaly_details": anomaly_details
                    }
                }
                
                collected_data.append(data_item)
            
            # Add options and futures data only for BTC and ETH
            if asset in ["BTC", "ETH"]:
                # Add options data
                if "options" in self.data_types:
                    options_data = await self._collect_options_data(asset)
                    collected_data.extend(options_data)
                
                # Add futures data
                if "futures" in self.data_types:
                    futures_data = await self._collect_futures_data(asset)
                    collected_data.extend(futures_data)
        
        # Calculate overall market volatility
        market_volatility = self._calculate_market_volatility(collected_data)
        
        # Get cost optimizer to update volatility
        try:
            cost_optimizer = await get_cost_optimizer()
            cost_optimizer.update_market_volatility(market_volatility)
            self.logger.info(f"Updated mock market volatility: {market_volatility:.4f}")
        except Exception as e:
            self.logger.error(f"Error updating market volatility: {e}")
        
        self.logger.info(f"Generated {len(collected_data)} mock financial data items")
        return collected_data