"""Onchain sentiment analysis.

This module provides functionality for analyzing sentiment from on-chain metrics
such as wallet activity, transaction volume, and blockchain data.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.blockchain_client import BlockchainClient
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class OnchainSentimentAgent(BaseSentimentAgent):
    """Analysis agent for onchain sentiment.
    
    This agent processes sentiment data from on-chain metrics
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the onchain sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "onchain_sentiment")
        
        # Onchain metrics to monitor
        self.metrics = config.get(
            f"analysis_agents.{agent_id}.metrics", 
            ["LargeTransactions", "ActiveAddresses"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            3600  # Default: 1 hour
        )
        
        # API clients (will be initialized during _initialize)
        self.blockchain_client = None
        
        # NLP service (will be set by manager)
        self.nlp_service = None
    
    async def _initialize(self) -> None:
        """Initialize the onchain sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing onchain sentiment agent",
                       metrics=self.metrics)
                       
        # Initialize API clients
        try:
            # Get API keys from config
            blockchain_com_api_key = config.get("sentiment.apis.blockchain_com.api_key", "")
            glassnode_api_key = config.get("sentiment.apis.glassnode.api_key", "")
            
            # Blockchain client
            self.blockchain_client = BlockchainClient(
                blockchain_com_api_key=blockchain_com_api_key,
                glassnode_api_key=glassnode_api_key
            )
            
            # Log which providers are being used
            active_providers = []
            if blockchain_com_api_key:
                active_providers.append("Blockchain.com")
            if glassnode_api_key:
                active_providers.append("Glassnode")
                
            if active_providers:
                self.logger.info("Initialized blockchain API clients", 
                              providers=", ".join(active_providers))
            else:
                self.logger.warning("No blockchain API keys provided, using mock data")
            
        except Exception as e:
            self.logger.error("Failed to initialize blockchain API client", error=str(e))
    
    def set_nlp_service(self, nlp_service: NLPService) -> None:
        """Set the NLP service for sentiment analysis.
        
        Args:
            nlp_service: The NLP service to use
        """
        self.nlp_service = nlp_service
        self.logger.info("NLP service set for onchain sentiment agent")
    
    async def _start(self) -> None:
        """Start the onchain sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for onchain sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the onchain sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Close blockchain client
        if hasattr(self, "blockchain_client") and self.blockchain_client:
            try:
                await self.blockchain_client.close()
                self.logger.debug("Blockchain client closed")
            except Exception as e:
                self.logger.warning("Error closing blockchain client", error=str(e))
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update onchain sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # Only process symbols that have on-chain data available
                    # (typically just BTC, ETH, and some major coins)
                    base_currency = symbol.split('/')[0]
                    if base_currency in ["BTC", "ETH"]:
                        await self._analyze_onchain_metrics(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Onchain sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in onchain sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_onchain_metrics(self, symbol: str) -> None:
        """Analyze onchain metrics for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "onchain", self.update_interval):
            return
            
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
        
        try:
            # Fetch on-chain metrics
            
            # Large transactions count and volume
            large_tx_data = await self.blockchain_client.get_large_transactions(
                asset=base_currency,
                time_period="24h"
            )
            large_tx_count = large_tx_data.get("count", 0)
            large_tx_volume = large_tx_data.get("volume", 0)
            
            # Active addresses
            active_addr_data = await self.blockchain_client.get_active_addresses(
                asset=base_currency,
                time_period="24h"
            )
            active_addresses = active_addr_data.get("count", 0)
            active_addr_change = active_addr_data.get("change_percentage", 0)
            
            # Network hash rate (for PoW chains)
            hash_rate_data = None
            hash_rate_change = 0
            if base_currency == "BTC":
                hash_rate_data = await self.blockchain_client.get_hash_rate(
                    asset=base_currency,
                    time_period="7d"
                )
                hash_rate_change = hash_rate_data.get("change_percentage", 0) if hash_rate_data else 0
            
            # Exchange reserves
            exchange_data = await self.blockchain_client.get_exchange_reserves(
                asset=base_currency,
                time_period="7d"
            )
            exchange_reserves = exchange_data.get("reserves", 0)
            exchange_reserves_change = exchange_data.get("change_percentage", 0)
            
            # Normalize metrics to sentiment scores (0-1)
            
            # Large transactions (normalize based on historical averages)
            large_tx_normalized = min(1.0, large_tx_volume / large_tx_data.get("average_volume", large_tx_volume))
            
            # Active addresses growth (-100% to +100%, normalize to 0-1)
            active_addr_normalized = 0.5 + (active_addr_change / 200)  # Convert to 0-1 scale
            
            # Hash rate change (-100% to +100%, normalize to 0-1)
            hash_rate_normalized = 0.5 + (hash_rate_change / 200)  # Convert to 0-1 scale
            
            # Exchange reserves change (-100% to +100%, normalize to 0-1)
            # Negative means tokens leaving exchanges (bullish)
            exchange_reserves_normalized = 0.5 - (exchange_reserves_change / 200)
            
            # Combine metrics into a sentiment score
            # Each metric is weighted differently
            sentiment_metrics = {
                "large_transactions": large_tx_normalized,
                "active_addresses": active_addr_normalized,
                "hash_rate": hash_rate_normalized if hash_rate_data else 0.5,
                "exchange_reserves": exchange_reserves_normalized
            }
            
            # Calculate weighted sentiment
            metric_weights = {
                "large_transactions": 0.3,
                "active_addresses": 0.3,
                "hash_rate": 0.2 if hash_rate_data else 0,
                "exchange_reserves": 0.2
            }
            
            # Adjust weights if hash rate is not available
            if not hash_rate_data:
                total_weight = sum(metric_weights.values())
                for key in metric_weights:
                    metric_weights[key] = metric_weights[key] / total_weight
            
            sentiment_value = sum(
                sentiment_metrics[metric] * metric_weights[metric]
                for metric in sentiment_metrics
            ) / sum(metric_weights.values())
            
            # Calculate confidence based on data quality
            confidence = 0.7  # Base confidence
            
            # Determine direction
            if sentiment_value > 0.6:
                direction = "bullish"
            elif sentiment_value < 0.4:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "large_transactions_count": large_tx_count,
                "large_transactions_volume": large_tx_volume,
                "active_addresses": active_addresses,
                "active_addresses_change": active_addr_change,
                "hash_rate_change": hash_rate_change if hash_rate_data else None,
                "exchange_reserves": exchange_reserves,
                "exchange_reserves_change": exchange_reserves_change,
                "metrics": self.metrics
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="onchain",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Publish event if significant shift or strong signal
            if sentiment_shift > self.sentiment_shift_threshold or confidence > 0.85:
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    # Exchange outflows are often a strong signal
                    is_strong_signal = exchange_reserves_change < -5.0  # 5% outflow
                    
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_strong_signal,
                        sources=self.metrics,
                        details=additional_data
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing onchain metrics", 
                           symbol=symbol,
                           error=str(e))
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to onchain metrics.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
            
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
            
        # Check if we have onchain sentiment data for this symbol
        if symbol not in self.sentiment_cache or "onchain" not in self.sentiment_cache[symbol]:
            return
            
        # Get the latest onchain sentiment
        sentiment_data = self.sentiment_cache[symbol]["onchain"]
        exchange_reserves_change = sentiment_data.get("exchange_reserves_change", 0)
        
        # Get price data from candles
        closes = [candle.close for candle in candles]
        
        # If there's a significant exchange outflow and price is stagnant or falling,
        # that's a potential divergence and accumulation signal
        if len(closes) >= 20 and exchange_reserves_change < -3.0:  # 3% outflow
            # Calculate short-term trend
            short_term_change = (closes[-1] / closes[-20]) - 1  # 20-period return
            
            # If price is not rising despite exchange outflows
            if short_term_change < 0.02:  # Less than 2% gain
                # This could be an accumulation signal
                await self.publish_sentiment_event(
                    symbol=symbol,
                    direction="bullish",
                    value=0.75,  # Quite bullish
                    confidence=0.85,
                    timeframe=timeframe,
                    is_extreme=False,
                    signal_type="divergence",
                    sources=self.metrics,
                    details={
                        "exchange_reserves_change": exchange_reserves_change,
                        "price_change": short_term_change,
                        "event_type": "accumulation_divergence"
                    }
                )