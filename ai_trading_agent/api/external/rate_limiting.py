"""
Rate limiting and quota management for External API Gateway.

This module implements rate limiting and quota management capabilities
for external API access, with different limits based on partner tiers.
"""
import time
import asyncio
import logging
from typing import Dict, Optional, Any, Union
from datetime import datetime, timedelta
import sqlite3
import json
import uuid

# Import partner tier configuration
from .config import PartnerTier, PartnerTierConfig

# Setup logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API requests.
    
    Implements a token bucket algorithm for rate limiting different
    partner tiers with different limits.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        # Storage for rate limit data
        # {identifier: {"tokens": float, "last_refill": float, "tier": str}}
        self.limiters: Dict[str, Dict[str, Any]] = {}
        
        # Default limits per tier (requests per second)
        self.tier_limits = {
            PartnerTier.PUBLIC: 5,      # 5 requests per second for unauthenticated
            PartnerTier.BASIC: 10,      # 10 requests per second for basic tier
            PartnerTier.PREMIUM: 50,    # 50 requests per second for premium tier
            PartnerTier.ENTERPRISE: 200 # 200 requests per second for enterprise tier
        }
        
        # Burst allowance (multiplier for bucket size)
        self.burst_multiplier = 2
        
        logger.info("Rate limiter initialized")
    
    async def check_rate_limit(
        self,
        identifier: str,
        tier: Union[PartnerTier, str],
        cost: float = 1.0
    ) -> bool:
        """
        Check if a request is within rate limits.
        
        Args:
            identifier: API key or IP address
            tier: Partner tier
            cost: Cost of this request (default=1.0)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Convert string tier to enum if needed
        if isinstance(tier, str):
            tier = PartnerTier(tier)
        
        # Get rate limit for this tier
        limit = self.tier_limits.get(tier, self.tier_limits[PartnerTier.PUBLIC])
        bucket_size = limit * self.burst_multiplier
        
        # Get or create limiter for this identifier
        if identifier not in self.limiters:
            self.limiters[identifier] = {
                "tokens": bucket_size,
                "last_refill": time.time(),
                "tier": tier.value
            }
        
        limiter = self.limiters[identifier]
        
        # Update tier if different
        if limiter["tier"] != tier.value:
            limiter["tier"] = tier.value
        
        # Refill tokens based on time elapsed
        current_time = time.time()
        time_elapsed = current_time - limiter["last_refill"]
        token_refill = time_elapsed * limit
        
        limiter["tokens"] = min(bucket_size, limiter["tokens"] + token_refill)
        limiter["last_refill"] = current_time
        
        # Check if enough tokens for this request
        if limiter["tokens"] >= cost:
            limiter["tokens"] -= cost
            return True
        else:
            logger.warning(f"Rate limit exceeded for {identifier} (tier: {tier.value})")
            return False
    
    def update_tier_limit(self, tier: PartnerTier, limit: float) -> None:
        """
        Update the rate limit for a specific tier.
        
        Args:
            tier: Partner tier
            limit: New rate limit (requests per second)
        """
        self.tier_limits[tier] = limit
        logger.info(f"Updated rate limit for tier {tier.value}: {limit} req/s")


class QuotaManager:
    """
    Quota manager for API requests.
    
    Tracks and enforces monthly request quotas for different partner tiers.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the quota manager.
        
        Args:
            db_path: Path to SQLite database for persistent storage
        """
        self.db_path = db_path
        
        # In-memory quota usage if no DB provided
        # {identifier: {"usage": int, "reset_time": float}}
        self.quota_usage: Dict[str, Dict[str, Any]] = {}
        
        # Default monthly quotas per tier
        self.tier_quotas = {
            PartnerTier.PUBLIC: 1000,       # 1,000 requests per month
            PartnerTier.BASIC: 100000,      # 100,000 requests per month
            PartnerTier.PREMIUM: 1000000,   # 1 million requests per month
            PartnerTier.ENTERPRISE: None    # Unlimited for enterprise tier
        }
        
        logger.info("Quota manager initialized")
    
    async def check_quota(
        self,
        identifier: str,
        tier: Union[PartnerTier, str]
    ) -> bool:
        """
        Check if a request is within monthly quota.
        
        Args:
            identifier: API key or partner ID
            tier: Partner tier
            
        Returns:
            True if request is allowed, False if quota exceeded
        """
        # Convert string tier to enum if needed
        if isinstance(tier, str):
            tier = PartnerTier(tier)
        
        # Get quota for this tier
        quota = self.tier_quotas.get(tier)
        
        # Enterprise tier has unlimited quota
        if quota is None:
            return True
        
        # If using database, check quota there
        if self.db_path:
            return await self._check_db_quota(identifier, quota)
        
        # Otherwise use in-memory tracking
        return await self._check_memory_quota(identifier, quota)
    
    async def _check_memory_quota(self, identifier: str, quota: int) -> bool:
        """Check quota using in-memory tracking."""
        # Get or create usage tracker for this identifier
        if identifier not in self.quota_usage:
            self.quota_usage[identifier] = {
                "usage": 0,
                "reset_time": self._get_next_month_start()
            }
        
        usage_info = self.quota_usage[identifier]
        
        # Check if quota period has reset
        current_time = time.time()
        if current_time >= usage_info["reset_time"]:
            usage_info["usage"] = 0
            usage_info["reset_time"] = self._get_next_month_start()
        
        # Check if quota exceeded
        return usage_info["usage"] < quota
    
    async def _check_db_quota(self, identifier: str, quota: int) -> bool:
        """Check quota using database tracking."""
        # Implementation for database-backed quota checking would go here
        # This is a simplified placeholder
        return True
    
    async def track_usage(
        self,
        identifier: str,
        endpoint: str,
        count: int = 1
    ) -> None:
        """
        Track API usage for quota management.
        
        Args:
            identifier: API key or partner ID
            endpoint: API endpoint that was accessed
            count: Number of requests to count (default=1)
        """
        # If using database, track usage there
        if self.db_path:
            await self._track_db_usage(identifier, endpoint, count)
            return
        
        # Otherwise use in-memory tracking
        if identifier in self.quota_usage:
            usage_info = self.quota_usage[identifier]
            
            # Check if quota period has reset
            current_time = time.time()
            if current_time >= usage_info["reset_time"]:
                usage_info["usage"] = count
                usage_info["reset_time"] = self._get_next_month_start()
            else:
                usage_info["usage"] += count
    
    async def _track_db_usage(
        self,
        identifier: str,
        endpoint: str,
        count: int
    ) -> None:
        """Track usage in database."""
        # Implementation for database-backed usage tracking would go here
        # This is a simplified placeholder
        pass
    
    def _get_next_month_start(self) -> float:
        """Get the timestamp for the start of the next month."""
        now = datetime.now()
        
        # If we're in December, go to January of next year
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)
        
        return next_month.timestamp()
    
    def update_tier_quota(
        self,
        tier: PartnerTier,
        quota: Optional[int]
    ) -> None:
        """
        Update the monthly quota for a specific tier.
        
        Args:
            tier: Partner tier
            quota: New monthly quota (None for unlimited)
        """
        self.tier_quotas[tier] = quota
        
        if quota is None:
            logger.info(f"Updated quota for tier {tier.value}: unlimited")
        else:
            logger.info(f"Updated quota for tier {tier.value}: {quota} req/month")
