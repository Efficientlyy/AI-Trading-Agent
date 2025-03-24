"""Provider failover mechanism for LLM services.

This module provides a robust failover system for LLM provider APIs, 
automatically detecting failures and switching to alternative providers
when needed, ensuring continuous service availability.
"""

import asyncio
import json
import time
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.common.caching import Cache
from src.analysis_agents.sentiment.monitoring_alerts import (
    AlertSeverity, AlertType, alert_manager
)


class ProviderStatus(Enum):
    """Status values for LLM providers."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderStats:
    """Statistics for a provider."""
    
    def __init__(self, provider_name: str):
        """Initialize provider statistics.
        
        Args:
            provider_name: Name of the provider
        """
        self.provider_name = provider_name
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.latency_ms_sum = 0
        self.tokens_processed = 0
        self.last_success_time: Optional[datetime] = None
        self.last_error_time: Optional[datetime] = None
        self.last_error_message: Optional[str] = None
        self.consecutive_errors = 0
        
    def record_success(self, latency_ms: float, tokens: int = 0) -> None:
        """Record a successful request.
        
        Args:
            latency_ms: Request latency in milliseconds
            tokens: Number of tokens processed
        """
        self.request_count += 1
        self.success_count += 1
        self.latency_ms_sum += latency_ms
        self.tokens_processed += tokens
        self.last_success_time = datetime.utcnow()
        self.consecutive_errors = 0
        
    def record_error(self, error_message: str) -> None:
        """Record a failed request.
        
        Args:
            error_message: Error message
        """
        self.request_count += 1
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
        self.last_error_message = error_message
        self.consecutive_errors += 1
        
    def get_success_rate(self) -> float:
        """Get the success rate.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count
    
    def get_average_latency(self) -> float:
        """Get the average latency.
        
        Returns:
            Average latency in milliseconds
        """
        if self.success_count == 0:
            return 0.0
        return self.latency_ms_sum / self.success_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "provider": self.provider_name,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.get_success_rate(),
            "average_latency_ms": self.get_average_latency(),
            "tokens_processed": self.tokens_processed,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_error_message": self.last_error_message,
            "consecutive_errors": self.consecutive_errors
        }


class ProviderFailoverManager:
    """Manages LLM provider failover and routing."""
    
    def __init__(self):
        """Initialize the provider failover manager."""
        self.logger = get_logger("analysis_agents", "provider_failover")
        
        # Configuration
        self.consecutive_errors_threshold = config.get("llm.failover.consecutive_errors_threshold", 3)
        self.error_window_seconds = config.get("llm.failover.error_window_seconds", 300)
        self.error_rate_threshold = config.get("llm.failover.error_rate_threshold", 0.25)
        self.recovery_check_interval = config.get("llm.failover.recovery_check_interval", 60)
        self.recovery_threshold = config.get("llm.failover.recovery_threshold", 3)
        self.circuit_breaker_reset_time = config.get("llm.failover.circuit_breaker_reset_time", 300)
        
        # State
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.provider_status: Dict[str, ProviderStatus] = {}
        self.circuit_breaker_reset_times: Dict[str, datetime] = {}
        self.active_recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # Provider priorities and weights
        self.provider_priorities: Dict[str, int] = {
            "openai": config.get("llm.failover.priorities.openai", 1),
            "anthropic": config.get("llm.failover.priorities.anthropic", 2),
            "azure": config.get("llm.failover.priorities.azure", 3)
        }
        
        # Model mappings for failover
        self.model_alternatives: Dict[str, List[Tuple[str, str]]] = {}
        self._initialize_model_alternatives()
        
        # Fallback response cache
        fallback_cache_ttl = config.get("llm.fallback_cache_ttl", 86400)  # 24 hours default
        self.fallback_cache = Cache(ttl=fallback_cache_ttl)
        
        # Provider lock
        self.locks: Dict[str, asyncio.Lock] = {}
    
    def _initialize_model_alternatives(self) -> None:
        """Initialize model alternatives mappings.
        
        This defines which models can substitute for each other in case of failures.
        Format: {model: [(provider, model), ...]}
        """
        self.model_alternatives = {
            # OpenAI models with alternatives
            "gpt-4o": [
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-opus"),
                ("anthropic", "claude-3-sonnet"),
                ("azure", "gpt-4")
            ],
            "gpt-4-turbo": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-opus"),
                ("anthropic", "claude-3-sonnet"),
                ("azure", "gpt-4")
            ],
            "gpt-3.5-turbo": [
                ("anthropic", "claude-3-haiku"),
                ("azure", "gpt-3.5-turbo")
            ],
            
            # Anthropic models with alternatives
            "claude-3-opus": [
                ("openai", "gpt-4o"),
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-sonnet"),
                ("azure", "gpt-4")
            ],
            "claude-3-sonnet": [
                ("anthropic", "claude-3-opus"),
                ("openai", "gpt-4o"),
                ("openai", "gpt-4-turbo"),
                ("azure", "gpt-4")
            ],
            "claude-3-haiku": [
                ("openai", "gpt-3.5-turbo"),
                ("azure", "gpt-3.5-turbo")
            ],
            
            # Azure models with alternatives
            "azure-gpt-4": [
                ("openai", "gpt-4o"),
                ("openai", "gpt-4-turbo"),
                ("anthropic", "claude-3-opus")
            ],
            "azure-gpt-3.5-turbo": [
                ("openai", "gpt-3.5-turbo"),
                ("anthropic", "claude-3-haiku")
            ]
        }
    
    async def initialize(self) -> None:
        """Initialize the provider failover manager."""
        self.logger.info("Initializing provider failover manager")
        
        # Initialize provider stats and status
        for provider in self.provider_priorities.keys():
            self.provider_stats[provider] = ProviderStats(provider)
            self.provider_status[provider] = ProviderStatus.HEALTHY
            self.locks[provider] = asyncio.Lock()
        
        # Subscribe to relevant events
        event_bus.subscribe("llm_api_request", self.handle_api_request_event)
        event_bus.subscribe("llm_api_error", self.handle_api_error_event)
        
        # Load cached fallbacks if any
        self._load_fallback_cache()
    
    def _load_fallback_cache(self) -> None:
        """Load fallback responses from persistent storage if available."""
        try:
            cache_file = config.get("llm.fallback_cache_file", "data/cache/fallback_responses.json")
            import os
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                
                # Populate cache
                for key, value in cache_data.items():
                    self.fallback_cache.set(key, value)
                
                self.logger.info(f"Loaded {len(cache_data)} fallback responses from cache")
        except Exception as e:
            self.logger.warning(f"Failed to load fallback cache: {str(e)}")
    
    async def save_fallback_cache(self) -> None:
        """Save fallback responses to persistent storage."""
        try:
            cache_file = config.get("llm.fallback_cache_file", "data/cache/fallback_responses.json")
            import os
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Get all cache data
            cache_data = {}
            for key, item in self.fallback_cache.items():
                cache_data[key] = item
            
            # Save to file
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Saved {len(cache_data)} fallback responses to cache file")
        except Exception as e:
            self.logger.warning(f"Failed to save fallback cache: {str(e)}")
    
    async def handle_api_request_event(self, event: Event) -> None:
        """Handle API request events.
        
        Args:
            event: The API request event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        model = data.get("model", "unknown")
        operation = data.get("operation", "unknown")
        is_success = data.get("success", True)
        latency_ms = data.get("latency_ms", 0)
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        error_message = data.get("error", "")
        
        # Ensure we have stats for this provider
        if provider not in self.provider_stats:
            self.provider_stats[provider] = ProviderStats(provider)
        
        # Update stats
        if is_success:
            self.provider_stats[provider].record_success(
                latency_ms=latency_ms, 
                tokens=input_tokens + output_tokens
            )
        else:
            self.provider_stats[provider].record_error(error_message)
            
            # Check if we need to mark provider as unhealthy
            await self._check_provider_health(provider)
    
    async def handle_api_error_event(self, event: Event) -> None:
        """Handle API error events.
        
        Args:
            event: The API error event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        error_message = data.get("error", "Unknown error")
        
        # Ensure we have stats for this provider
        if provider not in self.provider_stats:
            self.provider_stats[provider] = ProviderStats(provider)
        
        # Record error
        self.provider_stats[provider].record_error(error_message)
        
        # Check if we need to mark provider as unhealthy
        await self._check_provider_health(provider)
    
    async def _check_provider_health(self, provider: str) -> None:
        """Check provider health and update status.
        
        Args:
            provider: The provider to check
        """
        # Skip if provider not tracked
        if provider not in self.provider_stats:
            return
        
        stats = self.provider_stats[provider]
        
        # Check consecutive errors
        if stats.consecutive_errors >= self.consecutive_errors_threshold:
            await self._mark_provider_unhealthy(
                provider,
                f"Consecutive errors threshold exceeded: {stats.consecutive_errors}/{self.consecutive_errors_threshold}"
            )
            return
        
        # Check error rate within window
        error_window_start = datetime.utcnow() - timedelta(seconds=self.error_window_seconds)
        if (stats.last_error_time and stats.last_error_time >= error_window_start and 
                stats.get_success_rate() < (1 - self.error_rate_threshold)):
            await self._mark_provider_degraded(
                provider,
                f"Error rate threshold exceeded: {1 - stats.get_success_rate():.2f}/{self.error_rate_threshold}"
            )
    
    async def _mark_provider_degraded(self, provider: str, reason: str) -> None:
        """Mark a provider as degraded.
        
        Args:
            provider: The provider to mark
            reason: Reason for degradation
        """
        # Skip if already in worse state
        if provider in self.provider_status and self.provider_status[provider] == ProviderStatus.UNHEALTHY:
            return
        
        self.provider_status[provider] = ProviderStatus.DEGRADED
        self.logger.warning(f"Provider {provider} marked as degraded: {reason}")
        
        # Create alert if not already created
        alert_id = f"provider_degraded_{provider}"
        
        await alert_manager.create_alert(
            alert_type=AlertType.API_FAILURE,
            severity=AlertSeverity.WARNING,
            source=f"provider_failover_{provider}",
            message=f"LLM provider {provider} is degraded: {reason}",
            details={
                "provider": provider,
                "status": "degraded",
                "reason": reason,
                "stats": self.provider_stats[provider].to_dict()
            },
            related_entities=[provider]
        )
    
    async def _mark_provider_unhealthy(self, provider: str, reason: str) -> None:
        """Mark a provider as unhealthy and initiate failover.
        
        Args:
            provider: The provider to mark
            reason: Reason for unhealthy status
        """
        async with self.locks[provider]:
            # Skip if already unhealthy
            if provider in self.provider_status and self.provider_status[provider] == ProviderStatus.UNHEALTHY:
                return
            
            self.provider_status[provider] = ProviderStatus.UNHEALTHY
            self.circuit_breaker_reset_times[provider] = datetime.utcnow() + timedelta(seconds=self.circuit_breaker_reset_time)
            
            self.logger.error(f"Provider {provider} marked as unhealthy: {reason}")
            
            # Create alert
            await alert_manager.create_alert(
                alert_type=AlertType.API_FAILURE,
                severity=AlertSeverity.ERROR,
                source=f"provider_failover_{provider}",
                message=f"LLM provider {provider} is unhealthy: {reason}",
                details={
                    "provider": provider,
                    "status": "unhealthy",
                    "reason": reason,
                    "circuit_breaker_reset_time": self.circuit_breaker_reset_times[provider].isoformat(),
                    "stats": self.provider_stats[provider].to_dict()
                },
                related_entities=[provider]
            )
            
            # Start recovery check task if not already running
            if provider not in self.active_recovery_tasks or self.active_recovery_tasks[provider].done():
                self.active_recovery_tasks[provider] = asyncio.create_task(
                    self._recovery_check_loop(provider)
                )
    
    async def _mark_provider_healthy(self, provider: str) -> None:
        """Mark a provider as healthy.
        
        Args:
            provider: The provider to mark
        """
        async with self.locks[provider]:
            # Skip if already healthy
            if provider in self.provider_status and self.provider_status[provider] == ProviderStatus.HEALTHY:
                return
            
            previous_status = self.provider_status.get(provider)
            self.provider_status[provider] = ProviderStatus.HEALTHY
            
            if provider in self.circuit_breaker_reset_times:
                del self.circuit_breaker_reset_times[provider]
                
            self.logger.info(f"Provider {provider} marked as healthy (was {previous_status.value if previous_status else 'unknown'})")
            
            # Create recovery alert
            await alert_manager.create_alert(
                alert_type=AlertType.API_FAILURE,
                severity=AlertSeverity.INFO,
                source=f"provider_failover_{provider}",
                message=f"LLM provider {provider} has recovered and is now healthy",
                details={
                    "provider": provider,
                    "status": "healthy",
                    "previous_status": previous_status.value if previous_status else "unknown",
                    "stats": self.provider_stats[provider].to_dict()
                },
                related_entities=[provider]
            )
    
    async def _recovery_check_loop(self, provider: str) -> None:
        """Run recovery checks for an unhealthy provider.
        
        Args:
            provider: The provider to check
        """
        self.logger.info(f"Starting recovery check loop for provider {provider}")
        
        # Track consecutive successful pings
        successful_pings = 0
        
        while self.provider_status.get(provider) != ProviderStatus.HEALTHY:
            try:
                # Wait for interval
                await asyncio.sleep(self.recovery_check_interval)
                
                # Check if circuit breaker reset time has passed
                now = datetime.utcnow()
                reset_time = self.circuit_breaker_reset_times.get(provider)
                
                if reset_time and now >= reset_time:
                    self.logger.info(f"Circuit breaker reset time reached for {provider}, attempting ping")
                    
                    # Ping the provider
                    success = await self._ping_provider(provider)
                    
                    if success:
                        successful_pings += 1
                        self.logger.info(f"Successful ping for {provider}: {successful_pings}/{self.recovery_threshold}")
                    else:
                        successful_pings = 0
                        # Reset circuit breaker time
                        self.circuit_breaker_reset_times[provider] = datetime.utcnow() + timedelta(seconds=self.circuit_breaker_reset_time)
                    
                    # If enough successful pings, mark as healthy
                    if successful_pings >= self.recovery_threshold:
                        await self._mark_provider_healthy(provider)
                        break
            except Exception as e:
                self.logger.error(f"Error in recovery check for {provider}: {str(e)}")
                # Don't break the loop on error
        
        self.logger.info(f"Recovery check loop for provider {provider} completed")
    
    async def _ping_provider(self, provider: str) -> bool:
        """Ping a provider to check if it's healthy.
        
        Args:
            provider: The provider to ping
            
        Returns:
            True if ping was successful
        """
        try:
            import aiohttp
            
            if provider == "openai":
                url = "https://api.openai.com/v1/models"
                headers = {"Authorization": f"Bearer {config.get('apis.openai.api_key')}"}
            elif provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {"x-api-key": config.get('apis.anthropic.api_key'), "anthropic-version": "2023-06-01"}
            elif provider == "azure":
                url = f"{config.get('apis.azure_openai.endpoint')}/openai/deployments?api-version=2023-05-15"
                headers = {"api-key": config.get('apis.azure_openai.api_key')}
            else:
                self.logger.warning(f"Unknown provider for ping: {provider}")
                return False
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as response:
                    elapsed_time = time.time() - start_time
                    
                    if response.status < 300:
                        self.logger.info(f"Ping successful for {provider} in {elapsed_time:.2f}s")
                        # Update stats
                        self.provider_stats[provider].record_success(latency_ms=elapsed_time * 1000)
                        return True
                    else:
                        content = response.text()
                        self.logger.warning(f"Ping failed for {provider}: {response.status} - {content[:100]}")
                        self.provider_stats[provider].record_error(f"HTTP {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.warning(f"Ping failed for {provider}: {str(e)}")
            self.provider_stats[provider].record_error(str(e))
            return False
    
    async def select_provider_for_model(self, requested_model: str) -> Tuple[str, str]:
        """Select the best provider and model based on current health status.
        
        Args:
            requested_model: The originally requested model
            
        Returns:
            Tuple of (provider, model)
        """
        # Get provider for requested model
        original_provider = None
        for provider, models in self._get_provider_models().items():
            if requested_model in models:
                original_provider = provider
                break
        
        if not original_provider:
            # If model not found, return as is and let the caller handle it
            self.logger.warning(f"Unknown model requested: {requested_model}")
            return ("unknown", requested_model)
        
        # Check if original provider is healthy
        if (original_provider in self.provider_status and 
                self.provider_status[original_provider] == ProviderStatus.HEALTHY):
            # Original provider is healthy, use it
            return (original_provider, requested_model)
        
        # Original provider is not healthy, find alternatives
        alternatives = self.model_alternatives.get(requested_model, [])
        
        # Add the original model as last resort if not in alternatives
        if not any(alt[0] == original_provider and alt[1] == requested_model for alt in alternatives):
            alternatives.append((original_provider, requested_model))
        
        # Sort alternatives by provider priority and filter unhealthy providers
        sorted_alternatives = sorted(
            [
                (provider, model) for provider, model in alternatives
                if provider in self.provider_status and self.provider_status[provider] != ProviderStatus.UNHEALTHY
            ],
            key=lambda x: self.provider_priorities.get(x[0], 999)
        )
        
        if sorted_alternatives:
            # Return the best alternative
            best_alternative = sorted_alternatives[0]
            
            # Log if we're using a different provider or model
            if best_alternative[0] != original_provider or best_alternative[1] != requested_model:
                self.logger.info(
                    f"Failover: Routing request from {original_provider}/{requested_model} to {best_alternative[0]}/{best_alternative[1]}"
                )
            
            return best_alternative
        
        # If all alternatives are unhealthy, use the original provider but log the issue
        self.logger.warning(
            f"All providers for model {requested_model} are unhealthy. Using original provider {original_provider} as fallback."
        )
        return (original_provider, requested_model)
    
    def get_fallback_response(self, prompt_type: str, text_hash: str) -> Optional[Dict[str, Any]]:
        """Get a fallback response from cache if available.
        
        Args:
            prompt_type: Type of prompt (e.g., "sentiment_analysis")
            text_hash: Hash of the input text
            
        Returns:
            Cached response or None if not found
        """
        cache_key = f"{prompt_type}:{text_hash}"
        return self.fallback_cache.get(cache_key)
    
    async def store_fallback_response(self, prompt_type: str, text_hash: str, response: Dict[str, Any]) -> None:
        """Store a response for potential fallback use.
        
        Args:
            prompt_type: Type of prompt
            text_hash: Hash of the input text
            response: Response to store
        """
        cache_key = f"{prompt_type}:{text_hash}"
        self.fallback_cache.set(cache_key, response)
        
        # Periodically save the cache to disk
        if random.random() < 0.1:  # 10% chance to save on each store operation
            self.save_fallback_cache()
    
    def _get_provider_models(self) -> Dict[str, List[str]]:
        """Get a mapping of providers to their available models.
        
        Returns:
            Dictionary of provider to list of models
        """
        return {
            "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "azure": ["azure-gpt-4", "azure-gpt-3.5-turbo"]
        }
    
    def get_provider_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers.
        
        Returns:
            Dictionary of provider health statuses
        """
        health_status = {}
        
        for provider in self.provider_stats:
            health_status[provider] = {
                "status": self.provider_status.get(provider, ProviderStatus.HEALTHY).value,
                "stats": self.provider_stats[provider].to_dict(),
                "circuit_breaker_reset_time": (
                    self.circuit_breaker_reset_times[provider].isoformat()
                    if provider in self.circuit_breaker_reset_times
                    else None
                )
            }
        
        return health_status


# Singleton instance
provider_failover_manager = ProviderFailoverManager()