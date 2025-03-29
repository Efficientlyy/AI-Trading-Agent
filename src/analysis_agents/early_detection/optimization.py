"""
Cost Optimization for the Early Event Detection System.

This module implements various cost optimization strategies as outlined in
the cost-optimized implementation plan, including:
1. Tiered model usage for LLM API calls
2. Caching mechanisms for API responses
3. Request batching for data collection
4. Adaptive sampling based on market conditions
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import lru_cache

from src.common.config import config
from src.common.logging import get_logger


class APIUsageTracker:
    """Tracks API usage and costs for different services."""
    
    def __init__(self):
        """Initialize the API usage tracker."""
        self.logger = get_logger("early_detection", "api_usage_tracker")
        self.usage = {
            "llm": {
                "gpt-3.5-turbo": {"calls": 0, "tokens": 0, "cost": 0.0},
                "gpt-4-turbo": {"calls": 0, "tokens": 0, "cost": 0.0}
            },
            "twitter": {"calls": 0, "cost": 0.0},
            "reddit": {"calls": 0, "cost": 0.0},
            "news": {"calls": 0, "cost": 0.0},
            "financial": {"calls": 0, "cost": 0.0}
        }
        
        # Pricing information (per 1K tokens or per call)
        self.pricing = {
            "llm": {
                "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
                "gpt-4-turbo": 0.03,     # $0.03 per 1K tokens
            },
            "twitter": 0.0,  # Free tier
            "reddit": 0.0,   # Free tier
            "news": 0.01,    # Estimated cost per call
            "financial": 0.05  # Estimated cost per call
        }
        
        # Daily and monthly limits
        self.limits = {
            "llm": {
                "gpt-3.5-turbo": {"daily": 1000000, "monthly": 5000000},  # tokens
                "gpt-4-turbo": {"daily": 200000, "monthly": 1000000}      # tokens
            },
            "twitter": {"daily": 500, "monthly": 10000},  # calls
            "reddit": {"daily": 1000, "monthly": 20000},  # calls
            "news": {"daily": 1000, "monthly": 15000},    # calls
            "financial": {"daily": 500, "monthly": 10000}  # calls
        }
        
        # Usage period tracking
        self.current_day = datetime.now().day
        self.current_month = datetime.now().month
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def track_llm_usage(self, model: str, tokens: int):
        """Track LLM API usage.
        
        Args:
            model: The model used (e.g., "gpt-3.5-turbo", "gpt-4-turbo")
            tokens: The number of tokens used
        """
        self._check_period_reset()
        
        if model not in self.usage["llm"]:
            model = "gpt-3.5-turbo"  # Default to cheaper model
        
        self.usage["llm"][model]["calls"] += 1
        self.usage["llm"][model]["tokens"] += tokens
        
        # Calculate cost
        cost_per_k = self.pricing["llm"][model]
        cost = (tokens / 1000) * cost_per_k
        self.usage["llm"][model]["cost"] += cost
        
        # Log usage
        self.logger.debug(f"LLM API usage: {model}, {tokens} tokens, ${cost:.4f}")
        
        # Check if approaching limits
        daily_limit = self.limits["llm"][model]["daily"]
        monthly_limit = self.limits["llm"][model]["monthly"]
        
        if self.usage["llm"][model]["tokens"] > monthly_limit * 0.8:
            self.logger.warning(f"Approaching monthly limit for {model}: {self.usage['llm'][model]['tokens']} tokens used out of {monthly_limit}")
    
    def track_api_usage(self, api_type: str):
        """Track API usage for non-LLM services.
        
        Args:
            api_type: The type of API ("twitter", "reddit", "news", "financial")
        """
        self._check_period_reset()
        
        if api_type not in self.usage:
            self.logger.warning(f"Unknown API type: {api_type}")
            return
        
        self.usage[api_type]["calls"] += 1
        
        # Calculate cost
        if api_type in self.pricing:
            cost = self.pricing[api_type]
            self.usage[api_type]["cost"] += cost
        
        # Log usage
        self.logger.debug(f"{api_type.capitalize()} API call: ${self.usage[api_type]['cost']:.4f} total")
        
        # Check if approaching limits
        if api_type in self.limits:
            daily_limit = self.limits[api_type]["daily"]
            monthly_limit = self.limits[api_type]["monthly"]
            
            if self.usage[api_type]["calls"] > daily_limit * 0.8:
                self.logger.warning(f"Approaching daily limit for {api_type}: {self.usage[api_type]['calls']} calls out of {daily_limit}")
    
    def _check_period_reset(self):
        """Check if we need to reset daily or monthly counters."""
        now = datetime.now()
        
        # Check for day change
        if now.day != self.current_day:
            self._reset_daily_counters()
            self.current_day = now.day
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Check for month change
        if now.month != self.current_month:
            self._reset_monthly_counters()
            self.current_month = now.month
    
    def _reset_daily_counters(self):
        """Reset daily usage counters."""
        self.logger.info("Resetting daily API usage counters")
        # Daily reset logic would go here
        # For this implementation, we're just tracking totals
    
    def _reset_monthly_counters(self):
        """Reset monthly usage counters."""
        self.logger.info("Resetting monthly API usage counters")
        
        # Reset all counters
        for api_type in self.usage:
            if isinstance(self.usage[api_type], dict):
                # For LLM with multiple models
                for model in self.usage[api_type]:
                    if isinstance(self.usage[api_type][model], dict):
                        self.usage[api_type][model]["calls"] = 0
                        self.usage[api_type][model]["tokens"] = 0
                        self.usage[api_type][model]["cost"] = 0.0
            else:
                # For simple API types
                self.usage[api_type]["calls"] = 0
                self.usage[api_type]["cost"] = 0.0
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get a usage and cost report.
        
        Returns:
            Dictionary with usage statistics and costs
        """
        total_cost = 0.0
        
        # Calculate total cost
        for api_type, data in self.usage.items():
            if api_type == "llm":
                for model, model_data in data.items():
                    total_cost += model_data["cost"]
            else:
                total_cost += data.get("cost", 0.0)
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cost": total_cost,
            "api_usage": self.usage,
            "projected_monthly_cost": total_cost * 30 / datetime.now().day  # Simple projection
        }
        
        return report


class CachedLLMClient:
    """LLM client with caching and tiered model selection."""
    
    def __init__(self, api_usage_tracker: APIUsageTracker):
        """Initialize the cached LLM client.
        
        Args:
            api_usage_tracker: Tracker for API usage
        """
        self.logger = get_logger("early_detection", "cached_llm_client")
        self.api_tracker = api_usage_tracker
        
        # Cache settings
        self.cache_ttl = config.get("early_detection.optimization.llm_cache_ttl", 3600)  # 1 hour default
        self.cache_size = config.get("early_detection.optimization.llm_cache_size", 1000)
        
        # Cache storage
        self.cache = {}
        self.cache_timestamps = {}
        
        # Model selection thresholds
        self.gpt4_confidence_threshold = config.get("early_detection.optimization.gpt4_confidence_threshold", 0.7)
        self.gpt4_impact_threshold = config.get("early_detection.optimization.gpt4_impact_threshold", 0.8)
    
    async def query(self, prompt: str, use_cache: bool = True, force_model: Optional[str] = None) -> Dict[str, Any]:
        """Query the LLM with caching and model selection.
        
        Args:
            prompt: The prompt to send to the LLM
            use_cache: Whether to use the cache
            force_model: Force a specific model ("gpt-3.5-turbo" or "gpt-4-turbo")
            
        Returns:
            The LLM response
        """
        # Generate cache key
        cache_key = self._generate_cache_key(prompt)
        
        # Check cache if enabled
        if use_cache and cache_key in self.cache:
            # Check if cache entry is still valid
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug("Cache hit for LLM query")
                return self.cache[cache_key]
        
        # Select model if not forced
        model = force_model
        if not model:
            # Use cheaper model by default
            model = "gpt-3.5-turbo"
            
            # Check if prompt contains indicators for using the more powerful model
            if self._should_use_gpt4(prompt):
                model = "gpt-4-turbo"
        
        # Simulate calling the LLM API
        # In a real implementation, this would call the OpenAI API or equivalent
        response = await self._simulate_llm_call(prompt, model)
        
        # Track API usage
        tokens = self._estimate_tokens(prompt) + self._estimate_tokens(json.dumps(response))
        self.api_tracker.track_llm_usage(model, tokens)
        
        # Cache the response if caching is enabled
        if use_cache:
            self._update_cache(cache_key, response)
        
        return response
    
    async def _simulate_llm_call(self, prompt: str, model: str) -> Dict[str, Any]:
        """Simulate an LLM API call.
        
        In a real implementation, this would call the OpenAI API or equivalent.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            
        Returns:
            Simulated response
        """
        # Add a small delay to simulate API latency
        await asyncio.sleep(0.1)
        
        # For demo purposes, generate a mock response
        is_gpt4 = model == "gpt-4-turbo"
        confidence_modifier = 0.2 if is_gpt4 else 0.0
        
        response = {
            "model": model,
            "content": f"Simulated response from {model}",
            "metadata": {
                "usage": {
                    "prompt_tokens": self._estimate_tokens(prompt),
                    "completion_tokens": 150 if is_gpt4 else 100,
                    "total_tokens": self._estimate_tokens(prompt) + (150 if is_gpt4 else 100)
                }
            },
            "analysis": {
                "confidence": min(0.75 + confidence_modifier, 1.0),
                "detected_events": [
                    {
                        "title": "Simulated event detection",
                        "description": "This is a simulated event detected by the LLM",
                        "confidence": min(0.7 + confidence_modifier, 1.0),
                        "impact": min(0.6 + confidence_modifier, 1.0)
                    }
                ]
            }
        }
        
        return response
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a cache key for a prompt.
        
        Args:
            prompt: The prompt to generate a key for
            
        Returns:
            Cache key string
        """
        # Use hash of the prompt as cache key
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _update_cache(self, key: str, response: Dict[str, Any]):
        """Update the cache with a new response.
        
        Args:
            key: Cache key
            response: Response to cache
        """
        # Add to cache
        self.cache[key] = response
        self.cache_timestamps[key] = time.time()
        
        # Trim cache if it exceeds the configured size
        if len(self.cache) > self.cache_size:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:100]
            for old_key, _ in oldest_keys:
                if old_key in self.cache:
                    del self.cache[old_key]
                if old_key in self.cache_timestamps:
                    del self.cache_timestamps[old_key]
            
            self.logger.debug(f"Trimmed LLM cache by removing {len(oldest_keys)} oldest entries")
    
    def _should_use_gpt4(self, prompt: str) -> bool:
        """Determine whether to use GPT-4 based on the prompt content.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            True if GPT-4 should be used, False otherwise
        """
        # Check for high-priority indicators in the prompt
        high_priority_terms = [
            "critical", "urgent", "high impact", "significant", "major", 
            "breaking", "important", "critical event", "market crash"
        ]
        
        low_priority_terms = [
            "routine", "regular", "minor", "small", "trivial",
            "background", "standard", "normal"
        ]
        
        # Count occurrences of high and low priority terms
        high_priority_count = sum(term in prompt.lower() for term in high_priority_terms)
        low_priority_count = sum(term in prompt.lower() for term in low_priority_terms)
        
        # Decision logic
        if "force_gpt4" in prompt.lower():
            return True
        elif "force_gpt3" in prompt.lower():
            return False
        elif high_priority_count > 2 and low_priority_count == 0:
            return True
        else:
            return False
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        This is a rough approximation. In a real implementation,
        you would use a proper tokenizer.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token is about 4 characters on average
        return max(1, len(text) // 4)


class RequestBatcher:
    """Batches API requests to reduce the number of calls."""
    
    def __init__(self, api_usage_tracker: APIUsageTracker, batch_size: int = 10, batch_window: float = 1.0):
        """Initialize the request batcher.
        
        Args:
            api_usage_tracker: Tracker for API usage
            batch_size: Maximum number of requests in a batch
            batch_window: Time window in seconds to wait for batching
        """
        self.logger = get_logger("early_detection", "request_batcher")
        self.api_tracker = api_usage_tracker
        self.batch_size = batch_size
        self.batch_window = batch_window
        
        # Batch queues for different API types
        self.queues = {
            "twitter": [],
            "reddit": [],
            "news": [],
            "financial": []
        }
        
        # Pending futures for batch requests
        self.pending_futures = {
            "twitter": [],
            "reddit": [],
            "news": [],
            "financial": []
        }
        
        # Batch processing tasks
        self.batch_tasks = {}
        
        # Flag to indicate if the batcher is running
        self.is_running = False
    
    async def start(self):
        """Start the request batcher."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start batch processing tasks
        for api_type in self.queues:
            self.batch_tasks[api_type] = asyncio.create_task(self._process_batch_queue(api_type))
        
        self.logger.info("Request batcher started")
    
    async def stop(self):
        """Stop the request batcher."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel batch processing tasks
        for api_type, task in self.batch_tasks.items():
            if not task.done():
                task.cancel()
        
        # Process any remaining requests
        for api_type in self.queues:
            if self.queues[api_type]:
                await self._execute_batch(api_type)
        
        self.logger.info("Request batcher stopped")
    
    async def add_request(self, api_type: str, request_func: Callable, *args, **kwargs) -> Any:
        """Add a request to the batch queue.
        
        Args:
            api_type: Type of API ("twitter", "reddit", "news", "financial")
            request_func: Function to execute the request
            *args: Args for the request function
            **kwargs: Kwargs for the request function
            
        Returns:
            Future for the request result
        """
        if api_type not in self.queues:
            self.logger.warning(f"Unknown API type: {api_type}")
            # Execute request directly
            return await request_func(*args, **kwargs)
        
        # Create future for the request
        future = asyncio.Future()
        
        # Add request to the queue
        self.queues[api_type].append((future, request_func, args, kwargs))
        self.pending_futures[api_type].append(future)
        
        # If queue reaches batch size, trigger immediate processing
        if len(self.queues[api_type]) >= self.batch_size:
            # Force flush of the batch
            await self._execute_batch(api_type)
        
        return await future
    
    async def _process_batch_queue(self, api_type: str):
        """Process the batch queue for a specific API type.
        
        Args:
            api_type: The API type to process
        """
        self.logger.debug(f"Started batch queue processor for {api_type}")
        
        while self.is_running:
            try:
                # Wait for batch window
                await asyncio.sleep(self.batch_window)
                
                # Process queue if not empty
                if self.queues[api_type]:
                    await self._execute_batch(api_type)
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                self.logger.error(f"Error processing {api_type} batch queue: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(1.0)
        
        self.logger.debug(f"Stopped batch queue processor for {api_type}")
    
    async def _execute_batch(self, api_type: str):
        """Execute a batch of requests for a specific API type.
        
        Args:
            api_type: The API type to execute
        """
        if not self.queues[api_type]:
            return
        
        # Get current batch
        batch = self.queues[api_type].copy()
        # Clear queue
        self.queues[api_type] = []
        
        self.logger.debug(f"Executing batch of {len(batch)} {api_type} requests")
        
        # Track API usage (only count as one call for the whole batch)
        self.api_tracker.track_api_usage(api_type)
        
        try:
            # In a real implementation, this would make a single batched API call
            # Here, we execute each request individually
            for future, request_func, args, kwargs in batch:
                try:
                    # Execute request
                    result = await request_func(*args, **kwargs)
                    
                    # Set result if future is not done
                    if not future.done():
                        future.set_result(result)
                
                except Exception as e:
                    self.logger.error(f"Error executing {api_type} request: {e}")
                    
                    # Set exception if future is not done
                    if not future.done():
                        future.set_exception(e)
        
        except Exception as e:
            self.logger.error(f"Error executing {api_type} batch: {e}")
            
            # Set exception for all futures
            for future, _, _, _ in batch:
                if not future.done():
                    future.set_exception(e)


class AdaptiveSampler:
    """Adjusts data collection frequency based on market conditions."""
    
    def __init__(self):
        """Initialize the adaptive sampler."""
        self.logger = get_logger("early_detection", "adaptive_sampler")
        
        # Default sampling intervals (in seconds)
        self.default_intervals = {
            "twitter": 300,  # 5 minutes
            "reddit": 600,   # 10 minutes
            "news": 900,     # 15 minutes
            "financial": 300  # 5 minutes
        }
        
        # Current sampling intervals
        self.current_intervals = self.default_intervals.copy()
        
        # Market volatility tracking
        self.volatility_level = 0.5  # 0-1 scale, higher means more volatile
        
        # Last sampling times
        self.last_sample_times = {
            "twitter": 0,
            "reddit": 0,
            "news": 0,
            "financial": 0
        }
    
    def update_volatility(self, volatility: float):
        """Update the market volatility level.
        
        Args:
            volatility: Market volatility level (0-1)
        """
        # Ensure volatility is in the valid range
        volatility = max(0.0, min(1.0, volatility))
        
        # Update volatility level
        self.volatility_level = volatility
        
        # Adjust sampling intervals based on volatility
        for source in self.current_intervals:
            # High volatility -> shorter intervals
            # Low volatility -> longer intervals
            if volatility > 0.7:
                # High volatility - sample more frequently
                self.current_intervals[source] = int(self.default_intervals[source] * 0.5)
            elif volatility < 0.3:
                # Low volatility - sample less frequently
                self.current_intervals[source] = int(self.default_intervals[source] * 1.5)
            else:
                # Normal volatility - use default intervals
                self.current_intervals[source] = self.default_intervals[source]
        
        self.logger.debug(f"Updated sampling intervals based on volatility {volatility:.2f}: {self.current_intervals}")
    
    def should_sample(self, source: str) -> bool:
        """Check if a source should be sampled based on adaptive intervals.
        
        Args:
            source: The data source to check
            
        Returns:
            True if the source should be sampled, False otherwise
        """
        if source not in self.current_intervals:
            return True
        
        # Get current time
        current_time = time.time()
        
        # Get last sample time
        last_time = self.last_sample_times.get(source, 0)
        
        # Get interval for this source
        interval = self.current_intervals.get(source, 300)
        
        # Check if enough time has passed
        if current_time - last_time >= interval:
            # Update last sample time
            self.last_sample_times[source] = current_time
            return True
        
        return False
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get information about current sampling configuration.
        
        Returns:
            Dictionary with sampling information
        """
        return {
            "volatility_level": self.volatility_level,
            "current_intervals": self.current_intervals,
            "default_intervals": self.default_intervals,
            "last_sample_times": {
                source: datetime.fromtimestamp(ts).isoformat() if ts > 0 else None
                for source, ts in self.last_sample_times.items()
            }
        }


class CostOptimizer:
    """Central manager for cost optimization strategies."""
    
    def __init__(self):
        """Initialize the cost optimizer."""
        self.logger = get_logger("early_detection", "cost_optimizer")
        
        # Create API usage tracker
        self.api_tracker = APIUsageTracker()
        
        # Create LLM client
        self.llm_client = CachedLLMClient(self.api_tracker)
        
        # Create request batcher
        self.request_batcher = RequestBatcher(
            self.api_tracker,
            batch_size=config.get("early_detection.optimization.batch_size", 10),
            batch_window=config.get("early_detection.optimization.batch_window", 1.0)
        )
        
        # Create adaptive sampler
        self.adaptive_sampler = AdaptiveSampler()
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the cost optimizer."""
        self.logger.info("Initializing cost optimizer")
        
        # Start request batcher
        await self.request_batcher.start()
        
        self.is_initialized = True
        self.logger.info("Cost optimizer initialized")
    
    async def shutdown(self):
        """Shut down the cost optimizer."""
        if not self.is_initialized:
            return
        
        self.logger.info("Shutting down cost optimizer")
        
        # Stop request batcher
        await self.request_batcher.stop()
        
        self.is_initialized = False
        self.logger.info("Cost optimizer shut down")
    
    async def llm_query(self, prompt: str, use_cache: bool = True, force_model: Optional[str] = None) -> Dict[str, Any]:
        """Query the LLM with cost optimization.
        
        Args:
            prompt: The prompt to send to the LLM
            use_cache: Whether to use the cache
            force_model: Force a specific model
            
        Returns:
            The LLM response
        """
        return await self.llm_client.query(prompt, use_cache, force_model)
    
    async def api_request(self, api_type: str, request_func: Callable, *args, **kwargs) -> Any:
        """Make an API request with cost optimization.
        
        Args:
            api_type: Type of API
            request_func: Function to execute the request
            *args: Args for the request function
            **kwargs: Kwargs for the request function
            
        Returns:
            The API response
        """
        # Check if we should sample this source
        if not self.adaptive_sampler.should_sample(api_type):
            self.logger.debug(f"Skipping {api_type} request due to adaptive sampling")
            return None
        
        # Add request to batch queue
        return await self.request_batcher.add_request(api_type, request_func, *args, **kwargs)
    
    def update_market_volatility(self, volatility: float):
        """Update market volatility for adaptive sampling.
        
        Args:
            volatility: Market volatility level (0-1)
        """
        self.adaptive_sampler.update_volatility(volatility)
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get API usage and cost report.
        
        Returns:
            Dictionary with usage report
        """
        return self.api_tracker.get_usage_report()
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get adaptive sampling information.
        
        Returns:
            Dictionary with sampling information
        """
        return self.adaptive_sampler.get_sampling_info()


# Global instance of the cost optimizer
_cost_optimizer = None


async def get_cost_optimizer() -> CostOptimizer:
    """Get the global cost optimizer instance.
    
    Returns:
        The cost optimizer instance
    """
    global _cost_optimizer
    
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
        _cost_optimizer.initialize()
    
    return _cost_optimizer