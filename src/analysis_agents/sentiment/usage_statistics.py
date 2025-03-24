"""Usage statistics tracking for LLM-based sentiment analysis.

This module tracks usage statistics for LLM API providers, including token counts,
costs, and usage patterns to help optimize resource utilization.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, DefaultDict
from collections import defaultdict

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event


class APIUsageTracker:
    """Tracks API usage statistics across providers."""
    
    def __init__(self):
        """Initialize the API usage tracker."""
        self.logger = get_logger("analysis_agents", "usage_statistics")
        
        # Define token costs for different models
        self.token_costs = {
            # OpenAI models
            "gpt-4o": {"input": 0.00005, "output": 0.00015},  # per token
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},
            
            # Anthropic models
            "claude-3-opus": {"input": 0.00001, "output": 0.00003},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
            
            # Default fallback
            "default": {"input": 0.00001, "output": 0.00003},
        }
        
        # Daily and time-based usage stats
        self.daily_stats: Dict[str, Dict[str, Any]] = {}  # date -> stats
        self.hourly_stats: Dict[str, Dict[int, Dict[str, Any]]] = {}  # date -> hour -> stats
        
        # Provider and model stats
        self.provider_stats: Dict[str, Dict[str, Any]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        # Operation type stats (sentiment_analysis, event_detection, etc.)
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Set up storage paths
        self.storage_dir = config.get("sentiment_analysis.usage_statistics.storage_dir", "data/usage_statistics")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["daily", "hourly", "providers", "models", "operations"]:
            os.makedirs(os.path.join(self.storage_dir, subdir), exist_ok=True)
            
        # Daily stats rolling window (for memory)
        self.stats_window_days = config.get("sentiment_analysis.usage_statistics.window_days", 90)
        
        # Last save time
        self.last_save_time = 0
        self.save_interval = config.get("sentiment_analysis.usage_statistics.save_interval", 300)  # 5 minutes
    
    async def initialize(self) -> None:
        """Initialize the usage tracker."""
        self.logger.info("Initializing API usage tracker")
        
        # Load existing statistics
        self._load_statistics()
        
        # Subscribe to API request events
        event_bus.subscribe("llm_api_request", self.handle_api_request)
    
    async def _load_statistics(self) -> None:
        """Load statistics from storage."""
        try:
            # Load daily stats for the last window_days
            start_date = (datetime.utcnow() - timedelta(days=self.stats_window_days)).date()
            current_date = datetime.utcnow().date()
            
            while start_date <= current_date:
                date_str = start_date.isoformat()
                daily_file = os.path.join(self.storage_dir, "daily", f"{date_str}.json")
                
                if os.path.exists(daily_file):
                    with open(daily_file, "r") as f:
                        self.daily_stats[date_str] = json.load(f)
                        
                    # Also load hourly stats if available
                    hourly_file = os.path.join(self.storage_dir, "hourly", f"{date_str}.json")
                    if os.path.exists(hourly_file):
                        with open(hourly_file, "r") as f:
                            self.hourly_stats[date_str] = json.load(f)
                
                start_date += timedelta(days=1)
            
            # Load provider stats
            provider_file = os.path.join(self.storage_dir, "providers", "stats.json")
            if os.path.exists(provider_file):
                with open(provider_file, "r") as f:
                    self.provider_stats = json.load(f)
            
            # Load model stats
            model_file = os.path.join(self.storage_dir, "models", "stats.json")
            if os.path.exists(model_file):
                with open(model_file, "r") as f:
                    self.model_stats = json.load(f)
            
            # Load operation stats
            operation_file = os.path.join(self.storage_dir, "operations", "stats.json")
            if os.path.exists(operation_file):
                with open(operation_file, "r") as f:
                    self.operation_stats = json.load(f)
            
            self.logger.info(f"Loaded usage statistics for {len(self.daily_stats)} days")
            
        except Exception as e:
            self.logger.error(f"Error loading usage statistics: {str(e)}")
    
    async def handle_api_request(self, event: Event) -> None:
        """Handle API request events.
        
        Args:
            event: The API request event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        model = data.get("model", "unknown")
        operation = data.get("operation", "unknown")
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        latency_ms = data.get("latency_ms", 0)
        success = data.get("success", True)
        
        # Update statistics
        await self._update_statistics(
            provider, model, operation, input_tokens, output_tokens, 
            latency_ms, success
        )
        
        # Check if we should save statistics (throttle saves)
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.last_save_time = current_time
            self._save_statistics()
    
    async def _update_statistics(
        self, 
        provider: str, 
        model: str, 
        operation: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool
    ) -> None:
        """Update usage statistics.
        
        Args:
            provider: API provider
            model: Model name
            operation: Operation type
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Latency in milliseconds
            success: Whether the request was successful
        """
        # Get current date and hour
        now = datetime.utcnow()
        date_str = now.date().isoformat()
        hour = now.hour
        
        # Calculate cost
        model_costs = self.token_costs.get(model, self.token_costs["default"])
        input_cost = input_tokens * model_costs["input"]
        output_cost = output_tokens * model_costs["output"]
        total_cost = input_cost + output_cost
        
        # Initialize daily stats for today if not exists
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0},
                "providers": {},
                "models": {},
                "operations": {}
            }
        
        # Initialize hourly stats for today if not exists
        if date_str not in self.hourly_stats:
            self.hourly_stats[date_str] = {}
        
        if hour not in self.hourly_stats[date_str]:
            self.hourly_stats[date_str][hour] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0},
                "providers": {},
                "models": {},
                "operations": {}
            }
        
        # Update daily stats
        daily = self.daily_stats[date_str]
        daily["requests"]["total"] += 1
        daily["requests"]["success" if success else "failure"] += 1
        daily["tokens"]["input"] += input_tokens
        daily["tokens"]["output"] += output_tokens
        daily["tokens"]["total"] += input_tokens + output_tokens
        daily["costs"]["input"] += input_cost
        daily["costs"]["output"] += output_cost
        daily["costs"]["total"] += total_cost
        
        if success:
            daily["latency"]["total_ms"] += latency_ms
            daily["latency"]["count"] += 1
            daily["latency"]["average_ms"] = daily["latency"]["total_ms"] / daily["latency"]["count"]
        
        # Initialize provider stats if not exists
        if provider not in daily["providers"]:
            daily["providers"][provider] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0}
            }
        
        # Update provider stats
        provider_daily = daily["providers"][provider]
        provider_daily["requests"]["total"] += 1
        provider_daily["requests"]["success" if success else "failure"] += 1
        provider_daily["tokens"]["input"] += input_tokens
        provider_daily["tokens"]["output"] += output_tokens
        provider_daily["tokens"]["total"] += input_tokens + output_tokens
        provider_daily["costs"]["input"] += input_cost
        provider_daily["costs"]["output"] += output_cost
        provider_daily["costs"]["total"] += total_cost
        
        if success:
            provider_daily["latency"]["total_ms"] += latency_ms
            provider_daily["latency"]["count"] += 1
            provider_daily["latency"]["average_ms"] = (
                provider_daily["latency"]["total_ms"] / provider_daily["latency"]["count"]
            )
        
        # Initialize model stats if not exists
        if model not in daily["models"]:
            daily["models"][model] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0}
            }
        
        # Update model stats
        model_daily = daily["models"][model]
        model_daily["requests"]["total"] += 1
        model_daily["requests"]["success" if success else "failure"] += 1
        model_daily["tokens"]["input"] += input_tokens
        model_daily["tokens"]["output"] += output_tokens
        model_daily["tokens"]["total"] += input_tokens + output_tokens
        model_daily["costs"]["input"] += input_cost
        model_daily["costs"]["output"] += output_cost
        model_daily["costs"]["total"] += total_cost
        
        if success:
            model_daily["latency"]["total_ms"] += latency_ms
            model_daily["latency"]["count"] += 1
            model_daily["latency"]["average_ms"] = (
                model_daily["latency"]["total_ms"] / model_daily["latency"]["count"]
            )
        
        # Initialize operation stats if not exists
        if operation not in daily["operations"]:
            daily["operations"][operation] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0}
            }
        
        # Update operation stats
        operation_daily = daily["operations"][operation]
        operation_daily["requests"]["total"] += 1
        operation_daily["requests"]["success" if success else "failure"] += 1
        operation_daily["tokens"]["input"] += input_tokens
        operation_daily["tokens"]["output"] += output_tokens
        operation_daily["tokens"]["total"] += input_tokens + output_tokens
        operation_daily["costs"]["input"] += input_cost
        operation_daily["costs"]["output"] += output_cost
        operation_daily["costs"]["total"] += total_cost
        
        if success:
            operation_daily["latency"]["total_ms"] += latency_ms
            operation_daily["latency"]["count"] += 1
            operation_daily["latency"]["average_ms"] = (
                operation_daily["latency"]["total_ms"] / operation_daily["latency"]["count"]
            )
        
        # Update hourly stats with same patterns
        hourly = self.hourly_stats[date_str][hour]
        hourly["requests"]["total"] += 1
        hourly["requests"]["success" if success else "failure"] += 1
        hourly["tokens"]["input"] += input_tokens
        hourly["tokens"]["output"] += output_tokens
        hourly["tokens"]["total"] += input_tokens + output_tokens
        hourly["costs"]["input"] += input_cost
        hourly["costs"]["output"] += output_cost
        hourly["costs"]["total"] += total_cost
        
        if success:
            hourly["latency"]["total_ms"] += latency_ms
            hourly["latency"]["count"] += 1
            hourly["latency"]["average_ms"] = hourly["latency"]["total_ms"] / hourly["latency"]["count"]
        
        # Update provider stats for this hour
        if provider not in hourly["providers"]:
            hourly["providers"][provider] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0}
            }
        
        provider_hourly = hourly["providers"][provider]
        provider_hourly["requests"]["total"] += 1
        provider_hourly["requests"]["success" if success else "failure"] += 1
        provider_hourly["tokens"]["input"] += input_tokens
        provider_hourly["tokens"]["output"] += output_tokens
        provider_hourly["tokens"]["total"] += input_tokens + output_tokens
        provider_hourly["costs"]["input"] += input_cost
        provider_hourly["costs"]["output"] += output_cost
        provider_hourly["costs"]["total"] += total_cost
        
        if success:
            provider_hourly["latency"]["total_ms"] += latency_ms
            provider_hourly["latency"]["count"] += 1
            provider_hourly["latency"]["average_ms"] = (
                provider_hourly["latency"]["total_ms"] / provider_hourly["latency"]["count"]
            )
            
        # Update model stats for this hour
        if model not in hourly["models"]:
            hourly["models"][model] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0}
            }
        
        model_hourly = hourly["models"][model]
        model_hourly["requests"]["total"] += 1
        model_hourly["requests"]["success" if success else "failure"] += 1
        model_hourly["tokens"]["input"] += input_tokens
        model_hourly["tokens"]["output"] += output_tokens
        model_hourly["tokens"]["total"] += input_tokens + output_tokens
        model_hourly["costs"]["input"] += input_cost
        model_hourly["costs"]["output"] += output_cost
        model_hourly["costs"]["total"] += total_cost
        
        if success:
            model_hourly["latency"]["total_ms"] += latency_ms
            model_hourly["latency"]["count"] += 1
            model_hourly["latency"]["average_ms"] = (
                model_hourly["latency"]["total_ms"] / model_hourly["latency"]["count"]
            )
        
        # Update overall provider stats
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0},
                "first_seen": now.isoformat(),
                "last_seen": now.isoformat()
            }
        
        provider_stats = self.provider_stats[provider]
        provider_stats["requests"]["total"] += 1
        provider_stats["requests"]["success" if success else "failure"] += 1
        provider_stats["tokens"]["input"] += input_tokens
        provider_stats["tokens"]["output"] += output_tokens
        provider_stats["tokens"]["total"] += input_tokens + output_tokens
        provider_stats["costs"]["input"] += input_cost
        provider_stats["costs"]["output"] += output_cost
        provider_stats["costs"]["total"] += total_cost
        provider_stats["last_seen"] = now.isoformat()
        
        if success:
            provider_stats["latency"]["total_ms"] += latency_ms
            provider_stats["latency"]["count"] += 1
            provider_stats["latency"]["average_ms"] = (
                provider_stats["latency"]["total_ms"] / provider_stats["latency"]["count"]
            )
        
        # Update overall model stats
        if model not in self.model_stats:
            self.model_stats[model] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0},
                "first_seen": now.isoformat(),
                "last_seen": now.isoformat()
            }
        
        model_stats = self.model_stats[model]
        model_stats["requests"]["total"] += 1
        model_stats["requests"]["success" if success else "failure"] += 1
        model_stats["tokens"]["input"] += input_tokens
        model_stats["tokens"]["output"] += output_tokens
        model_stats["tokens"]["total"] += input_tokens + output_tokens
        model_stats["costs"]["input"] += input_cost
        model_stats["costs"]["output"] += output_cost
        model_stats["costs"]["total"] += total_cost
        model_stats["last_seen"] = now.isoformat()
        
        if success:
            model_stats["latency"]["total_ms"] += latency_ms
            model_stats["latency"]["count"] += 1
            model_stats["latency"]["average_ms"] = (
                model_stats["latency"]["total_ms"] / model_stats["latency"]["count"]
            )
        
        # Update overall operation stats
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                "requests": {"total": 0, "success": 0, "failure": 0},
                "tokens": {"input": 0, "output": 0, "total": 0},
                "costs": {"input": 0.0, "output": 0.0, "total": 0.0},
                "latency": {"total_ms": 0, "count": 0, "average_ms": 0},
                "first_seen": now.isoformat(),
                "last_seen": now.isoformat()
            }
        
        operation_stats = self.operation_stats[operation]
        operation_stats["requests"]["total"] += 1
        operation_stats["requests"]["success" if success else "failure"] += 1
        operation_stats["tokens"]["input"] += input_tokens
        operation_stats["tokens"]["output"] += output_tokens
        operation_stats["tokens"]["total"] += input_tokens + output_tokens
        operation_stats["costs"]["input"] += input_cost
        operation_stats["costs"]["output"] += output_cost
        operation_stats["costs"]["total"] += total_cost
        operation_stats["last_seen"] = now.isoformat()
        
        if success:
            operation_stats["latency"]["total_ms"] += latency_ms
            operation_stats["latency"]["count"] += 1
            operation_stats["latency"]["average_ms"] = (
                operation_stats["latency"]["total_ms"] / operation_stats["latency"]["count"]
            )
            
        # Check if we need to publish a cost event for alerts
        await self._check_cost_thresholds(provider, date_str)
    
    async def _check_cost_thresholds(self, provider: str, date_str: str) -> None:
        """Check if cost thresholds have been exceeded and publish events.
        
        Args:
            provider: API provider
            date_str: Date string
        """
        # Get daily stats
        daily_stats = self.daily_stats.get(date_str, {})
        if not daily_stats:
            return
        
        # Get provider daily stats
        provider_stats = daily_stats.get("providers", {}).get(provider, {})
        if not provider_stats:
            return
        
        # Get total daily cost for this provider
        daily_cost = provider_stats.get("costs", {}).get("total", 0.0)
        
        # Calculate weekly cost (last 7 days including today)
        weekly_cost = daily_cost
        today = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        for i in range(1, 7):  # Last 6 days before today
            day = today - timedelta(days=i)
            day_str = day.isoformat()
            
            if day_str in self.daily_stats:
                day_stats = self.daily_stats[day_str]
                provider_day_stats = day_stats.get("providers", {}).get(provider, {})
                
                if provider_day_stats:
                    weekly_cost += provider_day_stats.get("costs", {}).get("total", 0.0)
        
        # Publish cost event for alerting
        event_bus.publish(
            "sentiment_api_cost",
            {
                "provider": provider,
                "daily_cost": daily_cost,
                "weekly_cost": weekly_cost,
                "date": date_str,
                "details": {
                    "tokens": provider_stats.get("tokens", {}),
                    "requests": provider_stats.get("requests", {})
                }
            }
        )
    
    async def _save_statistics(self) -> None:
        """Save statistics to storage."""
        try:
            # Save daily stats
            for date_str, stats in self.daily_stats.items():
                daily_file = os.path.join(self.storage_dir, "daily", f"{date_str}.json")
                with open(daily_file, "w") as f:
                    json.dump(stats, f, indent=2)
            
            # Save hourly stats
            for date_str, hours in self.hourly_stats.items():
                hourly_file = os.path.join(self.storage_dir, "hourly", f"{date_str}.json")
                with open(hourly_file, "w") as f:
                    json.dump(hours, f, indent=2)
            
            # Save provider stats
            provider_file = os.path.join(self.storage_dir, "providers", "stats.json")
            with open(provider_file, "w") as f:
                json.dump(self.provider_stats, f, indent=2)
            
            # Save model stats
            model_file = os.path.join(self.storage_dir, "models", "stats.json")
            with open(model_file, "w") as f:
                json.dump(self.model_stats, f, indent=2)
            
            # Save operation stats
            operation_file = os.path.join(self.storage_dir, "operations", "stats.json")
            with open(operation_file, "w") as f:
                json.dump(self.operation_stats, f, indent=2)
            
            self.logger.debug("Saved usage statistics")
            
        except Exception as e:
            self.logger.error(f"Error saving usage statistics: {str(e)}")
    
    def get_daily_usage(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily usage statistics for the past N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of daily usage statistics
        """
        result = []
        
        # Get last N days
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days-1)
        
        # Iterate through dates
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            
            if date_str in self.daily_stats:
                stats = self.daily_stats[date_str]
                result.append({
                    "date": date_str,
                    "requests": stats["requests"]["total"],
                    "tokens": stats["tokens"]["total"],
                    "cost": stats["costs"]["total"],
                    "providers": stats["providers"]
                })
            else:
                # Add empty stats for days with no data
                result.append({
                    "date": date_str,
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "providers": {}
                })
            
            current_date += timedelta(days=1)
        
        return result
    
    def get_hourly_usage(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get hourly usage statistics for a specific date.
        
        Args:
            date_str: Date string (YYYY-MM-DD) or None for today
            
        Returns:
            List of hourly usage statistics
        """
        if date_str is None:
            date_str = datetime.utcnow().date().isoformat()
        
        result = []
        
        if date_str in self.hourly_stats:
            for hour in range(24):
                if hour in self.hourly_stats[date_str]:
                    stats = self.hourly_stats[date_str][hour]
                    result.append({
                        "hour": hour,
                        "requests": stats["requests"]["total"],
                        "tokens": stats["tokens"]["total"],
                        "cost": stats["costs"]["total"],
                        "providers": stats["providers"]
                    })
                else:
                    # Add empty stats for hours with no data
                    result.append({
                        "hour": hour,
                        "requests": 0,
                        "tokens": 0,
                        "cost": 0.0,
                        "providers": {}
                    })
        else:
            # Add empty stats for all hours if no data for this date
            for hour in range(24):
                result.append({
                    "hour": hour,
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "providers": {}
                })
        
        return result
    
    def get_provider_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics by provider.
        
        Returns:
            Provider usage statistics
        """
        return self.provider_stats
    
    def get_model_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics by model.
        
        Returns:
            Model usage statistics
        """
        return self.model_stats
    
    def get_operation_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics by operation type.
        
        Returns:
            Operation usage statistics
        """
        return self.operation_stats
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get cost summary for the past N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Cost summary
        """
        # Get daily usage
        daily_usage = self.get_daily_usage(days)
        
        # Calculate totals
        total_cost = sum(day["cost"] for day in daily_usage)
        total_tokens = sum(day["tokens"] for day in daily_usage)
        total_requests = sum(day["requests"] for day in daily_usage)
        
        # Get breakdown by provider
        provider_costs = defaultdict(float)
        provider_tokens = defaultdict(int)
        provider_requests = defaultdict(int)
        
        for day in daily_usage:
            for provider, stats in day["providers"].items():
                provider_costs[provider] += stats.get("costs", {}).get("total", 0.0)
                provider_tokens[provider] += stats.get("tokens", {}).get("total", 0)
                provider_requests[provider] += stats.get("requests", {}).get("total", 0)
        
        # Format provider breakdown
        providers = [
            {
                "provider": provider,
                "cost": cost,
                "tokens": provider_tokens[provider],
                "requests": provider_requests[provider],
                "percent": (cost / total_cost * 100) if total_cost > 0 else 0
            }
            for provider, cost in provider_costs.items()
        ]
        
        # Sort by cost descending
        providers.sort(key=lambda x: x["cost"], reverse=True)
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "daily_average": total_cost / days if days > 0 else 0,
            "providers": providers,
            "days": days,
            "start_date": (datetime.utcnow().date() - timedelta(days=days-1)).isoformat(),
            "end_date": datetime.utcnow().date().isoformat()
        }
    
    def get_usage_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on usage patterns.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Get cost summary
        cost_summary = self.get_cost_summary(30)
        
        # Check for inefficient model usage
        models = list(self.model_stats.items())
        models.sort(key=lambda x: x[1]["costs"]["total"], reverse=True)
        
        for model, stats in models[:3]:  # Focus on top 3 models by cost
            # Check cost per token
            tokens = stats["tokens"]["total"]
            cost = stats["costs"]["total"]
            
            if tokens > 0 and "gpt-4" in model and cost > 10:
                # Suggest using 3.5 for less critical tasks
                recommendations.append({
                    "type": "model_selection",
                    "model": model,
                    "message": f"Consider using gpt-3.5-turbo for less critical tasks instead of {model}",
                    "potential_savings": f"Up to {cost * 0.3:.2f} USD",
                    "priority": "medium" if cost > 50 else "low"
                })
            
            # Check for temperature settings
            if model in ["gpt-4o", "gpt-4-turbo"] and tokens > 100000:
                recommendations.append({
                    "type": "parameter_tuning",
                    "model": model,
                    "message": "Use temperature adjustment for deterministic tasks (0.0-0.2)",
                    "potential_savings": "Improved consistency, fewer retries",
                    "priority": "medium"
                })
        
        # Check for usage patterns
        hourly_data = self.get_hourly_usage()
        peak_hour = max(hourly_data, key=lambda x: x["requests"])
        
        if peak_hour["requests"] > 100:
            recommendations.append({
                "type": "request_distribution",
                "hour": peak_hour["hour"],
                "message": f"High request concentration at hour {peak_hour['hour']}. Consider distributing requests more evenly.",
                "potential_savings": "Lower rate limit issues, improved system stability",
                "priority": "low"
            })
        
        # Check for excessive token usage
        for operation, stats in self.operation_stats.items():
            avg_tokens_per_request = 0
            if stats["requests"]["total"] > 0:
                avg_tokens_per_request = stats["tokens"]["total"] / stats["requests"]["total"]
            
            if avg_tokens_per_request > 2000 and stats["requests"]["total"] > 10:
                recommendations.append({
                    "type": "prompt_optimization",
                    "operation": operation,
                    "message": f"High token usage ({avg_tokens_per_request:.0f} per request) for {operation}. Consider refining prompts.",
                    "potential_savings": f"Up to {stats['costs']['total'] * 0.2:.2f} USD",
                    "priority": "high" if stats["costs"]["total"] > 20 else "medium"
                })
        
        return recommendations


# Singleton instance
usage_tracker = APIUsageTracker()