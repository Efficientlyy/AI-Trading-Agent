"""
Prometheus metrics for the LLM Oversight service.

This module provides a metrics collector for monitoring LLM oversight performance
and integrating with the existing Prometheus monitoring infrastructure.
"""

import time
import logging
from typing import Dict, Any, Optional
from enum import Enum
from prometheus_client import Counter, Histogram, Gauge, Summary

# Configure logging
logger = logging.getLogger(__name__)


class OversightMetricsCollector:
    """
    Collects and exposes metrics for the LLM Oversight service.
    
    This class creates and updates Prometheus metrics that track oversight
    performance, response times, and system health.
    """
    
    def __init__(self):
        """Initialize metrics collectors."""
        # Request counters
        self.total_requests = Counter(
            'llm_oversight_requests_total',
            'Total number of requests to the oversight service',
            ['endpoint']
        )
        
        self.failed_requests = Counter(
            'llm_oversight_requests_failed',
            'Number of failed requests to the oversight service',
            ['endpoint', 'error_type']
        )
        
        # Decision validation metrics
        self.validation_decisions = Counter(
            'llm_oversight_validation_decisions',
            'Decisions made by the oversight system',
            ['action']  # approve, reject, modify
        )
        
        self.validation_latency = Histogram(
            'llm_oversight_validation_latency',
            'Latency of validation requests',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Market analysis metrics
        self.market_analysis_latency = Histogram(
            'llm_oversight_market_analysis_latency',
            'Latency of market analysis requests',
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.market_regimes_detected = Counter(
            'llm_oversight_market_regimes',
            'Market regimes detected by oversight',
            ['regime_type']  # bull, bear, neutral, volatile
        )
        
        # LLM interaction metrics
        self.llm_tokens_used = Counter(
            'llm_oversight_tokens_used',
            'Number of tokens consumed by LLM requests',
            ['provider', 'model', 'request_type']  # provider, model, prompt type
        )
        
        self.llm_request_errors = Counter(
            'llm_oversight_llm_errors',
            'Errors encountered when making LLM API calls',
            ['provider', 'error_type']
        )
        
        self.llm_request_latency = Histogram(
            'llm_oversight_llm_latency',
            'Latency of LLM API calls',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # System health metrics
        self.health_status = Gauge(
            'llm_oversight_health',
            'Health status of the oversight service (1=healthy, 0=unhealthy)'
        )
        
        # Per-model confidence metrics
        self.decision_confidence = Summary(
            'llm_oversight_confidence',
            'Confidence levels for oversight decisions',
            ['action']
        )
        
        # Evaluation metrics for accuracy tracking
        self.decision_outcomes = Counter(
            'llm_oversight_decision_outcomes',
            'Outcomes of oversight decisions (whether they were correct)',
            ['action', 'outcome']  # action=approve/reject/modify, outcome=correct/incorrect
        )
    
    def record_request(self, endpoint: str) -> None:
        """
        Record a request to an endpoint.
        
        Args:
            endpoint: Name of the endpoint being called
        """
        self.total_requests.labels(endpoint=endpoint).inc()
    
    def record_failed_request(self, endpoint: str, error_type: str) -> None:
        """
        Record a failed request.
        
        Args:
            endpoint: Name of the endpoint being called
            error_type: Type of error encountered
        """
        self.failed_requests.labels(endpoint=endpoint, error_type=error_type).inc()
    
    def record_validation_decision(self, action: str, latency: float, confidence: float) -> None:
        """
        Record a validation decision.
        
        Args:
            action: Decision action (approve, reject, modify)
            latency: Request processing time in seconds
            confidence: Confidence level of the decision (0-1)
        """
        self.validation_decisions.labels(action=action).inc()
        self.validation_latency.observe(latency)
        self.decision_confidence.labels(action=action).observe(confidence)
    
    def record_market_analysis(self, regime: str, latency: float) -> None:
        """
        Record a market analysis result.
        
        Args:
            regime: Detected market regime
            latency: Request processing time in seconds
        """
        self.market_regimes_detected.labels(regime_type=regime).inc()
        self.market_analysis_latency.observe(latency)
    
    def record_llm_tokens(
        self, provider: str, model: str, request_type: str, token_count: int
    ) -> None:
        """
        Record LLM token usage.
        
        Args:
            provider: LLM provider (e.g., OpenAI, Anthropic)
            model: LLM model used
            request_type: Type of request (e.g., validation, analysis)
            token_count: Number of tokens used
        """
        self.llm_tokens_used.labels(
            provider=provider, model=model, request_type=request_type
        ).inc(token_count)
    
    def record_llm_error(self, provider: str, error_type: str) -> None:
        """
        Record an LLM API error.
        
        Args:
            provider: LLM provider (e.g., OpenAI, Anthropic)
            error_type: Type of error encountered
        """
        self.llm_request_errors.labels(provider=provider, error_type=error_type).inc()
    
    def time_llm_request(self) -> 'LLMRequestTimer':
        """
        Create a timer for LLM requests.
        
        Returns:
            Timer context manager
        """
        return LLMRequestTimer(self.llm_request_latency)
    
    def update_health_status(self, is_healthy: bool) -> None:
        """
        Update service health status.
        
        Args:
            is_healthy: Whether the service is healthy
        """
        self.health_status.set(1 if is_healthy else 0)
    
    def record_decision_outcome(self, action: str, was_correct: bool) -> None:
        """
        Record the outcome of a decision.
        
        Args:
            action: Decision action (approve, reject, modify)
            was_correct: Whether the decision was correct
        """
        outcome = "correct" if was_correct else "incorrect"
        self.decision_outcomes.labels(action=action, outcome=outcome).inc()


class LLMRequestTimer:
    """Context manager for timing LLM requests."""
    
    def __init__(self, histogram):
        """Initialize with a histogram."""
        self.histogram = histogram
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration."""
        duration = time.time() - self.start_time
        self.histogram.observe(duration)


# Create a singleton instance for use throughout the service
metrics = OversightMetricsCollector()
