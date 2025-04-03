# Provider Failover System

The Provider Failover System is a critical component of the sentiment analysis architecture that ensures robust, uninterrupted operation by automatically detecting LLM API failures and seamlessly transitioning to alternative providers.

## Overview

The Provider Failover System provides the following capabilities:

1. **Health Monitoring**: Continuously tracks the health of each LLM provider (OpenAI, Anthropic, Azure)
2. **Automatic Failover**: Redirects requests to healthy providers when a provider becomes unavailable
3. **Circuit Breaker Pattern**: Temporarily stops requests to failing providers to prevent cascading failures
4. **Fallback Responses**: Provides cached responses when all providers are unavailable
5. **Health Dashboard**: Visualizes provider health, failover events, and performance metrics
6. **Self-healing**: Periodically tests unhealthy providers for recovery

## Architecture

The Provider Failover System consists of the following components:

### Provider Failover Manager

The heart of the system is the `ProviderFailoverManager`, which:

- Tracks the health status of each provider (`HEALTHY`, `DEGRADED`, `UNHEALTHY`)
- Maintains statistics on request success/failure rates and latency
- Implements the circuit breaker pattern for unhealthy providers
- Provides intelligent routing based on provider health and priority
- Maintains a fallback response cache for emergencies

### LLM Service Integration

The LLM Service is modified to:

- Consult the failover manager before making API calls
- Switch to alternative providers when the primary provider is unhealthy
- Record API usage metrics for monitoring
- Use fallback cached responses when all providers are unavailable
- Report health events to the monitoring system

### Provider Health Dashboard

A dedicated dashboard that:

- Displays real-time provider health status
- Shows success rates, latency, and request volumes over time
- Provides detailed metrics for each provider
- Visualizes failover events and circuit breaker status

## Configuration

The Provider Failover System can be configured via the following settings in `config/sentiment_analysis.yaml`:

```yaml
llm:
  failover:
    # How many consecutive errors trigger unhealthy status
    consecutive_errors_threshold: 3
    
    # Time window for calculating error rate (seconds)
    error_window_seconds: 300
    
    # Error rate that triggers degraded status (0.0-1.0)
    error_rate_threshold: 0.25
    
    # How often to check if unhealthy providers have recovered (seconds)
    recovery_check_interval: 60
    
    # How many successful pings needed to consider a provider recovered
    recovery_threshold: 3
    
    # How long to wait before testing an unhealthy provider (seconds)
    circuit_breaker_reset_time: 300
    
    # Provider priorities (lower number = higher priority)
    priorities:
      openai: 1
      anthropic: 2
      azure: 3
      
    # Fallback response cache TTL (seconds)
    fallback_cache_ttl: 86400
```

## Integration with Alerting System

The Provider Failover System integrates with the Alerting System to:

1. Generate alerts when a provider becomes degraded or unhealthy
2. Record detailed diagnostic information for troubleshooting
3. Alert when a provider recovers from failure
4. Notify when all providers are unavailable

## How Failover Works

1. When a request is made to the LLM service, the provider failover manager is consulted for the best provider to use based on:
   - Provider health status
   - Configured provider priorities
   - Model compatibility

2. If the original provider is healthy, it is used as normal

3. If the original provider is unhealthy:
   - An alternative provider is selected based on priority
   - The request is transparently redirected to the alternative provider
   - The response is processed normally

4. If all providers are unhealthy:
   - The system attempts to use a cached response from previous successful calls
   - If no cached response is available, a generic fallback response is returned

5. The health status of each provider is continually updated based on:
   - Consecutive errors (immediately marks as UNHEALTHY)
   - Error rate within a time window (marks as DEGRADED)
   - Recovery checks (periodically tests unhealthy providers)

## Circuit Breaker Pattern

The system implements the circuit breaker pattern to prevent overwhelming unhealthy services:

1. **Closed State**: Normal operation, requests flow through
2. **Open State**: Provider is marked unhealthy, requests are redirected
3. **Half-Open State**: After a reset timeout, test requests are sent to check recovery

## Fallback Response Caching

To ensure operation even when all providers are unavailable:

1. Successful responses are cached with a hash of the input
2. If all providers fail, cached responses are used when available
3. Cached responses expire after a configurable TTL
4. Fallback responses are persisted to disk for reliability across restarts

## Model Alternatives Mapping

The system maintains a mapping of model alternatives for failover:

```
gpt-4o → gpt-4-turbo → claude-3-opus → claude-3-sonnet → azure-gpt-4
```

This ensures that when a provider fails, an appropriate alternative model is selected that can perform the same function.

## Running the Provider Health Dashboard

To monitor the health of all providers, run:

```bash
python run_provider_health_dashboard.py --port 8051
```

Then open a browser to `http://localhost:8051`

## Metrics Tracked

For each provider, the system tracks:

- Request count
- Success count and rate
- Error count and consecutive errors
- Average latency
- Tokens processed
- Last success and error timestamps
- Error messages

## Technical Implementation Details

1. **Event-based Architecture**: Uses the event bus for communication between components
2. **Asynchronous Programming**: Implemented with `asyncio` for non-blocking operations
3. **Thread Safety**: Uses locks to prevent race conditions during status updates
4. **Persistent Caching**: Caches fallback responses to disk for reliability
5. **Health Check Protocol**: Implements lightweight ping operations for checking provider health
6. **Timeout Handling**: Sets appropriate timeouts for API calls to prevent hanging requests