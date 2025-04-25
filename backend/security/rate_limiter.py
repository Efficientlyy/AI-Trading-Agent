"""
API Rate Limiter

This module provides rate limiting functionality for the API to prevent abuse,
protect against DDoS attacks, and ensure fair usage of resources.
"""

import time
import logging
from typing import Dict, Optional, Tuple, Any, Callable, Awaitable, List
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
import hashlib
import json

from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Setup logging
logger = logging.getLogger(__name__)

# Import security audit logging
from .audit_logging import log_security_event, SecurityEventType, log_rate_limit_event

@dataclass
class RateLimitRule:
    """Defines a rate limit rule with number of requests allowed in a time window"""
    name: str
    requests_per_window: int
    window_seconds: int
    scope: str = "default"  # Can be 'default', 'ip', 'user', 'endpoint', etc.
    exempt_paths: list = field(default_factory=list)
    exempt_methods: list = field(default_factory=list)
    penalty_factor: float = 1.0  # Multiplier for repeat offenders
    max_penalty_factor: float = 10.0  # Maximum penalty multiplier


@dataclass
class RateLimitState:
    """Tracks the state of a rate limit for a specific key"""
    requests: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)
    violation_count: int = 0  # Count of rate limit violations
    current_penalty: float = 1.0  # Current penalty multiplier


class RateLimiter:
    """
    Rate limiter that tracks and enforces request limits based on configurable rules.
    """
    
    def __init__(self):
        self.rules: Dict[str, RateLimitRule] = {}
        self.states: Dict[str, Dict[str, RateLimitState]] = {}
        self.cleanup_task = None
        self.api_key_rules: Dict[str, RateLimitRule] = {}  # Rules specific to API keys
    
    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule"""
        self.rules[rule.name] = rule
        if rule.scope not in self.states:
            self.states[rule.scope] = {}
    
    def add_api_key_rule(self, api_key: str, rule: RateLimitRule) -> None:
        """Add a rate limiting rule for a specific API key"""
        self.api_key_rules[api_key] = rule
        scope = f"api_key:{api_key}"
        if scope not in self.states:
            self.states[scope] = {}
    
    def generate_key(self, rule: RateLimitRule, request: Request) -> str:
        """Generate a unique key for the request based on the rule scope"""
        if rule.scope == "ip":
            # Use client IP as key
            client_host = request.client.host if request.client else "unknown"
            # Check for X-Forwarded-For header for clients behind proxies
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                # Use the first IP in the list (client IP)
                client_host = forwarded_for.split(",")[0].strip()
            return f"{rule.name}:ip:{client_host}"
        elif rule.scope == "user":
            # Use user ID as key if authenticated, or IP if not
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                return f"{rule.name}:user:{user_id}"
            else:
                client_host = request.client.host if request.client else "unknown"
                return f"{rule.name}:ip:{client_host}"
        elif rule.scope == "endpoint":
            # Use path and method as key
            return f"{rule.name}:{request.method}:{request.url.path}"
        elif rule.scope == "user_endpoint":
            # Use user ID and endpoint as key
            user_id = getattr(request.state, "user_id", None)
            client_host = request.client.host if request.client else "unknown"
            key_base = f"user:{user_id}" if user_id else f"ip:{client_host}"
            return f"{rule.name}:{key_base}:{request.method}:{request.url.path}"
        elif rule.scope == "api_key":
            # Use API key as key
            api_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
            if api_key:
                # Hash the API key before using it in the key to avoid exposing it in logs
                hashed_key = hashlib.sha256(api_key.encode()).hexdigest()[:8]
                return f"{rule.name}:api_key:{hashed_key}"
            else:
                # Fallback to IP if no API key
                client_host = request.client.host if request.client else "unknown"
                return f"{rule.name}:ip:{client_host}"
        else:
            # Default to using the rule name as key (global rate limit)
            return rule.name
    
    def is_path_exempt(self, rule: RateLimitRule, path: str) -> bool:
        """Check if a path is exempt from rate limiting for this rule"""
        for exempt_path in rule.exempt_paths:
            if exempt_path.endswith('*'):
                prefix = exempt_path[:-1]
                if path.startswith(prefix):
                    return True
            elif path == exempt_path:
                return True
        return False
    
    def is_method_exempt(self, rule: RateLimitRule, method: str) -> bool:
        """Check if a method is exempt from rate limiting for this rule"""
        return method in rule.exempt_methods
    
    def apply_progressive_penalty(self, state: RateLimitState, rule: RateLimitRule) -> None:
        """Apply a progressive penalty to repeat violators"""
        if state.violation_count > 0:
            # Increase the penalty factor based on violation count and rule configuration
            state.current_penalty = min(
                rule.max_penalty_factor,
                1.0 + (state.violation_count * (rule.penalty_factor - 1.0))
            )
        else:
            state.current_penalty = 1.0
    
    def get_adjusted_limit(self, rule: RateLimitRule, state: RateLimitState) -> int:
        """Get the adjusted request limit after applying penalties"""
        adjusted_limit = int(rule.requests_per_window / state.current_penalty)
        return max(1, adjusted_limit)  # Ensure at least 1 request is allowed
    
    async def check_rate_limit(self, request: Request) -> Tuple[bool, Optional[RateLimitRule], Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if request exceeds any rate limits.
        
        Returns:
            Tuple containing:
            - Boolean indicating if request is allowed
            - The rule that was violated (if any)
            - The key that was used for the check (if any)
            - Dict with rate limit headers (if any)
        """
        now = time.time()
        headers = {}
        
        # Check for API key-specific rules first
        api_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if api_key and api_key in self.api_key_rules:
            rule = self.api_key_rules[api_key]
            scope = f"api_key:{api_key}"
            
            # Generate key for this rule and request
            key = f"{rule.name}:{api_key}"
            
            # Get or create state for this key
            if key not in self.states.get(scope, {}):
                if scope not in self.states:
                    self.states[scope] = {}
                self.states[scope][key] = RateLimitState(requests=0, window_start=now, last_request=now)
            
            state = self.states[scope][key]
            
            # Check if window has expired and reset if needed
            if now - state.window_start > rule.window_seconds:
                state.requests = 0
                state.window_start = now
                # Keep violation count and penalty to persist between windows
            
            # Apply progressive penalty
            self.apply_progressive_penalty(state, rule)
            adjusted_limit = self.get_adjusted_limit(rule, state)
            
            # Check if limit is exceeded
            if state.requests >= adjusted_limit:
                # Calculate reset time
                reset_at = state.window_start + rule.window_seconds
                reset_seconds = int(reset_at - now) + 1  # Round up
                
                # Increment violation count
                state.violation_count += 1
                
                # Set rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(adjusted_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                    "Retry-After": str(reset_seconds)
                }
                
                # Log rate limit event
                user_id = getattr(request.state, "user_id", None)
                client_host = request.client.host if request.client else "unknown"
                
                log_rate_limit_event(
                    limit_type="api_key",
                    client_ip=client_host,
                    endpoint=request.url.path,
                    current_usage=state.requests,
                    limit=adjusted_limit,
                    user_id=user_id,
                    request=request
                )
                
                # Request exceeds rate limit
                return False, rule, key, headers
            
            # Update state with this request
            state.requests += 1
            state.last_request = now
            
            # Set rate limit headers for this rule
            headers = {
                "X-RateLimit-Limit": str(adjusted_limit),
                "X-RateLimit-Remaining": str(adjusted_limit - state.requests),
                "X-RateLimit-Reset": str(int(state.window_start + rule.window_seconds - now))
            }
        
        # Check regular rules
        for rule_name, rule in self.rules.items():
            # Skip exempt paths and methods
            if self.is_path_exempt(rule, request.url.path) or self.is_method_exempt(rule, request.method):
                continue
            
            # Generate key for this rule and request
            key = self.generate_key(rule, request)
            
            # Get or create state for this key
            if key not in self.states.get(rule.scope, {}):
                if rule.scope not in self.states:
                    self.states[rule.scope] = {}
                self.states[rule.scope][key] = RateLimitState(requests=0, window_start=now, last_request=now)
            
            state = self.states[rule.scope][key]
            
            # Check if window has expired and reset if needed
            if now - state.window_start > rule.window_seconds:
                state.requests = 0
                state.window_start = now
                # Keep violation count and penalty to persist between windows
            
            # Apply progressive penalty
            self.apply_progressive_penalty(state, rule)
            adjusted_limit = self.get_adjusted_limit(rule, state)
            
            # Check if limit is exceeded
            if state.requests >= adjusted_limit:
                # Calculate reset time
                reset_at = state.window_start + rule.window_seconds
                reset_seconds = int(reset_at - now) + 1  # Round up
                
                # Increment violation count
                state.violation_count += 1
                
                # Set rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(adjusted_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                    "Retry-After": str(reset_seconds)
                }
                
                # Log rate limit event
                user_id = getattr(request.state, "user_id", None)
                client_host = request.client.host if request.client else "unknown"
                
                log_rate_limit_event(
                    limit_type=rule.scope,
                    client_ip=client_host,
                    endpoint=request.url.path,
                    current_usage=state.requests,
                    limit=adjusted_limit,
                    user_id=user_id,
                    request=request
                )
                
                # Apply additional restrictions if repeated violations
                if state.violation_count > 3:
                    # Log as potential abuse
                    log_security_event(
                        event_type=SecurityEventType.RATE_LIMIT_RESTRICTION_APPLIED,
                        message=f"Repeated rate limit violations detected",
                        user_id=user_id,
                        request=request,
                        details={
                            "rule": rule.name,
                            "violation_count": state.violation_count,
                            "client_ip": client_host,
                            "endpoint": request.url.path,
                        },
                        severity="WARNING"
                    )
                
                # Request exceeds rate limit
                return False, rule, key, headers
            
            # Update state with this request
            state.requests += 1
            state.last_request = now
            
            # Set rate limit headers for this rule
            headers = {
                "X-RateLimit-Limit": str(adjusted_limit),
                "X-RateLimit-Remaining": str(adjusted_limit - state.requests),
                "X-RateLimit-Reset": str(int(state.window_start + rule.window_seconds - now))
            }
        
        # All rules passed
        return True, None, None, headers
    
    async def cleanup_expired_states(self) -> None:
        """Clean up expired rate limit states to prevent memory leaks"""
        try:
            while True:
                now = time.time()
                for scope, states in self.states.items():
                    expired_keys = []
                    for key, state in states.items():
                        # If the state's window has expired and it's been 
                        # at least 2x the window since the last request, remove it
                        rule_name = key.split(':')[0]
                        if rule_name in self.rules:
                            rule = self.rules[rule_name]
                            if (now - state.window_start > rule.window_seconds and
                                now - state.last_request > rule.window_seconds * 2):
                                expired_keys.append(key)
                    
                    # Remove expired keys
                    for key in expired_keys:
                        states.pop(key, None)
                
                # Clean up every minute
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Task was cancelled
            logger.info("Rate limiter cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in rate limiter cleanup task: {str(e)}")
    
    async def start_cleanup_task(self) -> None:
        """Start the background task to clean up expired states"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self.cleanup_expired_states())
    
    async def stop_cleanup_task(self) -> None:
        """Stop the cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    def remove_api_key_rules(self, api_key: str) -> None:
        """Remove all rate limiting rules for a specific API key"""
        keys_to_remove = []
        for rule_name in list(self.rules.keys()):
            if f"api_key_{api_key}" in rule_name:
                keys_to_remove.append(rule_name)
        
        for key in keys_to_remove:
            del self.rules[key]
        
        # Also clean up any API key-specific state
        scope = f"api_key:{api_key}"
        if scope in self.states:
            del self.states[scope]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits on API requests.
    """
    
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next):
        # Static files and some paths might be exempt from rate limiting
        path = request.url.path
        
        # Check if the path should be rate limited (adjust as needed)
        should_rate_limit = path.startswith('/api') or path.startswith('/ws')
        
        if not should_rate_limit:
            # Skip rate limiting for this request
            return await call_next(request)
        
        # Check rate limits
        allowed, rule, key, headers = await self.limiter.check_rate_limit(request)
        
        if not allowed:
            # Rate limit exceeded
            error_response = {
                "status": "error",
                "message": f"Rate limit exceeded. Try again in {headers.get('Retry-After', 'some')} seconds.",
                "code": "rate_limit_exceeded"
            }
            
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response
            )
            
            # Add rate limit headers
            for name, value in headers.items():
                response.headers[name] = value
            
            # Log the rate limit event
            client_host = request.client.host if request.client else "unknown"
            user_id = getattr(request.state, "user_id", None)
            
            # Detailed logging is now handled in the check_rate_limit method
            
            return response
        
        # Continue processing the request
        response = await call_next(request)
        
        # Add rate limit headers to the response
        if headers:
            for name, value in headers.items():
                response.headers[name] = value
        
        return response


# Add new functionality for API key management
class ApiKeyRateLimitManager:
    """
    Manager for API key rate limit rules and configurations.
    This allows for dynamic management of API key rate limits.
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.api_key_metadata = {}  # Stores metadata for each API key
    
    def register_api_key(self, 
                         api_key: str, 
                         owner_id: str, 
                         name: str = "Default", 
                         tier: str = "basic",
                         requests_per_minute: int = 60,
                         requests_per_day: int = 10000) -> None:
        """
        Register a new API key with rate limit settings
        
        Args:
            api_key: The API key
            owner_id: User ID of the key owner
            name: Name for this API key
            tier: Service tier (basic, premium, enterprise)
            requests_per_minute: Requests allowed per minute
            requests_per_day: Requests allowed per day
        """
        # Create per-minute rule
        minute_rule = RateLimitRule(
            name=f"api_key_{api_key}_minute",
            requests_per_window=requests_per_minute,
            window_seconds=60,
            scope="api_key",
            penalty_factor=1.5,
            max_penalty_factor=5.0
        )
        
        # Create per-day rule
        day_rule = RateLimitRule(
            name=f"api_key_{api_key}_day",
            requests_per_window=requests_per_day,
            window_seconds=86400,  # 24 hours
            scope="api_key",
            penalty_factor=1.5,
            max_penalty_factor=5.0
        )
        
        # Add rules to rate limiter
        self.rate_limiter.add_api_key_rule(api_key, minute_rule)
        self.rate_limiter.add_api_key_rule(api_key, day_rule)
        
        # Store metadata
        self.api_key_metadata[api_key] = {
            "name": name,
            "owner_id": owner_id,
            "tier": tier,
            "created_at": datetime.now().isoformat(),
            "limits": {
                "requests_per_minute": requests_per_minute,
                "requests_per_day": requests_per_day
            },
            "last_used": None,
            "total_requests": 0
        }
        
        # Log the registration
        log_security_event(
            event_type=SecurityEventType.API_KEY_REGISTERED,
            message=f"New API key registered: {name} ({tier} tier)",
            user_id=owner_id,
            details={
                "api_key_name": name,
                "tier": tier,
                "limits": {
                    "requests_per_minute": requests_per_minute,
                    "requests_per_day": requests_per_day
                }
            }
        )
    
    def update_api_key(self, 
                       api_key: str, 
                       name: str = None,
                       tier: str = None,
                       requests_per_minute: int = None,
                       requests_per_day: int = None) -> bool:
        """
        Update an existing API key's settings
        
        Args:
            api_key: The API key to update
            name: New name for this API key (optional)
            tier: New service tier (optional)
            requests_per_minute: New requests allowed per minute (optional)
            requests_per_day: New requests allowed per day (optional)
            
        Returns:
            bool: True if update was successful
        """
        if api_key not in self.api_key_metadata:
            return False
        
        metadata = self.api_key_metadata[api_key]
        
        # Update metadata fields
        if name:
            metadata["name"] = name
        
        if tier:
            metadata["tier"] = tier
        
        # Update rate limits if specified
        if requests_per_minute is not None:
            # Create new per-minute rule
            minute_rule = RateLimitRule(
                name=f"api_key_{api_key}_minute",
                requests_per_window=requests_per_minute,
                window_seconds=60,
                scope="api_key",
                penalty_factor=1.5,
                max_penalty_factor=5.0
            )
            self.rate_limiter.add_api_key_rule(api_key, minute_rule)
            metadata["limits"]["requests_per_minute"] = requests_per_minute
        
        if requests_per_day is not None:
            # Create new per-day rule
            day_rule = RateLimitRule(
                name=f"api_key_{api_key}_day",
                requests_per_window=requests_per_day,
                window_seconds=86400,  # 24 hours
                scope="api_key",
                penalty_factor=1.5,
                max_penalty_factor=5.0
            )
            self.rate_limiter.add_api_key_rule(api_key, day_rule)
            metadata["limits"]["requests_per_day"] = requests_per_day
        
        # Log the update
        log_security_event(
            event_type=SecurityEventType.API_KEY_UPDATED,
            message=f"API key updated: {metadata['name']} ({metadata['tier']} tier)",
            user_id=metadata["owner_id"],
            details={
                "api_key_name": metadata["name"],
                "tier": metadata["tier"],
                "limits": metadata["limits"]
            }
        )
        
        return True
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key
        
        Args:
            api_key: The API key to revoke
            
        Returns:
            bool: True if revocation was successful
        """
        if api_key not in self.api_key_metadata:
            return False
        
        metadata = self.api_key_metadata[api_key]
        
        # Remove API key rules from rate limiter
        self.rate_limiter.remove_api_key_rules(api_key)
        
        # Log the revocation
        log_security_event(
            event_type=SecurityEventType.API_KEY_REVOKED,
            message=f"API key revoked: {metadata['name']} ({metadata['tier']} tier)",
            user_id=metadata["owner_id"],
            details={
                "api_key_name": metadata["name"],
                "tier": metadata["tier"],
                "created_at": metadata["created_at"],
                "total_requests": metadata["total_requests"]
            },
            severity="WARNING"
        )
        
        # Remove metadata
        del self.api_key_metadata[api_key]
        
        return True
    
    def track_api_key_usage(self, api_key: str, request: Request) -> None:
        """
        Track usage of an API key
        
        Args:
            api_key: The API key being used
            request: The current request
        """
        if api_key in self.api_key_metadata:
            metadata = self.api_key_metadata[api_key]
            metadata["last_used"] = datetime.now().isoformat()
            metadata["total_requests"] += 1
    
    def get_api_key_metadata(self, api_key: str) -> Optional[Dict]:
        """
        Get metadata for an API key
        
        Args:
            api_key: The API key to get metadata for
            
        Returns:
            Dict or None: The API key metadata or None if not found
        """
        return self.api_key_metadata.get(api_key)
    
    def list_api_keys(self, owner_id: str = None) -> List[Dict]:
        """
        List all API keys, optionally filtered by owner
        
        Args:
            owner_id: Optional user ID to filter by
            
        Returns:
            List[Dict]: List of API key metadata
        """
        if owner_id:
            return [
                {
                    "key_id": key[-8:],  # Only show last 8 chars for security
                    **meta
                }
                for key, meta in self.api_key_metadata.items()
                if meta["owner_id"] == owner_id
            ]
        else:
            return [
                {
                    "key_id": key[-8:],  # Only show last 8 chars for security
                    **meta
                }
                for key, meta in self.api_key_metadata.items()
            ]


# Create a singleton instance of the rate limiter
rate_limiter = RateLimiter()

# Create the API key manager instance
api_key_manager = ApiKeyRateLimitManager(rate_limiter)

# Add dependency to get API key manager
def get_api_key_manager():
    return api_key_manager

# Add middleware to identify and validate API keys
async def check_api_key(request: Request, call_next):
    """Middleware to identify and validate API keys in requests"""
    # Check for API key in header or query param
    api_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    
    if api_key:
        # Store API key in request state for later use
        request.state.api_key = api_key
        
        # Check if API key is valid and get metadata
        metadata = api_key_manager.get_api_key_metadata(api_key)
        
        if metadata:
            # API key is valid
            request.state.api_key_valid = True
            request.state.api_key_metadata = metadata
            
            # Track API key usage
            api_key_manager.track_api_key_usage(api_key, request)
        else:
            # Invalid API key
            request.state.api_key_valid = False
            
            # Log invalid API key attempt
            client_host = request.client.host if request.client else "unknown"
            log_security_event(
                event_type=SecurityEventType.INVALID_API_KEY,
                message="Invalid API key used in request",
                details={
                    "client_ip": client_host,
                    "endpoint": request.url.path,
                    "api_key_preview": api_key[-8:] if len(api_key) >= 8 else api_key
                },
                severity="WARNING"
            )
    
    # Continue processing the request
    response = await call_next(request)
    return response

# Configure default rate limit rules
rate_limiter.add_rule(
    RateLimitRule(
        name="global",
        requests_per_window=1000,
        window_seconds=60,  # 1000 requests per minute globally
        scope="default",
        exempt_paths=["/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/_debug/*"],
        exempt_methods=["OPTIONS"]
    )
)

rate_limiter.add_rule(
    RateLimitRule(
        name="ip_basic",
        requests_per_window=100,
        window_seconds=60,  # 100 requests per minute per IP
        scope="ip",
        exempt_paths=["/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/_debug/*"],
        exempt_methods=["OPTIONS"],
        penalty_factor=1.5,  # Increase penalty for repeat offenders
        max_penalty_factor=5.0  # Cap at 5x penalty
    )
)

rate_limiter.add_rule(
    RateLimitRule(
        name="user_basic",
        requests_per_window=300,
        window_seconds=60,  # 300 requests per minute per user
        scope="user",
        exempt_paths=["/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/_debug/*"],
        exempt_methods=["OPTIONS"]
    )
)

rate_limiter.add_rule(
    RateLimitRule(
        name="trading_endpoints",
        requests_per_window=20,
        window_seconds=60,  # 20 trading requests per minute
        scope="user_endpoint",
        exempt_paths=[],
        exempt_methods=[]
    )
)

# Add rule specifically for API authentication endpoints to prevent brute force
rate_limiter.add_rule(
    RateLimitRule(
        name="auth_endpoints",
        requests_per_window=5,
        window_seconds=60,  # 5 auth requests per minute
        scope="ip",
        exempt_paths=[],
        exempt_methods=[],
        penalty_factor=2.0,  # Steep penalty increase for repeated auth failures
        max_penalty_factor=10.0  # Up to 10x penalty (0.5 req/min)
    )
)

# Start the cleanup task at application startup
async def start_rate_limiter_cleanup():
    await rate_limiter.start_cleanup_task()

# Stop the cleanup task at application shutdown
async def stop_rate_limiter_cleanup():
    await rate_limiter.stop_cleanup_task()

# Export the middleware for use in the application
def get_rate_limit_middleware():
    return RateLimitMiddleware(limiter=rate_limiter)

# Export the rate limiter for use in the application
def get_rate_limiter():
    return rate_limiter