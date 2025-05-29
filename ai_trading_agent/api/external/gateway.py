"""
External API Gateway for AI Trading Agent.

This module implements the main gateway for external third-party access to the
AI Trading Agent system, providing secure, rate-limited, and versioned APIs.
"""
import logging
import time
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import json
import hashlib
import hmac

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import health monitoring for status checks
from ...monitoring.health import HealthMonitor, HealthStatus

# Rate limiting and authentication utilities
from .auth import APIKeyAuth, OAuth2Auth, JWTAuth, APIKeyStore
from .rate_limiting import RateLimiter, QuotaManager

# Configuratio for different partner tiers
from .config import PartnerTierConfig, PartnerTier, APIConfig

# Setup logging
logger = logging.getLogger(__name__)


# API Gateway class
class ExternalAPIGateway:
    """
    External API Gateway that provides secure access to AI Trading Agent services
    for third-party partners and developers.
    
    Features:
    - Authentication and authorization via API keys and OAuth
    - Rate limiting and quota management
    - Request validation and sanitization
    - Versioned endpoints
    - Usage tracking and analytics
    - Automatic documentation
    """
    
    def __init__(
        self,
        api_config: APIConfig,
        health_monitor: HealthMonitor,
        custom_middlewares: Optional[List[Callable]] = None
    ):
        """
        Initialize the External API Gateway.
        
        Args:
            api_config: Configuration for the API
            health_monitor: Reference to the system health monitor
            custom_middlewares: List of custom middleware functions to apply
        """
        self.app = FastAPI(
            title="AI Trading Agent External API",
            description="Partner API for integrating with AI Trading Agent",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        self.config = api_config
        self.health_monitor = health_monitor
        
        # Initialize authentication components
        self.api_key_store = APIKeyStore(self.config.api_key_db_path)
        self.api_key_auth = APIKeyAuth(self.api_key_store)
        self.oauth_auth = OAuth2Auth(
            token_url=self.config.oauth_token_url,
            client_ids=self.config.oauth_client_ids
        )
        self.jwt_auth = JWTAuth(
            secret_key=self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
            token_expire_minutes=self.config.jwt_token_expire_minutes
        )
        
        # Initialize rate limiting
        self.rate_limiter = RateLimiter()
        self.quota_manager = QuotaManager()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_allow_origins,
            allow_credentials=self.config.cors_allow_credentials,
            allow_methods=self.config.cors_allow_methods,
            allow_headers=self.config.cors_allow_headers,
        )
        
        # Add custom middlewares
        if custom_middlewares:
            for middleware in custom_middlewares:
                self.app.middleware("http")(middleware)
                
        # Register default middlewares
        self.app.middleware("http")(self.log_request_middleware)
        self.app.middleware("http")(self.rate_limit_middleware)
        
        # Register routes
        self._register_routes()
        
        # Register with health monitor
        self.health_monitor.register_component(
            "external_api_gateway",
            self.check_health,
            interval_seconds=30
        )
        
        logger.info("External API Gateway initialized")
    
    async def log_request_middleware(self, request: Request, call_next):
        """Log incoming requests and their processing time."""
        request_id = str(uuid.uuid4())
        logger.info(f"Request started: {request_id} - {request.method} {request.url.path}")
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"Request completed: {request_id} - {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.4f}s"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request_id} - {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.4f}s"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id}
            )
    
    async def rate_limit_middleware(self, request: Request, call_next):
        """Implement rate limiting per API key or IP address."""
        # Extract API key from request
        api_key = request.headers.get("X-API-Key")
        
        if api_key:
            # Get partner tier for API key
            partner_info = await self.api_key_store.get_partner_info(api_key)
            
            if partner_info:
                tier = partner_info.get("tier", PartnerTier.BASIC)
                
                # Check rate limit
                if not await self.rate_limiter.check_rate_limit(api_key, tier):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"}
                    )
                
                # Check quota
                if not await self.quota_manager.check_quota(api_key, tier):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Monthly quota exceeded"}
                    )
        else:
            # Use IP address for rate limiting if no API key
            client_ip = request.client.host
            
            # Apply stricter limits to unauthenticated requests
            if not await self.rate_limiter.check_rate_limit(client_ip, PartnerTier.PUBLIC):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
        
        # Process the request
        response = await call_next(request)
        
        # Track usage
        if api_key:
            await self.quota_manager.track_usage(api_key, request.url.path)
        
        return response
    
    def _register_routes(self):
        """Register all routes for the external API."""
        # Health check endpoint (public)
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """Check if the API is operational."""
            status = await self.check_health()
            if status == HealthStatus.HEALTHY:
                return {"status": "healthy"}
            elif status == HealthStatus.DEGRADED:
                return {"status": "degraded"}
            else:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy"}
                )
        
        # API documentation
        @self.app.get("/api-info", tags=["System"])
        async def api_info():
            """Get information about the API and available endpoints."""
            return {
                "name": "AI Trading Agent External API",
                "version": "1.0.0",
                "documentation": "/docs",
                "endpoints": {
                    "health": "/health",
                    "api-info": "/api-info",
                    "market-data": "/v1/market-data",
                    "trading": "/v1/trading",
                    "analytics": "/v1/analytics",
                    "signals": "/v1/signals",
                    "auth": "/v1/auth"
                }
            }
        
        # Authentication endpoint
        @self.app.post("/v1/auth/token", tags=["Authentication"])
        async def get_token(request: Request):
            """
            Get a JWT token using API key authentication.
            
            This token can be used for subsequent requests instead of the API key.
            """
            api_key = request.headers.get("X-API-Key")
            
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key required"
                )
            
            partner_info = await self.api_key_store.get_partner_info(api_key)
            
            if not partner_info:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
            
            # Generate JWT token
            token = self.jwt_auth.create_access_token(
                data={"partner_id": partner_info.get("partner_id")}
            )
            
            return {"access_token": token, "token_type": "bearer"}
    
    async def check_health(self) -> HealthStatus:
        """
        Check the health of the External API Gateway.
        
        Returns:
            HealthStatus indicating the current health state
        """
        # Check API key store
        try:
            if not await self.api_key_store.is_healthy():
                logger.warning("API key store is not healthy")
                return HealthStatus.DEGRADED
            
            # Add more health checks as needed
            
            return HealthStatus.HEALTHY
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthStatus.UNHEALTHY
    
    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application instance.
        
        Returns:
            FastAPI application
        """
        return self.app
