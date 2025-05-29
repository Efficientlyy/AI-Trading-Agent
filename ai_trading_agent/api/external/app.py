"""
External API Gateway for AI Trading Agent.

This module implements the main FastAPI application for the External API Gateway,
integrating authentication, rate limiting, health monitoring, and API endpoints.
"""
import logging
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from fastapi.openapi.utils import get_openapi

# Import configuration and components
from .config import APIConfig, create_default_config
from .auth import APIKeyAuth, JWTAuth, setup_auth_routes
from .rate_limiting import RateLimiter, QuotaManager
from .health import HealthMonitor, HealthMiddleware

# Import API endpoints
from .endpoints import routers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExternalAPIGateway:
    """
    External API Gateway for AI Trading Agent.
    
    This class implements the FastAPI application for the External API Gateway,
    integrating all components and endpoints.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API Gateway.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = APIConfig.from_yaml(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = create_default_config()
            logger.info("Using default configuration")
            
            # Save default config if path provided
            if config_path:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                self.config.to_yaml(config_path)
                logger.info(f"Saved default configuration to {config_path}")
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AI Trading Agent API",
            description="External API Gateway for AI Trading Agent",
            version="0.1.0",
            docs_url=self.config.docs_url,
            redoc_url=self.config.redoc_url,
            openapi_url=self.config.openapi_url,
            debug=self.config.debug_mode
        )
        
        # Setup components
        self._setup_components()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("External API Gateway initialized")
    
    def _setup_components(self):
        """Initialize and setup gateway components."""
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Update rate limits from config
        for tier, tier_config in self.config.tier_configs.items():
            self.rate_limiter.update_tier_limit(tier, tier_config.rate_limit)
        
        # Initialize quota manager
        self.quota_manager = QuotaManager(
            db_path=self.config.api_key_db_path
        )
        
        # Update quotas from config
        for tier, tier_config in self.config.tier_configs.items():
            self.quota_manager.update_tier_quota(tier, tier_config.monthly_quota)
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(config={
            "monitoring_interval": 60,  # seconds
            "max_history_length": 100,
            "alert_thresholds": {
                "response_time_ms": 1000,
                "error_rate_percent": 5,
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90
            }
        })
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_allow_origins,
            allow_credentials=self.config.cors_allow_credentials,
            allow_methods=self.config.cors_allow_methods,
            allow_headers=self.config.cors_allow_headers,
        )
        
        # Add health monitoring middleware
        @self.app.middleware("http")
        async def health_middleware(request: Request, call_next):
            """Track request performance in health monitor."""
            start_time = request.state.start_time = None
            try:
                # Record start time
                import time
                start_time = request.state.start_time = time.time()
                
                # Process request
                response = await call_next(request)
                
                # Track request
                if start_time:
                    response_time_ms = (time.time() - start_time) * 1000
                    self.health_monitor.track_request(
                        endpoint=request.url.path,
                        response_time_ms=response_time_ms,
                        status_code=response.status_code,
                        method=request.method
                    )
                
                return response
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                
                # Track request with error
                if start_time:
                    response_time_ms = (time.time() - start_time) * 1000
                    self.health_monitor.track_request(
                        endpoint=request.url.path,
                        response_time_ms=response_time_ms,
                        status_code=500,
                        method=request.method
                    )
                
                raise
        
        # Add rate limiting middleware
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            """Apply rate limiting to requests."""
            # Skip rate limiting for internal paths
            path = request.url.path
            if path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi"):
                return await call_next(request)
            
            # Get API key from header
            api_key = request.headers.get("X-API-Key")
            if api_key is None:
                # Use IP address for rate limiting unauthenticated requests
                identifier = request.client.host
                tier = "public"
            else:
                identifier = api_key
                # This is a placeholder - in a real implementation, we'd look up
                # the partner tier from the API key database
                tier = "basic"
            
            # Check rate limit
            if self.config.enable_rate_limiting:
                allowed = await self.rate_limiter.check_rate_limit(identifier, tier)
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Please try again later."}
                    )
            
            # Check quota
            if self.config.enable_quota_management:
                allowed = await self.quota_manager.check_quota(identifier, tier)
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Monthly quota exceeded."}
                    )
            
            # Process request
            response = await call_next(request)
            
            # Track quota usage
            if self.config.enable_quota_management and api_key:
                await self.quota_manager.track_usage(
                    identifier=api_key,
                    endpoint=path
                )
            
            return response
    
    def _setup_routes(self):
        """Setup API routes and endpoints."""
        # Create main router with prefix
        main_router = APIRouter(prefix=self.config.api_path_prefix)
        
        # Add health check endpoints
        health_router = APIRouter(prefix="/health", tags=["Health"])
        
        @health_router.get("", summary="Get API health status")
        async def health_check():
            """Get the current health status of the API."""
            return self.health_monitor.get_health_status()
        
        @health_router.get("/metrics", summary="Get system metrics")
        async def system_metrics():
            """Get detailed system metrics."""
            if not self.health_monitor.system_metrics_history:
                await self.health_monitor.collect_system_metrics()
            return {
                "current": self.health_monitor.system_metrics_history[-1] if self.health_monitor.system_metrics_history else None,
                "history": self.health_monitor.system_metrics_history[-10:] if len(self.health_monitor.system_metrics_history) > 10 else self.health_monitor.system_metrics_history
            }
        
        @health_router.get("/endpoints", summary="Get endpoint performance stats")
        async def endpoint_stats():
            """Get detailed endpoint performance statistics."""
            return self.health_monitor.get_endpoint_stats()
        
        # Add health routes
        main_router.include_router(health_router)
        
        # Add authentication routes
        auth_router = setup_auth_routes()
        main_router.include_router(auth_router)
        
        # Add all domain-specific endpoint routers
        for router in routers:
            main_router.include_router(router)
        
        # Add main router to app
        self.app.include_router(main_router)
    
    def start(self):
        """Start the health monitoring."""
        import asyncio
        asyncio.create_task(self.health_monitor.start_monitoring())
        logger.info("External API Gateway started")
    
    def stop(self):
        """Stop the health monitoring."""
        import asyncio
        asyncio.create_task(self.health_monitor.stop_monitoring())
        logger.info("External API Gateway stopped")
    
    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application.
        
        Returns:
            FastAPI application
        """
        return self.app
    
    def custom_openapi(self):
        """
        Generate custom OpenAPI schema.
        
        Returns:
            OpenAPI schema
        """
        if self.app.openapi_schema:
            return self.app.openapi_schema
        
        openapi_schema = get_openapi(
            title="AI Trading Agent API",
            version="0.1.0",
            description="External API Gateway for AI Trading Agent",
            routes=self.app.routes,
        )
        
        # Customize OpenAPI schema
        # Add security schemes
        openapi_schema["components"] = {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        }
        
        # Apply security to all operations
        for path in openapi_schema["paths"].values():
            for operation in path.values():
                operation["security"] = [
                    {"ApiKeyAuth": []},
                    {"BearerAuth": []}
                ]
        
        self.app.openapi_schema = openapi_schema
        return self.app.openapi_schema


# Function to create the FastAPI app
def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application
    """
    gateway = ExternalAPIGateway(config_path)
    app = gateway.get_app()
    
    # Set custom OpenAPI schema
    app.openapi = gateway.custom_openapi
    
    # Set startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Startup event handler."""
        gateway.start()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler."""
        gateway.stop()
    
    return app
