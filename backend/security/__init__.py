"""
Security package for the AI Trading Agent API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .rate_limiter import get_rate_limit_middleware, check_api_key
from .cors import configure_cors
from .rate_limiter import start_rate_limiter_cleanup, stop_rate_limiter_cleanup

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self' wss://*; "
            "frame-ancestors 'none'; "
            "form-action 'self';"
        )
        
        # HTTP Strict Transport Security (HSTS)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Click-jacking protection
        response.headers["X-Frame-Options"] = "DENY"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        
        return response


def configure_security(app: FastAPI) -> None:
    """
    Configure all security settings for the application
    
    Args:
        app: The FastAPI application to configure
    """
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify domains instead of "*"
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add API key middleware
    app.middleware("http")(check_api_key)
    
    # Add rate limiting middleware
    app.add_middleware(get_rate_limit_middleware())
    
    # Add other security middlewares as needed
    # e.g., app.add_middleware(SecurityHeadersMiddleware)
    # e.g., app.add_middleware(CSRFMiddleware)
    
    # Register events for rate limiter cleanup
    @app.on_event("startup")
    async def startup_event():
        await start_rate_limiter_cleanup()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await stop_rate_limiter_cleanup()
    
    return None