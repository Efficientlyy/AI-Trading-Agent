"""
CORS Configuration

This module provides Cross-Origin Resource Sharing (CORS) configuration
to securely control which domains can access the API.
"""

import logging
from typing import List, Optional, Dict, Any
import os
from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class CORSConfig(BaseModel):
    """Configuration model for CORS settings"""
    allow_origins: List[str]
    allow_origin_regex: Optional[str] = None
    allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    allow_headers: List[str] = ["*"]
    allow_credentials: bool = True
    expose_headers: List[str] = []
    max_age: int = 600


class Environment(str, Enum):
    """Application deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Get the current environment"""
    env = os.environ.get("APP_ENV", "development").lower()
    
    if env in ["prod", "production"]:
        return Environment.PRODUCTION
    elif env in ["staging", "test"]:
        return Environment.STAGING
    else:
        return Environment.DEVELOPMENT


def get_cors_config() -> CORSConfig:
    """
    Get CORS configuration based on the environment.
    
    Returns:
        CORSConfig: CORS configuration for the current environment
    """
    environment = get_environment()
    
    # Default headers to expose
    expose_headers = [
        "Content-Type", 
        "X-RateLimit-Limit", 
        "X-RateLimit-Remaining", 
        "X-RateLimit-Reset"
    ]
    
    if environment == Environment.PRODUCTION:
        # Strict CORS for production
        return CORSConfig(
            allow_origins=[
                "https://ai-trading-agent.com",
                "https://www.ai-trading-agent.com",
                "https://app.ai-trading-agent.com"
            ],
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_credentials=True,
            expose_headers=expose_headers,
            max_age=3600  # 1 hour
        )
    
    elif environment == Environment.STAGING:
        # Less restrictive for staging
        return CORSConfig(
            allow_origins=[
                "https://staging.ai-trading-agent.com",
                "https://test.ai-trading-agent.com",
                "http://localhost:3000",
                "http://localhost:8000",
            ],
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            allow_credentials=True,
            expose_headers=expose_headers,
            max_age=1800  # 30 minutes
        )
    
    else:  # DEVELOPMENT
        # Most permissive for local development
        return CORSConfig(
            allow_origins=["*"],  # Allow all origins in development
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
            allow_credentials=True,
            expose_headers=expose_headers,
            max_age=600  # 10 minutes
        )


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    config = get_cors_config()
    logger.info(f"Configuring CORS with origins: {config.allow_origins}")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_origin_regex=config.allow_origin_regex,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        allow_credentials=config.allow_credentials,
        expose_headers=config.expose_headers,
        max_age=config.max_age,
    )