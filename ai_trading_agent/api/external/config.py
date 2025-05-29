"""
Configuration for External API Gateway.

This module defines configuration settings for the External API Gateway,
including partner tiers, rate limits, and API settings.
"""
import os
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from pydantic import BaseModel, Field, validator
import yaml


class PartnerTier(str, Enum):
    """Partner tiers for external API access."""
    PUBLIC = "public"          # Unauthenticated or free tier
    BASIC = "basic"            # Basic tier for simple integrations
    PREMIUM = "premium"        # Premium tier for advanced integrations
    ENTERPRISE = "enterprise"  # Enterprise tier for high-volume partners


class PartnerTierConfig(BaseModel):
    """Configuration for a partner tier."""
    tier: PartnerTier
    rate_limit: float = Field(..., description="Requests per second")
    burst_multiplier: float = Field(2.0, description="Multiplier for burst capacity")
    monthly_quota: Optional[int] = Field(None, description="Monthly request quota (None for unlimited)")
    allowed_endpoints: List[str] = Field(["*"], description="List of allowed endpoints ('*' for all)")
    max_request_size: int = Field(1024 * 1024, description="Maximum request size in bytes")
    max_response_size: int = Field(5 * 1024 * 1024, description="Maximum response size in bytes")
    include_historical_data: bool = Field(False, description="Whether historical data access is included")
    include_real_time_data: bool = Field(False, description="Whether real-time data access is included")
    include_signals: bool = Field(False, description="Whether trading signals are included")
    include_analytics: bool = Field(False, description="Whether advanced analytics are included")
    include_backtesting: bool = Field(False, description="Whether backtesting capabilities are included")
    max_concurrent_requests: int = Field(10, description="Maximum concurrent requests")
    support_level: str = Field("standard", description="Support level (standard, priority, dedicated)")
    
    @validator("allowed_endpoints")
    def validate_endpoints(cls, v):
        """Validate allowed endpoints."""
        if "*" in v:
            return ["*"]  # If wildcard is present, it's all that matters
        return v


class APIConfig(BaseModel):
    """Configuration for the External API Gateway."""
    # General settings
    debug_mode: bool = Field(False, description="Enable debug mode")
    api_path_prefix: str = Field("/api", description="Path prefix for all API endpoints")
    
    # Documentation settings
    docs_url: str = Field("/docs", description="URL for Swagger UI")
    redoc_url: str = Field("/redoc", description="URL for ReDoc UI")
    openapi_url: str = Field("/openapi.json", description="URL for OpenAPI schema")
    
    # CORS settings
    cors_allow_origins: List[str] = Field(["*"], description="Allowed origins for CORS")
    cors_allow_credentials: bool = Field(True, description="Allow credentials for CORS")
    cors_allow_methods: List[str] = Field(["*"], description="Allowed methods for CORS")
    cors_allow_headers: List[str] = Field(["*"], description="Allowed headers for CORS")
    
    # Authentication settings
    api_key_db_path: str = Field("./data/api_keys.db", description="Path to API key database")
    oauth_token_url: str = Field("/token", description="URL for OAuth token endpoint")
    oauth_client_ids: Set[str] = Field(default_factory=set, description="Valid OAuth client IDs")
    jwt_secret_key: str = Field("", description="Secret key for JWT tokens")
    jwt_algorithm: str = Field("HS256", description="Algorithm for JWT tokens")
    jwt_token_expire_minutes: int = Field(60, description="JWT token expiration time in minutes")
    
    # Tier settings
    tier_configs: Dict[PartnerTier, PartnerTierConfig] = Field(
        default_factory=dict,
        description="Configuration for each partner tier"
    )
    
    # Rate limiting and quota settings
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    enable_quota_management: bool = Field(True, description="Enable quota management")
    
    # Logging settings
    log_all_requests: bool = Field(True, description="Log all API requests")
    log_response_time: bool = Field(True, description="Log response time for requests")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "APIConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            APIConfig instance
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Process tier configs
        if "tier_configs" in config_data:
            tier_configs = {}
            for tier_name, tier_data in config_data["tier_configs"].items():
                try:
                    tier = PartnerTier(tier_name)
                    tier_data["tier"] = tier
                    tier_configs[tier] = PartnerTierConfig(**tier_data)
                except ValueError:
                    # Skip invalid tier names
                    continue
            
            config_data["tier_configs"] = tier_configs
        
        return cls(**config_data)
    
    def to_yaml(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            file_path: Path to save configuration
        """
        # Convert to dictionary
        config_dict = self.dict()
        
        # Convert tier configs to dictionary
        if "tier_configs" in config_dict:
            tier_configs = {}
            for tier, config in config_dict["tier_configs"].items():
                if isinstance(tier, PartnerTier):
                    tier_name = tier.value
                else:
                    tier_name = str(tier)
                
                # Convert config to dict and remove tier field
                config_dict = config
                if isinstance(config, PartnerTierConfig):
                    config_dict = config.dict()
                
                if "tier" in config_dict:
                    del config_dict["tier"]
                
                tier_configs[tier_name] = config_dict
            
            config_dict["tier_configs"] = tier_configs
        
        # Convert set to list for serialization
        if "oauth_client_ids" in config_dict:
            config_dict["oauth_client_ids"] = list(config_dict["oauth_client_ids"])
        
        # Save to file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def create_default_config() -> APIConfig:
    """
    Create a default API configuration.
    
    Returns:
        Default APIConfig instance
    """
    # Create basic tier configs
    tier_configs = {
        PartnerTier.PUBLIC: PartnerTierConfig(
            tier=PartnerTier.PUBLIC,
            rate_limit=5.0,
            monthly_quota=1000,
            allowed_endpoints=["/v1/market-data/public/*"],
            include_historical_data=False,
            include_real_time_data=False,
            include_signals=False,
            include_analytics=False,
            include_backtesting=False,
            max_concurrent_requests=5,
            support_level="community"
        ),
        PartnerTier.BASIC: PartnerTierConfig(
            tier=PartnerTier.BASIC,
            rate_limit=10.0,
            monthly_quota=100000,
            allowed_endpoints=["*"],
            include_historical_data=True,
            include_real_time_data=False,
            include_signals=False,
            include_analytics=False,
            include_backtesting=False,
            max_concurrent_requests=10,
            support_level="standard"
        ),
        PartnerTier.PREMIUM: PartnerTierConfig(
            tier=PartnerTier.PREMIUM,
            rate_limit=50.0,
            monthly_quota=1000000,
            allowed_endpoints=["*"],
            include_historical_data=True,
            include_real_time_data=True,
            include_signals=True,
            include_analytics=True,
            include_backtesting=False,
            max_concurrent_requests=20,
            support_level="priority"
        ),
        PartnerTier.ENTERPRISE: PartnerTierConfig(
            tier=PartnerTier.ENTERPRISE,
            rate_limit=200.0,
            monthly_quota=None,  # Unlimited
            allowed_endpoints=["*"],
            include_historical_data=True,
            include_real_time_data=True,
            include_signals=True,
            include_analytics=True,
            include_backtesting=True,
            max_concurrent_requests=50,
            support_level="dedicated"
        )
    }
    
    # Create default configuration
    return APIConfig(
        debug_mode=False,
        api_path_prefix="/api",
        cors_allow_origins=["*"],
        api_key_db_path="./data/api_keys.db",
        jwt_secret_key=os.urandom(32).hex(),  # Generate random secret key
        tier_configs=tier_configs
    )
