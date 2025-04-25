"""
Authentication Module

This module provides JWT-based authentication with role-based access control
for securing API endpoints. It includes utilities for token generation,
validation, and permission verification.
"""

import os
import time
import logging
from typing import Dict, Optional, List, Any, Callable, Union
from datetime import datetime, timedelta
import uuid

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError

# Setup logging
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "user": "Standard user access",
        "admin": "Administrator access",
        "api": "API access",
        "trading": "Trading operations",
        "readonly": "Read-only access"
    }
)

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: int  # Unix timestamp


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []
    jti: Optional[str] = None  # JWT ID for token revocation


class AuthConfig:
    """Configuration for JWT authentication"""
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    @classmethod
    def init(cls):
        """Initialize with secure defaults if not configured"""
        if not cls.SECRET_KEY:
            # Load from file or generate
            secret_key_path = os.path.join("config", "security", "jwt_secret.key")
            if os.path.exists(secret_key_path):
                with open(secret_key_path, "r") as f:
                    cls.SECRET_KEY = f.read().strip()
            else:
                # Generate a strong key and save it
                import secrets
                cls.SECRET_KEY = secrets.token_hex(32)
                os.makedirs(os.path.dirname(secret_key_path), exist_ok=True)
                with open(secret_key_path, "w") as f:
                    f.write(cls.SECRET_KEY)
                logger.warning(f"Generated new JWT secret key at {secret_key_path}")


class AuthManager:
    """
    Manages authentication, token generation, and validation.
    """
    
    def __init__(self):
        # Initialize configuration
        AuthConfig.init()
        
        # Set of revoked tokens by JTI
        self.revoked_tokens: Dict[str, int] = {}  # jti -> expiry timestamp
        
        # Schedule periodic cleanup of expired revoked tokens
        self._schedule_cleanup()
    
    def _schedule_cleanup(self):
        """Schedule periodic cleanup of expired revoked tokens"""
        # In a real application, use a background task
        pass
    
    def _clean_revoked_tokens(self):
        """Remove expired token JTIs from the revoked list"""
        current_time = int(time.time())
        expired = [jti for jti, expiry in self.revoked_tokens.items() 
                  if expiry < current_time]
        
        for jti in expired:
            self.revoked_tokens.pop(jti, None)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hashed password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password for storage"""
        return pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None,
        scopes: List[str] = ["user"]
    ) -> str:
        """
        Create a new JWT access token
        
        Args:
            data: Dictionary of claims to include in the token
            expires_delta: Optional custom expiration time
            scopes: List of scopes/permissions for this token
            
        Returns:
            str: Encoded JWT token
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        # Add token claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "scope": " ".join(scopes),
            "jti": str(uuid.uuid4())  # Unique token ID for revocation
        })
        
        # Sign and encode token
        return jwt.encode(
            to_encode, 
            AuthConfig.SECRET_KEY, 
            algorithm=AuthConfig.ALGORITHM
        )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT refresh token with longer expiry
        
        Args:
            data: Dictionary of claims to include in the token
            
        Returns:
            str: Encoded JWT refresh token
        """
        expires_delta = timedelta(days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS)
        return self.create_access_token(
            data=data,
            expires_delta=expires_delta,
            scopes=["refresh"]
        )
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding its JTI to the revoked list
        
        Args:
            token: Token to revoke
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Decode the token to get the JTI and expiry
            payload = jwt.decode(
                token, 
                AuthConfig.SECRET_KEY, 
                algorithms=[AuthConfig.ALGORITHM]
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if not jti or not exp:
                logger.warning("Token missing JTI or expiry - cannot revoke")
                return False
            
            # Add to revoked tokens list
            self.revoked_tokens[jti] = exp
            return True
            
        except JWTError:
            logger.warning("Failed to decode token for revocation")
            return False
    
    async def get_current_user(
        self, 
        security_scopes: SecurityScopes,
        token: str = Depends(oauth2_scheme),
        request: Request = None
    ) -> Dict[str, Any]:
        """
        Validate token and return user information
        
        Args:
            security_scopes: Required scopes for the endpoint
            token: JWT token to validate
            request: Optional request for additional context
            
        Returns:
            Dict: User information from the token
            
        Raises:
            HTTPException: If token is invalid or user doesn't have required permissions
        """
        # Prepare authentication error with appropriate scopes
        authenticate_value = f"Bearer scope=\"{security_scopes.scope_str}\"" if security_scopes.scopes else "Bearer"
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )
        
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                AuthConfig.SECRET_KEY, 
                algorithms=[AuthConfig.ALGORITHM]
            )
            
            # Extract token data
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            jti: str = payload.get("jti")
            
            if username is None or user_id is None:
                logger.warning("Token missing required claims")
                raise credentials_exception
            
            # Check if token has been revoked
            if jti in self.revoked_tokens:
                logger.warning(f"Attempt to use revoked token for user {username}")
                raise credentials_exception
            
            # Parse token scopes
            token_scopes = payload.get("scope", "").split()
            token_data = TokenData(
                username=username,
                user_id=user_id,
                scopes=token_scopes,
                jti=jti
            )
            
            # Verify required scopes
            for scope in security_scopes.scopes:
                if scope not in token_data.scopes:
                    logger.warning(
                        f"User {username} attempted to access {request.url if request else 'endpoint'} "
                        f"without required scope: {scope}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Not enough permissions. Required: {scope}",
                        headers={"WWW-Authenticate": authenticate_value},
                    )
            
            # Return user data
            return {
                "username": token_data.username,
                "user_id": token_data.user_id,
                "scopes": token_data.scopes
            }
            
        except JWTError:
            logger.warning("JWT token validation error")
            raise credentials_exception
        except ValidationError:
            logger.warning("Token data validation error")
            raise credentials_exception


# Create a singleton instance
auth_manager = AuthManager()


# Dependency functions
async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    request: Request = None
) -> Dict[str, Any]:
    """Dependency to get current user from token"""
    return await auth_manager.get_current_user(security_scopes, token, request)


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Dependency to get current active user"""
    # Here you could check if user is active in DB
    return current_user


# Scope-specific dependencies
async def require_admin(
    current_user: Dict[str, Any] = Depends(SecurityScopes(["admin"]).scope(get_current_user))
) -> Dict[str, Any]:
    """Require admin access"""
    return current_user


async def require_trading_permission(
    current_user: Dict[str, Any] = Depends(SecurityScopes(["trading"]).scope(get_current_user))
) -> Dict[str, Any]:
    """Require trading permission"""
    return current_user


# Utility functions
def get_auth_manager() -> AuthManager:
    """Get the singleton AuthManager instance"""
    return auth_manager