"""
Authentication utilities for External API Gateway.

This module implements various authentication methods for external partners,
including API key authentication, OAuth2, and JWT token management.
"""
import os
import json
import time
import logging
import secrets
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
import hashlib
import hmac
import uuid

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from passlib.context import CryptContext

# Import partner tier configuration
from .config import PartnerTier

# Setup logging
logger = logging.getLogger(__name__)


class PartnerInfo(BaseModel):
    """Model for partner information."""
    partner_id: str
    name: str
    email: str
    tier: PartnerTier
    created_at: datetime
    quota_limit: int
    active: bool


class APIKey(BaseModel):
    """Model for API key information."""
    key_id: str
    api_key: str
    partner_id: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked: bool = False


class APIKeyStore:
    """
    Store and manage API keys for external partners.
    
    Provides functionality to create, validate, and revoke API keys.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the API key store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure the database and required tables exist."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create partners table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partners (
                partner_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                quota_limit INTEGER NOT NULL,
                active INTEGER NOT NULL
            )
        """)
        
        # Create api_keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                partner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT,
                expires_at TEXT,
                revoked INTEGER NOT NULL,
                FOREIGN KEY (partner_id) REFERENCES partners (partner_id)
            )
        """)
        
        # Create usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                request_count INTEGER NOT NULL,
                FOREIGN KEY (api_key) REFERENCES api_keys (api_key)
            )
        """)
        
        # Commit and close
        conn.commit()
        conn.close()
        
        logger.info(f"API key database initialized at {self.db_path}")
    
    async def create_partner(
        self,
        name: str,
        email: str,
        tier: PartnerTier = PartnerTier.BASIC,
        quota_limit: int = 10000
    ) -> PartnerInfo:
        """
        Create a new partner.
        
        Args:
            name: Partner name
            email: Contact email
            tier: Partner tier
            quota_limit: Monthly request quota
            
        Returns:
            PartnerInfo object with created partner information
        """
        partner_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO partners (partner_id, name, email, tier, created_at, quota_limit, active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (partner_id, name, email, tier.value, created_at.isoformat(), quota_limit, 1)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created new partner: {name} ({partner_id}) with tier {tier.value}")
        
        return PartnerInfo(
            partner_id=partner_id,
            name=name,
            email=email,
            tier=tier,
            created_at=created_at,
            quota_limit=quota_limit,
            active=True
        )
    
    async def create_api_key(
        self,
        partner_id: str,
        expires_in_days: Optional[int] = 365
    ) -> APIKey:
        """
        Create a new API key for a partner.
        
        Args:
            partner_id: Partner ID
            expires_in_days: Days until the key expires, or None for no expiration
            
        Returns:
            APIKey object with the new API key information
        """
        # Check if partner exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT partner_id FROM partners WHERE partner_id = ?",
            (partner_id,)
        )
        
        if not cursor.fetchone():
            conn.close()
            raise ValueError(f"Partner {partner_id} does not exist")
        
        # Generate API key
        key_id = str(uuid.uuid4())
        api_key = secrets.token_urlsafe(32)
        created_at = datetime.utcnow()
        
        # Calculate expiration date
        expires_at = None
        if expires_in_days is not None:
            expires_at = created_at + timedelta(days=expires_in_days)
        
        # Insert into database
        cursor.execute(
            """
            INSERT INTO api_keys (key_id, api_key, partner_id, created_at, expires_at, revoked)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key_id,
                api_key,
                partner_id,
                created_at.isoformat(),
                expires_at.isoformat() if expires_at else None,
                0
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created new API key for partner {partner_id}")
        
        return APIKey(
            key_id=key_id,
            api_key=api_key,
            partner_id=partner_id,
            created_at=created_at,
            expires_at=expires_at,
            revoked=False
        )
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if the key was revoked, False if it doesn't exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE api_keys SET revoked = 1 WHERE api_key = ?",
            (api_key,)
        )
        
        revoked = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if revoked:
            logger.info(f"Revoked API key")
        else:
            logger.warning(f"Attempted to revoke non-existent API key")
        
        return revoked
    
    async def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if the key is valid, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get key information
        cursor.execute(
            """
            SELECT expires_at, revoked
            FROM api_keys
            WHERE api_key = ?
            """,
            (api_key,)
        )
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False
        
        expires_at, revoked = result
        
        # Check if revoked
        if revoked:
            conn.close()
            return False
        
        # Check if expired
        if expires_at:
            expires_at_dt = datetime.fromisoformat(expires_at)
            if expires_at_dt < datetime.utcnow():
                conn.close()
                return False
        
        # Update last used timestamp
        cursor.execute(
            "UPDATE api_keys SET last_used = ? WHERE api_key = ?",
            (datetime.utcnow().isoformat(), api_key)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    async def get_partner_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Get partner information for an API key.
        
        Args:
            api_key: API key
            
        Returns:
            Partner information or None if the key is invalid
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get partner information
        cursor.execute(
            """
            SELECT p.partner_id, p.name, p.email, p.tier, p.created_at, p.quota_limit, p.active
            FROM partners p
            JOIN api_keys k ON p.partner_id = k.partner_id
            WHERE k.api_key = ? AND k.revoked = 0
            """,
            (api_key,)
        )
        
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            return None
        
        partner_id, name, email, tier, created_at, quota_limit, active = result
        
        return {
            "partner_id": partner_id,
            "name": name,
            "email": email,
            "tier": tier,
            "created_at": created_at,
            "quota_limit": quota_limit,
            "active": bool(active)
        }
    
    async def is_healthy(self) -> bool:
        """
        Check if the API key store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple query to check connectivity
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            conn.close()
            
            return result is not None
        except Exception as e:
            logger.error(f"API key store health check failed: {str(e)}")
            return False


class APIKeyAuth:
    """API key authentication handler for FastAPI."""
    
    def __init__(self, api_key_store: APIKeyStore):
        """
        Initialize API key authentication.
        
        Args:
            api_key_store: API key store for validation
        """
        self.api_key_store = api_key_store
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    async def __call__(self, api_key: str = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))):
        """
        Validate API key authentication.
        
        Args:
            api_key: API key from request header
            
        Returns:
            Partner information if authenticated
            
        Raises:
            HTTPException if authentication fails
        """
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        valid = await self.api_key_store.validate_api_key(api_key)
        
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        partner_info = await self.api_key_store.get_partner_info(api_key)
        
        if not partner_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        return partner_info


class JWTAuth:
    """JWT token authentication handler for FastAPI."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_expire_minutes: int = 60
    ):
        """
        Initialize JWT authentication.
        
        Args:
            secret_key: Secret key for JWT encryption
            algorithm: JWT algorithm
            token_expire_minutes: Token expiration time in minutes
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_minutes = token_expire_minutes
        self.bearer_scheme = HTTPBearer()
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Create a new JWT access token.
        
        Args:
            data: Token payload data
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        # Set expiration time
        if self.token_expire_minutes:
            expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
            to_encode.update({"exp": expire})
        
        # Add issued at time
        to_encode.update({"iat": datetime.utcnow()})
        
        # Encode token
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=True))
    ) -> Dict[str, Any]:
        """
        Validate JWT token authentication.
        
        Args:
            credentials: Bearer token credentials
            
        Returns:
            Token payload if authenticated
            
        Raises:
            HTTPException if authentication fails
        """
        try:
            # Decode token
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check if token has expired
            if "exp" in payload and payload["exp"] < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )


class OAuth2Auth:
    """OAuth2 authentication handler for FastAPI."""
    
    def __init__(
        self,
        token_url: str,
        client_ids: Set[str]
    ):
        """
        Initialize OAuth2 authentication.
        
        Args:
            token_url: URL for token endpoint
            client_ids: Set of valid client IDs
        """
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl=token_url)
        self.client_ids = client_ids
    
    async def __call__(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
        """
        Validate OAuth2 authentication.
        
        Args:
            token: OAuth2 token
            
        Returns:
            Token payload if authenticated
            
        Raises:
            HTTPException if authentication fails
        """
        # This is a simplified implementation
        # In a real-world scenario, this would validate with an OAuth provider
        
        # For now, we'll just check if the token is in our list of valid client IDs
        if token not in self.client_ids:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid OAuth token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return placeholder payload
        return {"client_id": token}
