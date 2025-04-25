"""
Secure Credential Storage Module

This module provides secure storage and access for sensitive credentials, including:
- Exchange API keys
- Database credentials
- External service access tokens
- Encryption keys

It supports multiple storage backends with appropriate encryption.
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time
from datetime import datetime, timedelta
from enum import Enum

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from pydantic import BaseModel, SecretStr, validator

# Setup logging
logger = logging.getLogger(__name__)


class CredentialType(str, Enum):
    """Types of credentials that can be stored"""
    EXCHANGE_API_KEY = "exchange_api_key"
    EXCHANGE_SECRET = "exchange_secret"
    DATABASE = "database"
    JWT_SECRET = "jwt_secret"
    EXTERNAL_API_KEY = "external_api_key"
    ENCRYPTION_KEY = "encryption_key"
    OTHER = "other"


class Credential(BaseModel):
    """Model representing a stored credential"""
    name: str
    type: CredentialType
    value: SecretStr
    metadata: Dict[str, Any] = {}
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    @validator("created_at", "updated_at", pre=True, always=True)
    def set_timestamps(cls, v):
        return v or datetime.now()

    def is_expired(self) -> bool:
        """Check if credential has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def get_secret_value(self) -> str:
        """Get the actual secret value"""
        return self.value.get_secret_value()


class CredentialManager:
    """
    Abstract base class for credential storage backends.
    """
    
    def get(self, name: str) -> Optional[Credential]:
        """Get a credential by name"""
        raise NotImplementedError
    
    def store(self, credential: Credential) -> bool:
        """Store a credential"""
        raise NotImplementedError
    
    def delete(self, name: str) -> bool:
        """Delete a credential"""
        raise NotImplementedError
    
    def list(self, credential_type: Optional[CredentialType] = None) -> Dict[str, Credential]:
        """List all credentials, optionally filtered by type"""
        raise NotImplementedError


class FileCredentialManager(CredentialManager):
    """
    Store credentials in an encrypted file.
    """
    
    def __init__(self, file_path: str, master_key: Optional[str] = None):
        """
        Initialize file-based credential storage.
        
        Args:
            file_path: Path to the credential file
            master_key: Master encryption key (will generate if not provided)
        """
        self.file_path = Path(file_path)
        self._credentials: Dict[str, Credential] = {}
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get or generate the encryption key
        self._encryption_key = self._get_encryption_key(master_key)
        
        # Initialize Fernet cipher for encryption/decryption
        self._cipher = Fernet(self._encryption_key)
        
        # Load credentials if file exists
        if self.file_path.exists():
            self._load()
    
    def _get_encryption_key(self, master_key: Optional[str] = None) -> bytes:
        """
        Get or generate the encryption key.
        
        Args:
            master_key: Optional master key to derive encryption key from
            
        Returns:
            bytes: The encryption key
        """
        if master_key:
            # Derive a key from the master key
            salt = b'aitrading_salt'  # In production, use a secure random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            return base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        
        # Try to get from environment variable
        env_key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
        if env_key:
            try:
                # Ensure it's valid base64
                decoded = base64.urlsafe_b64decode(env_key.encode())
                if len(decoded) == 32:  # 256 bits
                    return env_key.encode()
            except Exception:
                logger.warning("Invalid encryption key in environment variable")
        
        # If no environment key or it's invalid, generate a new one
        logger.warning("Generating new encryption key - all existing credentials will be inaccessible")
        return Fernet.generate_key()
    
    def _load(self) -> None:
        """Load credentials from file"""
        try:
            # Read and decrypt the file
            encrypted_data = self.file_path.read_bytes()
            decrypted_data = self._cipher.decrypt(encrypted_data)
            
            # Parse JSON
            creds_dict = json.loads(decrypted_data.decode())
            
            # Convert to Credential objects
            self._credentials = {}
            for name, data in creds_dict.items():
                # Convert value to SecretStr
                if isinstance(data["value"], str):
                    data["value"] = SecretStr(data["value"])
                
                # Convert date strings to datetime
                for date_field in ["expires_at", "created_at", "updated_at"]:
                    if date_field in data and data[date_field]:
                        data[date_field] = datetime.fromisoformat(data[date_field])
                
                self._credentials[name] = Credential(**data)
            
            logger.info(f"Loaded {len(self._credentials)} credentials from {self.file_path}")
            
        except InvalidToken:
            logger.error("Failed to decrypt credentials file - wrong encryption key")
            self._credentials = {}
        except Exception as e:
            logger.error(f"Error loading credentials file: {str(e)}")
            self._credentials = {}
    
    def _save(self) -> None:
        """Save credentials to file"""
        try:
            # Convert credentials to dict
            creds_dict = {}
            for name, cred in self._credentials.items():
                # Convert to dict and handle SecretStr
                cred_dict = cred.dict()
                cred_dict["value"] = cred.get_secret_value()
                
                # Convert datetime to ISO format
                for date_field in ["expires_at", "created_at", "updated_at"]:
                    if cred_dict[date_field]:
                        cred_dict[date_field] = cred_dict[date_field].isoformat()
                
                creds_dict[name] = cred_dict
            
            # Encrypt and save
            json_data = json.dumps(creds_dict).encode()
            encrypted_data = self._cipher.encrypt(json_data)
            self.file_path.write_bytes(encrypted_data)
            
            logger.debug(f"Saved {len(self._credentials)} credentials to {self.file_path}")
            
        except Exception as e:
            logger.error(f"Error saving credentials file: {str(e)}")
            raise
    
    def get(self, name: str) -> Optional[Credential]:
        """
        Get a credential by name.
        
        Args:
            name: Name of the credential
            
        Returns:
            Credential if found, None otherwise
        """
        credential = self._credentials.get(name)
        
        # Check for expired credentials
        if credential and credential.is_expired():
            logger.info(f"Credential '{name}' has expired, returning None")
            return None
            
        return credential
    
    def store(self, credential: Credential) -> bool:
        """
        Store a credential.
        
        Args:
            credential: The credential to store
            
        Returns:
            bool: True if stored successfully
        """
        # Update or add the credential
        self._credentials[credential.name] = credential
        
        # Save to file
        self._save()
        return True
    
    def delete(self, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            name: Name of the credential to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        if name in self._credentials:
            del self._credentials[name]
            self._save()
            return True
            
        return False
    
    def list(self, credential_type: Optional[CredentialType] = None) -> Dict[str, Credential]:
        """
        List all credentials, optionally filtered by type.
        
        Args:
            credential_type: Optional type to filter by
            
        Returns:
            Dict[str, Credential]: Dictionary of credentials
        """
        result = {}
        
        for name, cred in self._credentials.items():
            # Skip expired credentials
            if cred.is_expired():
                continue
                
            # Filter by type if specified
            if credential_type is None or cred.type == credential_type:
                result[name] = cred
                
        return result


class KeyringCredentialManager(CredentialManager):
    """
    Store credentials using the system keyring.
    """
    
    def __init__(self, namespace: str = "ai_trading_agent"):
        """
        Initialize keyring-based credential storage.
        
        Args:
            namespace: Namespace for keyring storage
        """
        self.namespace = namespace
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load metadata cache from disk if possible
        self._load_metadata_cache()
    
    def _get_metadata_path(self) -> Path:
        """Get path to metadata cache file"""
        cache_dir = Path(os.path.expanduser("~/.aitrading"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "credential_metadata.json"
    
    def _load_metadata_cache(self) -> None:
        """Load metadata cache from disk"""
        try:
            metadata_path = self._get_metadata_path()
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self._metadata_cache = json.load(f)
                logger.debug(f"Loaded metadata cache for {len(self._metadata_cache)} credentials")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {str(e)}")
            self._metadata_cache = {}
    
    def _save_metadata_cache(self) -> None:
        """Save metadata cache to disk"""
        try:
            metadata_path = self._get_metadata_path()
            with open(metadata_path, "w") as f:
                json.dump(self._metadata_cache, f)
            logger.debug(f"Saved metadata cache for {len(self._metadata_cache)} credentials")
        except Exception as e:
            logger.error(f"Error saving metadata cache: {str(e)}")
    
    def get(self, name: str) -> Optional[Credential]:
        """
        Get a credential by name.
        
        Args:
            name: Name of the credential
            
        Returns:
            Credential if found, None otherwise
        """
        # Get the secret value from keyring
        key = f"{self.namespace}_{name}"
        value = keyring.get_password(self.namespace, key)
        
        if value is None:
            return None
            
        # Get metadata from cache
        metadata = self._metadata_cache.get(name, {})
        
        # Create credential object
        try:
            # Parse dates from metadata
            expires_at = None
            if "expires_at" in metadata and metadata["expires_at"]:
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                
            created_at = None
            if "created_at" in metadata and metadata["created_at"]:
                created_at = datetime.fromisoformat(metadata["created_at"])
                
            updated_at = None
            if "updated_at" in metadata and metadata["updated_at"]:
                updated_at = datetime.fromisoformat(metadata["updated_at"])
            
            credential = Credential(
                name=name,
                type=metadata.get("type", CredentialType.OTHER),
                value=SecretStr(value),
                metadata=metadata.get("metadata", {}),
                expires_at=expires_at,
                created_at=created_at,
                updated_at=updated_at,
            )
            
            # Check for expired credentials
            if credential.is_expired():
                logger.info(f"Credential '{name}' has expired, returning None")
                return None
                
            return credential
            
        except Exception as e:
            logger.error(f"Error creating credential object: {str(e)}")
            return None
    
    def store(self, credential: Credential) -> bool:
        """
        Store a credential.
        
        Args:
            credential: The credential to store
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Store value in keyring
            key = f"{self.namespace}_{credential.name}"
            keyring.set_password(self.namespace, key, credential.get_secret_value())
            
            # Store metadata in cache
            self._metadata_cache[credential.name] = {
                "type": credential.type,
                "metadata": credential.metadata,
                "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
                "created_at": credential.created_at.isoformat() if credential.created_at else None,
                "updated_at": datetime.now().isoformat()
            }
            
            # Save metadata cache
            self._save_metadata_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing credential: {str(e)}")
            return False
    
    def delete(self, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            name: Name of the credential to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            # Delete from keyring
            key = f"{self.namespace}_{name}"
            keyring.delete_password(self.namespace, key)
            
            # Remove from metadata cache
            if name in self._metadata_cache:
                del self._metadata_cache[name]
                self._save_metadata_cache()
                
            return True
            
        except keyring.errors.PasswordDeleteError:
            return False
        except Exception as e:
            logger.error(f"Error deleting credential: {str(e)}")
            return False
    
    def list(self, credential_type: Optional[CredentialType] = None) -> Dict[str, Credential]:
        """
        List all credentials, optionally filtered by type.
        
        Args:
            credential_type: Optional type to filter by
            
        Returns:
            Dict[str, Credential]: Dictionary of credentials
        """
        result = {}
        
        # Get all credentials from metadata cache
        for name in self._metadata_cache:
            # Get full credential
            cred = self.get(name)
            
            # Skip expired or missing credentials
            if cred is None:
                continue
                
            # Filter by type if specified
            if credential_type is None or cred.type == credential_type:
                result[name] = cred
                
        return result


def create_credential_manager(storage_type: str = "file") -> CredentialManager:
    """
    Factory function to create the appropriate credential manager.
    
    Args:
        storage_type: Type of storage ("file" or "keyring")
        
    Returns:
        CredentialManager: The credential manager instance
    """
    if storage_type == "file":
        # Get file path from environment or use default
        file_path = os.environ.get("CREDENTIAL_FILE", "~/.aitrading/credentials.enc")
        file_path = os.path.expanduser(file_path)
        
        # Get master key from environment
        master_key = os.environ.get("CREDENTIAL_MASTER_KEY")
        
        return FileCredentialManager(file_path, master_key)
    
    elif storage_type == "keyring":
        # Get namespace from environment or use default
        namespace = os.environ.get("CREDENTIAL_NAMESPACE", "ai_trading_agent")
        
        return KeyringCredentialManager(namespace)
    
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


# Default credential manager instance
credential_manager = create_credential_manager(
    os.environ.get("CREDENTIAL_STORAGE_TYPE", "file")
)


def get_credential_manager() -> CredentialManager:
    """Get the default credential manager instance"""
    return credential_manager