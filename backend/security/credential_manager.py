"""
Credential Manager

This module provides secure storage and management for sensitive credentials,
including API keys, secrets, and access tokens. It implements credential rotation
policies and secure encryption for stored credentials.
"""

import os
import base64
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import uuid
import secrets
import hashlib
from enum import Enum

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Setup logging
logger = logging.getLogger(__name__)


class CredentialType(str, Enum):
    """Types of credentials that can be managed"""
    API_KEY = "api_key"
    DATABASE = "database"
    JWT = "jwt"
    EXCHANGE = "exchange"
    THIRD_PARTY = "third_party"


class CredentialManager:
    """
    Manages secure storage, retrieval, and rotation of sensitive credentials.
    """
    
    def __init__(self, 
                 encryption_key: Optional[str] = None,
                 storage_path: Optional[str] = None,
                 auto_rotate: bool = True,
                 rotation_days: int = 90):
        """
        Initialize the credential manager.
        
        Args:
            encryption_key: Master key for encrypting credentials. If None, will be loaded from env var
            storage_path: Path to store encrypted credentials. If None, uses default location
            auto_rotate: Whether to automatically rotate credentials based on expiration
            rotation_days: Default number of days before credentials should be rotated
        """
        self.auto_rotate = auto_rotate
        self.rotation_days = rotation_days
        
        # Set up encryption key
        self._init_encryption_key(encryption_key)
        
        # Set up storage path
        self._init_storage_path(storage_path)
        
        # Load existing credentials
        self.credentials = self._load_credentials()
        
        # Check for credentials needing rotation
        if self.auto_rotate:
            self._check_rotation_needed()
    
    def _init_encryption_key(self, encryption_key: Optional[str]) -> None:
        """Initialize the encryption key"""
        # Use provided key, environment variable, or generate one
        if encryption_key:
            self.master_key = encryption_key
        elif os.environ.get("CREDENTIAL_ENCRYPTION_KEY"):
            self.master_key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
        else:
            # Look for key file
            key_path = Path("./config/security/master.key")
            if key_path.exists():
                with open(key_path, "rb") as key_file:
                    self.master_key = key_file.read().decode("utf-8").strip()
            else:
                # Generate new key if none exists
                logger.warning("No encryption key provided or found. Generating a new one.")
                self.master_key = self._generate_encryption_key()
                
                # Ensure directory exists
                key_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the key for future use
                with open(key_path, "wb") as key_file:
                    key_file.write(self.master_key.encode("utf-8"))
                
                logger.info(f"New encryption key generated and saved to {key_path}")
                logger.warning("Backup this key immediately! Losing it will result in loss of all credentials.")
        
        # Initialize Fernet cipher with derived key
        salt = b"AI_Trading_Agent_Salt"  # Fixed salt for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self.cipher = Fernet(key)
    
    def _init_storage_path(self, storage_path: Optional[str]) -> None:
        """Initialize the storage path for credentials"""
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("./config/security/credentials")
        
        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_encryption_key(self) -> str:
        """Generate a new random encryption key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from storage"""
        credentials_file = self.storage_path / "credentials.enc"
        
        if not credentials_file.exists():
            logger.info("No credentials file found. Starting with empty credentials.")
            return {}
        
        try:
            with open(credentials_file, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except (InvalidToken, json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return {}
    
    def _save_credentials(self) -> bool:
        """Save credentials to storage"""
        try:
            credentials_data = json.dumps(self.credentials).encode("utf-8")
            encrypted_data = self.cipher.encrypt(credentials_data)
            
            credentials_file = self.storage_path / "credentials.enc"
            
            # Write to temporary file first and then rename for atomicity
            temp_file = self.storage_path / f"credentials.{uuid.uuid4()}.tmp"
            with open(temp_file, "wb") as f:
                f.write(encrypted_data)
            
            # Replace the original file
            if os.path.exists(credentials_file):
                os.replace(temp_file, credentials_file)
            else:
                os.rename(temp_file, credentials_file)
            
            return True
        except Exception as e:
            logger.error(f"Error saving credentials: {str(e)}")
            return False
    
    def _check_rotation_needed(self) -> None:
        """Check if any credentials need rotation"""
        now = datetime.now()
        credentials_to_rotate = []
        
        for cred_id, cred_info in self.credentials.items():
            if "expiration" in cred_info:
                try:
                    expiration = datetime.fromisoformat(cred_info["expiration"])
                    if expiration <= now:
                        credentials_to_rotate.append(cred_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid expiration format for credential {cred_id}")
        
        # Log credentials needing rotation
        if credentials_to_rotate:
            logger.warning(f"The following credentials need rotation: {', '.join(credentials_to_rotate)}")
    
    def store_credential(self, 
                        credential_id: str, 
                        credential_type: CredentialType,
                        value: Any,
                        metadata: Optional[Dict[str, Any]] = None,
                        expiration_days: Optional[int] = None) -> bool:
        """
        Store a credential securely.
        
        Args:
            credential_id: Unique identifier for the credential
            credential_type: Type of credential
            value: The credential value to store
            metadata: Additional metadata about the credential
            expiration_days: Days until credential should be rotated
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Calculate expiration date
            if expiration_days is None:
                expiration_days = self.rotation_days
            
            expiration = datetime.now() + timedelta(days=expiration_days)
            
            # Create credential entry
            credential_entry = {
                "type": credential_type,
                "value": value,
                "created": datetime.now().isoformat(),
                "expiration": expiration.isoformat(),
                "metadata": metadata or {}
            }
            
            # Store the credential
            self.credentials[credential_id] = credential_entry
            
            # Save to storage
            success = self._save_credentials()
            return success
            
        except Exception as e:
            logger.error(f"Error storing credential {credential_id}: {str(e)}")
            return False
    
    def get_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a credential by ID.
        
        Args:
            credential_id: ID of the credential to retrieve
            
        Returns:
            Optional[Dict]: Credential data or None if not found
        """
        if credential_id not in self.credentials:
            return None
        
        credential = self.credentials[credential_id]
        
        # Check if rotation is needed
        if self.auto_rotate and "expiration" in credential:
            try:
                expiration = datetime.fromisoformat(credential["expiration"])
                if expiration <= datetime.now():
                    logger.warning(f"Credential {credential_id} has expired and needs rotation")
            except (ValueError, TypeError):
                logger.warning(f"Invalid expiration format for credential {credential_id}")
        
        return credential
    
    def get_credential_value(self, credential_id: str) -> Optional[Any]:
        """
        Retrieve just the value of a credential.
        
        Args:
            credential_id: ID of the credential to retrieve
            
        Returns:
            Any: The credential value or None if not found
        """
        credential = self.get_credential(credential_id)
        return credential["value"] if credential else None
    
    def delete_credential(self, credential_id: str) -> bool:
        """
        Delete a credential by ID.
        
        Args:
            credential_id: ID of the credential to delete
            
        Returns:
            bool: True if deleted successfully
        """
        if credential_id not in self.credentials:
            logger.warning(f"Credential {credential_id} not found for deletion")
            return False
        
        try:
            # Remove from memory
            del self.credentials[credential_id]
            
            # Save changes to storage
            return self._save_credentials()
        except Exception as e:
            logger.error(f"Error deleting credential {credential_id}: {str(e)}")
            return False
    
    def rotate_credential(self, 
                         credential_id: str, 
                         new_value: Any,
                         new_metadata: Optional[Dict[str, Any]] = None,
                         expiration_days: Optional[int] = None) -> bool:
        """
        Rotate a credential by updating its value and resetting expiration.
        
        Args:
            credential_id: ID of the credential to rotate
            new_value: New credential value
            new_metadata: New or updated metadata (None to keep existing)
            expiration_days: New expiration period in days (None for default)
            
        Returns:
            bool: True if rotated successfully
        """
        if credential_id not in self.credentials:
            logger.warning(f"Credential {credential_id} not found for rotation")
            return False
        
        try:
            credential = self.credentials[credential_id]
            
            # Calculate new expiration
            if expiration_days is None:
                expiration_days = self.rotation_days
            
            new_expiration = datetime.now() + timedelta(days=expiration_days)
            
            # Update credential
            credential["value"] = new_value
            credential["rotated"] = datetime.now().isoformat()
            credential["expiration"] = new_expiration.isoformat()
            
            # Update metadata if provided
            if new_metadata:
                credential["metadata"] = {
                    **(credential.get("metadata", {})),
                    **new_metadata
                }
            
            # Save changes
            return self._save_credentials()
            
        except Exception as e:
            logger.error(f"Error rotating credential {credential_id}: {str(e)}")
            return False
    
    def list_credentials(self, credential_type: Optional[CredentialType] = None) -> List[Dict[str, Any]]:
        """
        List all credentials or filter by type.
        
        Args:
            credential_type: Optional filter by credential type
            
        Returns:
            List[Dict]: List of credential entries (without values)
        """
        result = []
        
        for cred_id, cred_info in self.credentials.items():
            # Filter by type if specified
            if credential_type and cred_info.get("type") != credential_type:
                continue
            
            # Create a sanitized copy without the actual credential value
            sanitized = {
                "id": cred_id,
                "type": cred_info.get("type"),
                "created": cred_info.get("created"),
                "expiration": cred_info.get("expiration"),
                "metadata": cred_info.get("metadata", {})
            }
            
            if "rotated" in cred_info:
                sanitized["rotated"] = cred_info["rotated"]
            
            result.append(sanitized)
        
        return result
    
    def generate_api_key(self) -> Tuple[str, str]:
        """
        Generate a new API key and secret.
        
        Returns:
            Tuple[str, str]: API key and secret
        """
        # Generate a random 32-byte API key
        api_key = secrets.token_hex(16)
        
        # Generate a random 64-byte API secret
        api_secret = secrets.token_hex(32)
        
        return api_key, api_secret
    
    def store_api_credentials(self, 
                             user_id: int,
                             description: str = "API Key",
                             expiration_days: int = 90) -> Tuple[str, str, bool]:
        """
        Generate and store API credentials for a user.
        
        Args:
            user_id: User ID associated with the API key
            description: Description of what this key is for
            expiration_days: Days until key expiration
            
        Returns:
            Tuple[str, str, bool]: API key, API secret, success status
        """
        # Generate credentials
        api_key, api_secret = self.generate_api_key()
        
        # Create a secure hash of the API secret
        secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()
        
        # Generate a unique credential ID
        credential_id = f"api_key_{user_id}_{int(time.time())}"
        
        # Store in the credential manager
        metadata = {
            "user_id": user_id,
            "description": description,
            "secret_hash": secret_hash,
            "permissions": ["read", "trade"],
            "ip_restrictions": []
        }
        
        success = self.store_credential(
            credential_id=credential_id,
            credential_type=CredentialType.API_KEY,
            value=api_key,
            metadata=metadata,
            expiration_days=expiration_days
        )
        
        return api_key, api_secret, success
    
    def verify_api_credentials(self, api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
        """
        Verify API key and secret against stored credentials.
        
        Args:
            api_key: The API key to verify
            api_secret: The API secret to verify
            
        Returns:
            Optional[Dict]: Credential info if verified, None otherwise
        """
        # Find the credential with matching API key
        for cred_id, cred_info in self.credentials.items():
            if (cred_info.get("type") == CredentialType.API_KEY and
                cred_info.get("value") == api_key):
                
                # Get the stored secret hash
                secret_hash = cred_info.get("metadata", {}).get("secret_hash")
                if not secret_hash:
                    logger.error(f"API key {api_key} found but has no secret hash")
                    return None
                
                # Hash the provided secret and compare
                provided_hash = hashlib.sha256(api_secret.encode()).hexdigest()
                if provided_hash == secret_hash:
                    return cred_info
                else:
                    logger.warning(f"API key {api_key} found but secret verification failed")
                    return None
        
        logger.warning(f"API key {api_key} not found")
        return None


# Create a singleton instance
credential_manager = CredentialManager()

# Provide a function to get the singleton instance
def get_credential_manager() -> CredentialManager:
    """Get the singleton instance of CredentialManager"""
    return credential_manager