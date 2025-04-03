"""Security utilities for the AI Crypto Trading System.

This module provides security-related functionality, including:
- API key management (secure storage and retrieval)
- Encryption/decryption utilities
- Password hashing and validation
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

# Local imports
from src.common.config import config
from src.common.logging import get_logger

# Try to import cryptography, but handle the case where it's not installed
try:
    # For encryption
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Constants
DEFAULT_KEY_FILE = "credentials/api_keys.json"
DEFAULT_MASTER_KEY_FILE = "credentials/master.key"
SALT_FILE = "credentials/salt"


@dataclass
class ApiCredential:
    """Represents an API credential pair for an exchange."""
    exchange_id: str
    key: str
    secret: str
    passphrase: Optional[str] = None
    description: Optional[str] = None
    is_testnet: bool = False
    permissions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "exchange_id": self.exchange_id,
            "key": self.key,
            "secret": self.secret,
            "passphrase": self.passphrase,
            "description": self.description,
            "is_testnet": self.is_testnet,
            "permissions": self.permissions or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiCredential':
        """Create from dictionary representation."""
        return cls(
            exchange_id=data["exchange_id"],
            key=data["key"],
            secret=data["secret"],
            passphrase=data.get("passphrase"),
            description=data.get("description"),
            is_testnet=data.get("is_testnet", False),
            permissions=data.get("permissions", [])
        )


class ApiKeyManager:
    """Manager for API keys and secrets.
    
    This class provides secure storage and retrieval of API keys and secrets.
    Keys are stored encrypted in a JSON file.
    """
    
    def __init__(self, 
                 key_file: Optional[str] = None, 
                 master_key_file: Optional[str] = None,
                 salt_file: Optional[str] = None,
                 password: Optional[str] = None):
        """Initialize the API key manager.
        
        Args:
            key_file: Path to the key storage file (relative to project root)
            master_key_file: Path to the master key file
            salt_file: Path to the salt file
            password: Master password for generating the encryption key
        """
        self.logger = get_logger("common", "security")
        
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning(
                "Cryptography package not available. API key encryption disabled. "
                "Install with 'pip install cryptography'."
            )
        
        # Get paths from config or use defaults
        self.key_file = key_file or config.get("security.api_keys.file", DEFAULT_KEY_FILE)
        self.master_key_file = master_key_file or config.get(
            "security.api_keys.master_key_file", DEFAULT_MASTER_KEY_FILE
        )
        self.salt_file = salt_file or config.get("security.api_keys.salt_file", SALT_FILE)
        
        # Get project root
        self.project_root = self._get_project_root()
        
        # Ensure paths are absolute
        self.key_file = os.path.join(self.project_root, self.key_file)
        self.master_key_file = os.path.join(self.project_root, self.master_key_file)
        self.salt_file = os.path.join(self.project_root, self.salt_file)
        
        # Initialize the encryption system
        self.fernet = None
        if CRYPTOGRAPHY_AVAILABLE:
            self.fernet = self._initialize_encryption(password)
        
        # Load existing credentials or initialize empty dict
        self.credentials: Dict[str, ApiCredential] = {}
        self._load_credentials()
        
        self.logger.info("API Key Manager initialized")
    
    def _get_project_root(self) -> str:
        """Get the absolute path to the project root directory."""
        # Get the directory of the current file and go up 2 levels
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    def _initialize_encryption(self, password: Optional[str] = None) -> Optional[Any]:
        """Initialize the encryption system.
        
        Args:
            password: Master password for generating the encryption key
            
        Returns:
            Fernet encryption object or None if encryption not available
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
            
        # Ensure credentials directory exists
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        # Check if master key exists
        if os.path.exists(self.master_key_file):
            # Load existing master key
            with open(self.master_key_file, "rb") as f:
                key = f.read()
        else:
            # Generate and save new master key if no password provided
            if password is None:
                self.logger.info("Generating new master key")
                key = Fernet.generate_key()
                with open(self.master_key_file, "wb") as f:
                    f.write(key)
            else:
                # Generate key from password
                self.logger.info("Generating master key from password")
                
                # Create or load salt
                if os.path.exists(self.salt_file):
                    with open(self.salt_file, "rb") as f:
                        salt = f.read()
                else:
                    salt = os.urandom(16)
                    with open(self.salt_file, "wb") as f:
                        f.write(salt)
                
                # Create key derivation function
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                
                # Derive key from password
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                
                # Save the key
                with open(self.master_key_file, "wb") as f:
                    f.write(key)
        
        return Fernet(key)
    
    def _load_credentials(self) -> None:
        """Load credentials from storage file."""
        if not os.path.exists(self.key_file):
            self.logger.info("Key storage file not found, initializing empty")
            return
        
        try:
            # Read the file
            with open(self.key_file, "rb") as f:
                data = f.read()
            
            # Decrypt if encryption is available
            if self.fernet:
                decrypted_data = self.fernet.decrypt(data).decode("utf-8")
            else:
                # Use plaintext if encryption is not available
                decrypted_data = data.decode("utf-8")
            
            # Parse JSON
            creds_data = json.loads(decrypted_data)
            
            # Convert to ApiCredential objects
            for exchange_id, cred_dict in creds_data.items():
                self.credentials[exchange_id] = ApiCredential.from_dict(cred_dict)
            
            self.logger.info(f"Loaded {len(self.credentials)} API credentials")
        
        except Exception as e:
            self.logger.error(f"Error loading credentials: {str(e)}")
            # Initialize empty credentials
            self.credentials = {}
    
    def _save_credentials(self) -> None:
        """Save credentials to storage file."""
        try:
            # Convert credentials to dictionary
            creds_dict = {
                exchange_id: cred.to_dict() 
                for exchange_id, cred in self.credentials.items()
            }
            
            # Convert to JSON
            json_data = json.dumps(creds_dict, indent=2)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            # Encrypt if encryption is available
            if self.fernet:
                data = self.fernet.encrypt(json_data.encode("utf-8"))
            else:
                # Use plaintext if encryption is not available
                self.logger.warning("Saving credentials without encryption")
                data = json_data.encode("utf-8")
            
            # Save to file
            with open(self.key_file, "wb") as f:
                f.write(data)
            
            self.logger.info(f"Saved {len(self.credentials)} API credentials")
        
        except Exception as e:
            self.logger.error(f"Error saving credentials: {str(e)}")
            raise
    
    def add_credential(self, credential: ApiCredential) -> None:
        """Add or update a credential.
        
        Args:
            credential: The API credential to add or update
        """
        self.credentials[credential.exchange_id] = credential
        self._save_credentials()
        self.logger.info(f"Added/updated API credential for {credential.exchange_id}")
    
    def get_credential(self, exchange_id: str) -> Optional[ApiCredential]:
        """Get a credential by exchange ID.
        
        Args:
            exchange_id: The exchange identifier
            
        Returns:
            The API credential or None if not found
        """
        return self.credentials.get(exchange_id)
    
    def remove_credential(self, exchange_id: str) -> bool:
        """Remove a credential.
        
        Args:
            exchange_id: The exchange identifier
            
        Returns:
            True if removed, False if not found
        """
        if exchange_id in self.credentials:
            del self.credentials[exchange_id]
            self._save_credentials()
            self.logger.info(f"Removed API credential for {exchange_id}")
            return True
        
        return False
    
    def list_credentials(self) -> List[str]:
        """List all stored credential exchange IDs.
        
        Returns:
            List of exchange IDs
        """
        return list(self.credentials.keys())
    
    def clear_all_credentials(self) -> None:
        """Clear all stored credentials."""
        self.credentials = {}
        self._save_credentials()
        self.logger.info("Cleared all API credentials")


# Singleton instance
_api_key_manager = None


def get_api_key_manager(
    key_file: Optional[str] = None,
    master_key_file: Optional[str] = None,
    password: Optional[str] = None
) -> ApiKeyManager:
    """Get the API key manager singleton instance.
    
    Args:
        key_file: Path to the key storage file (optional)
        master_key_file: Path to the master key file (optional)
        password: Master password for key generation (optional)
        
    Returns:
        ApiKeyManager instance
    """
    global _api_key_manager
    
    if _api_key_manager is None:
        _api_key_manager = ApiKeyManager(
            key_file=key_file,
            master_key_file=master_key_file,
            password=password
        )
    
    return _api_key_manager 