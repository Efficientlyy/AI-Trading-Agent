"""
API Key Management System.

This module provides secure storage and retrieval of exchange API credentials.
It implements encryption for sensitive information and a simple interface for
managing API keys for various exchanges.
"""

import os
import json
import base64
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ApiCredential:
    """API credential for an exchange."""
    
    exchange_id: str
    key: str
    secret: str
    passphrase: Optional[str] = None
    description: Optional[str] = None
    is_testnet: bool = False
    permissions: List[str] = field(default_factory=list)


class ApiKeyManager:
    """Manager for securely storing and retrieving API keys."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the API key manager.
        
        Args:
            storage_path: Optional path to the storage directory.
                          If not provided, defaults to ~/.trading_bot/keys
        """
        self._storage_dir = storage_path or Path.home() / ".trading_bot" / "keys"
        logger.debug(f"Using storage directory: {self._storage_dir}")
        
        # Create the storage directory if it doesn't exist
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._master_key_file = self._storage_dir / "master.key"
        self._credential_file = self._storage_dir / "credentials.enc"
        
        logger.debug(f"Master key file: {self._master_key_file}")
        logger.debug(f"Credential file: {self._credential_file}")
        
        # Generate or load master key
        self._master_key = self._get_or_create_master_key()
        
        # Initialize encryption
        self._fernet = Fernet(self._master_key)
        
        # Load credentials if they exist
        self._credentials: Dict[str, ApiCredential] = {}
        self._load_credentials()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get the master key from file or create a new one if it doesn't exist.
        
        Returns:
            The master key as bytes.
        """
        if self._master_key_file.exists():
            # Load existing key
            logger.debug("Loading existing master key")
            with open(self._master_key_file, "rb") as f:
                return f.read()
        
        # Create a new key
        logger.debug("Creating new master key")
        key = Fernet.generate_key()
        
        # Save the key
        with open(self._master_key_file, "wb") as f:
            f.write(key)
        
        logger.debug("Master key created and saved")
        return key
    
    def _load_credentials(self) -> None:
        """Load credentials from the encrypted file."""
        if not self._credential_file.exists():
            logger.debug("No credential file found, starting with empty credentials")
            return
        
        try:
            logger.debug("Loading credentials from file")
            with open(self._credential_file, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt the data
            decrypted_data = self._fernet.decrypt(encrypted_data).decode('utf-8')
            credential_dicts = json.loads(decrypted_data)
            
            # Convert to ApiCredential objects
            self._credentials = {
                cred_dict["exchange_id"]: ApiCredential(**cred_dict)
                for cred_dict in credential_dicts
            }
            
            logger.info(f"Loaded {len(self._credentials)} API credentials")
            for exchange_id in self._credentials:
                logger.debug(f"Loaded credential for {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            # In case of error, start with empty credentials
            self._credentials = {}
    
    def _save_credentials(self) -> None:
        """Save credentials to the encrypted file."""
        try:
            logger.debug("Saving credentials to file")
            # Convert to dictionaries
            credential_dicts = [asdict(cred) for cred in self._credentials.values()]
            
            # Convert to JSON
            json_data = json.dumps(credential_dicts)
            
            # Encrypt the data
            encrypted_data = self._fernet.encrypt(json_data.encode('utf-8'))
            
            # Save to file
            with open(self._credential_file, "wb") as f:
                f.write(encrypted_data)
            
            logger.info(f"Saved {len(self._credentials)} API credentials")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}", exc_info=True)
    
    def add_credential(self, credential: ApiCredential) -> None:
        """Add or update an API credential.
        
        Args:
            credential: The API credential to add or update.
        """
        logger.debug(f"Adding credential for {credential.exchange_id}")
        self._credentials[credential.exchange_id] = credential
        self._save_credentials()
        logger.info(f"Added API credential for {credential.exchange_id}")
    
    def get_credential(self, exchange_id: str) -> Optional[ApiCredential]:
        """Get an API credential by exchange ID.
        
        Args:
            exchange_id: The exchange ID.
            
        Returns:
            The API credential, or None if not found.
        """
        logger.debug(f"Getting credential for {exchange_id}")
        credential = self._credentials.get(exchange_id)
        if credential:
            logger.debug(f"Found credential for {exchange_id}")
        else:
            logger.debug(f"No credential found for {exchange_id}")
        return credential
    
    def remove_credential(self, exchange_id: str) -> bool:
        """Remove an API credential.
        
        Args:
            exchange_id: The exchange ID.
            
        Returns:
            True if the credential was removed, False if not found.
        """
        logger.debug(f"Removing credential for {exchange_id}")
        if exchange_id in self._credentials:
            del self._credentials[exchange_id]
            self._save_credentials()
            logger.info(f"Removed API credential for {exchange_id}")
            return True
        logger.debug(f"No credential found to remove for {exchange_id}")
        return False
    
    def list_credentials(self) -> List[str]:
        """Get the list of exchange IDs with stored credentials.
        
        Returns:
            A list of exchange IDs.
        """
        logger.debug(f"Listing credentials, found {len(self._credentials)} entries")
        return list(self._credentials.keys())


# Singleton instance
_API_KEY_MANAGER: Optional[ApiKeyManager] = None


def get_api_key_manager() -> ApiKeyManager:
    """Get the singleton API key manager instance.
    
    Returns:
        The API key manager instance.
    """
    global _API_KEY_MANAGER
    if _API_KEY_MANAGER is None:
        logger.debug("Creating new API key manager instance")
        _API_KEY_MANAGER = ApiKeyManager()
    return _API_KEY_MANAGER 