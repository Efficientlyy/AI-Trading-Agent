"""
API Key Manager

This module provides a class for managing API keys for various services.
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ApiKeyCredential:
    """API Key Credential class"""
    
    def __init__(self, provider: str, key: str, secret: str = None, passphrase: str = None):
        """
        Initialize API key credential.
        
        Args:
            provider: Provider name
            key: API key
            secret: API secret
            passphrase: API passphrase (optional)
        """
        self.provider = provider
        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        self.created_at = datetime.now().isoformat()
        self.last_used = None
        self.last_validated = None
        self.is_valid = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert credential to dictionary.
        
        Returns:
            Dictionary representation of credential
        """
        return {
            'provider': self.provider,
            'key': self.key,
            'secret': self.secret,
            'passphrase': self.passphrase,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'last_validated': self.last_validated,
            'is_valid': self.is_valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiKeyCredential':
        """
        Create credential from dictionary.
        
        Args:
            data: Dictionary representation of credential
            
        Returns:
            ApiKeyCredential instance
        """
        credential = cls(
            provider=data.get('provider'),
            key=data.get('key'),
            secret=data.get('secret'),
            passphrase=data.get('passphrase')
        )
        credential.created_at = data.get('created_at')
        credential.last_used = data.get('last_used')
        credential.last_validated = data.get('last_validated')
        credential.is_valid = data.get('is_valid')
        return credential

class ApiKeyManager:
    """
    API Key Manager class for managing API keys for various services.
    """
    
    def __init__(self, credentials_file: str = None):
        """
        Initialize API key manager.
        
        Args:
            credentials_file: Path to credentials file
        """
        self.credentials_file = credentials_file or os.path.join('config', 'credentials.json')
        self.credentials = {}
        self.load_credentials()
    
    def load_credentials(self) -> bool:
        """
        Load credentials from file.
        
        Returns:
            bool: True if credentials were loaded successfully, False otherwise
        """
        if not os.path.exists(self.credentials_file):
            logger.info(f"Credentials file not found: {self.credentials_file}")
            return False
        
        try:
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)
            
            for provider, cred_data in data.items():
                self.credentials[provider] = ApiKeyCredential.from_dict(cred_data)
            
            logger.info(f"Loaded {len(self.credentials)} API credentials")
            return True
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return False
    
    def save_credentials(self) -> bool:
        """
        Save credentials to file.
        
        Returns:
            bool: True if credentials were saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            
            # Convert credentials to dictionary
            data = {provider: cred.to_dict() for provider, cred in self.credentials.items()}
            
            # Save to file
            with open(self.credentials_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Saved {len(self.credentials)} API credentials")
            return True
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False
    
    def add_credential(self, provider: str, key: str, secret: str = None, passphrase: str = None) -> bool:
        """
        Add a new credential.
        
        Args:
            provider: Provider name
            key: API key
            secret: API secret
            passphrase: API passphrase (optional)
            
        Returns:
            bool: True if credential was added successfully, False otherwise
        """
        try:
            self.credentials[provider] = ApiKeyCredential(
                provider=provider,
                key=key,
                secret=secret,
                passphrase=passphrase
            )
            
            return self.save_credentials()
        except Exception as e:
            logger.error(f"Error adding credential: {e}")
            return False
    
    def get_credential(self, provider: str) -> Optional[ApiKeyCredential]:
        """
        Get credential for provider.
        
        Args:
            provider: Provider name
            
        Returns:
            ApiKeyCredential: Credential for provider, or None if not found
        """
        return self.credentials.get(provider)
    
    def delete_credential(self, provider: str) -> bool:
        """
        Delete credential for provider.
        
        Args:
            provider: Provider name
            
        Returns:
            bool: True if credential was deleted successfully, False otherwise
        """
        if provider in self.credentials:
            del self.credentials[provider]
            return self.save_credentials()
        return False
    
    def validate_credential(self, provider: str) -> Tuple[bool, str]:
        """
        Validate credential for provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        credential = self.get_credential(provider)
        if not credential:
            return False, f"No credential found for {provider}"
        
        # Update last validated timestamp
        credential.last_validated = datetime.now().isoformat()
        
        # Validate based on provider
        if provider == 'binance':
            is_valid, message = self._validate_binance_credentials(credential.key, credential.secret)
        elif provider == 'coinbase':
            is_valid, message = self._validate_coinbase_credentials(credential.key, credential.secret)
        elif provider == 'kraken':
            is_valid, message = self._validate_kraken_credentials(credential.key, credential.secret)
        elif provider == 'bitvavo':
            is_valid, message = self._validate_bitvavo_credentials(credential.key, credential.secret)
        else:
            is_valid, message = False, f"Validation not implemented for {provider}"
        
        # Update credential validity
        credential.is_valid = is_valid
        self.save_credentials()
        
        return is_valid, message
    
    def _validate_binance_credentials(self, key: str, secret: str) -> Tuple[bool, str]:
        """
        Validate Binance API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if key and secret are provided
        if not key or not secret:
            return False, "API key and secret are required"
        
        # Check key format (Binance API keys are 64 characters)
        if not re.match(r'^[A-Za-z0-9]{64}$', key):
            return False, "Invalid API key format"
        
        # Check secret format (Binance API secrets are 64 characters)
        if not re.match(r'^[A-Za-z0-9]{64}$', secret):
            return False, "Invalid API secret format"
        
        return True, "Binance API credentials are valid"
    
    def _validate_coinbase_credentials(self, key: str, secret: str) -> Tuple[bool, str]:
        """
        Validate Coinbase API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if key and secret are provided
        if not key or not secret:
            return False, "API key and secret are required"
        
        # Check key format (Coinbase API keys are alphanumeric)
        if not re.match(r'^[A-Za-z0-9]+$', key):
            return False, "Invalid API key format"
        
        # Check secret format (Coinbase API secrets are base64 encoded)
        try:
            base64.b64decode(secret)
        except Exception:
            return False, "Invalid API secret format"
        
        return True, "Coinbase API credentials are valid"
    
    def _validate_kraken_credentials(self, key: str, secret: str) -> Tuple[bool, str]:
        """
        Validate Kraken API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if key and secret are provided
        if not key or not secret:
            return False, "API key and secret are required"
        
        # Check key format (Kraken API keys start with 'K')
        if not key.startswith('K'):
            return False, "Invalid API key format"
        
        # Check secret format (Kraken API secrets are base64 encoded)
        try:
            base64.b64decode(secret)
        except Exception:
            return False, "Invalid API secret format"
        
        return True, "Kraken API credentials are valid"
    
    def _validate_bitvavo_credentials(self, key: str, secret: str) -> Tuple[bool, str]:
        """
        Validate Bitvavo API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        # Check if key and secret are provided
        if not key or not secret:
            return False, "API key and secret are required"
        
        # Check key format (Bitvavo API keys are alphanumeric with hyphens)
        if not re.match(r'^[A-Za-z0-9\-]+$', key):
            return False, "API key should contain only alphanumeric characters and hyphens"
        
        # Check secret format (Bitvavo API secrets are alphanumeric with hyphens)
        if not re.match(r'^[A-Za-z0-9\-]+$', secret):
            return False, "API secret should contain only alphanumeric characters and hyphens"
        
        return True, "Bitvavo API credentials are valid"
    
    def add_binance_credentials(self, key: str, secret: str) -> bool:
        """
        Add Binance API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            bool: True if credentials were added successfully, False otherwise
        """
        # Validate credentials
        is_valid, message = self._validate_binance_credentials(key, secret)
        if not is_valid:
            logger.error(f"Invalid Binance credentials: {message}")
            return False
        
        # Add credentials
        return self.add_credential('binance', key, secret)
    
    def add_coinbase_credentials(self, key: str, secret: str, passphrase: str) -> bool:
        """
        Add Coinbase API credentials.
        
        Args:
            key: API key
            secret: API secret
            passphrase: API passphrase
            
        Returns:
            bool: True if credentials were added successfully, False otherwise
        """
        # Validate credentials
        is_valid, message = self._validate_coinbase_credentials(key, secret)
        if not is_valid:
            logger.error(f"Invalid Coinbase credentials: {message}")
            return False
        
        # Add credentials
        return self.add_credential('coinbase', key, secret, passphrase)
    
    def add_kraken_credentials(self, key: str, secret: str) -> bool:
        """
        Add Kraken API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            bool: True if credentials were added successfully, False otherwise
        """
        # Validate credentials
        is_valid, message = self._validate_kraken_credentials(key, secret)
        if not is_valid:
            logger.error(f"Invalid Kraken credentials: {message}")
            return False
        
        # Add credentials
        return self.add_credential('kraken', key, secret)
    
    def add_bitvavo_credentials(self, key: str, secret: str) -> bool:
        """
        Add Bitvavo API credentials.
        
        Args:
            key: API key
            secret: API secret
            
        Returns:
            bool: True if credentials were added successfully, False otherwise
        """
        # Validate credentials
        is_valid, message = self._validate_bitvavo_credentials(key, secret)
        if not is_valid:
            logger.error(f"Invalid Bitvavo credentials: {message}")
            return False
        
        # Add credentials
        return self.add_credential('bitvavo', key, secret)
    
    def test_binance_connection(self) -> Tuple[bool, str]:
        """
        Test connection to Binance API.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Get credentials
        credential = self.get_credential('binance')
        if not credential:
            return False, "No Binance credentials found"
        
        # Test connection
        try:
            # Import Binance connector
            from src.execution.exchange.binance import BinanceExchangeConnector
            
            # Create connector
            connector = BinanceExchangeConnector(api_key=credential.key, api_secret=credential.secret)
            
            # Test connection
            success = connector.initialize()
            
            if success:
                return True, "Successfully connected to Binance API"
            else:
                return False, "Failed to connect to Binance API"
        except Exception as e:
            logger.error(f"Error testing Binance connection: {e}")
            return False, f"Error testing Binance connection: {str(e)}"
    
    def test_coinbase_connection(self) -> Tuple[bool, str]:
        """
        Test connection to Coinbase API.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Get credentials
        credential = self.get_credential('coinbase')
        if not credential:
            return False, "No Coinbase credentials found"
        
        # Test connection
        try:
            # Import Coinbase connector
            from src.execution.exchange.coinbase import CoinbaseExchangeConnector
            
            # Create connector
            connector = CoinbaseExchangeConnector(
                api_key=credential.key,
                api_secret=credential.secret,
                passphrase=credential.passphrase
            )
            
            # Test connection
            success = connector.initialize()
            
            if success:
                return True, "Successfully connected to Coinbase API"
            else:
                return False, "Failed to connect to Coinbase API"
        except Exception as e:
            logger.error(f"Error testing Coinbase connection: {e}")
            return False, f"Error testing Coinbase connection: {str(e)}"
    
    def test_kraken_connection(self) -> Tuple[bool, str]:
        """
        Test connection to Kraken API.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Get credentials
        credential = self.get_credential('kraken')
        if not credential:
            return False, "No Kraken credentials found"
        
        # Test connection
        try:
            # Import Kraken connector
            from src.execution.exchange.kraken import KrakenExchangeConnector
            
            # Create connector
            connector = KrakenExchangeConnector(api_key=credential.key, api_secret=credential.secret)
            
            # Test connection
            success = connector.initialize()
            
            if success:
                return True, "Successfully connected to Kraken API"
            else:
                return False, "Failed to connect to Kraken API"
        except Exception as e:
            logger.error(f"Error testing Kraken connection: {e}")
            return False, f"Error testing Kraken connection: {str(e)}"
    
    def test_bitvavo_connection(self) -> Tuple[bool, str]:
        """
        Test connection to Bitvavo API.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Get credentials
        credential = self.get_credential('bitvavo')
        if not credential:
            return False, "No Bitvavo credentials found"
        
        # Test connection
        try:
            # Import Bitvavo connector
            from src.execution.exchange.bitvavo import BitvavoConnector
            
            # Create connector
            connector = BitvavoConnector(api_key=credential.key, api_secret=credential.secret)
            
            # Test connection
            success = connector.initialize()
            
            if success:
                return True, "Successfully connected to Bitvavo API"
            else:
                return False, "Failed to connect to Bitvavo API"
        except Exception as e:
            logger.error(f"Error testing Bitvavo connection: {e}")
            return False, f"Error testing Bitvavo connection: {str(e)}"