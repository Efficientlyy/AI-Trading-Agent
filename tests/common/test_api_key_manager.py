"""Tests for the API key management system."""

import os
import tempfile
import unittest
from pathlib import Path

from src.common.security import ApiCredential, ApiKeyManager


class TestApiKeyManager(unittest.TestCase):
    """Test cases for the API key manager."""
    
    def setUp(self):
        """Set up the test case with a temporary directory for storage."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_manager = ApiKeyManager(storage_path=self.temp_dir)
        
        # Test credential
        self.test_credential = ApiCredential(
            exchange_id="test_exchange",
            key="test_api_key_123",
            secret="test_api_secret_456",
            passphrase="test_passphrase",
            description="Test credential",
            is_testnet=True,
            permissions=["read", "trade"]
        )
    
    def tearDown(self):
        """Clean up after the test."""
        # Delete the credential file
        if os.path.exists(self.temp_dir / "credentials.enc"):
            os.remove(self.temp_dir / "credentials.enc")
        
        # Delete the master key file
        if os.path.exists(self.temp_dir / "master.key"):
            os.remove(self.temp_dir / "master.key")
        
        # Delete the temporary directory
        os.rmdir(self.temp_dir)
    
    def test_add_credential(self):
        """Test adding a credential."""
        self.key_manager.add_credential(self.test_credential)
        
        # Check if the credential file was created
        self.assertTrue(os.path.exists(self.temp_dir / "credentials.enc"))
        
        # List credentials
        credentials = self.key_manager.list_credentials()
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0], "test_exchange")
    
    def test_get_credential(self):
        """Test getting a credential."""
        # Add the credential
        self.key_manager.add_credential(self.test_credential)
        
        # Get the credential
        credential = self.key_manager.get_credential("test_exchange")
        
        # Check the credential
        self.assertIsNotNone(credential)
        self.assertEqual(credential.exchange_id, "test_exchange")
        self.assertEqual(credential.key, "test_api_key_123")
        self.assertEqual(credential.secret, "test_api_secret_456")
        self.assertEqual(credential.passphrase, "test_passphrase")
        self.assertEqual(credential.description, "Test credential")
        self.assertTrue(credential.is_testnet)
        self.assertEqual(credential.permissions, ["read", "trade"])
    
    def test_get_nonexistent_credential(self):
        """Test getting a nonexistent credential."""
        # Get a nonexistent credential
        credential = self.key_manager.get_credential("nonexistent")
        
        # Check that no credential was returned
        self.assertIsNone(credential)
    
    def test_remove_credential(self):
        """Test removing a credential."""
        # Add the credential
        self.key_manager.add_credential(self.test_credential)
        
        # Remove the credential
        success = self.key_manager.remove_credential("test_exchange")
        
        # Check that the credential was removed
        self.assertTrue(success)
        self.assertEqual(len(self.key_manager.list_credentials()), 0)
    
    def test_remove_nonexistent_credential(self):
        """Test removing a nonexistent credential."""
        # Try to remove a nonexistent credential
        success = self.key_manager.remove_credential("nonexistent")
        
        # Check that no credential was removed
        self.assertFalse(success)
    
    def test_persistence(self):
        """Test that credentials are persisted to disk and can be loaded again."""
        # Add the credential
        self.key_manager.add_credential(self.test_credential)
        
        # Create a new key manager with the same storage path
        new_key_manager = ApiKeyManager(storage_path=self.temp_dir)
        
        # Check that the credential was loaded
        credentials = new_key_manager.list_credentials()
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0], "test_exchange")
        
        # Get the credential
        credential = new_key_manager.get_credential("test_exchange")
        
        # Check the credential
        self.assertIsNotNone(credential)
        self.assertEqual(credential.exchange_id, "test_exchange")
        self.assertEqual(credential.key, "test_api_key_123")


if __name__ == "__main__":
    unittest.main() 