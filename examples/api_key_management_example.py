#!/usr/bin/env python3
"""
API Key Management Example.

This script demonstrates how to use the API key management system to:
1. Add and store encrypted API keys securely
2. Retrieve stored API keys
3. Remove API keys
4. List all available exchange credentials
"""

import sys
import os
import getpass
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_key_example")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Change the working directory to the project root to make config loading work
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the API key management system
from src.common.security import get_api_key_manager, ApiCredential


def add_exchange_key(
    exchange_id: str, 
    key: str, 
    secret: str, 
    passphrase: Optional[str] = None,
    description: Optional[str] = None,
    is_testnet: bool = False
) -> None:
    """Add an exchange API key to the secure storage.
    
    Args:
        exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
        key: API key
        secret: API secret
        passphrase: Additional passphrase (if required by the exchange)
        description: Optional description (e.g., 'Main account', 'Testing only')
        is_testnet: Whether this is for a testnet/sandbox account
    """
    logger.info(f"Adding API key for {exchange_id}")
    
    # Get the API key manager
    key_manager = get_api_key_manager()
    
    # Create a new credential
    credential = ApiCredential(
        exchange_id=exchange_id,
        key=key,
        secret=secret,
        passphrase=passphrase,
        description=description,
        is_testnet=is_testnet,
        permissions=[]  # Permissions not implemented yet
    )
    
    # Add the credential to secure storage
    key_manager.add_credential(credential)
    
    print(f"Added API key for {exchange_id}")
    if description:
        print(f"Description: {description}")
    if is_testnet:
        print("This is a testnet/sandbox account")


def list_exchanges() -> None:
    """List all exchanges with stored API keys."""
    logger.info("Listing stored API keys")
    
    # Get the API key manager
    key_manager = get_api_key_manager()
    
    # Get the list of exchange IDs
    exchange_ids = key_manager.list_credentials()
    
    logger.debug(f"Found {len(exchange_ids)} credentials")
    
    if not exchange_ids:
        print("No API keys stored.")
        return
    
    print("Stored API keys:")
    for exchange_id in exchange_ids:
        credential = key_manager.get_credential(exchange_id)
        if credential:
            print(f"- {exchange_id}" + 
                  (f" ({credential.description})" if credential.description else "") +
                  (" [TESTNET]" if credential.is_testnet else ""))


def get_exchange_key(exchange_id: str) -> None:
    """Get and display API key details for an exchange.
    
    Args:
        exchange_id: Exchange identifier
    """
    logger.info(f"Getting API key details for {exchange_id}")
    
    # Get the API key manager
    key_manager = get_api_key_manager()
    
    # Get the credential
    credential = key_manager.get_credential(exchange_id)
    
    if not credential:
        print(f"No API key found for {exchange_id}.")
        return
    
    print(f"API Key details for {exchange_id}:")
    print(f"Key: {credential.key}")
    # Mask the secret for security, showing only first few characters
    masked_secret = credential.secret[:4] + "*" * (len(credential.secret) - 4) if len(credential.secret) > 4 else "****"
    print(f"Secret: {masked_secret}")
    
    if credential.passphrase:
        print(f"Passphrase: {'*' * len(credential.passphrase)}")
    
    if credential.description:
        print(f"Description: {credential.description}")
    
    print(f"Testnet: {'Yes' if credential.is_testnet else 'No'}")


def remove_exchange_key(exchange_id: str) -> None:
    """Remove an API key from secure storage.
    
    Args:
        exchange_id: Exchange identifier
    """
    logger.info(f"Removing API key for {exchange_id}")
    
    # Get the API key manager
    key_manager = get_api_key_manager()
    
    # Remove the credential
    success = key_manager.remove_credential(exchange_id)
    
    if success:
        print(f"Removed API key for {exchange_id}.")
    else:
        print(f"No API key found for {exchange_id}.")


def interactive_add_key() -> None:
    """Interactively add a new API key."""
    logger.info("Starting interactive API key addition")
    
    print("Adding a new API key")
    print("--------------------")
    
    # Get exchange ID
    exchange_id = input("Exchange ID (e.g., binance, coinbase): ").strip().lower()
    
    # Get API key and secret
    key = input("API Key: ").strip()
    secret = getpass.getpass("API Secret: ").strip()
    
    # Ask for optional passphrase
    passphrase_required = input("Does this exchange require a passphrase? (y/n): ").strip().lower()
    passphrase = None
    if passphrase_required in ("y", "yes"):
        passphrase = getpass.getpass("API Passphrase: ").strip()
    
    # Get description
    description = input("Description (optional): ").strip() or None
    
    # Ask if testnet
    is_testnet = input("Is this for a testnet/sandbox account? (y/n): ").strip().lower() in ("y", "yes")
    
    # Add the key
    add_exchange_key(
        exchange_id=exchange_id,
        key=key,
        secret=secret,
        passphrase=passphrase,
        description=description,
        is_testnet=is_testnet
    )


def main() -> None:
    """Run the API key management example."""
    logger.info("Starting API key management example")
    
    parser = argparse.ArgumentParser(description="API Key Management Example")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new API key")
    add_parser.add_argument("--exchange", "-e", required=True, help="Exchange ID")
    add_parser.add_argument("--key", "-k", required=True, help="API Key")
    add_parser.add_argument("--secret", "-s", required=True, help="API Secret")
    add_parser.add_argument("--passphrase", "-p", help="API Passphrase (if required)")
    add_parser.add_argument("--description", "-d", help="Description")
    add_parser.add_argument("--testnet", "-t", action="store_true", help="Is testnet/sandbox")
    
    # Interactive add command
    subparsers.add_parser("interactive", help="Add a new API key interactively")
    
    # List command
    subparsers.add_parser("list", help="List all stored API keys")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get API key details")
    get_parser.add_argument("exchange", help="Exchange ID")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an API key")
    remove_parser.add_argument("exchange", help="Exchange ID")
    
    # Parse arguments
    args = parser.parse_args()
    
    logger.debug(f"Command: {args.command}")
    
    # Handle commands
    if args.command == "add":
        add_exchange_key(
            exchange_id=args.exchange,
            key=args.key,
            secret=args.secret,
            passphrase=args.passphrase,
            description=args.description,
            is_testnet=args.testnet
        )
    elif args.command == "interactive":
        interactive_add_key()
    elif args.command == "list":
        list_exchanges()
    elif args.command == "get":
        get_exchange_key(args.exchange)
    elif args.command == "remove":
        remove_exchange_key(args.exchange)
    else:
        parser.print_help()
    
    logger.info("API key management example completed")


if __name__ == "__main__":
    main() 