#!/usr/bin/env python3
"""
Utility script for setting up exchange API credentials.

This script provides a secure way to add and validate exchange API keys
for use with the AI Crypto Trading System.
"""

import os
import sys
import asyncio
import argparse
import getpass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.common.config import config
from src.common.logging import setup_logging, get_logger
from src.common.security.api_keys import APIKeyManager


# Set up logging
setup_logging()
logger = get_logger("security", "credentials")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Exchange Credentials Setup Utility")
    
    # Main subparsers for different operations
    subparsers = parser.add_subparsers(title="commands", dest="command")
    
    # Add exchange keys
    add_parser = subparsers.add_parser("add", help="Add exchange API keys")
    add_parser.add_argument(
        "--exchange", 
        type=str, 
        required=True,
        choices=["binance", "coinbase", "kraken", "ftx", "kucoin"],
        help="Exchange to add credentials for"
    )
    add_parser.add_argument(
        "--key", "-k",
        type=str,
        help="API key (omit to enter interactively)"
    )
    add_parser.add_argument(
        "--secret", "-s",
        type=str,
        help="API secret (omit to enter interactively)"
    )
    add_parser.add_argument(
        "--passphrase", "-p",
        type=str,
        help="API passphrase (omit to enter interactively, if required)"
    )
    add_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the credentials by connecting to the exchange"
    )
    
    # List available exchanges and keys
    list_parser = subparsers.add_parser("list", help="List configured exchanges")
    list_parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Show partial keys (last 4 characters)"
    )
    
    # Delete exchange keys
    delete_parser = subparsers.add_parser("delete", help="Delete exchange API keys")
    delete_parser.add_argument(
        "--exchange", 
        type=str, 
        required=True,
        help="Exchange to delete credentials for"
    )
    delete_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion without prompting"
    )
    
    # Validate exchange keys
    validate_parser = subparsers.add_parser("validate", help="Validate exchange API keys")
    validate_parser.add_argument(
        "--exchange", 
        type=str,
        help="Specific exchange to validate (omit to validate all)"
    )
    
    return parser.parse_args()


async def add_exchange_keys(args, key_manager):
    """Add exchange API keys."""
    exchange = args.exchange.lower()
    
    # Get API key and secret
    api_key = args.key or getpass.getpass(f"Enter {exchange.title()} API Key: ")
    api_secret = args.secret or getpass.getpass(f"Enter {exchange.title()} API Secret: ")
    
    # Get passphrase if needed
    api_passphrase = None
    if exchange in ["coinbase", "kraken", "kucoin"]:
        api_passphrase = args.passphrase or getpass.getpass(
            f"Enter {exchange.title()} API Passphrase (leave empty if not required): "
        )
        if api_passphrase == "":
            api_passphrase = None
    
    # Create credentials dictionary
    credentials = {
        "api_key": api_key,
        "api_secret": api_secret
    }
    
    if api_passphrase:
        credentials["passphrase"] = api_passphrase
    
    # Save credentials
    success = await key_manager.set_keys(exchange, credentials)
    
    if success:
        logger.info(f"{exchange.title()} API credentials saved successfully")
        
        # Validate if requested
        if args.validate:
            valid = await key_manager.validate_keys(exchange)
            if valid:
                logger.info(f"{exchange.title()} API credentials validated successfully")
            else:
                logger.error(f"{exchange.title()} API credentials validation failed")
    else:
        logger.error(f"Failed to save {exchange.title()} API credentials")


async def list_exchanges(args, key_manager):
    """List configured exchanges."""
    logger.info("Listing configured exchanges")
    
    # Get exchanges with configured keys
    exchanges = await key_manager.list_exchanges()
    
    if not exchanges:
        print("No exchanges configured")
        return
    
    print("\nConfigured Exchanges:")
    print("====================")
    
    for exchange in exchanges:
        if args.show_keys:
            keys = await key_manager.get_keys(exchange)
            if keys and "api_key" in keys:
                api_key = keys["api_key"]
                # Show only last 4 characters
                masked_key = f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"
                print(f"{exchange.title()}: {masked_key}")
            else:
                print(f"{exchange.title()}: [No API key]")
        else:
            print(f"{exchange.title()}")
    
    print()


async def delete_exchange_keys(args, key_manager):
    """Delete exchange API keys."""
    exchange = args.exchange.lower()
    
    # Check if credentials exist
    if not await key_manager.has_keys(exchange):
        logger.error(f"No credentials found for {exchange.title()}")
        return
    
    # Confirm deletion
    if not args.confirm:
        confirm = input(f"Are you sure you want to delete credentials for {exchange.title()}? (y/N): ")
        if confirm.lower() != "y":
            logger.info("Deletion cancelled")
            return
    
    # Delete credentials
    success = await key_manager.delete_keys(exchange)
    
    if success:
        logger.info(f"{exchange.title()} API credentials deleted successfully")
    else:
        logger.error(f"Failed to delete {exchange.title()} API credentials")


async def validate_exchange_keys(args, key_manager):
    """Validate exchange API keys."""
    if args.exchange:
        # Validate specific exchange
        exchange = args.exchange.lower()
        if not await key_manager.has_keys(exchange):
            logger.error(f"No credentials found for {exchange.title()}")
            return
        
        valid = await key_manager.validate_keys(exchange)
        if valid:
            logger.info(f"{exchange.title()} API credentials validated successfully")
        else:
            logger.error(f"{exchange.title()} API credentials validation failed")
    else:
        # Validate all exchanges
        exchanges = await key_manager.list_exchanges()
        
        if not exchanges:
            logger.info("No exchanges configured to validate")
            return
        
        results = {}
        for exchange in exchanges:
            try:
                valid = await key_manager.validate_keys(exchange)
                results[exchange] = valid
                status = "VALID" if valid else "INVALID"
                logger.info(f"{exchange.title()} API credentials: {status}")
            except Exception as e:
                logger.error(f"Error validating {exchange.title()} API credentials: {str(e)}")
                results[exchange] = False
        
        # Print summary
        print("\nValidation Results:")
        print("===================")
        
        for exchange, valid in results.items():
            status = "VALID" if valid else "INVALID"
            print(f"{exchange.title()}: {status}")
        
        print()


async def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return 1
    
    try:
        # Initialize API key manager
        key_manager = APIKeyManager()
        await key_manager.initialize()
        
        # Execute requested command
        if args.command == "add":
            await add_exchange_keys(args, key_manager)
        elif args.command == "list":
            await list_exchanges(args, key_manager)
        elif args.command == "delete":
            await delete_exchange_keys(args, key_manager)
        elif args.command == "validate":
            await validate_exchange_keys(args, key_manager)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)