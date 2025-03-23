#!/usr/bin/env python
"""
Twitter Credentials Setup Script

This script helps set up Twitter API credentials for the sentiment analysis system.
It creates a .env file with the provided credentials and tests the connection.
"""

import os
import sys
import dotenv
import getpass

def setup_credentials():
    """Set up Twitter API credentials."""
    print("\n=== Twitter API Credentials Setup ===\n")
    print("This script will help you set up Twitter API credentials for sentiment analysis.")
    print("You'll need to enter your Twitter Developer credentials.")
    print("These will be saved to a .env file in the project root.")
    print("\nIMPORTANT: Never share your API credentials or commit the .env file to version control!\n")
    
    # Check if .env file already exists
    if os.path.exists(".env"):
        overwrite = input("A .env file already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Exiting without changes.")
            return
    
    # Get credentials
    api_key = getpass.getpass("Enter Twitter API Key: ")
    api_secret = getpass.getpass("Enter Twitter API Secret: ")
    access_token = getpass.getpass("Enter Twitter Access Token (or press Enter to skip): ")
    access_secret = getpass.getpass("Enter Twitter Access Token Secret (or press Enter to skip): ")
    
    # Check if essential credentials are provided
    if not api_key or not api_secret:
        print("\nERROR: API Key and Secret are required.")
        return
    
    # Create or update .env file
    with open(".env", "w") as f:
        f.write("# Twitter API Credentials\n")
        f.write(f"TWITTER_API_KEY={api_key}\n")
        f.write(f"TWITTER_API_SECRET={api_secret}\n")
        
        if access_token and access_secret:
            f.write(f"TWITTER_ACCESS_TOKEN={access_token}\n")
            f.write(f"TWITTER_ACCESS_SECRET={access_secret}\n")
        
        # Preserve any existing variables
        if os.path.exists(".env.example"):
            try:
                with open(".env.example", "r") as example:
                    for line in example:
                        if not line.strip() or line.strip().startswith("#"):
                            continue
                        if not line.startswith("TWITTER_"):
                            f.write(line)
            except Exception as e:
                print(f"Warning: Couldn't copy other variables from .env.example: {e}")
    
    print("\nCredentials saved to .env file.")
    
    # Test the credentials
    print("\nWould you like to test the Twitter API connection?")
    test = input("This will require the tweepy package installed (y/n): ").lower()
    
    if test == 'y':
        try:
            import tweepy
            
            # Load the credentials
            dotenv.load_dotenv()
            
            auth = tweepy.OAuth1UserHandler(
                os.getenv("TWITTER_API_KEY"),
                os.getenv("TWITTER_API_SECRET"),
                os.getenv("TWITTER_ACCESS_TOKEN"),
                os.getenv("TWITTER_ACCESS_SECRET")
            )
            
            api = tweepy.API(auth)
            
            # Test connection
            user = api.verify_credentials()
            print(f"\nConnection successful! Authenticated as: @{user.screen_name}")
            
            # Test search
            print("\nTesting search API...")
            tweets = api.search_tweets(q="#Bitcoin", count=5)
            print(f"Found {len(tweets)} tweets about #Bitcoin")
            
        except ImportError:
            print("\nCouldn't import tweepy. Install it with: pip install tweepy")
        except Exception as e:
            print(f"\nError testing API connection: {e}")
    
    print("\nSetup complete!")
    print("You can now use the Twitter sentiment analysis features.")

if __name__ == "__main__":
    setup_credentials()