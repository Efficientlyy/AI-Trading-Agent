#!/usr/bin/env python
"""
Test Twitter API Credentials

This script tests if your Twitter API credentials are working correctly
by making a sample API call to search for tweets.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tweepy
except ImportError:
    logger.error("Tweepy package not found. Install it with: pip install tweepy")
    sys.exit(1)


def test_tweepy_credentials():
    """Test Twitter credentials using tweepy directly."""
    logger.info("Testing Twitter API credentials with tweepy...")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get credentials from environment
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_secret = os.getenv("TWITTER_ACCESS_SECRET")
    
    # Check if credentials are available
    if not api_key or not api_secret:
        logger.error("API Key and Secret not found in environment variables.")
        logger.error("Make sure you have a .env file with TWITTER_API_KEY and TWITTER_API_SECRET.")
        return False
    
    try:
        # Set up authorization
        auth = tweepy.OAuth1UserHandler(
            api_key, api_secret, access_token, access_secret
        )
        
        # Create API object
        api = tweepy.API(auth)
        
        # Verify credentials
        user = api.verify_credentials()
        logger.info(f"Successfully authenticated as: @{user.screen_name}")
        
        # Test search
        logger.info("Testing search API...")
        tweets = api.search_tweets(q="#Bitcoin", count=5)
        logger.info(f"Found {len(tweets)} tweets about #Bitcoin")
        
        # Show a sample tweet
        if tweets:
            sample_tweet = tweets[0]
            logger.info(f"Sample tweet from @{sample_tweet.user.screen_name}: {sample_tweet.text[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Twitter API: {e}")
        return False


async def test_sentiment_client():
    """Test Twitter credentials using our sentiment client."""
    logger.info("Testing Twitter API with our SocialMediaSentiment client...")
    
    try:
        from src.analysis_agents.sentiment.social_media_sentiment import TwitterClient
        
        # Load environment variables
        dotenv.load_dotenv()
        
        # Create Twitter client
        twitter_client = TwitterClient(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.getenv("TWITTER_ACCESS_SECRET", "")
        )
        
        # Check if using real or mock data
        if twitter_client.use_mock:
            logger.warning("Using mock data - credentials may be missing or invalid")
        else:
            logger.info("Using real Twitter API connection")
        
        # Search for tweets
        query = "#Bitcoin OR $BTC"
        logger.info(f"Searching for tweets with query: {query}")
        tweets = await twitter_client.search_tweets(query=query, count=10)
        
        # Display results
        logger.info(f"Found {len(tweets)} tweets")
        if tweets:
            logger.info("Sample tweets:")
            for i, tweet in enumerate(tweets[:3]):
                logger.info(f"{i+1}. {tweet[:100]}...")
        
        return not twitter_client.use_mock
        
    except Exception as e:
        logger.error(f"Error testing sentiment client: {e}")
        return False


async def main():
    """Run the tests."""
    print("\n==== Twitter API Credential Test ====\n")
    
    # Test with tweepy directly
    tweepy_success = test_tweepy_credentials()
    
    print("\n")
    
    # Test with our sentiment client
    client_success = await test_sentiment_client()
    
    print("\n==== Test Results ====\n")
    
    if tweepy_success:
        print("✅ Tweepy test: SUCCESS")
    else:
        print("❌ Tweepy test: FAILED")
    
    if client_success:
        print("✅ Sentiment client test: SUCCESS (using real API)")
    else:
        print("❌ Sentiment client test: FAILED (using mock data)")
    
    print("\nNext steps:")
    if not tweepy_success and not client_success:
        print("- Check your Twitter API credentials in .env file")
        print("- Make sure you have created a Twitter Developer account")
        print("- Ensure your app has the right permissions")
        print("- Run the setup_twitter_credentials.py script to update your credentials")
    elif not client_success:
        print("- Check how credentials are being loaded in the sentiment client")
    else:
        print("- Your Twitter credentials are working correctly!")
        print("- You can now run the sentiment analysis examples")
        print("- Try: python examples/sentiment_real_integration_demo.py")


if __name__ == "__main__":
    asyncio.run(main())