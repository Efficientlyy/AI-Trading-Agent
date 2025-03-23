# Twitter API Setup Guide

## Overview

This guide provides detailed instructions for setting up Twitter API access for the AI Trading Agent's sentiment analysis system. The Twitter API allows the system to collect real-time tweets related to cryptocurrencies for sentiment analysis.

## Prerequisites

Before you begin, you'll need:

- A Twitter account
- A valid phone number (required for Twitter Developer account)
- A valid email address
- A credit card (for identity verification, but API access is free up to certain limits)

## Step 1: Create a Twitter Developer Account

1. Visit the [Twitter Developer Portal](https://developer.twitter.com/) and sign in with your Twitter account.

2. Click on "Apply" to apply for a developer account.

3. Select "Academic" as the account type if this is for research purposes, or "Business" if it's for commercial use.

4. Complete the application form with the following information:
   - **Use case**: Describe how you'll use the Twitter API (example: "Analyzing cryptocurrency market sentiment using social media data for algorithmic trading research")
   - **Application details**: Provide specific details about your application
   - **Intended use**: Explain how the data will be used and whether you'll display tweets to the public

5. Review and agree to the Twitter Developer Agreement and Policy.

6. Submit your application and wait for approval (usually 1-2 business days).

## Step 2: Create a Project and App

After your developer account is approved:

1. Go to the [Twitter Developer Portal Dashboard](https://developer.twitter.com/en/portal/dashboard).

2. Click on "Projects & Apps" in the left sidebar.

3. Click "Create Project" and provide the following:
   - **Project name**: "AI Trading Agent Sentiment Analysis"
   - **Use case**: Select "Making a bot" or "Academic research"
   - **Project description**: Briefly describe your project

4. Click "Next" and then "Create App" to create an app within your project.

5. Provide an app name (e.g., "AITradingAgent-Sentiment") and click "Complete".

6. You will now see your API keys and tokens. **IMPORTANT: Save these immediately!** They include:
   - API Key (Consumer Key)
   - API Key Secret (Consumer Secret)
   - Bearer Token

## Step 3: Generate Access Token and Secret

For user authentication (required for some endpoints), you'll need to generate access tokens:

1. In the Developer Portal, navigate to your app under "Projects & Apps".

2. Under "User authentication settings", click "Set up".

3. Configure the authentication settings:
   - **App permissions**: Select "Read" (we only need to read tweets, not post)
   - **Type of app**: Select "Web App, Automated App or Bot"
   - **Callback URL**: Enter a placeholder URL like `https://127.0.0.1/callback`
   - **Website URL**: Enter your project website or GitHub repository URL

4. Click "Save".

5. Navigate to "Keys and tokens" tab for your app.

6. Click "Generate" under "Access Token and Secret".

7. Save the generated Access Token and Access Token Secret.

## Step 4: Choose API Version and Endpoint

Twitter offers different API versions with different capabilities:

### Twitter API v2 (Recommended)

The v2 API is the latest version with improved features and better rate limits:

1. For **recent search** (tweets from the last 7 days), you'll use:
   ```
   https://api.twitter.com/2/tweets/search/recent
   ```

2. For **filtered stream** (real-time tweets matching criteria), you'll use:
   ```
   https://api.twitter.com/2/tweets/search/stream
   ```

### Twitter API v1.1 (Legacy)

If you need specific v1.1 endpoints:

1. For **standard search** (tweets from the last 7 days), you'll use:
   ```
   https://api.twitter.com/1.1/search/tweets.json
   ```

## Step 5: Configure API Credentials in Your Project

Store your API credentials securely using environment variables:

1. Create a `.env` file in your project root (ensure it's in `.gitignore`):
   ```
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_SECRET=your_access_token_secret
   TWITTER_BEARER_TOKEN=your_bearer_token
   ```

2. Load these environment variables in your application:
   ```python
   import os
   from dotenv import load_dotenv

   # Load environment variables
   load_dotenv()

   # Access credentials
   api_key = os.getenv("TWITTER_API_KEY")
   api_secret = os.getenv("TWITTER_API_SECRET")
   access_token = os.getenv("TWITTER_ACCESS_TOKEN")
   access_secret = os.getenv("TWITTER_ACCESS_SECRET")
   bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
   ```

## Step 6: Install and Configure Tweepy

Tweepy is a Python library that simplifies working with the Twitter API:

1. Install Tweepy:
   ```bash
   pip install tweepy
   ```

2. Basic usage example:
   ```python
   import tweepy

   # Authentication
   auth = tweepy.OAuth1UserHandler(
       api_key, api_secret, access_token, access_secret
   )
   api = tweepy.API(auth)

   # Search for tweets
   tweets = api.search_tweets(q="#BTC OR #Bitcoin", count=100)

   # Process tweets
   for tweet in tweets:
       print(f"{tweet.user.screen_name}: {tweet.text}")
   ```

3. For Twitter API v2 with Tweepy:
   ```python
   import tweepy

   # Authentication
   client = tweepy.Client(
       bearer_token=bearer_token,
       consumer_key=api_key,
       consumer_secret=api_secret,
       access_token=access_token,
       access_token_secret=access_secret
   )

   # Search for recent tweets
   response = client.search_recent_tweets(
       query="#BTC OR #Bitcoin",
       max_results=100
   )

   # Process tweets
   for tweet in response.data:
       print(tweet.text)
   ```

## Step 7: Understanding Rate Limits

Twitter API has rate limits that restrict the number of requests you can make:

### Standard API (Free Tier)

- **Search API**: 180 requests per 15-minute window
- **Filtered Stream API**: 50 rules per app and up to 25 connections per app
- **Tweet lookup**: 300 requests per 15-minute window

### Essential Access (Paid Tier)

- Higher rate limits
- Access to the full archive search

### Handling Rate Limits

1. Implement rate limit handling:
   ```python
   try:
       response = client.search_recent_tweets(query="#BTC", max_results=100)
   except tweepy.TooManyRequests:
       # Wait for 15 minutes before trying again
       print("Rate limit exceeded. Waiting for 15 minutes...")
       time.sleep(15 * 60)
   except tweepy.TwitterServerError:
       # Handle server errors
       print("Twitter server error. Retrying in 60 seconds...")
       time.sleep(60)
   ```

2. Track rate limit status:
   ```python
   # Get rate limit status (v1.1 API)
   rate_limit_status = api.rate_limit_status()
   search_limits = rate_limit_status['resources']['search']['/search/tweets']
   remaining = search_limits['remaining']
   reset_time = search_limits['reset']
   
   print(f"Remaining requests: {remaining}")
   print(f"Reset time: {datetime.fromtimestamp(reset_time)}")
   ```

## Step 8: Implementing Efficient Search

To maximize the value of your API requests:

### Optimize Search Queries

1. Use boolean operators:
   ```
   "#BTC OR #Bitcoin OR $BTC OR #crypto"
   ```

2. Filter by language:
   ```python
   client.search_recent_tweets(
       query="#BTC OR #Bitcoin",
       max_results=100,
       tweet_fields=["created_at", "lang"],
       expansions=["author_id"],
       user_fields=["username", "verified"],
       lang="en"  # English tweets only
   )
   ```

3. Exclude retweets to focus on original content:
   ```
   "#BTC OR #Bitcoin -is:retweet"
   ```

### Request Only Needed Fields

Specify which fields you need to reduce response size:

```python
client.search_recent_tweets(
    query="#BTC OR #Bitcoin",
    max_results=100,
    tweet_fields=["created_at", "text", "public_metrics"],
    expansions=["author_id"],
    user_fields=["username", "verified", "public_metrics"]
)
```

## Step 9: Testing Your Setup

Create a simple test script to verify your Twitter API integration:

```python
# test_twitter_api.py
import tweepy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access credentials
api_key = os.getenv("TWITTER_API_KEY")
api_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_secret = os.getenv("TWITTER_ACCESS_SECRET")
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

# Test authentication
try:
    # v1.1 API test
    auth = tweepy.OAuth1UserHandler(
        api_key, api_secret, access_token, access_secret
    )
    api = tweepy.API(auth)
    user = api.verify_credentials()
    print(f"Authentication successful for user: @{user.screen_name}")
    
    # v2 API test
    client = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_secret
    )
    me = client.get_me()
    print(f"Authentication successful for v2 API: @{me.data.username}")
    
    # Test search
    query = "#BTC OR #Bitcoin"
    print(f"Searching for: {query}")
    
    tweets = client.search_recent_tweets(
        query=query,
        max_results=10,
        tweet_fields=["created_at"]
    )
    
    if tweets.data:
        print(f"Found {len(tweets.data)} tweets")
        for tweet in tweets.data:
            print(f"- [{tweet.created_at}] {tweet.text[:50]}...")
    else:
        print("No tweets found.")
        
except Exception as e:
    print(f"Error: {e}")
```

Run the test:
```bash
python test_twitter_api.py
```

## Step 10: Implementing in the AI Trading Agent

Integrate the Twitter API with the AI Trading Agent's sentiment analysis system:

1. Create a dedicated Twitter client class in `src/analysis_agents/sentiment/twitter_client.py`:

```python
import tweepy
import asyncio
from typing import List, Dict, Any, Optional
import os
import time
import logging
from datetime import datetime, timedelta

from src.common.logging import get_logger

class TwitterClient:
    """Client for accessing Twitter API.
    
    This class provides methods for searching tweets, streaming tweets,
    and handling rate limits.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 access_token: Optional[str] = None,
                 access_secret: Optional[str] = None,
                 bearer_token: Optional[str] = None,
                 use_mock_data: bool = False):
        """Initialize the Twitter client.
        
        Args:
            api_key: Twitter API key (consumer key)
            api_secret: Twitter API key secret (consumer secret)
            access_token: Twitter access token
            access_secret: Twitter access token secret
            bearer_token: Twitter bearer token
            use_mock_data: Whether to use mock data instead of real API
        """
        self.logger = get_logger("sentiment", "twitter_client")
        
        # Use provided credentials or try environment variables
        self.api_key = api_key or os.getenv("TWITTER_API_KEY")
        self.api_secret = api_secret or os.getenv("TWITTER_API_SECRET")
        self.access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_secret = access_secret or os.getenv("TWITTER_ACCESS_SECRET")
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        # Check if credentials are available
        self.has_credentials = (
            self.api_key and self.api_secret and 
            (self.access_token and self.access_secret or self.bearer_token)
        )
        
        # Use mock data if specified or if credentials are missing
        self.use_mock_data = use_mock_data or not self.has_credentials
        
        if self.use_mock_data:
            self.logger.warning("Using mock Twitter data. Real API credentials not available or mock mode enabled.")
            self.api = None
            self.client = None
        else:
            try:
                # Initialize v1.1 API
                auth = tweepy.OAuth1UserHandler(
                    self.api_key, self.api_secret, self.access_token, self.access_secret
                )
                self.api = tweepy.API(auth)
                
                # Initialize v2 API
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret
                )
                
                # Test authentication
                user = self.api.verify_credentials()
                self.logger.info(f"Twitter API authentication successful: @{user.screen_name}")
                
            except Exception as e:
                self.logger.error(f"Twitter API authentication failed: {e}")
                self.api = None
                self.client = None
                self.use_mock_data = True
        
        # Rate limit handling
        self.rate_limits = {
            "search": {
                "remaining": 180,
                "reset_time": datetime.now() + timedelta(minutes=15)
            }
        }
    
    async def search_tweets(self, 
                           query: str, 
                           count: int = 100, 
                           result_type: str = "recent",
                           language: str = "en",
                           include_entities: bool = False) -> List[str]:
        """Search for tweets matching a query.
        
        Args:
            query: Search query
            count: Maximum number of tweets to return
            result_type: Type of results (recent, popular, mixed)
            language: Language filter
            include_entities: Whether to include entities
            
        Returns:
            List of tweet texts
        """
        if self.use_mock_data:
            return await self._get_mock_tweets(query, count)
        
        # Check rate limits
        if not self._check_rate_limit("search"):
            self.logger.warning("Rate limit exceeded, using mock data")
            return await self._get_mock_tweets(query, count)
        
        try:
            # Use v2 API if client is available
            if self.client:
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(count, 100),  # v2 API has max 100 per request
                    tweet_fields=["created_at", "text", "public_metrics", "lang"],
                    expansions=["author_id"],
                    user_fields=["username", "verified"],
                )
                
                if not response.data:
                    self.logger.warning(f"No tweets found for query: {query}")
                    return []
                
                # Extract tweet texts
                tweets = [tweet.text for tweet in response.data]
                self.logger.info(f"Found {len(tweets)} tweets for query: {query}")
                
                # Update rate limit (approximate)
                self._update_rate_limit("search", remaining=self.rate_limits["search"]["remaining"] - 1)
                
                return tweets
                
            # Fall back to v1.1 API
            elif self.api:
                tweets = self.api.search_tweets(
                    q=query,
                    count=count,
                    result_type=result_type,
                    lang=language,
                    include_entities=include_entities
                )
                
                # Extract tweet texts
                texts = [tweet.text for tweet in tweets]
                self.logger.info(f"Found {len(texts)} tweets for query: {query}")
                
                # Update rate limit
                rate_limit = self.api.rate_limit_status()["resources"]["search"]["/search/tweets"]
                self._update_rate_limit("search", 
                                      remaining=rate_limit["remaining"],
                                      reset_time=datetime.fromtimestamp(rate_limit["reset"]))
                
                return texts
            
            else:
                self.logger.error("No Twitter API client available")
                return await self._get_mock_tweets(query, count)
                
        except tweepy.TooManyRequests:
            self.logger.warning("Twitter API rate limit exceeded")
            self._update_rate_limit("search", remaining=0)
            return await self._get_mock_tweets(query, count)
            
        except Exception as e:
            self.logger.error(f"Error searching tweets: {e}")
            return await self._get_mock_tweets(query, count)
    
    async def _get_mock_tweets(self, query: str, count: int) -> List[str]:
        """Generate mock tweets for testing.
        
        Args:
            query: Search query
            count: Number of mock tweets to generate
            
        Returns:
            List of mock tweet texts
        """
        self.logger.info(f"Generating {count} mock tweets for query: {query}")
        
        # Extract keywords from query
        keywords = query.replace("OR", " ").replace("AND", " ")
        keywords = keywords.replace("#", "").replace("$", "").replace("@", "")
        keywords = [k.strip() for k in keywords.split() if len(k.strip()) > 0]
        
        if not keywords:
            keywords = ["crypto"]
        
        # Generate mock tweets
        mock_tweets = []
        sentiments = ["positive", "neutral", "negative"]
        
        positive_templates = [
            "Just bought more {}! To the moon! 🚀🚀 #bullish",
            "Feeling bullish on {} today. Price looking strong! 📈",
            "{} is showing great fundamentals. Long-term hold! 💎🙌",
            "This {} dip is a great buying opportunity! Loading up.",
            "Institutional adoption of {} is increasing. Bullish signal!"
        ]
        
        neutral_templates = [
            "What do you think about {} price action today?",
            "Interesting developments in the {} ecosystem.",
            "Watching {} closely. Could go either way from here.",
            "Anyone following the latest {} news?",
            "{} trading sideways. Waiting for a clear signal."
        ]
        
        negative_templates = [
            "Selling my {} bags. This doesn't look good. 📉",
            "{} showing weakness. Might be the start of a downtrend.",
            "Bearish on {} in the short term. Taking profits.",
            "Bad news for {}. Expect more downside.",
            "{} volume declining. Not a good sign for price."
        ]
        
        for i in range(count):
            keyword = keywords[i % len(keywords)]
            sentiment = sentiments[i % len(sentiments)]
            
            if sentiment == "positive":
                template = positive_templates[i % len(positive_templates)]
            elif sentiment == "neutral":
                template = neutral_templates[i % len(neutral_templates)]
            else:
                template = negative_templates[i % len(negative_templates)]
                
            tweet = template.format(keyword)
            mock_tweets.append(tweet)
        
        return mock_tweets
    
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if the rate limit allows a request.
        
        Args:
            endpoint: API endpoint to check
            
        Returns:
            True if request is allowed, False otherwise
        """
        if endpoint not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[endpoint]
        
        if limit_info["remaining"] <= 0:
            # Check if reset time has passed
            if datetime.now() >= limit_info["reset_time"]:
                # Reset rate limit
                self._update_rate_limit(endpoint, remaining=180)
                return True
            else:
                return False
        
        return True
    
    def _update_rate_limit(self, 
                          endpoint: str, 
                          remaining: int, 
                          reset_time: Optional[datetime] = None) -> None:
        """Update rate limit information.
        
        Args:
            endpoint: API endpoint
            remaining: Remaining requests
            reset_time: Time when the rate limit resets
        """
        if reset_time is None:
            reset_time = datetime.now() + timedelta(minutes=15)
            
        self.rate_limits[endpoint] = {
            "remaining": remaining,
            "reset_time": reset_time
        }
        
        if remaining <= 10:
            self.logger.warning(f"Twitter API rate limit for {endpoint} low: {remaining} remaining")
```

2. Use the Twitter client in the sentiment analysis system:

```python
from src.analysis_agents.sentiment.twitter_client import TwitterClient

async def analyze_twitter_sentiment(symbol: str) -> Dict[str, Any]:
    """Analyze Twitter sentiment for a cryptocurrency.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Initialize Twitter client
    twitter = TwitterClient()
    
    # Extract base currency from symbol
    base_currency = symbol.split('/')[0]
    
    # Create search query
    query = f"#{base_currency} OR ${base_currency} OR {base_currency} -is:retweet"
    
    # Search for tweets
    tweets = await twitter.search_tweets(query=query, count=100)
    
    # Analyze sentiment
    # ... (sentiment analysis code)
    
    return {
        "source": "twitter",
        "symbol": symbol,
        "tweet_count": len(tweets),
        "sentiment_value": sentiment_value,
        "confidence": confidence,
        "timestamp": datetime.now()
    }
```

## Common Issues and Troubleshooting

### Rate Limit Errors

**Issue**: Receiving rate limit exceeded errors.

**Solution**:
- Implement exponential backoff strategy for retries
- Cache results to reduce API calls
- Use streams for real-time data instead of repeated searches

### Authentication Errors

**Issue**: Authentication failures.

**Solutions**:
- Verify that credentials are correct
- Check if your Twitter Developer account is active
- Ensure you've agreed to the latest terms of service
- Confirm that the app has the correct permissions

### No Results Found

**Issue**: Searches return no results.

**Solutions**:
- Broaden your search query
- Check that you're not using operators incorrectly
- Verify that the topic is being discussed on Twitter
- Consider using more common hashtags or keywords

## Best Practices

1. **Rate Limit Handling**:
   - Implement robust rate limit tracking
   - Use exponential backoff for retries
   - Cache results when appropriate

2. **Error Handling**:
   - Implement comprehensive error handling for all API calls
   - Log errors with enough context for diagnosis
   - Provide graceful fallbacks (like mock data)

3. **Security**:
   - Never hardcode API credentials in your code
   - Use environment variables or a secrets manager
   - Rotate credentials periodically
   - Include credentials in .gitignore

4. **Performance**:
   - Request only the fields you need
   - Use pagination for large result sets
   - Implement response caching
   - Process tweets in batches

## Additional Resources

- [Twitter API Documentation](https://developer.twitter.com/en/docs)
- [Tweepy Documentation](https://docs.tweepy.org/)
- [Twitter API Rate Limits](https://developer.twitter.com/en/docs/twitter-api/rate-limits)
- [Twitter API Search Operators](https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query)

## Conclusion

You've now set up Twitter API access for the AI Trading Agent's sentiment analysis system. This integration enables real-time collection of social media sentiment data for cryptocurrencies, which can be used to generate trading signals and enhance trading strategies.

Remember to monitor your API usage to ensure you stay within rate limits, and consider upgrading to a paid tier if you need higher limits for production use.