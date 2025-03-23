# Setting Up Twitter API for Sentiment Analysis

## Prerequisites

1. A Twitter Developer account
2. API credentials (API Key, API Secret, Access Token, Access Token Secret)
3. Python and the `tweepy` package installed

## Getting Twitter Developer Credentials

If you don't already have Twitter Developer credentials:

1. Go to [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
2. Sign in with your Twitter account
3. Create a new project and app
4. Generate API keys and tokens
5. Make sure your app has appropriate permissions (read-only is sufficient for sentiment analysis)

## Setting Up Your Credentials

### Option 1: Using the Setup Script (Recommended)

1. Run the setup script:
   ```bash
   python setup_twitter_credentials.py
   ```

2. Follow the prompts to enter your credentials
3. The script will create a `.env` file with your credentials
4. The script can test your credentials to ensure they work

### Option 2: Manual Setup

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your Twitter credentials:
   ```
   TWITTER_API_KEY=your_api_key_here
   TWITTER_API_SECRET=your_api_secret_here
   TWITTER_ACCESS_TOKEN=your_access_token_here
   TWITTER_ACCESS_SECRET=your_access_token_secret_here
   ```

3. Save the file

## Testing Your Setup

To test if your Twitter credentials are working:

```bash
python examples/test_twitter_credentials.py
```

If successful, you should see output showing that the connection was established and some sample tweets were retrieved.

## Using Twitter Sentiment Analysis

With your credentials set up, you can now run the sentiment analysis examples:

```bash
python examples/sentiment_real_integration_demo.py
```

This will fetch real tweets about Bitcoin (or whichever cryptocurrency you specify) and analyze their sentiment, providing trading signals based on the social media sentiment.

## Security Notes

- NEVER commit your `.env` file to version control
- The `.gitignore` file is set up to exclude `.env` files
- If you accidentally expose your credentials, regenerate them immediately in the Twitter Developer Portal
- Consider using a dedicated Twitter Developer account for your trading bot