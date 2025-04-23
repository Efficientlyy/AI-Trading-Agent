import os
import unittest
from ai_trading_agent.sentiment_analysis.reddit_provider import RedditSentimentProvider

# Placeholder credentials - replace with actual environment variables or config loading
# For testing purposes, these can be dummy values if not actually connecting to Reddit
# Ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT are set in your environment
# or replace the os.environ.get calls with your actual credentials for local testing.
TEST_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "dummy_client_id")
TEST_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "dummy_client_secret")
TEST_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "AI Trading Agent Test")

class TestSentimentIntegration(unittest.TestCase):

    def test_reddit_provider_fetches_and_processes_sentiment(self):
        """
        Tests if the RedditSentimentProvider fetches data and the SentimentProcessor
        adds sentiment scores.
        """
        # Initialize the provider with placeholder/env credentials
        provider = RedditSentimentProvider(
            client_id=TEST_CLIENT_ID,
            client_secret=TEST_CLIENT_SECRET,
            user_agent=TEST_USER_AGENT,
            # Use a small limit for testing to avoid long fetch times
            subreddits=["stocks"], # Use a common subreddit for testing
            limit=5
        )

        # Fetch a small amount of data
        sentiment_data = provider.fetch_sentiment_data()

        # Assert that data was returned
        self.assertIsInstance(sentiment_data, list)
        self.assertGreater(len(sentiment_data), 0, "Should fetch at least one data entry")

        # Assert that each entry has a sentiment_score
        for entry in sentiment_data:
            self.assertIn('sentiment_score', entry, "Each entry should have a 'sentiment_score'")
            self.assertIsInstance(entry['sentiment_score'], float, "'sentiment_score' should be a float")
            # VADER scores are between -1.0 and 1.0
            self.assertGreaterEqual(entry['sentiment_score'], -1.0)
            self.assertLessEqual(entry['sentiment_score'], 1.0)

        print("\nSuccessfully fetched and processed sentiment data:")
        for i, entry in enumerate(sentiment_data[:3]): # Print first 3 entries as example
             print(f"Entry {i+1}: Text='{entry.get('text', 'N/A')[:100]}...', Score={entry.get('sentiment_score', 'N/A')}")


if __name__ == '__main__':
    unittest.main()
