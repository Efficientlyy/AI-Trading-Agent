# REST client for Alpha Vantage API

import requests
import logging
import time
from ai_trading_agent.config import ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.api_key = api_key

    def _make_request(self, params: dict) -> tuple[dict | None, str | None]:
        """Makes a request to the Alpha Vantage API. Returns (data, reason) where reason is None on success."""
        params['apikey'] = self.api_key
        response = None # Initialize response to None
        try:
            response = requests.get(self.BASE_URL, params=params)
            logger.debug(f"Alpha Vantage request URL: {response.url}") # Log the exact URL
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # Check for API specific error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API Error (status {response.status_code}): {data['Error Message']}")
                return None, f"api_error: {data['Error Message']}"
            if "Information" in data and "limit" in data["Information"].lower():
                logger.error(f"Alpha Vantage API rate limit likely reached. Full response data: {data}") 
                return None, f"rate_limit: {data['Information']}"

            return data, None

        except requests.exceptions.RequestException as e:
            # Log status code if response object exists
            status_code = response.status_code if response is not None else 'N/A'
            logger.error(f"HTTP Request failed (status {status_code}): {e}")
            return None, f"http_error: {e}"
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response: {response.text}")
            return None, f"json_error: {response.text[:100]}..." # Truncate long text
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error in _make_request: {e}", exc_info=True)
            return None, f"unexpected_error: {e}"

    def get_news_sentiment(self, tickers: list[str] | None = None, topics: list[str] | None = None, limit: int = 50) -> tuple[dict | None, str | None]:
        """
        Fetches news sentiment data from Alpha Vantage.

        Args:
            tickers: A list of ticker symbols (e.g., ["AAPL", "MSFT"]).
            topics: A list of topics (e.g., ["technology", "blockchain"]).
            limit: Number of results to return (default 50, max 1000).

        Returns:
            A tuple (data, reason). `data` is a dict on success, else None. `reason` is None on success, else an error string.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "limit": min(max(1, limit), 1000), # Ensure limit is within 1-1000
        }
        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)

        logger.info(f"Fetching news sentiment for tickers={tickers}, topics={topics}")
        return self._make_request(params)

# --- Example Usage --- 

def main():
    logging.basicConfig(level=logging.INFO)
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY not found in environment variables. Exiting.")
        return

    client = AlphaVantageClient(api_key=ALPHA_VANTAGE_API_KEY)

    # Example: Get sentiment for specific crypto tickers
    crypto_tickers = ["BTC", "ETH", "SOL"]
    sentiment_data, reason = client.get_news_sentiment(tickers=crypto_tickers, limit=10)

    if sentiment_data and 'feed' in sentiment_data:
        print(f"\n--- Sentiment for {crypto_tickers} ---")
        for item in sentiment_data['feed'][:5]: # Print first 5 articles
            title = item.get('title', 'N/A')
            overall_sentiment = item.get('overall_sentiment_label', 'N/A')
            print(f"- {title} ({overall_sentiment})")
    elif reason:
        print(f"\nFailed to retrieve sentiment data for crypto tickers: {reason}")
    else:
        print("\nCould not retrieve sentiment data for crypto tickers.")

    # Example: Get sentiment for blockchain topic
    # Note: Free tier might have limitations on topic/ticker combinations or frequency
    # time.sleep(15) # Add delay if hitting free tier rate limits (e.g., 5 calls/min)
    blockchain_sentiment, reason = client.get_news_sentiment(topics=["blockchain"], limit=5)
    if blockchain_sentiment and 'feed' in blockchain_sentiment:
         print(f"\n--- Sentiment for Blockchain Topic ---")
         for item in blockchain_sentiment['feed']:
             title = item.get('title', 'N/A')
             overall_sentiment = item.get('overall_sentiment_label', 'N/A')
             print(f"- {title} ({overall_sentiment})")
    elif reason:
        print(f"\nFailed to retrieve sentiment data for blockchain topic: {reason}")
    else:
         print("\nCould not retrieve sentiment data for blockchain topic.")

if __name__ == "__main__":
    main()
