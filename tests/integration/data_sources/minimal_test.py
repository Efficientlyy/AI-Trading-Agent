"""
Minimal test for Alpha Vantage client to avoid hitting rate limits.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_api_request():
    """Make a direct API request to Alpha Vantage to check if the API key is valid."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment variables.")
        return False
    
    import requests
    
    # Make a simple request to the Alpha Vantage API
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=blockchain&apikey={api_key}"
    logger.info(f"Making direct API request to: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        data = response.json()
        
        # Check if we got a valid response
        if "feed" in data:
            article_count = len(data["feed"])
            logger.info(f"Direct API request successful! Retrieved {article_count} articles")
            if article_count > 0:
                logger.info(f"First article title: {data['feed'][0]['title']}")
            return True
        else:
            # Check for error messages
            if "Note" in data:
                logger.error(f"API Note: {data['Note']}")
            elif "Information" in data:
                logger.error(f"API Information: {data['Information']}")
            elif "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
            else:
                logger.error(f"Unexpected API response: {data}")
            return False
    except Exception as e:
        logger.exception(f"Error making direct API request: {e}")
        return False

def main():
    # Check for API key
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment variables.")
        logger.info("Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return
    
    logger.info(f"Using Alpha Vantage API key: {api_key}")
    
    # First test with a direct API request
    logger.info("Testing direct API request...")
    direct_api_success = test_direct_api_request()
    
    if not direct_api_success:
        logger.error("Direct API request failed. Please check your API key and rate limits.")
        return
    
    # If direct API request succeeded, test our client
    try:
        # Create client
        client = AlphaVantageClient(api_key=api_key)
        logger.info("Created Alpha Vantage client successfully")
        
        # Test with a single topic query (uses cache if available)
        logger.info("Testing get_sentiment_by_topic with 'blockchain'...")
        result = client.get_sentiment_by_topic("blockchain", days_back=1, use_cache=True)
        
        if result is None:
            logger.error("Received None result from get_sentiment_by_topic")
            return
        
        # Check if we got data
        if "data" in result and result["data"] and "feed" in result["data"]:
            article_count = len(result["data"]["feed"])
            logger.info(f"Success! Retrieved {article_count} articles")
            
            # Show first article title if available
            if article_count > 0:
                logger.info(f"First article: {result['data']['feed'][0]['title']}")
        else:
            logger.warning("No data returned, possibly hit rate limit or API key issue")
            if "error" in result:
                logger.error(f"Error: {result['error']}")
            else:
                logger.error(f"Unexpected response format: {result}")
    except Exception as e:
        logger.exception(f"Error testing Alpha Vantage client: {e}")

if __name__ == "__main__":
    main()
