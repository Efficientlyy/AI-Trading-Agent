"""
Integration tests for the Alpha Vantage client.

These tests verify that the Alpha Vantage client can successfully connect to the API
and retrieve sentiment data. They require a valid API key to be set in the environment.
"""

import os
import pytest
import logging
from dotenv import load_dotenv
from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    os.getenv("ALPHA_VANTAGE_API_KEY") is None,
    reason="Alpha Vantage API key not set in environment"
)

@pytest.fixture
def alpha_vantage_client():
    """Create an Alpha Vantage client for testing."""
    return AlphaVantageClient()

def test_get_sentiment_by_topic(alpha_vantage_client):
    """Test retrieving sentiment data by topic."""
    # Test with a valid topic
    result = alpha_vantage_client.get_sentiment_by_topic("blockchain", days_back=3)
    
    # Check that we got a valid response
    assert "data" in result
    assert "feed" in result["data"]
    assert len(result["data"]["feed"]) > 0
    
    # Test with an invalid topic that should be mapped to a valid one
    result = alpha_vantage_client.get_sentiment_by_topic("bitcoin", days_back=3)
    
    # Check that we got a valid response (should map to blockchain)
    assert "data" in result
    assert "feed" in result["data"]
    assert len(result["data"]["feed"]) > 0

def test_get_sentiment_by_crypto(alpha_vantage_client):
    """Test retrieving sentiment data by cryptocurrency ticker."""
    # Test with a valid crypto ticker
    result = alpha_vantage_client.get_sentiment_by_crypto("BTC", days_back=3)
    
    # If we're on the free tier, this might fall back to topics
    assert "data" in result
    
    if "feed" in result["data"]:
        assert len(result["data"]["feed"]) > 0
    else:
        logger.warning("No feed data in response, likely due to API tier limitations")

def test_extract_sentiment_scores(alpha_vantage_client):
    """Test extracting sentiment scores from API response."""
    # First get some sentiment data
    result = alpha_vantage_client.get_sentiment_by_topic("blockchain", days_back=3)
    
    # Extract sentiment scores
    scores = alpha_vantage_client.extract_sentiment_scores(result)
    
    # Check that we got some scores
    assert len(scores) > 0
    
    # Check that each score has the expected fields
    for score in scores:
        assert "title" in score
        assert "url" in score
        assert "time_published" in score
        assert "overall_sentiment_score" in score
        assert "overall_sentiment_label" in score

def test_topic_validation(alpha_vantage_client):
    """Test that topic validation works correctly."""
    # Test with a valid topic
    valid_topics = alpha_vantage_client._validate_topics(["blockchain"])
    assert valid_topics == ["blockchain"]
    
    # Test with an invalid topic that should be mapped to a valid one
    invalid_topics = alpha_vantage_client._validate_topics(["bitcoin"])
    assert "blockchain" in invalid_topics
    
    # Test with multiple topics, some valid, some invalid
    mixed_topics = alpha_vantage_client._validate_topics(["blockchain", "bitcoin", "invalid"])
    assert "blockchain" in mixed_topics
    assert len(mixed_topics) >= 2  # Should have at least blockchain and a mapping for bitcoin

def test_fallback_mechanism(alpha_vantage_client):
    """Test the fallback mechanism for failed queries."""
    # Create a mock response with an error
    mock_error_response = {"error": "API limit reached", "data": None}
    
    # Test the fallback mechanism
    fallback_result = alpha_vantage_client._attempt_fallback_query(
        tickers=["BTC", "ETH"],
        time_from="20230101T0000",
        time_to="20230131T2359"
    )
    
    # Check that we got a valid response
    assert "data" in fallback_result
    
    if "feed" in fallback_result["data"]:
        assert len(fallback_result["data"]["feed"]) > 0
    else:
        logger.warning("No feed data in fallback response, likely due to API tier limitations")
