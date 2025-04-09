"""
Unit tests for the sentiment data collection module.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.sentiment_analysis.data_collection import (
    BaseSentimentCollector,
    TwitterSentimentCollector,
    RedditSentimentCollector,
    NewsAPISentimentCollector,
    FearGreedIndexCollector,
    SentimentCollectionService
)


class TestBaseSentimentCollector:
    """Tests for the BaseSentimentCollector class."""
    
    def test_validate_dates(self):
        """Test the _validate_dates method."""
        # Create a concrete subclass for testing
        class TestCollector(BaseSentimentCollector):
            def collect(self, symbols, start_date=None, end_date=None):
                return pd.DataFrame()
                
        collector = TestCollector({})
        
        # Test with both dates provided
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 7)
        validated_start, validated_end = collector._validate_dates(start, end)
        assert validated_start == start
        assert validated_end == end
        
        # Test with only end_date provided
        validated_start, validated_end = collector._validate_dates(None, end)
        assert validated_end == end
        assert validated_start == end - timedelta(days=7)
        
        # Test with neither date provided
        now = datetime.now()
        validated_start, validated_end = collector._validate_dates(None, None)
        assert (now - validated_end).total_seconds() < 5  # Within 5 seconds
        assert (validated_end - validated_start).days == 7
        
        # Test with start_date after end_date
        with pytest.raises(ValueError):
            collector._validate_dates(end, start)


class TestTwitterSentimentCollector:
    """Tests for the TwitterSentimentCollector class."""
    
    def test_initialization(self):
        """Test initialization of the collector."""
        config = {'api_key': 'test_key', 'api_secret': 'test_secret'}
        collector = TwitterSentimentCollector(config)
        assert collector.config == config
        assert collector.name == 'TwitterSentimentCollector'
        
    def test_collect(self):
        """Test the collect method returns a DataFrame with expected columns."""
        config = {'api_key': 'test_key', 'api_secret': 'test_secret'}
        collector = TwitterSentimentCollector(config)
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'source' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'volume' in result.columns
        
        # Check that all requested symbols are included
        assert set(result['symbol'].unique()) == set(symbols)
        
        # Check that all timestamps are within the requested range
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
        
        # Check that the source is correctly set
        assert all(result['source'] == 'twitter')


class TestRedditSentimentCollector:
    """Tests for the RedditSentimentCollector class."""
    
    def test_initialization(self):
        """Test initialization of the collector."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret'}
        collector = RedditSentimentCollector(config)
        assert collector.config == config
        assert collector.name == 'RedditSentimentCollector'
        
    def test_collect(self):
        """Test the collect method returns a DataFrame with expected columns."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret'}
        collector = RedditSentimentCollector(config)
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'source' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'volume' in result.columns
        assert 'subreddit' in result.columns
        
        # Check that all requested symbols are included
        assert set(result['symbol'].unique()) == set(symbols)
        
        # Check that all timestamps are within the requested range
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
        
        # Check that the source is correctly set
        assert all(result['source'] == 'reddit')


class TestNewsAPISentimentCollector:
    """Tests for the NewsAPISentimentCollector class."""
    
    def test_initialization(self):
        """Test initialization of the collector."""
        config = {'api_key': 'test_key'}
        collector = NewsAPISentimentCollector(config)
        assert collector.config == config
        assert collector.name == 'NewsAPISentimentCollector'
        
    def test_collect(self):
        """Test the collect method returns a DataFrame with expected columns."""
        config = {'api_key': 'test_key'}
        collector = NewsAPISentimentCollector(config)
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'source' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'volume' in result.columns
        assert 'title' in result.columns
        assert 'source_name' in result.columns
        assert 'url' in result.columns
        
        # Check that all requested symbols are included
        assert set(result['symbol'].unique()) == set(symbols)
        
        # Check that all timestamps are within the requested range
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
        
        # Check that the source is correctly set
        assert all(result['source'] == 'news')


class TestFearGreedIndexCollector:
    """Tests for the FearGreedIndexCollector class."""
    
    def test_initialization(self):
        """Test initialization of the collector."""
        config = {'api_key': 'test_key'}
        collector = FearGreedIndexCollector(config)
        assert collector.config == config
        assert collector.name == 'FearGreedIndexCollector'
        
    def test_collect(self):
        """Test the collect method returns a DataFrame with expected columns."""
        config = {'api_key': 'test_key'}
        collector = FearGreedIndexCollector(config)
        
        symbols = ['BTC', 'ETH']  # Not used directly by this collector
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)  # 7 days
        
        result = collector.collect(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'source' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'value' in result.columns
        assert 'classification' in result.columns
        
        # Check that the symbol is set to 'MARKET'
        assert all(result['symbol'] == 'MARKET')
        
        # Check that all timestamps are within the requested range
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
        
        # Check that the source is correctly set
        assert all(result['source'] == 'fear_greed_index')
        
        # Check that we have daily data (7 days)
        assert len(result) == 7


class TestSentimentCollectionService:
    """Tests for the SentimentCollectionService class."""
    
    def test_initialization(self):
        """Test initialization of the service."""
        config = {
            'collectors': {
                'twitter': {'api_key': 'test_key', 'api_secret': 'test_secret'},
                'reddit': {'client_id': 'test_id', 'client_secret': 'test_secret'},
                'news': {'api_key': 'test_key'},
                'fear_greed_index': {'api_key': 'test_key'}
            }
        }
        service = SentimentCollectionService(config)
        
        assert service.config == config
        assert len(service.collectors) == 4
        assert 'twitter' in service.collectors
        assert 'reddit' in service.collectors
        assert 'news' in service.collectors
        assert 'fear_greed_index' in service.collectors
        
    def test_collect_all(self):
        """Test the collect_all method combines data from all collectors."""
        config = {
            'collectors': {
                'twitter': {'api_key': 'test_key', 'api_secret': 'test_secret'},
                'reddit': {'client_id': 'test_id', 'client_secret': 'test_secret'}
            }
        }
        service = SentimentCollectionService(config)
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = service.collect_all(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'source' in result.columns
        assert 'sentiment_score' in result.columns
        assert 'volume' in result.columns
        
        # Check that all requested symbols are included
        assert set(result['symbol'].unique()) == set(symbols)
        
        # Check that all timestamps are within the requested range
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
        
        # Check that data from both collectors is included
        assert 'twitter' in result['source'].unique()
        assert 'reddit' in result['source'].unique()
        
    def test_collect_all_with_error(self):
        """Test the collect_all method handles errors from collectors."""
        config = {
            'collectors': {
                'twitter': {'api_key': 'test_key', 'api_secret': 'test_secret'},
                'reddit': {'client_id': 'test_id', 'client_secret': 'test_secret'}
            }
        }
        service = SentimentCollectionService(config)
        
        # Mock the Twitter collector to raise an exception
        service.collectors['twitter'].collect = MagicMock(side_effect=Exception("API error"))
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        # Should still return data from Reddit collector
        result = service.collect_all(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'reddit' in result['source'].unique()
        assert 'twitter' not in result['source'].unique()
