"""
Unit tests for the Reddit sentiment collector.

This file contains comprehensive tests for the RedditSentimentCollector class,
which is responsible for collecting sentiment data from Reddit using PRAW.
"""
import pytest
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock

from ai_trading_agent.sentiment_analysis.data_collection import RedditSentimentCollector
from ai_trading_agent.nlp_processing.sentiment_processor import SentimentProcessor


class TestRedditSentimentCollector:
    """Comprehensive tests for RedditSentimentCollector."""
    
    def test_initialization_with_config(self):
        """Test initialization with provided configuration."""
        # Test configuration with all parameters
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'user_agent': 'test_agent',
            'subreddits': ['test_sub1', 'test_sub2'],
            'keywords': ['test_keyword1', 'test_keyword2'],
            'comment_limit': 5,
            'post_limit': 50,
            'time_filter': 'week',
            'rate_limit_wait': 0.5
        }
        
        with patch('praw.Reddit') as mock_reddit, \
             patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor') as mock_processor:
            
            # Mock the reddit instance
            mock_reddit_instance = mock_reddit.return_value
            mock_reddit_instance.read_only = True
            
            # Initialize collector
            collector = RedditSentimentCollector(config)
            
            # Verify Reddit was initialized with correct parameters
            mock_reddit.assert_called_once_with(
                client_id='test_id',
                client_secret='test_secret',
                user_agent='test_agent',
                username='',
                password=''
            )
            
            # Verify configuration parameters were set correctly
            assert collector.subreddits == ['test_sub1', 'test_sub2']
            assert collector.keywords == ['test_keyword1', 'test_keyword2']
            assert collector.comment_limit == 5
            assert collector.post_limit == 50
            assert collector.time_filter == 'week'
            assert collector.rate_limit_wait == 0.5
            assert collector.client is mock_reddit_instance
            assert mock_reddit_instance.read_only is True

    def test_initialization_with_env_vars(self):
        """Test initialization with environment variables."""
        # Setup environment variables
        os.environ['REDDIT_CLIENT_ID'] = 'env_test_id'
        os.environ['REDDIT_CLIENT_SECRET'] = 'env_test_secret'
        os.environ['REDDIT_USER_AGENT'] = 'env_test_agent'
        
        # Minimal config that should use env vars as fallback
        config = {}
        
        with patch('praw.Reddit') as mock_reddit, \
             patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor'):
            
            # Initialize collector
            collector = RedditSentimentCollector(config)
            
            # Verify Reddit was initialized with environment variable values
            mock_reddit.assert_called_once_with(
                client_id='env_test_id',
                client_secret='env_test_secret',
                user_agent='env_test_agent',
                username='',
                password=''
            )
            
            # Verify default configuration parameters were set
            assert len(collector.subreddits) > 0
            assert collector.keywords == []
            assert collector.comment_limit == 10
            assert collector.post_limit == 100
            assert collector.time_filter == 'day'
            assert collector.rate_limit_wait == 1.0
        
        # Cleanup environment
        del os.environ['REDDIT_CLIENT_ID']
        del os.environ['REDDIT_CLIENT_SECRET']
        del os.environ['REDDIT_USER_AGENT']

    def test_initialization_with_import_error(self):
        """Test graceful handling of import errors."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret'}
        
        with patch('ai_trading_agent.sentiment_analysis.data_collection.RedditSentimentCollector._initialize_client') as mock_init:
            # Simulate import error
            mock_init.side_effect = ImportError("Could not import praw")
            
            collector = RedditSentimentCollector(config)
            
            # Client should be None after initialization failure
            assert collector.client is None

    def test_create_search_queries(self):
        """Test the _create_search_queries method."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret', 'keywords': ['buy', 'sell']}
        
        with patch('praw.Reddit'), \
             patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor'):
            
            collector = RedditSentimentCollector(config)
            
            # Test with common symbols that have mappings
            symbols = ['BTC', 'AAPL']
            queries = collector._create_search_queries(symbols)
            
            assert 'BTC' in queries
            assert 'AAPL' in queries
            assert '"BTC"' in queries['BTC']
            assert '"Bitcoin"' in queries['BTC']
            assert '"AAPL"' in queries['AAPL']
            assert '"Apple"' in queries['AAPL']
            
            # Check that keywords are included
            assert '"BTC buy"' in queries['BTC'] or '"BTC sell"' in queries['BTC']
            
            # Test with unknown symbol
            symbols = ['UNKNOWN']
            queries = collector._create_search_queries(symbols)
            
            assert 'UNKNOWN' in queries
            assert '"UNKNOWN"' in queries['UNKNOWN']
            assert '"UNKNOWN buy"' in queries['UNKNOWN'] or '"UNKNOWN sell"' in queries['UNKNOWN']

    def test_mock_data_generation(self):
        """Test the _generate_mock_data method."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret'}
        
        with patch('praw.Reddit'), \
             patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor'):
            
            collector = RedditSentimentCollector(config)
            
            symbols = ['BTC', 'ETH']
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 7)  # 7 days
            
            result = collector._generate_mock_data(symbols, start_date, end_date)
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            
            # Check basic structure
            assert 'timestamp' in result.columns
            assert 'symbol' in result.columns
            assert 'source' in result.columns
            assert 'sentiment_score' in result.columns
            assert 'volume' in result.columns
            assert 'subreddit' in result.columns
            assert 'post_count' in result.columns
            assert 'comment_count' in result.columns
            assert 'content' in result.columns
            assert 'url' in result.columns
            assert 'is_comment' in result.columns
            
            # Check that all requested symbols are included
            assert set(result['symbol'].unique()) == set(symbols)
            
            # Check that all timestamps are within the requested range
            assert all(start_date <= ts <= end_date for ts in result['timestamp'])
            
            # Check that sentiment scores are within valid range
            assert all(-1.0 <= score <= 1.0 for score in result['sentiment_score'])

    @patch('praw.Reddit')
    @patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor')
    def test_collect_client_none(self, mock_processor, mock_reddit):
        """Test collect method when client initialization failed."""
        config = {'client_id': 'test_id', 'client_secret': 'test_secret'}
        
        # Setup collector with client = None
        collector = RedditSentimentCollector(config)
        collector.client = None
        
        # Mock the _generate_mock_data method
        collector._generate_mock_data = MagicMock(return_value=pd.DataFrame({'mock': [1, 2, 3]}))
        
        symbols = ['BTC', 'ETH']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        # Should call _generate_mock_data when client is None
        collector._generate_mock_data.assert_called_once_with(symbols, start_date, end_date)
        assert result is collector._generate_mock_data.return_value

    @patch('praw.Reddit')
    @patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor')
    @patch('time.sleep')
    def test_collect_successful(self, mock_sleep, mock_processor, mock_reddit):
        """Test successful data collection."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'subreddits': ['testsubreddit'],
            'comment_limit': 2,
            'rate_limit_wait': 0.0  # No waiting for tests
        }
        
        # Setup collector
        collector = RedditSentimentCollector(config)
        
        # Mock the Reddit API interaction
        mock_subreddit = MagicMock()
        mock_submission = MagicMock()
        mock_comment = MagicMock()
        
        # Configure mocks
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        mock_subreddit.search.return_value = [mock_submission]
        
        # Configure submission properties
        mock_submission.title = "Bitcoin is great"
        mock_submission.selftext = "I think BTC will go up"
        mock_submission.created_utc = (datetime(2023, 1, 1) + timedelta(hours=6)).timestamp()
        mock_submission.url = "https://reddit.com/r/testsubreddit/post1"
        mock_submission.score = 100
        mock_submission.num_comments = 5
        mock_submission.id = "abc123"
        
        # Configure comments property with replace_more method and list access
        mock_submission.comments.replace_more = MagicMock()
        mock_comment.body = "I agree that BTC is great"
        mock_comment.created_utc = (datetime(2023, 1, 1) + timedelta(hours=7)).timestamp()
        mock_comment.permalink = "/r/testsubreddit/post1/comment1"
        mock_comment.score = 10
        mock_comment.id = "def456"
        type(mock_submission).comments = PropertyMock(return_value=[mock_comment])
        
        # Mock sentiment processor
        mock_processor.return_value.process_data.return_value = [{'sentiment_score': 0.8}]
        
        # Call collect method
        symbols = ['BTC']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 2  # One submission and one comment
        
        # Check submission row
        submission_row = result[~result['is_comment']].iloc[0]
        assert submission_row['symbol'] == 'BTC'
        assert submission_row['source'] == 'reddit'
        assert submission_row['subreddit'] == 'testsubreddit'
        assert 'Bitcoin is great' in submission_row['content']
        assert submission_row['url'] == "https://reddit.com/r/testsubreddit/post1"
        assert submission_row['score'] == 100
        assert submission_row['num_comments'] == 5
        assert submission_row['sentiment_score'] == 0.8
        assert submission_row['post_count'] == 1
        assert submission_row['comment_count'] == 1
        
        # Check comment row
        comment_row = result[result['is_comment']].iloc[0]
        assert comment_row['symbol'] == 'BTC'
        assert comment_row['source'] == 'reddit'
        assert comment_row['subreddit'] == 'testsubreddit'
        assert comment_row['content'] == "I agree that BTC is great"
        assert "comment1" in comment_row['url']
        assert comment_row['score'] == 10
        assert comment_row['post_id'] == "abc123"
        assert comment_row['comment_id'] == "def456"
        assert comment_row['sentiment_score'] == 0.8

    @patch('praw.Reddit')
    @patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor')
    def test_collect_with_search_error(self, mock_processor, mock_reddit):
        """Test error handling during search."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'subreddits': ['testsubreddit'],
            'comment_limit': 0  # No comments for simplicity
        }
        
        # Setup collector
        collector = RedditSentimentCollector(config)
        
        # Mock the Reddit API interaction
        mock_subreddit = MagicMock()
        
        # Configure mocks
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        mock_subreddit.search.side_effect = Exception("API Error")
        
        # Mock the _generate_mock_data method
        collector._generate_mock_data = MagicMock(return_value=pd.DataFrame({'mock': [1, 2, 3]}))
        
        # Call collect method
        symbols = ['BTC']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        # Should call _generate_mock_data when API error occurs
        collector._generate_mock_data.assert_called_once_with(symbols, start_date, end_date)
        assert result is collector._generate_mock_data.return_value

    @patch('praw.Reddit')
    @patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor')
    @patch('time.sleep')
    def test_collect_empty_results(self, mock_sleep, mock_processor, mock_reddit):
        """Test handling of empty search results."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'subreddits': ['testsubreddit'],
            'comment_limit': 0  # No comments for simplicity
        }
        
        # Setup collector
        collector = RedditSentimentCollector(config)
        
        # Mock the Reddit API interaction
        mock_subreddit = MagicMock()
        
        # Configure mocks
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        mock_subreddit.search.return_value = []  # Empty search results
        
        # Mock the _generate_mock_data method
        collector._generate_mock_data = MagicMock(return_value=pd.DataFrame({'mock': [1, 2, 3]}))
        
        # Call collect method
        symbols = ['BTC']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = collector.collect(symbols, start_date, end_date)
        
        # Should call _generate_mock_data when no results are found
        collector._generate_mock_data.assert_called_once_with(symbols, start_date, end_date)
        assert result is collector._generate_mock_data.return_value

    @patch('praw.Reddit')
    @patch('ai_trading_agent.nlp_processing.sentiment_processor.SentimentProcessor')
    @patch('time.sleep')
    def test_collect_date_filtering(self, mock_sleep, mock_processor, mock_reddit):
        """Test filtering by date range."""
        config = {
            'client_id': 'test_id',
            'client_secret': 'test_secret',
            'subreddits': ['testsubreddit'],
            'comment_limit': 0  # No comments for simplicity
        }
        
        # Setup collector
        collector = RedditSentimentCollector(config)
        
        # Mock the Reddit API interaction
        mock_subreddit = MagicMock()
        mock_submission_in_range = MagicMock()
        mock_submission_out_of_range = MagicMock()
        
        # Configure mocks
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        mock_subreddit.search.return_value = [mock_submission_in_range, mock_submission_out_of_range]
        
        # Configure submission properties
        mock_submission_in_range.title = "Bitcoin is great"
        mock_submission_in_range.selftext = "I think BTC will go up"
        mock_submission_in_range.created_utc = datetime(2023, 1, 1, 12, 0).timestamp()  # In range
        mock_submission_in_range.url = "https://reddit.com/r/testsubreddit/post1"
        mock_submission_in_range.score = 100
        mock_submission_in_range.num_comments = 5
        mock_submission_in_range.id = "abc123"
        mock_submission_in_range.comments = []
        
        mock_submission_out_of_range.title = "Bitcoin is old news"
        mock_submission_out_of_range.selftext = "BTC discussion"
        mock_submission_out_of_range.created_utc = datetime(2023, 1, 3, 12, 0).timestamp()  # Out of range
        mock_submission_out_of_range.url = "https://reddit.com/r/testsubreddit/post2"
        mock_submission_out_of_range.score = 50
        mock_submission_out_of_range.num_comments = 2
        mock_submission_out_of_range.id = "xyz789"
        mock_submission_out_of_range.comments = []
        
        # Mock sentiment processor
        mock_processor.return_value.process_data.return_value = [{'sentiment_score': 0.8}]
        
        # Call collect method
        symbols = ['BTC']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)  # End date excludes the out of range submission
        
        result = collector.collect(symbols, start_date, end_date)
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 1  # Only one submission should be included (the in-range one)
        
        # Check that we have the correct submission
        assert result.iloc[0]['content'].startswith("Bitcoin is great")
        assert result.iloc[0]['url'] == "https://reddit.com/r/testsubreddit/post1"


if __name__ == "__main__":
    pytest.main()