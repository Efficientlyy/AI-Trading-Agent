"""
Unit tests for the sentiment analysis service module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from unittest.mock import patch, MagicMock

from ai_trading_agent.sentiment_analysis.service import SentimentAnalysisService
from ai_trading_agent.trading_engine.models import Order


class TestSentimentAnalysisService:
    """Tests for the SentimentAnalysisService class."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for output
        self.test_output_dir = 'test_output'
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        # Clean up temporary files
        for file in os.listdir(self.test_output_dir):
            os.remove(os.path.join(self.test_output_dir, file))
        os.rmdir(self.test_output_dir)
    
    def test_initialization(self):
        """Test initialization of the service."""
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        assert service.config == config
        assert service.output_dir == self.test_output_dir
        assert hasattr(service, 'collection_service')
        assert hasattr(service, 'nlp_pipeline')
        assert hasattr(service, 'strategy')
        assert service.sentiment_cache == {}
    
    @patch('ai_trading_agent.sentiment_analysis.data_collection.SentimentCollectionService.collect_all')
    @patch('ai_trading_agent.sentiment_analysis.nlp_processing.NLPPipeline.process_dataframe')
    def test_collect_sentiment_data(self, mock_process_dataframe, mock_collect_all):
        """Test collection of sentiment data."""
        # Setup mock returns
        mock_sentiment_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'source': ['twitter'],
            'content': ['Test content'],
            'sentiment_score': [0.5]
        })
        mock_collect_all.return_value = mock_sentiment_data
        
        # Mock NLP processing to return the same data
        mock_process_dataframe.return_value = mock_sentiment_data
        
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        # Call the collect_sentiment_data method
        symbols = ['BTC/USD']
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        result = service.collect_sentiment_data(symbols, start_date, end_date)
        
        # Check that the collection service was called with the correct parameters
        mock_collect_all.assert_called_once_with(symbols, start_date, end_date)
        
        # Check that the NLP pipeline was called with the correct parameters
        mock_process_dataframe.assert_called_once()
        
        # Check that the result is the expected DataFrame
        assert result is mock_sentiment_data
        
        # Check that the result is cached
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        assert cache_key in service.sentiment_cache
        assert service.sentiment_cache[cache_key]['data'] is mock_sentiment_data
    
    @patch('ai_trading_agent.sentiment_analysis.data_collection.SentimentCollectionService.collect_all')
    def test_collect_sentiment_data_cache(self, mock_collect_all):
        """Test that sentiment data is cached and reused."""
        # Setup mock returns
        mock_sentiment_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'source': ['twitter'],
            'content': ['Test content'],
            'sentiment_score': [0.5]
        })
        mock_collect_all.return_value = mock_sentiment_data
        
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir,
            'cache_expiry_hours': 24
        }
        service = SentimentAnalysisService(config)
        
        # Call the collect_sentiment_data method
        symbols = ['BTC/USD']
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        # First call should hit the collection service
        result1 = service.collect_sentiment_data(symbols, start_date, end_date)
        assert mock_collect_all.call_count == 1
        
        # Second call with the same parameters should use the cache
        result2 = service.collect_sentiment_data(symbols, start_date, end_date)
        assert mock_collect_all.call_count == 1  # Still 1, not called again
        
        # Check that both results are the same
        assert result1 is result2
        
        # Force refresh should bypass the cache
        result3 = service.collect_sentiment_data(symbols, start_date, end_date, force_refresh=True)
        assert mock_collect_all.call_count == 2
    
    @patch('ai_trading_agent.sentiment_analysis.service.SentimentAnalysisService.collect_sentiment_data')
    @patch('ai_trading_agent.sentiment_analysis.strategy.SentimentStrategy.preprocess_data')
    @patch('ai_trading_agent.sentiment_analysis.strategy.SentimentStrategy.generate_signals')
    def test_generate_trading_signals(self, mock_generate_signals, mock_preprocess_data, mock_collect_sentiment_data):
        """Test generation of trading signals."""
        # Setup mock returns
        mock_sentiment_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'source': ['twitter'],
            'sentiment_score': [0.5]
        })
        mock_collect_sentiment_data.return_value = mock_sentiment_data
        
        mock_processed_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'close': [40000],
            'sentiment_score': [0.5]
        })
        mock_preprocess_data.return_value = mock_processed_data
        
        mock_signals = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'signal': [1],
            'position_size': [0.1]
        })
        mock_generate_signals.return_value = mock_signals
        
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        # Call the generate_trading_signals method
        market_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'open': [39000],
            'high': [41000],
            'low': [38000],
            'close': [40000],
            'volume': [1000]
        })
        symbols = ['BTC/USD']
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        result = service.generate_trading_signals(market_data, symbols, start_date, end_date)
        
        # Check that the collect_sentiment_data method was called with the correct parameters
        mock_collect_sentiment_data.assert_called_once_with(symbols, start_date, end_date, False)
        
        # Check that the preprocess_data method was called with the correct parameters
        mock_preprocess_data.assert_called_once_with(market_data, mock_sentiment_data)
        
        # Check that the generate_signals method was called with the correct parameters
        mock_generate_signals.assert_called_once_with(mock_processed_data)
        
        # Check that the result is the expected DataFrame
        assert result is mock_signals
    
    @patch('ai_trading_agent.sentiment_analysis.strategy.SentimentStrategy.generate_orders')
    def test_generate_orders(self, mock_generate_orders):
        """Test generation of orders from signals."""
        # Setup mock returns
        mock_orders = [
            MagicMock(symbol='BTC/USD', side='buy', quantity=0.1),
            MagicMock(symbol='ETH/USD', side='sell', quantity=0.5)
        ]
        mock_generate_orders.return_value = mock_orders
        
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        # Call the generate_orders method
        signals = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USD'],
            'signal': [1],
            'position_size': [0.1]
        })
        timestamp = datetime.now()
        current_positions = {'BTC/USD': {'side': 'long', 'quantity': 0.1}}
        
        result = service.generate_orders(signals, timestamp, current_positions)
        
        # Check that the generate_orders method was called with the correct parameters
        mock_generate_orders.assert_called_once_with(signals, timestamp, current_positions)
        
        # Check that the result is the expected list of orders
        assert result is mock_orders
    
    @patch('ai_trading_agent.sentiment_analysis.strategy.SentimentStrategy.update_trade_history')
    def test_update_trade_history(self, mock_update_trade_history):
        """Test updating trade history."""
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {},
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        # Call the update_trade_history method
        trade_result = {'symbol': 'BTC/USD', 'profit': 100}
        service.update_trade_history(trade_result)
        
        # Check that the update_trade_history method was called with the correct parameters
        mock_update_trade_history.assert_called_once_with(trade_result)
    
    @patch('ai_trading_agent.sentiment_analysis.service.SentimentAnalysisService.collect_sentiment_data')
    def test_get_sentiment_summary(self, mock_collect_sentiment_data):
        """Test getting a summary of sentiment data."""
        # Setup mock returns
        mock_sentiment_data = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now()],
            'symbol': ['BTC/USD', 'BTC/USD'],
            'source': ['twitter', 'reddit'],
            'sentiment_score': [0.5, 0.3]
        })
        mock_collect_sentiment_data.return_value = mock_sentiment_data
        
        config = {
            'data_collection': {'collectors': {}},
            'nlp_processing': {},
            'strategy': {
                'source_weights': {
                    'twitter': 0.6,
                    'reddit': 0.4
                }
            },
            'output_dir': self.test_output_dir
        }
        service = SentimentAnalysisService(config)
        
        # Call the get_sentiment_summary method
        symbols = ['BTC/USD']
        days = 7
        
        result = service.get_sentiment_summary(symbols, days)
        
        # Check that the collect_sentiment_data method was called with the correct parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        mock_collect_sentiment_data.assert_called_once()
        
        # Check that the result is a dictionary with the expected structure
        assert isinstance(result, dict)
        assert 'BTC/USD' in result
        assert 'overall_score' in result['BTC/USD']
        assert 'source_sentiment' in result['BTC/USD']
        assert 'twitter' in result['BTC/USD']['source_sentiment']
        assert 'reddit' in result['BTC/USD']['source_sentiment']
