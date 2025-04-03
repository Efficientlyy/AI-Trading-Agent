"""
Unit tests for the BitvavoConnector class.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import time
from datetime import datetime

from src.execution.exchange.bitvavo import BitvavoConnector

class TestBitvavoConnector(unittest.TestCase):
    """Test cases for the BitvavoConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.connector = BitvavoConnector(api_key=self.api_key, api_secret=self.api_secret)
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertEqual(self.connector.api_key, self.api_key)
        self.assertEqual(self.connector.api_secret, self.api_secret)
        self.assertEqual(self.connector.base_url, "https://api.bitvavo.com/v2")
        self.assertEqual(self.connector.session.headers["Bitvavo-Access-Key"], self.api_key)
        self.assertEqual(self.connector.rate_limit_remaining, 1000)
    
    def test_generate_signature(self):
        """Test signature generation."""
        timestamp = 1617184835000
        method = "GET"
        url_path = "/v2/time"
        
        signature = self.connector._generate_signature(timestamp, method, url_path)
        
        # Verify signature is a string with correct length (SHA-256 hex digest)
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)
    
    @patch('requests.Session.request')
    def test_request_success(self, mock_request):
        """Test successful API request."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test_data"}
        mock_response.headers = {
            "Bitvavo-Ratelimit-Remaining": "999",
            "Bitvavo-Ratelimit-ResetAt": str(int(time.time() * 1000) + 60000)
        }
        mock_request.return_value = mock_response
        
        # Make request
        result = self.connector._request("GET", "/test")
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/test")
        
        # Verify result
        self.assertEqual(result, {"success": True, "data": "test_data"})
        
        # Verify rate limit tracking
        self.assertEqual(self.connector.rate_limit_remaining, 999)
    
    @patch('requests.Session.request')
    def test_request_error(self, mock_request):
        """Test API request with error response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "Bad request"}'
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Make request
        result = self.connector._request("GET", "/test")
        
        # Verify result contains error
        self.assertIn("error", result)
    
    @patch('requests.Session.request')
    def test_get_time(self, mock_request):
        """Test get_time method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"time": 1617184835000}
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Call method
        result = self.connector.get_time()
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/time")
        
        # Verify result
        self.assertEqual(result, {"time": 1617184835000})
    
    @patch('requests.Session.request')
    def test_get_markets(self, mock_request):
        """Test get_markets method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"market": "BTC-EUR", "status": "trading"},
            {"market": "ETH-EUR", "status": "trading"}
        ]
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Call method
        result = self.connector.get_markets()
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/markets")
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["market"], "BTC-EUR")
    
    @patch('requests.Session.request')
    def test_get_ticker(self, mock_request):
        """Test get_ticker method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"market": "BTC-EUR", "price": "50000"}
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Call method
        result = self.connector.get_ticker("BTC/EUR")
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/ticker/price")
        self.assertEqual(kwargs["params"], {"market": "BTC-EUR"})
        
        # Verify result
        self.assertEqual(result["market"], "BTC-EUR")
        self.assertEqual(result["price"], "50000")
    
    @patch('requests.Session.request')
    def test_get_order_book(self, mock_request):
        """Test get_order_book method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "market": "BTC-EUR",
            "bids": [["50000", "1.0"]],
            "asks": [["51000", "1.0"]]
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Call method
        result = self.connector.get_order_book("BTC/EUR", depth=10)
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/orderbook")
        self.assertEqual(kwargs["params"], {"market": "BTC-EUR", "depth": 10})
        
        # Verify result
        self.assertEqual(result["market"], "BTC-EUR")
        self.assertEqual(len(result["bids"]), 1)
        self.assertEqual(len(result["asks"]), 1)
    
    @patch('requests.Session.request')
    def test_create_order(self, mock_request):
        """Test create_order method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": "12345",
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "limit",
            "amount": "1.0",
            "price": "50000"
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        # Call method
        result = self.connector.create_order(
            symbol="BTC/EUR",
            side="buy",
            order_type="limit",
            amount=1.0,
            price=50000
        )
        
        # Verify request was made correctly
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["url"], "https://api.bitvavo.com/v2/order")
        self.assertEqual(kwargs["json"], {
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "limit",
            "amount": "1.0",
            "price": "50000"
        })
        
        # Verify result
        self.assertEqual(result["orderId"], "12345")
        self.assertEqual(result["market"], "BTC-EUR")
    
    def test_normalize_symbol(self):
        """Test normalize_symbol method."""
        # Test with standard format
        result = self.connector.normalize_symbol("BTC/EUR")
        self.assertEqual(result, "BTC-EUR")
        
        # Test with already normalized format
        result = self.connector.normalize_symbol("BTC-EUR")
        self.assertEqual(result, "BTC-EUR")
    
    def test_standardize_symbol(self):
        """Test standardize_symbol method."""
        # Test with normalized format
        result = self.connector.standardize_symbol("BTC-EUR")
        self.assertEqual(result, "BTC/EUR")
        
        # Test with already standard format
        result = self.connector.standardize_symbol("BTC/EUR")
        self.assertEqual(result, "BTC/EUR")

if __name__ == "__main__":
    unittest.main()