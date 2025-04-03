"""
Integration tests for the Bitvavo integration.
"""

import unittest
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the BitvavoConnector class
from src.execution.exchange.bitvavo import BitvavoConnector

class TestBitvavoIntegration(unittest.TestCase):
    """Integration tests for the Bitvavo integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Load test configuration
        cls.config_path = os.path.join('config', 'test', 'bitvavo_test_config.json')
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.config_path), exist_ok=True)
        
        # Check if config file exists
        if not os.path.exists(cls.config_path):
            # Create default test config
            cls.config = {
                "test_mode": True,
                "api_credentials": {
                    "key": os.environ.get("BITVAVO_TEST_API_KEY", ""),
                    "secret": os.environ.get("BITVAVO_TEST_API_SECRET", "")
                },
                "test_pairs": ["BTC/EUR", "ETH/EUR", "XRP/EUR"],
                "test_order_size": {
                    "BTC/EUR": 0.001,
                    "ETH/EUR": 0.01,
                    "XRP/EUR": 10
                },
                "max_test_order_value_eur": 50
            }
            
            # Save config
            with open(cls.config_path, 'w') as f:
                json.dump(cls.config, f, indent=4)
                
            logger.info(f"Created default test config at {cls.config_path}")
        else:
            # Load existing config
            with open(cls.config_path, 'r') as f:
                cls.config = json.load(f)
                
            logger.info(f"Loaded test config from {cls.config_path}")
        
        # Check if API credentials are available
        if not cls.config["api_credentials"]["key"] or not cls.config["api_credentials"]["secret"]:
            logger.warning("API credentials not found in config or environment variables")
            logger.warning("Some tests will be skipped")
            cls.skip_api_tests = True
        else:
            cls.skip_api_tests = False
            
            # Create connector
            cls.connector = BitvavoConnector(
                api_key=cls.config["api_credentials"]["key"],
                api_secret=cls.config["api_credentials"]["secret"]
            )
            
            # Initialize connector
            success = cls.connector.initialize()
            if not success:
                logger.warning("Failed to initialize connector")
                cls.skip_api_tests = True
    
    def setUp(self):
        """Set up test fixtures before each test."""
        if self.skip_api_tests:
            self.skipTest("API credentials not available")
    
    def test_get_time(self):
        """Test getting server time."""
        result = self.connector.get_time()
        
        # Verify result
        self.assertIn("time", result)
        self.assertIsInstance(result["time"], int)
        
        # Log result
        logger.info(f"Server time: {datetime.fromtimestamp(result['time'] / 1000)}")
    
    def test_get_markets(self):
        """Test getting markets."""
        result = self.connector.get_markets()
        
        # Verify result
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Verify test pairs are available
        markets = [self.connector.normalize_symbol(pair) for pair in self.config["test_pairs"]]
        available_markets = [market["market"] for market in result]
        
        for market in markets:
            self.assertIn(market, available_markets, f"Market {market} not available")
        
        # Log result
        logger.info(f"Found {len(result)} markets")
        logger.info(f"Test pairs available: {all(market in available_markets for market in markets)}")
    
    def test_get_ticker(self):
        """Test getting ticker."""
        for pair in self.config["test_pairs"]:
            result = self.connector.get_ticker(pair)
            
            # Verify result
            self.assertIn("market", result)
            self.assertIn("price", result)
            
            # Log result
            logger.info(f"Ticker for {pair}: {result['price']}")
    
    def test_get_order_book(self):
        """Test getting order book."""
        for pair in self.config["test_pairs"]:
            result = self.connector.get_order_book(pair)
            
            # Verify result
            self.assertIn("market", result)
            self.assertIn("bids", result)
            self.assertIn("asks", result)
            
            # Verify bids and asks
            self.assertIsInstance(result["bids"], list)
            self.assertIsInstance(result["asks"], list)
            
            # Log result
            logger.info(f"Order book for {pair}: {len(result['bids'])} bids, {len(result['asks'])} asks")
    
    def test_get_trades(self):
        """Test getting trades."""
        for pair in self.config["test_pairs"]:
            result = self.connector.get_trades(pair, limit=10)
            
            # Verify result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 10)
            
            if len(result) > 0:
                # Verify trade fields
                trade = result[0]
                self.assertIn("id", trade)
                self.assertIn("timestamp", trade)
                self.assertIn("amount", trade)
                self.assertIn("price", trade)
                self.assertIn("side", trade)
            
            # Log result
            logger.info(f"Trades for {pair}: {len(result)} trades")
    
    def test_get_candles(self):
        """Test getting candles."""
        for pair in self.config["test_pairs"]:
            result = self.connector.get_candles(pair, interval="1h", limit=10)
            
            # Verify result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 10)
            
            if len(result) > 0:
                # Verify candle fields
                candle = result[0]
                self.assertEqual(len(candle), 6)  # [timestamp, open, high, low, close, volume]
            
            # Log result
            logger.info(f"Candles for {pair}: {len(result)} candles")
    
    @unittest.skipIf(not os.environ.get("BITVAVO_TEST_TRADING", False), "Trading tests disabled")
    def test_create_and_cancel_order(self):
        """Test creating and canceling an order."""
        # Use BTC/EUR for testing
        pair = "BTC/EUR"
        
        # Get current price
        ticker = self.connector.get_ticker(pair)
        price = float(ticker["price"])
        
        # Calculate test order size and price
        size = self.config["test_order_size"][pair]
        bid_price = price * 0.9  # 10% below current price
        
        # Create limit buy order
        order = self.connector.create_order(
            symbol=pair,
            side="buy",
            order_type="limit",
            amount=size,
            price=bid_price
        )
        
        # Verify order
        self.assertIn("orderId", order)
        self.assertEqual(order["market"], self.connector.normalize_symbol(pair))
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["orderType"], "limit")
        
        # Log order
        logger.info(f"Created order: {order['orderId']}")
        
        # Cancel order
        result = self.connector.cancel_order(pair, order["orderId"])
        
        # Verify result
        self.assertIn("orderId", result)
        
        # Log result
        logger.info(f"Canceled order: {result['orderId']}")
    
    def test_symbol_conversion(self):
        """Test symbol conversion."""
        # Test pairs
        pairs = [
            ("BTC/EUR", "BTC-EUR"),
            ("ETH/BTC", "ETH-BTC"),
            ("XRP/USD", "XRP-USD")
        ]
        
        for standard, normalized in pairs:
            # Test normalization
            result = self.connector.normalize_symbol(standard)
            self.assertEqual(result, normalized)
            
            # Test standardization
            result = self.connector.standardize_symbol(normalized)
            self.assertEqual(result, standard)

if __name__ == "__main__":
    unittest.main()