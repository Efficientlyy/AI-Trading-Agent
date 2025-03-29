"""Tests for the Order model."""

import unittest
from unittest import mock

from src.common.datetime_utils import utc_now
from src.models.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce


class TestOrder(unittest.TestCase):
    """Tests for the Order model."""

    def test_order_initialization(self):
        """Test that an order can be initialized with correct values."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )

        # Check order is initialized correctly
        self.assertEqual(order.exchange, "binance")
        self.assertEqual(order.symbol, "BTC/USDT")
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 0.1)
        self.assertEqual(order.price, 50000.0)
        self.assertEqual(order.time_in_force, TimeInForce.GTC)  # Default time in force
        self.assertEqual(order.status, OrderStatus.CREATED)  # Default status
        self.assertTrue(isinstance(order.id, str))
        self.assertTrue(order.id)  # ID is not empty
        self.assertIsNotNone(order.created_at)
        self.assertTrue(order.created_at.tzinfo)  # created_at has timezone info

    def test_is_active(self):
        """Test the is_active method."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )

        # Test when order status is CREATED
        order.status = OrderStatus.CREATED
        self.assertFalse(order.is_active())

        # Test when order status is PENDING
        order.status = OrderStatus.PENDING
        self.assertTrue(order.is_active())

        # Test when order status is OPEN
        order.status = OrderStatus.OPEN
        self.assertTrue(order.is_active())

        # Test when order status is PARTIALLY_FILLED
        order.status = OrderStatus.PARTIALLY_FILLED
        self.assertTrue(order.is_active())

        # Test when order status is FILLED
        order.status = OrderStatus.FILLED
        self.assertFalse(order.is_active())

    def test_is_complete(self):
        """Test the is_complete method."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )

        # Test when order status is CREATED
        order.status = OrderStatus.CREATED
        self.assertFalse(order.is_complete())

        # Test when order status is FILLED
        order.status = OrderStatus.FILLED
        self.assertTrue(order.is_complete())

        # Test when order status is CANCELLED
        order.status = OrderStatus.CANCELLED
        self.assertTrue(order.is_complete())

        # Test when order status is REJECTED
        order.status = OrderStatus.REJECTED
        self.assertTrue(order.is_complete())

        # Test when order status is EXPIRED
        order.status = OrderStatus.EXPIRED
        self.assertTrue(order.is_complete())

    def test_remaining_quantity(self):
        """Test the remaining_quantity method."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )

        # No fills yet
        self.assertEqual(order.remaining_quantity(), 0.1)

        # Partially filled
        order.filled_quantity = 0.04
        self.assertAlmostEqual(order.remaining_quantity(), 0.06, places=10)

        # Fully filled
        order.filled_quantity = 0.1
        self.assertEqual(order.remaining_quantity(), 0.0)

        # Over-filled (should not happen but handle gracefully)
        order.filled_quantity = 0.11
        self.assertEqual(order.remaining_quantity(), 0.0)

    def test_fill_percent(self):
        """Test the fill_percent method."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )

        # No fills yet
        self.assertEqual(order.fill_percent(), 0.0)

        # 40% filled
        order.filled_quantity = 0.04
        self.assertEqual(order.fill_percent(), 40.0)

        # Fully filled
        order.filled_quantity = 0.1
        self.assertEqual(order.fill_percent(), 100.0)

        # Edge case: zero quantity
        order.quantity = 0.0
        self.assertEqual(order.fill_percent(), 0.0)

    def test_update_status(self):
        """Test updating order status."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        
        # Mock utc_now to control the timestamp in tests
        mock_now = utc_now()
        
        with mock.patch('src.common.datetime_utils.utc_now', return_value=mock_now):
            order.update_status(
                new_status=OrderStatus.PARTIALLY_FILLED,
                filled_qty=0.05,
                avg_price=50010.0,
                exchange_id="exchange123"
            )
        
        # Check order was updated correctly
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(order.updated_at.replace(microsecond=0), mock_now.replace(microsecond=0))
        self.assertEqual(order.filled_quantity, 0.05)
        self.assertEqual(order.average_fill_price, 50010.0)
        self.assertEqual(order.exchange_order_id, "exchange123")

    def test_update_fill(self):
        """Test updating order fill information."""
        order = Order(
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            status=OrderStatus.OPEN
        )
        
        # Test partial fill
        order.update_fill(0.05, 50100.0)
        self.assertEqual(order.filled_quantity, 0.05)
        self.assertEqual(order.average_fill_price, 50100.0)
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        
        # Test complete fill
        order.update_fill(0.1, 50050.0)
        self.assertEqual(order.filled_quantity, 0.1)
        self.assertEqual(order.average_fill_price, 50050.0)
        self.assertEqual(order.status, OrderStatus.FILLED)
