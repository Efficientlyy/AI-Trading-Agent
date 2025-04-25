"""
Test module for Order initialization.

This module tests the initialization of the Order class to help debug validation errors.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, OrderStatus


def test_order_initialization():
    """Test basic Order initialization."""
    print("\nTesting Order initialization...")
    
    # Create an order with float values
    try:
        order1 = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
            created_at=datetime.now()
        )
        print(f"Order 1 created successfully: {order1}")
        print(f"Order 1 attributes: {dir(order1)}")
        print(f"Order 1 type: {order1.type}")
        print(f"Order 1 order_type: {order1.order_type}")
        
        assert order1.symbol == "BTC"
        assert order1.side == OrderSide.BUY
        assert order1.type == OrderType.MARKET
        assert order1.quantity == 0.1
        assert order1.price == 50000.0
    except Exception as e:
        print(f"Error creating Order 1: {e}")
        assert False, f"Order creation with float values failed: {e}"
    
    # Try with Decimal values
    print("\nTesting Order with Decimal values...")
    try:
        order2 = Order(
            symbol="ETH",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("3000"),
            created_at=datetime.now()
        )
        print(f"Order 2 created successfully: {order2}")
        print(f"Order 2 quantity type: {type(order2.quantity)}")
        print(f"Order 2 price type: {type(order2.price)}")
        decimal_works = True
    except Exception as e:
        print(f"Error with Decimal values: {e}")
        decimal_works = False
    
    # We'll check if Decimal works or not, but won't assert either way
    print(f"Decimal values work: {decimal_works}")
    
    # Try with order_type instead of type
    print("\nTesting Order with order_type instead of type...")
    try:
        order3 = Order(
            symbol="LTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=100.0,
            created_at=datetime.now()
        )
        print(f"Order 3 created successfully: {order3}")
        print(f"Order 3 type: {order3.type}")
        print(f"Order 3 order_type: {order3.order_type}")
        order_type_works = True
    except Exception as e:
        print(f"Error with order_type: {e}")
        order_type_works = False
    
    print(f"order_type alias works: {order_type_works}")
    
    # Try with timestamp instead of created_at
    print("\nTesting Order with timestamp instead of created_at...")
    try:
        order4 = Order(
            symbol="XRP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.1,
            price=1.0,
            timestamp=datetime.now()
        )
        print(f"Order 4 created successfully: {order4}")
        timestamp_works = True
    except Exception as e:
        print(f"Error with timestamp: {e}")
        timestamp_works = False
    
    print(f"timestamp instead of created_at works: {timestamp_works}")


if __name__ == "__main__":
    test_order_initialization()
