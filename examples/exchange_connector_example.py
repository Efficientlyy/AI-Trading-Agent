#!/usr/bin/env python3
"""
Exchange Connector Example.

This script demonstrates how to use the mock exchange connector to interact with
a simulated exchange. It shows how to:
1. Initialize the exchange connector
2. Get exchange information and account balances
3. Retrieve market data (tickers, orderbooks)
4. Create and cancel orders
5. Monitor order status and trades
"""

import asyncio
import sys
import os
import random
from decimal import Decimal
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Change the working directory to the project root to make config loading work
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from src.execution.exchange.mock import MockExchangeConnector


async def print_exchange_info(connector):
    """Print information about the exchange."""
    print("\n=== Exchange Information ===")
    
    # Get exchange info
    exchange_info = await connector.get_exchange_info()
    print(f"Exchange: {exchange_info['name']}")
    print(f"Available symbols: {', '.join(exchange_info['symbols'])}")
    print(f"Timeframes: {', '.join(exchange_info['timeframes'])}")
    print(f"Maker fee: {exchange_info['fees']['maker']}")
    print(f"Taker fee: {exchange_info['fees']['taker']}")


async def print_account_balances(connector):
    """Print account balances."""
    print("\n=== Account Balances ===")
    
    # Get account balances
    balances = await connector.get_account_balance()
    
    # Print balances with non-zero amounts
    for asset, amount in sorted(balances.items()):
        if amount > Decimal("0"):
            print(f"{asset}: {amount}")


async def print_market_data(connector, symbol):
    """Print market data for a symbol."""
    print(f"\n=== Market Data for {symbol} ===")
    
    # Get ticker information
    ticker = await connector.get_ticker(symbol)
    if ticker:
        print(f"Current price: {ticker['price']}")
        print(f"Bid: {ticker['bid']}")
        print(f"Ask: {ticker['ask']}")
        print(f"24h Volume: {ticker['volume']}")
    else:
        print(f"No ticker data available for {symbol}")
    
    # Get order book
    orderbook = await connector.get_orderbook(symbol, limit=5)
    if orderbook and 'bids' in orderbook and 'asks' in orderbook:
        print("\nOrder Book (top 5 levels):")
        
        print("Bids:")
        for price, size in orderbook['bids'][:5]:
            print(f"  {price:.2f} - {size:.4f}")
        
        print("Asks:")
        for price, size in orderbook['asks'][:5]:
            print(f"  {price:.2f} - {size:.4f}")
    else:
        print(f"No order book data available for {symbol}")


async def place_and_monitor_orders(connector, symbol):
    """Place various order types and monitor their status."""
    print(f"\n=== Placing Orders for {symbol} ===")
    
    # Get the current price
    ticker = await connector.get_ticker(symbol)
    if not ticker:
        print(f"Cannot get current price for {symbol}")
        return
    
    current_price = ticker['price']
    print(f"Current price: {current_price}")
    
    # Create a market buy order
    market_buy_order = Order(
        exchange=connector.exchange_id,
        symbol=symbol,
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=0.01,  # Small BTC quantity
        time_in_force=TimeInForce.GTC
    )
    
    print("\nPlacing market buy order...")
    success, order_id, error = await connector.create_order(market_buy_order)
    
    if success:
        print(f"Market buy order placed successfully. ID: {order_id}")
        
        # Get the order details
        await asyncio.sleep(0.5)  # Wait for order processing
        order_details = await connector.get_order(order_id, symbol)
        
        if order_details:
            print(f"Order status: {order_details['status']}")
            print(f"Filled quantity: {order_details['filled_quantity']}")
            print(f"Average fill price: {order_details['average_fill_price']}")
    else:
        print(f"Failed to place market buy order: {error}")
    
    # Create a limit sell order above the current price
    limit_price = current_price * Decimal("1.02")  # 2% above current price
    
    limit_sell_order = Order(
        exchange=connector.exchange_id,
        symbol=symbol,
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=0.01,
        price=float(limit_price),
        time_in_force=TimeInForce.GTC
    )
    
    print("\nPlacing limit sell order...")
    success, limit_order_id, error = await connector.create_order(limit_sell_order)
    
    if success:
        print(f"Limit sell order placed successfully. ID: {limit_order_id}")
        print(f"Limit price: {limit_price} (2% above current price)")
        
        # Get initial order details
        order_details = await connector.get_order(limit_order_id, symbol)
        print(f"Order status: {order_details['status']}")
        
        # Cancel the limit order after a delay
        await asyncio.sleep(2)
        print("\nCancelling limit sell order...")
        cancel_success, cancel_error = await connector.cancel_order(limit_order_id, symbol)
        
        if cancel_success:
            print("Limit order cancelled successfully")
            
            # Get the updated order details
            order_details = await connector.get_order(limit_order_id, symbol)
            print(f"Updated order status: {order_details['status']}")
        else:
            print(f"Failed to cancel limit order: {cancel_error}")
    else:
        print(f"Failed to place limit sell order: {error}")
    
    # Get all open orders
    await asyncio.sleep(0.5)
    open_orders = await connector.get_open_orders(symbol)
    print(f"\nOpen orders for {symbol}: {len(open_orders)}")
    
    for order in open_orders:
        print(f"  Order ID: {order['id']}, Type: {order['type']}, Side: {order['side']}, Status: {order['status']}")


async def print_trade_history(connector, symbol):
    """Print trade history for a symbol."""
    print(f"\n=== Trade History for {symbol} ===")
    
    # Get trade history
    trades = await connector.get_trade_history(symbol, limit=5)
    
    if trades:
        print(f"Recent trades (last {len(trades)}):")
        for trade in trades:
            print(f"  Time: {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Side: {trade['side']}")
            print(f"  Price: {trade['price']}")
            print(f"  Quantity: {trade['quantity']}")
            print(f"  Fee: {trade['fee']} {trade['fee_currency']}")
            print()
    else:
        print(f"No trade history available for {symbol}")


async def simulate_price_changes(connector, symbol, duration_seconds=15):
    """Simulate price changes over time."""
    print(f"\n=== Simulating Price Changes for {symbol} ===")
    print(f"Monitoring prices for {duration_seconds} seconds...")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    initial_price = (await connector.get_ticker(symbol))['price']
    print(f"Initial price: {initial_price}")
    
    while datetime.now() < end_time:
        await asyncio.sleep(1)
        ticker = await connector.get_ticker(symbol)
        price = ticker['price']
        change_pct = (price - initial_price) / initial_price * 100
        print(f"Current price: {price} ({change_pct:+.2f}%)")


async def main():
    """Run the exchange connector example."""
    print("=== Exchange Connector Example ===")
    
    # Initialize the mock exchange connector
    connector = MockExchangeConnector(
        exchange_id="mockexchange",
        initial_balances={
            "BTC": 0.5,
            "ETH": 5.0,
            "USDT": 10000.0
        },
        initial_prices={
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0
        },
        latency_ms=200,  # 200ms simulated latency
        fill_probability=0.8,  # 80% chance of limit orders filling
        price_volatility=0.005  # 0.5% price volatility
    )
    
    # Initialize the connector
    print("Initializing exchange connector...")
    success = await connector.initialize()
    
    if not success:
        print("Failed to initialize exchange connector")
        return
    
    try:
        # Choose a symbol to work with
        symbol = "BTC/USDT"
        
        # Print exchange information
        await print_exchange_info(connector)
        
        # Print account balances
        await print_account_balances(connector)
        
        # Print market data
        await print_market_data(connector, symbol)
        
        # Place and monitor orders
        await place_and_monitor_orders(connector, symbol)
        
        # Print trade history
        await print_trade_history(connector, symbol)
        
        # Simulate price changes
        await simulate_price_changes(connector, symbol, duration_seconds=10)
        
        # Print final account balances
        await print_account_balances(connector)
        
    finally:
        # Shutdown the connector
        print("\nShutting down exchange connector...")
        await connector.shutdown()
        print("Exchange connector shut down")


if __name__ == "__main__":
    asyncio.run(main()) 