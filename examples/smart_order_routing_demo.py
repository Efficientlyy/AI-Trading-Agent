#!/usr/bin/env python
"""Smart Order Routing (SOR) Demo

This script demonstrates the Smart Order Routing (SOR) algorithm using multiple
mock exchange connectors. It shows how the algorithm analyzes orderbook liquidity
across exchanges and routes orders to achieve the best overall execution.

The demo simulates:
1. Multiple exchanges with different prices and liquidity profiles
2. Order routing across these exchanges based on best available prices
3. Fee-aware routing decisions
4. Execution progress monitoring
5. Comparison of SOR vs. single-exchange execution

This is useful for understanding how Smart Order Routing can improve execution
quality in a multi-exchange trading environment.
"""

import asyncio
import logging
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Mapping

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.logging import get_logger
from src.execution.exchange.mock import MockExchangeConnector
from src.execution.exchange.base import BaseExchangeConnector
from src.execution.interface import ExchangeInterface
from src.execution.algorithms.smart_order_routing import SmartOrderRouter
from src.models.order import OrderSide
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("examples.smart_order_routing_demo")


def create_mock_exchanges() -> List[MockExchangeConnector]:
    """Create multiple mock exchange connectors with different price/liquidity profiles."""
    
    # Exchange A - Higher prices but good liquidity
    exchange_a = MockExchangeConnector(
        exchange_id="exchange_a",
        is_paper_trading=True,
        latency_ms=50,
        fill_probability=0.98,
        price_volatility=0.001
    )
    
    # Exchange B - Lower prices but with less liquidity
    exchange_b = MockExchangeConnector(
        exchange_id="exchange_b",
        is_paper_trading=True,
        latency_ms=150,
        fill_probability=0.90,
        price_volatility=0.003
    )
    
    # Exchange C - Medium prices and medium liquidity
    exchange_c = MockExchangeConnector(
        exchange_id="exchange_c",
        is_paper_trading=True,
        latency_ms=100,
        fill_probability=0.95,
        price_volatility=0.002
    )
    
    # Customize prices to create a more interesting demo
    exchange_a.prices = {
        "BTC/USDT": Decimal("50100"),
        "ETH/USDT": Decimal("3020"),
        "SOL/USDT": Decimal("105")
    }
    
    exchange_b.prices = {
        "BTC/USDT": Decimal("49900"),
        "ETH/USDT": Decimal("2980"),
        "SOL/USDT": Decimal("95")
    }
    
    exchange_c.prices = {
        "BTC/USDT": Decimal("50000"),
        "ETH/USDT": Decimal("3000"),
        "SOL/USDT": Decimal("100")
    }
    
    # Set different fee structures
    exchange_a.exchange_info["fees"] = {
        "maker": Decimal("0.0008"),  # 0.08%
        "taker": Decimal("0.0010")   # 0.10%
    }
    
    exchange_b.exchange_info["fees"] = {
        "maker": Decimal("0.0010"),  # 0.10%
        "taker": Decimal("0.0015")   # 0.15%
    }
    
    exchange_c.exchange_info["fees"] = {
        "maker": Decimal("0.0005"),  # 0.05%
        "taker": Decimal("0.0008")   # 0.08%
    }
    
    return [exchange_a, exchange_b, exchange_c]


async def execute_on_single_exchange(exchange_interface, exchange_id, symbol, side, quantity):
    """Execute an order on a single exchange for comparison."""
    logger.info(f"Executing {quantity} {symbol} {side} on {exchange_id} only")
    
    market_price = await exchange_interface.get_market_price(exchange_id, symbol)
    if market_price is None:
        logger.error(f"Failed to get market price on {exchange_id}")
        return
    
    logger.info(f"Market price on {exchange_id}: {market_price}")
    
    # Create a market order
    start_time = time.time()
    success, order, error = await exchange_interface.create_market_order(
        exchange_id=exchange_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        client_order_id=f"single_{int(time.time())}"
    )
    
    if not success or order is None:
        logger.error(f"Failed to create order on {exchange_id}: {error}")
        return
    
    logger.info(f"Order created: {order.exchange_order_id}")
    
    # Wait for a short time to let the order process
    await asyncio.sleep(0.5)
    
    # Get order status
    if order.exchange_order_id is None:
        logger.error("Order ID is None, cannot get status")
        return
        
    status = await exchange_interface.get_order_status(
        exchange_id=exchange_id,
        order_id=order.exchange_order_id,
        symbol=symbol
    )
    
    execution_time = time.time() - start_time
    
    if status:
        logger.info(f"Order executed in {execution_time:.2f} seconds")
        logger.info(f"Status: {status.get('status')}")
        logger.info(f"Filled: {status.get('filled_quantity')} at {status.get('average_price')}")
        return {
            "exchange": exchange_id,
            "filled_quantity": status.get('filled_quantity', 0),
            "average_price": status.get('average_price', 0),
            "execution_time": execution_time
        }
    else:
        logger.error("Failed to get order status")
        return None


async def run_sor_demo():
    """Run a demonstration of the Smart Order Router."""
    logger.info("Starting Smart Order Routing Demo")
    
    # Create mock exchange connectors
    exchange_connectors = create_mock_exchanges()
    
    # Initialize all exchange connectors
    for connector in exchange_connectors:
        await connector.initialize()
    
    # Build exchange map - use Mapping for covariance
    # Note: In a real application, you would use different types of exchange connectors
    exchange_map: Dict[str, BaseExchangeConnector] = {}
    for conn in exchange_connectors:
        exchange_map[conn.exchange_id] = conn
    
    # Create the exchange interface
    exchange_interface = ExchangeInterface(exchange_map)
    
    # Create the smart order router
    sor = SmartOrderRouter(exchange_interface)
    
    try:
        # Demo setup
        symbol = "BTC/USDT"
        side = OrderSide.BUY
        quantity = 1.0  # BTC
        
        # Display available exchanges and their prices
        logger.info("Exchange Information:")
        for exchange_id in exchange_map.keys():
            price = await exchange_interface.get_market_price(exchange_id, symbol)
            connector = exchange_interface.get_connector(exchange_id)
            if connector and isinstance(connector, MockExchangeConnector):
                fees = connector.exchange_info["fees"]
                logger.info(f"  {exchange_id}: Price = {price}, Taker Fee = {float(fees['taker'])*100}%")
            else:
                logger.info(f"  {exchange_id}: Price = {price}")
        
        logger.info("\n")
        logger.info("=== Part 1: Using SOR to execute an order ===")
        
        # Execute using SOR
        logger.info(f"Routing order for {quantity} {symbol} {side} across all exchanges")
        success, job_id, error = await sor.route_order(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            consider_fees=True,
            execution_style="aggressive"
        )
        
        if not success or job_id is None:
            logger.error(f"Failed to start SOR: {error}")
            return
        
        logger.info(f"SOR job started with ID: {job_id}")
        
        # Monitor the SOR execution
        completed = False
        start_time = time.time()
        
        while not completed and (time.time() - start_time) < 30:
            job_status = sor.get_job_status(job_id)
            
            if job_status is None:
                logger.error("Failed to get SOR job status")
                break
            
            logger.info(f"SOR Progress: {job_status['percent_complete']:.2f}% complete")
            logger.info(f"Status: {job_status['status']}")
            
            completed = job_status["is_completed"]
            
            if completed:
                # Log final results
                logger.info("SOR execution completed")
                logger.info(f"Total filled: {job_status['filled_quantity']} {symbol}")
                logger.info(f"Average price: {job_status['average_execution_price']}")
                
                # Show execution across exchanges
                logger.info("Execution breakdown by exchange:")
                orders_by_exchange = {}
                
                for order in job_status["orders"]:
                    exchange_id = order["exchange_id"]
                    if exchange_id not in orders_by_exchange:
                        orders_by_exchange[exchange_id] = []
                    orders_by_exchange[exchange_id].append(order)
                
                for exchange_id, orders in orders_by_exchange.items():
                    total_filled = sum(order["filled_quantity"] for order in orders)
                    total_value = sum(order["filled_quantity"] * order["average_price"] for order in orders if order["filled_quantity"] > 0)
                    avg_price = total_value / total_filled if total_filled > 0 else 0
                    
                    percentage = (total_filled / job_status['filled_quantity']) * 100 if job_status['filled_quantity'] > 0 else 0
                    
                    logger.info(f"  {exchange_id}: {total_filled} {symbol} ({percentage:.1f}%) at avg price {avg_price:.2f}")
            else:
                await asyncio.sleep(1)
        
        sor_result = {
            "filled_quantity": job_status.get("filled_quantity", 0) if job_status else 0,
            "average_price": job_status.get("average_execution_price", 0) if job_status else 0,
            "execution_time": time.time() - start_time
        }
        
        if not completed:
            logger.warning("SOR execution did not complete within the expected time")
        
        logger.info("\n")
        logger.info("=== Part 2: Comparing SOR to single-exchange execution ===")
        
        # Execute on individual exchanges for comparison
        single_results = {}
        for exchange_id in exchange_map.keys():
            result = await execute_on_single_exchange(
                exchange_interface, 
                exchange_id, 
                symbol, 
                side, 
                quantity
            )
            if result:
                single_results[exchange_id] = result
        
        # Compare results
        logger.info("\n")
        logger.info("=== Execution Comparison Results ===")
        
        if sor_result.get("filled_quantity", 0) > 0:
            logger.info(f"SOR: {sor_result['filled_quantity']} {symbol} at {sor_result['average_price']} (in {sor_result['execution_time']:.2f}s)")
            
            # Calculate total cost (with fees)
            sor_cost = sor_result['filled_quantity'] * sor_result['average_price']
            logger.info(f"SOR Total Cost: {sor_cost:.2f} USDT")
            
            # Compare with single exchanges
            logger.info("\nSingle exchange results:")
            for exchange_id, result in single_results.items():
                if result.get("filled_quantity", 0) > 0:
                    price = result["average_price"]
                    cost = result['filled_quantity'] * price
                    
                    price_diff = ((price - sor_result['average_price']) / sor_result['average_price']) * 100
                    cost_diff = ((cost - sor_cost) / sor_cost) * 100
                    
                    logger.info(f"{exchange_id}: {result['filled_quantity']} {symbol} at {price} (in {result['execution_time']:.2f}s)")
                    logger.info(f"  Price difference vs SOR: {price_diff:+.2f}%")
                    logger.info(f"  Cost difference vs SOR: {cost_diff:+.2f}%")
        
        logger.info("\nSOR Advantages:")
        logger.info("1. Optimized execution across multiple venues")
        logger.info("2. Reduced market impact by splitting orders")
        logger.info("3. Takes advantage of price disparities between exchanges")
        logger.info("4. Fee-aware routing decisions")
        logger.info("5. More resilient to temporary liquidity gaps")
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
    
    finally:
        # Shut down all exchange connectors
        for connector in exchange_connectors:
            await connector.shutdown()
        
        logger.info("Smart Order Routing Demo completed")


async def run_exchange_comparison_demo():
    """Run a more detailed exchange comparison demo."""
    logger.info("Starting Exchange Comparison Demo")
    
    # Create mock exchange connectors
    exchange_connectors = create_mock_exchanges()
    
    # Initialize all exchange connectors
    for connector in exchange_connectors:
        await connector.initialize()
    
    # Build exchange map - use Mapping for covariance
    # Note: In a real application, you would use different types of exchange connectors
    exchange_map: Dict[str, BaseExchangeConnector] = {}
    for conn in exchange_connectors:
        exchange_map[conn.exchange_id] = conn
    
    # Create the exchange interface
    exchange_interface = ExchangeInterface(exchange_map)
    
    try:
        # Display orderbook data for each exchange
        symbol = "BTC/USDT"
        logger.info(f"Orderbook Comparison for {symbol}:")
        
        for exchange_id in exchange_map.keys():
            orderbook = await exchange_interface.get_orderbook_snapshot(exchange_id, symbol, depth=5)
            
            if not orderbook:
                logger.warning(f"Failed to get orderbook for {exchange_id}")
                continue
            
            asks = orderbook.get("asks", [])
            bids = orderbook.get("bids", [])
            
            logger.info(f"\n{exchange_id} Orderbook:")
            
            logger.info("  Asks (Sell Orders):")
            for i, (price, size) in enumerate(asks[:5]):
                logger.info(f"    {i+1}: {size} @ {price}")
            
            logger.info("  Bids (Buy Orders):")
            for i, (price, size) in enumerate(bids[:5]):
                logger.info(f"    {i+1}: {size} @ {price}")
            
            # Calculate and display total liquidity
            ask_liquidity = sum(float(size) for _, size in asks[:5])
            bid_liquidity = sum(float(size) for _, size in bids[:5])
            
            logger.info(f"  Total Ask Liquidity (Top 5 Levels): {ask_liquidity:.4f} BTC")
            logger.info(f"  Total Bid Liquidity (Top 5 Levels): {bid_liquidity:.4f} BTC")
            
            # Get fee information
            connector = exchange_interface.get_connector(exchange_id)
            if connector and isinstance(connector, MockExchangeConnector):
                fees = connector.exchange_info["fees"]
                logger.info(f"  Maker Fee: {float(fees['maker'])*100:.3f}%")
                logger.info(f"  Taker Fee: {float(fees['taker'])*100:.3f}%")
            else:
                logger.info("  Fee information not available")
    
    except Exception as e:
        logger.error(f"Error during exchange comparison demo: {str(e)}")
    
    finally:
        # Shut down all exchange connectors
        for connector in exchange_connectors:
            await connector.shutdown()
        
        logger.info("Exchange Comparison Demo completed")


async def run_demo():
    """Run the full Smart Order Routing demo."""
    await run_exchange_comparison_demo()
    logger.info("\n" + "="*50 + "\n")
    await run_sor_demo()


if __name__ == "__main__":
    asyncio.run(run_demo()) 