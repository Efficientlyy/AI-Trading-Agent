#!/usr/bin/env python
"""
Mock Data Testing Script for Trading-Agent System

This script tests the system with mock data and edge cases to ensure
robustness and proper handling of unusual scenarios.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mock_data_test")

# Import required modules
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading import PaperTradingSystem
from paper_trading_extension import EnhancedPaperTradingSystem
from flash_trading_signals_extension import ExtendedFlashTradingSignals
from trading_session_manager import TradingSessionManager

class MockDataTester:
    """Test system components with mock data and edge cases"""
    
    def __init__(self):
        """Initialize mock data tester"""
        self.client = OptimizedMEXCClient()
        self.paper_trading = EnhancedPaperTradingSystem(self.client)
        self.signals = ExtendedFlashTradingSignals()  # Using the extended signals class
        self.session_manager = TradingSessionManager()
        
        logger.info("Mock data tester initialized")
    
    def generate_mock_market_data(self, symbol="BTCUSDC", num_candles=100, volatility=0.02):
        """Generate mock market data
        
        Args:
            symbol: Trading pair symbol
            num_candles: Number of candles to generate
            volatility: Price volatility factor
            
        Returns:
            DataFrame with mock OHLCV data
        """
        logger.info(f"Generating mock market data for {symbol} with {num_candles} candles")
        
        # Start with a reasonable price for the symbol
        if "BTC" in symbol:
            start_price = 105000.0
        elif "ETH" in symbol:
            start_price = 2600.0
        elif "SOL" in symbol:
            start_price = 150.0
        else:
            start_price = 100.0
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=num_candles)
        timestamps = [start_time + timedelta(minutes=i) for i in range(num_candles)]
        
        # Generate prices with random walk
        prices = [start_price]
        for i in range(1, num_candles):
            # Random price change with specified volatility
            change = np.random.normal(0, volatility * prices[-1])
            new_price = max(0.01, prices[-1] + change)  # Ensure price is positive
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i in range(num_candles):
            price = prices[i]
            # Generate open, high, low, close with some randomness
            open_price = price * (1 + np.random.normal(0, 0.002))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.005)))
            close_price = price * (1 + np.random.normal(0, 0.002))
            
            # Generate volume with some randomness
            volume = abs(np.random.normal(10, 5)) * (1 + abs(np.random.normal(0, 0.5)))
            
            data.append({
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        logger.info(f"Generated mock market data: {len(df)} rows")
        return df
    
    def generate_mock_order_book(self, symbol="BTCUSDC", depth=20, spread_bps=10, imbalance=0.0):
        """Generate mock order book
        
        Args:
            symbol: Trading pair symbol
            depth: Order book depth
            spread_bps: Spread in basis points
            imbalance: Order book imbalance (-1.0 to 1.0, negative means more asks)
            
        Returns:
            Dictionary with mock order book data
        """
        logger.info(f"Generating mock order book for {symbol} with imbalance={imbalance}")
        
        # Start with a reasonable price for the symbol
        if "BTC" in symbol:
            mid_price = 105000.0
        elif "ETH" in symbol:
            mid_price = 2600.0
        elif "SOL" in symbol:
            mid_price = 150.0
        else:
            mid_price = 100.0
        
        # Calculate spread
        spread = mid_price * spread_bps / 10000
        
        # Calculate bid and ask prices
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Generate bids and asks
        bids = []
        asks = []
        
        # Adjust volume based on imbalance
        bid_volume_factor = 1.0 + imbalance
        ask_volume_factor = 1.0 - imbalance
        
        for i in range(depth):
            # Generate bid with decreasing price and random volume
            bid_price_level = bid_price * (1 - i * 0.0005)
            bid_volume = abs(np.random.normal(0.1, 0.05)) * bid_volume_factor
            bids.append([str(bid_price_level), str(bid_volume)])
            
            # Generate ask with increasing price and random volume
            ask_price_level = ask_price * (1 + i * 0.0005)
            ask_volume = abs(np.random.normal(0.1, 0.05)) * ask_volume_factor
            asks.append([str(ask_price_level), str(ask_volume)])
        
        # Create order book
        order_book = {
            "bids": bids,
            "asks": asks
        }
        
        logger.info(f"Generated mock order book with {len(bids)} bids and {len(asks)} asks")
        return order_book
    
    def test_signal_generation_with_mock_data(self):
        """Test signal generation with mock data"""
        logger.info("Testing signal generation with mock data")
        
        # Generate mock market data with different volatility levels
        low_vol_data = self.generate_mock_market_data("BTCUSDC", 100, 0.005)
        med_vol_data = self.generate_mock_market_data("BTCUSDC", 100, 0.02)
        high_vol_data = self.generate_mock_market_data("BTCUSDC", 100, 0.05)
        
        # Generate mock order books with different imbalances
        balanced_book = self.generate_mock_order_book("BTCUSDC", 20, 10, 0.0)
        buy_imbalance_book = self.generate_mock_order_book("BTCUSDC", 20, 10, 0.7)
        sell_imbalance_book = self.generate_mock_order_book("BTCUSDC", 20, 10, -0.7)
        
        # Test signal generation with different market conditions
        test_cases = [
            {"name": "Low volatility, balanced", "data": low_vol_data, "book": balanced_book},
            {"name": "Low volatility, buy imbalance", "data": low_vol_data, "book": buy_imbalance_book},
            {"name": "Low volatility, sell imbalance", "data": low_vol_data, "book": sell_imbalance_book},
            {"name": "Medium volatility, balanced", "data": med_vol_data, "book": balanced_book},
            {"name": "Medium volatility, buy imbalance", "data": med_vol_data, "book": buy_imbalance_book},
            {"name": "Medium volatility, sell imbalance", "data": med_vol_data, "book": sell_imbalance_book},
            {"name": "High volatility, balanced", "data": high_vol_data, "book": balanced_book},
            {"name": "High volatility, buy imbalance", "data": high_vol_data, "book": buy_imbalance_book},
            {"name": "High volatility, sell imbalance", "data": high_vol_data, "book": sell_imbalance_book},
        ]
        
        results = []
        for case in test_cases:
            logger.info(f"Testing case: {case['name']}")
            
            # Create market state from mock data
            market_state = {
                "symbol": "BTCUSDC",
                "timestamp": int(time.time() * 1000),
                "bid_price": float(case["book"]["bids"][0][0]),
                "ask_price": float(case["book"]["asks"][0][0]),
                "mid_price": (float(case["book"]["bids"][0][0]) + float(case["book"]["asks"][0][0])) / 2,
                "spread": float(case["book"]["asks"][0][0]) - float(case["book"]["bids"][0][0]),
                "spread_bps": (float(case["book"]["asks"][0][0]) - float(case["book"]["bids"][0][0])) / 
                             ((float(case["book"]["bids"][0][0]) + float(case["book"]["asks"][0][0])) / 2) * 10000,
                "order_imbalance": sum([float(b[1]) for b in case["book"]["bids"][:5]]) / 
                                  (sum([float(b[1]) for b in case["book"]["bids"][:5]]) + 
                                   sum([float(a[1]) for a in case["book"]["asks"][:5]])) * 2 - 1,
                "price_history": case["data"]["close"].tolist()[-100:],
                "timestamp_history": [int(t.timestamp() * 1000) for t in case["data"].index.tolist()[-100:]],
                "volume_history": case["data"]["volume"].tolist()[-100:],
                "last_trade_price": case["data"]["close"].iloc[-1],
                "last_trade_size": case["data"]["volume"].iloc[-1] / 10,
                "last_trade_side": "BUY" if np.random.random() > 0.5 else "SELL",
                "last_trade_time": int(case["data"].index[-1].timestamp() * 1000),
                "momentum": (case["data"]["close"].iloc[-1] / case["data"]["close"].iloc[-20] - 1) * 100,
                "volatility": case["data"]["close"].pct_change().std() * 100,
                "trend": np.mean(np.diff(case["data"]["close"].iloc[-20:].values)) / case["data"]["close"].iloc[-20] * 100
            }
            
            # Generate signals with different threshold levels
            thresholds = [
                {"imbalance": 0.25, "volatility": 0.08, "momentum": 0.06},  # Default
                {"imbalance": 0.15, "volatility": 0.05, "momentum": 0.04},  # Medium
                {"imbalance": 0.05, "volatility": 0.02, "momentum": 0.02}   # Sensitive
            ]
            
            case_results = {"case": case["name"], "signals": []}
            for i, threshold in enumerate(thresholds):
                # Set thresholds
                self.signals.imbalance_threshold = threshold["imbalance"]
                self.signals.volatility_threshold = threshold["volatility"]
                self.signals.momentum_threshold = threshold["momentum"]
                
                # Generate signals
                signals = self.signals.generate_signals_from_state(market_state)
                
                case_results["signals"].append({
                    "threshold_level": i,
                    "thresholds": threshold,
                    "signals_count": len(signals),
                    "signals": signals
                })
                
                logger.info(f"  Threshold level {i}: {len(signals)} signals generated")
            
            results.append(case_results)
        
        logger.info("Signal generation test completed")
        return results
    
    def test_paper_trading_with_mock_data(self):
        """Test paper trading with mock data"""
        logger.info("Testing paper trading with mock data")
        
        # Reset paper trading system
        self.paper_trading = EnhancedPaperTradingSystem(self.client)
        
        # Test cases
        test_cases = [
            {"symbol": "BTCUSDC", "side": "BUY", "type": "LIMIT", "quantity": 0.001, "price": 105000.0},
            {"symbol": "BTCUSDC", "side": "BUY", "type": "LIMIT", "quantity": 0.002, "price": 104500.0},
            {"symbol": "ETHUSDC", "side": "BUY", "type": "LIMIT", "quantity": 0.01, "price": 2600.0},
            {"symbol": "ETHUSDC", "side": "BUY", "type": "LIMIT", "quantity": 0.02, "price": 2550.0},
            {"symbol": "BTCUSDC", "side": "SELL", "type": "LIMIT", "quantity": 0.001, "price": 105500.0},
            {"symbol": "ETHUSDC", "side": "SELL", "type": "LIMIT", "quantity": 0.01, "price": 2650.0}
        ]
        
        # Place orders
        orders = []
        for case in test_cases:
            logger.info(f"Placing order: {case}")
            order = self.paper_trading.place_order(
                case["symbol"], case["side"], case["type"], case["quantity"], case["price"]
            )
            orders.append(order)
            logger.info(f"Order placed: {order}")
        
        # Check account state
        account = self.paper_trading.get_account_info()
        logger.info(f"Account state after placing orders: {account}")
        
        # Execute some orders
        executed_orders = []
        for i, order in enumerate(orders):
            if i % 2 == 0:  # Execute every other order
                logger.info(f"Executing order: {order['orderId']}")
                result = self.paper_trading.execute_order(order["orderId"])
                executed_orders.append(result)
                logger.info(f"Order executed: {result}")
        
        # Check account state after execution
        account = self.paper_trading.get_account_info()
        logger.info(f"Account state after executing orders: {account}")
        
        # Cancel remaining orders
        canceled_orders = []
        for i, order in enumerate(orders):
            if i % 2 != 0:  # Cancel orders that weren't executed
                logger.info(f"Canceling order: {order['orderId']}")
                result = self.paper_trading.cancel_order(order["orderId"])
                canceled_orders.append(result)
                logger.info(f"Order canceled: {result}")
        
        # Check final account state
        account = self.paper_trading.get_account_info()
        logger.info(f"Final account state: {account}")
        
        return {
            "placed_orders": orders,
            "executed_orders": executed_orders,
            "canceled_orders": canceled_orders,
            "final_account": account
        }
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        logger.info("Testing edge cases and error handling")
        
        edge_cases = []
        
        # Test case 1: Invalid symbol
        logger.info("Testing invalid symbol")
        try:
            result = self.client.get_ticker("INVALIDUSDC")
            edge_cases.append({"case": "Invalid symbol", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Invalid symbol", "result": "Exception", "error": str(e)})
        
        # Test case 2: Zero quantity order
        logger.info("Testing zero quantity order")
        try:
            result = self.paper_trading.place_order("BTCUSDC", "BUY", "LIMIT", 0, 105000.0)
            edge_cases.append({"case": "Zero quantity order", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Zero quantity order", "result": "Exception", "error": str(e)})
        
        # Test case 3: Negative price order
        logger.info("Testing negative price order")
        try:
            result = self.paper_trading.place_order("BTCUSDC", "BUY", "LIMIT", 0.001, -105000.0)
            edge_cases.append({"case": "Negative price order", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Negative price order", "result": "Exception", "error": str(e)})
        
        # Test case 4: Invalid order type
        logger.info("Testing invalid order type")
        try:
            result = self.paper_trading.place_order("BTCUSDC", "BUY", "INVALID", 0.001, 105000.0)
            edge_cases.append({"case": "Invalid order type", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Invalid order type", "result": "Exception", "error": str(e)})
        
        # Test case 5: Insufficient balance
        logger.info("Testing insufficient balance")
        try:
            result = self.paper_trading.place_order("BTCUSDC", "BUY", "LIMIT", 1000, 105000.0)
            edge_cases.append({"case": "Insufficient balance", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Insufficient balance", "result": "Exception", "error": str(e)})
        
        # Test case 6: Invalid session name
        logger.info("Testing invalid session name")
        try:
            result = self.session_manager.get_session_parameters("INVALID")
            edge_cases.append({"case": "Invalid session name", "result": "No exception", "data": result})
        except Exception as e:
            edge_cases.append({"case": "Invalid session name", "result": "Exception", "error": str(e)})
        
        logger.info("Edge case testing completed")
        return edge_cases

def main():
    """Main function"""
    logger.info("Starting mock data testing")
    
    tester = MockDataTester()
    
    # Test signal generation with mock data
    signal_results = tester.test_signal_generation_with_mock_data()
    
    # Test paper trading with mock data
    trading_results = tester.test_paper_trading_with_mock_data()
    
    # Test edge cases
    edge_case_results = tester.test_edge_cases()
    
    # Save results
    results = {
        "signal_generation": signal_results,
        "paper_trading": trading_results,
        "edge_cases": edge_case_results
    }
    
    with open("mock_data_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Mock data testing completed")
    logger.info("Results saved to mock_data_test_results.json")

if __name__ == "__main__":
    main()
