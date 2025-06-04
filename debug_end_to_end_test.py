#!/usr/bin/env python
"""
Modified End-to-End Test for Trading-Agent System with Granular Logging

This script breaks down the end-to-end test into smaller components with
detailed logging to identify where the process is hanging.
"""

import os
import sys
import json
import time
import logging
import unittest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import traceback
import uuid

# Import Trading-Agent components
from env_loader import load_environment_variables
from enhanced_flash_trading_signals import EnhancedFlashTradingSignals
from enhanced_dl_integration_fixed import EnhancedPatternRecognitionIntegration
from enhanced_dl_model_fixed import EnhancedPatternRecognitionModel
from enhanced_feature_adapter_fixed import EnhancedFeatureAdapter
from execution_optimization import (
    OrderRouter, SmartOrderRouter, ExecutionOptimizer, 
    LatencyProfiler, Order, OrderType, OrderSide, OrderStatus
)
from rl_agent_fixed_v4 import TradingRLAgent
from optimized_mexc_client import OptimizedMEXCClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug_end_to_end_test.log")
    ]
)
logger = logging.getLogger("debug_end_to_end_test")

# Load environment variables
env_vars = load_environment_variables()

def test_component_initialization():
    """Test initialization of all components"""
    logger.info("=== TESTING COMPONENT INITIALIZATION ===")
    
    try:
        logger.info("Initializing MEXC client...")
        start_time = time.time()
        mexc_client = OptimizedMEXCClient(
            api_key=env_vars['MEXC_API_KEY'],
            api_secret=env_vars['MEXC_SECRET_KEY']
        )
        logger.info(f"MEXC client initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Testing MEXC client connection...")
        start_time = time.time()
        server_time = mexc_client.get_server_time()
        logger.info(f"Server time: {server_time} (retrieved in {time.time() - start_time:.2f} seconds)")
        
        logger.info("Initializing pattern recognition model...")
        start_time = time.time()
        model = EnhancedPatternRecognitionModel(input_dim=16, sequence_length=40)
        logger.info(f"Pattern recognition model initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing feature adapter...")
        start_time = time.time()
        feature_adapter = EnhancedFeatureAdapter()
        logger.info(f"Feature adapter initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing pattern recognition integration...")
        start_time = time.time()
        pattern_recognition = EnhancedPatternRecognitionIntegration(
            model=model,
            feature_adapter=feature_adapter,
            confidence_threshold=0.45
        )
        logger.info(f"Pattern recognition integration initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing trading signals...")
        start_time = time.time()
        trading_signals = EnhancedFlashTradingSignals(
            client_instance=mexc_client,
            pattern_recognition=pattern_recognition
        )
        logger.info(f"Trading signals initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing RL agent...")
        start_time = time.time()
        rl_agent = TradingRLAgent(state_dim=10, action_dim=3)
        logger.info(f"RL agent initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing latency profiler...")
        start_time = time.time()
        latency_profiler = LatencyProfiler()
        latency_profiler.set_threshold("order_submission", 100)  # 100ms threshold
        logger.info(f"Latency profiler initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("Initializing order router...")
        start_time = time.time()
        order_router = OrderRouter(
            client_instance=mexc_client,
            latency_profiler=latency_profiler
        )
        logger.info(f"Order router initialized in {time.time() - start_time:.2f} seconds")
        
        logger.info("All components initialized successfully")
        return {
            "mexc_client": mexc_client,
            "model": model,
            "feature_adapter": feature_adapter,
            "pattern_recognition": pattern_recognition,
            "trading_signals": trading_signals,
            "rl_agent": rl_agent,
            "latency_profiler": latency_profiler,
            "order_router": order_router
        }
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.error(traceback.format_exc())
        return None

def test_market_data_retrieval(components):
    """Test market data retrieval"""
    logger.info("=== TESTING MARKET DATA RETRIEVAL ===")
    
    try:
        mexc_client = components["mexc_client"]
        
        symbol = "BTCUSDT"
        timeframe = "5m"
        limit = 500  # Reduced from 1000 to avoid potential timeouts
        
        logger.info(f"Retrieving {limit} candles for {symbol} {timeframe}...")
        start_time = time.time()
        
        candles = mexc_client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(candles)} candles in {elapsed:.2f} seconds")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Print sample data
        logger.info(f"Sample data:\n{df.head()}")
        
        # Save to file for analysis
        df.to_csv("test_market_data.csv", index=False)
        logger.info(f"Saved {len(df)} candles to test_market_data.csv")
        
        return df
    except Exception as e:
        logger.error(f"Error retrieving market data: {e}")
        logger.error(traceback.format_exc())
        return None

def test_pattern_recognition(components, market_data):
    """Test pattern recognition"""
    logger.info("=== TESTING PATTERN RECOGNITION ===")
    
    try:
        pattern_recognition = components["pattern_recognition"]
        
        symbol = "BTC/USDC"
        timeframe = "5m"
        
        logger.info(f"Detecting patterns for {symbol} {timeframe}...")
        start_time = time.time()
        
        # Add progress logging to pattern detection
        logger.info("Starting pattern detection process...")
        signals = []
        
        try:
            logger.info(f"Input data shape: {market_data.shape}")
            logger.info(f"Input data columns: {market_data.columns.tolist()}")
            logger.info(f"First few rows of input data:\n{market_data.head()}")
            
            # Add timeout mechanism
            max_time = 60  # 60 seconds timeout
            
            def detect_with_timeout():
                return pattern_recognition.detect_patterns(
                    market_data, symbol, timeframe, max_patterns=50
                )
            
            # Run pattern detection with timeout
            logger.info(f"Running pattern detection with {max_time}s timeout...")
            start_detect = time.time()
            
            # Use a separate thread with timeout
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def worker():
                try:
                    result = detect_with_timeout()
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", e))
            
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            
            # Wait for result or timeout
            thread.join(max_time)
            
            if thread.is_alive():
                logger.error(f"Pattern detection timed out after {max_time} seconds")
                return None
            
            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success":
                    signals = result
                    logger.info(f"Pattern detection completed in {time.time() - start_detect:.2f} seconds")
                else:
                    logger.error(f"Pattern detection failed: {result}")
                    logger.error(traceback.format_exc())
                    return None
            else:
                logger.error("No result from pattern detection thread")
                return None
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            logger.error(traceback.format_exc())
            return None
        
        elapsed = time.time() - start_time
        
        if signals:
            logger.info(f"Detected {len(signals)} pattern signals in {elapsed:.2f} seconds")
            
            # Log sample signals
            for i, signal in enumerate(signals[:5]):
                logger.info(f"Sample signal {i+1}: {signal}")
            
            # Save signals to file
            with open("test_pattern_signals.json", "w") as f:
                json.dump(signals, f, default=str)
            
            logger.info(f"Saved {len(signals)} pattern signals to test_pattern_signals.json")
        else:
            logger.warning(f"No pattern signals detected in {elapsed:.2f} seconds")
        
        return signals
    except Exception as e:
        logger.error(f"Error in pattern recognition: {e}")
        logger.error(traceback.format_exc())
        return None

def test_signal_generation(components):
    """Test signal generation"""
    logger.info("=== TESTING SIGNAL GENERATION ===")
    
    try:
        trading_signals = components["trading_signals"]
        
        symbol = "BTC/USDC"
        timeframe = "5m"
        limit = 500  # Reduced from 1000 to avoid potential timeouts
        max_signals = 50  # Limit signals for faster test completion
        
        logger.info(f"Generating signals for {symbol} {timeframe}...")
        start_time = time.time()
        
        # Add timeout mechanism
        max_time = 60  # 60 seconds timeout
        
        def generate_with_timeout():
            return trading_signals.get_signals(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                max_signals=max_signals
            )
        
        # Run signal generation with timeout
        logger.info(f"Running signal generation with {max_time}s timeout...")
        start_generate = time.time()
        
        # Use a separate thread with timeout
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def worker():
            try:
                result = generate_with_timeout()
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        # Wait for result or timeout
        thread.join(max_time)
        
        if thread.is_alive():
            logger.error(f"Signal generation timed out after {max_time} seconds")
            return None
        
        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "success":
                signals = result
                logger.info(f"Signal generation completed in {time.time() - start_generate:.2f} seconds")
            else:
                logger.error(f"Signal generation failed: {result}")
                logger.error(traceback.format_exc())
                return None
        else:
            logger.error("No result from signal generation thread")
            return None
        
        elapsed = time.time() - start_time
        
        if signals:
            logger.info(f"Generated {len(signals)} signals in {elapsed:.2f} seconds")
            
            # Log sample signals
            for i, signal in enumerate(signals[:5]):
                logger.info(f"Sample signal {i+1}: {signal}")
            
            # Save signals to file
            with open("test_trading_signals.json", "w") as f:
                json.dump(signals, f, default=str)
            
            logger.info(f"Saved {len(signals)} signals to test_trading_signals.json")
        else:
            logger.warning(f"No signals generated in {elapsed:.2f} seconds")
        
        return signals
    except Exception as e:
        logger.error(f"Error in signal generation: {e}")
        logger.error(traceback.format_exc())
        return None

def test_decision_generation(components, signals):
    """Test decision generation"""
    logger.info("=== TESTING DECISION GENERATION ===")
    
    if not signals:
        logger.warning("No signals available for decision generation")
        return None
    
    try:
        rl_agent = components["rl_agent"]
        
        logger.info(f"Generating decisions for {len(signals)} signals...")
        start_time = time.time()
        
        decisions = []
        for i, signal in enumerate(signals[:max(50, len(signals))]):  # Limit to 50 signals for faster test completion
            logger.debug(f"Processing signal {i+1}/{min(50, len(signals))}...")
            
            # Convert signal to state
            state = rl_agent.signal_to_state(signal)
            
            # Get action
            action = rl_agent.get_action(state)
            
            # Convert action to decision
            decision = rl_agent.action_to_decision(action, signal)
            
            if decision:
                decisions.append(decision)
        
        elapsed = time.time() - start_time
        
        if decisions:
            logger.info(f"Generated {len(decisions)} decisions in {elapsed:.2f} seconds")
            
            # Log sample decisions
            for i, decision in enumerate(decisions[:5]):
                logger.info(f"Sample decision {i+1}: {decision}")
            
            # Save decisions to file
            with open("test_trading_decisions.json", "w") as f:
                json.dump(decisions, f, default=str)
            
            logger.info(f"Saved {len(decisions)} decisions to test_trading_decisions.json")
        else:
            logger.warning(f"No decisions generated in {elapsed:.2f} seconds")
        
        return decisions
    except Exception as e:
        logger.error(f"Error in decision generation: {e}")
        logger.error(traceback.format_exc())
        return None

def test_order_execution(components, decisions):
    """Test order execution"""
    logger.info("=== TESTING ORDER EXECUTION ===")
    
    if not decisions:
        logger.warning("No decisions available for order execution")
        return None
    
    try:
        order_router = components["order_router"]
        
        logger.info(f"Executing orders for {len(decisions)} decisions...")
        start_time = time.time()
        
        orders = []
        for i, decision in enumerate(decisions[:min(10, len(decisions))]):  # Limit to 10 decisions for faster test completion
            logger.debug(f"Processing decision {i+1}/{min(10, len(decisions))}...")
            
            # Create order from decision
            symbol = decision.get("symbol", "BTC/USDC")
            side = OrderSide.BUY if decision.get("action", "buy").lower() == "buy" else OrderSide.SELL
            quantity = decision.get("quantity", 0.001)  # Small quantity for testing
            price = decision.get("price", 50000.0)
            
            order = Order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=OrderType.MARKET,
                side=side
            )
            
            # Submit order
            result = order_router.submit_order(order)
            
            if result:
                orders.append(result)
        
        elapsed = time.time() - start_time
        
        if orders:
            logger.info(f"Executed {len(orders)} orders in {elapsed:.2f} seconds")
            
            # Log sample orders
            for i, order in enumerate(orders[:5]):
                logger.info(f"Sample order {i+1}: {order}")
            
            # Save orders to file (convert to dict for JSON serialization)
            order_dicts = []
            for order in orders:
                order_dict = {
                    "symbol": order.symbol,
                    "quantity": float(order.quantity),
                    "price": float(order.price) if order.price is not None else None,
                    "type": str(order.order_type),
                    "side": str(order.side),
                    "status": str(order.status)
                }
                order_dicts.append(order_dict)
            
            with open("test_executed_orders.json", "w") as f:
                json.dump(order_dicts, f)
            
            logger.info(f"Saved {len(orders)} orders to test_executed_orders.json")
        else:
            logger.warning(f"No orders executed in {elapsed:.2f} seconds")
        
        return orders
    except Exception as e:
        logger.error(f"Error in order execution: {e}")
        logger.error(traceback.format_exc())
        return None

def run_tests():
    """Run all tests"""
    logger.info("Starting debug end-to-end tests...")
    
    # Test component initialization
    components = test_component_initialization()
    if not components:
        logger.error("Component initialization failed, aborting tests")
        return False
    
    # Test market data retrieval
    market_data = test_market_data_retrieval(components)
    if market_data is None:
        logger.error("Market data retrieval failed, aborting tests")
        return False
    
    # Test pattern recognition
    pattern_signals = test_pattern_recognition(components, market_data)
    
    # Test signal generation (independent of pattern recognition)
    trading_signals = test_signal_generation(components)
    
    # Use either pattern signals or trading signals for decision generation
    signals_for_decisions = pattern_signals if pattern_signals else trading_signals
    
    if not signals_for_decisions:
        logger.warning("No signals available for decision generation, skipping remaining tests")
        return False
    
    # Test decision generation
    decisions = test_decision_generation(components, signals_for_decisions)
    
    if not decisions:
        logger.warning("No decisions generated, skipping order execution test")
        return False
    
    # Test order execution
    orders = test_order_execution(components, decisions)
    
    logger.info("Debug end-to-end tests completed")
    return True

if __name__ == "__main__":
    run_tests()
