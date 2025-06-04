#!/usr/bin/env python
"""
Long Duration Test for Flash Trading System

This script runs a 24-hour test cycle of the flash trading system,
collecting comprehensive metrics across all trading sessions.
"""

import os
import time
import logging
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from threading import Thread, Event

# Add parent directory to path to import system modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling utilities
from error_handling_utils import (
    safe_get, safe_get_nested, safe_list_access,
    validate_api_response, log_exception,
    parse_float_safely, parse_int_safely,
    handle_api_error, APIResponseValidationError
)

from flash_trading_signals import SignalGenerator
from trading_session_manager import TradingSessionManager
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading import PaperTradingSystem
from flash_trading_config import FlashTradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("long_duration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("long_duration_test")

class LongDurationTest:
    """Runs a long duration test of the flash trading system"""
    
    def __init__(self, duration_hours=24, env_path=None, config_path=None):
        """Initialize test with configuration"""
        self.env_path = env_path or ".env-secure/.env"
        self.config_path = config_path
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        
        # Initialize components
        self.client = OptimizedMEXCClient(env_path=self.env_path)
        self.config = FlashTradingConfig(config_path=self.config_path)
        self.session_manager = TradingSessionManager()
        self.signal_generator = None
        self.paper_trading = None
        
        # Test metrics
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "signals_by_session": {"ASIA": 0, "EUROPE": 0, "US": 0},
            "trades_by_session": {"ASIA": 0, "EUROPE": 0, "US": 0},
            "wins_by_session": {"ASIA": 0, "EUROPE": 0, "US": 0},
            "losses_by_session": {"ASIA": 0, "EUROPE": 0, "US": 0},
            "pnl_by_session": {"ASIA": 0.0, "EUROPE": 0.0, "US": 0.0},
            "latency_by_session": {"ASIA": [], "EUROPE": [], "US": []},
            "system_metrics": {
                "cpu_usage": [],
                "memory_usage": [],
                "api_calls": 0,
                "errors": 0
            },
            "hourly_metrics": []
        }
        
        # Control flags
        self.running = False
        self.stop_event = Event()
    
    def setup(self):
        """Set up test environment"""
        logger.info(f"Setting up {self.duration_hours}-hour test environment")
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            client=self.client,
            env_path=self.env_path,
            config=self.config.get_signal_generation_config()
        )
        
        # Initialize paper trading system
        self.paper_trading = PaperTradingSystem(
            client=self.client,
            config=self.config
        )
        
        # Reset paper trading state to ensure clean test
        self.paper_trading.reset()
        
        # Connect signal generator to paper trading
        self._connect_components()
        
        logger.info("Test environment setup complete")
        return True
    
    def _connect_components(self):
        """Connect signal generator to paper trading"""
        # Create decision handler
        def handle_decision(symbol, decision):
            if not decision:
                return
                
            # Get current session
            current_session = self.session_manager.get_current_session_name()
            
            # Record decision
            self.metrics["signals_by_session"][current_session] += 1
            
            # Execute paper trade
            start_time = time.time()
            
            try:
                # Extract decision parameters
                symbol = decision["symbol"]
                side = decision["side"]
                order_type = decision.get("order_type", "LIMIT")
                quantity = decision["size"]
                price = decision.get("price")
                
                # Place order
                order = self.paper_trading.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price
                )
                
                if order:
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Record trade and latency
                    self.metrics["trades_by_session"][current_session] += 1
                    self.metrics["latency_by_session"][current_session].append(latency_ms)
                    
                    logger.info(f"Trade executed: {side} {quantity} {symbol} @ {price} in {current_session} session")
                    
                    # TODO: Track P&L when order is filled
                
            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                self.metrics["system_metrics"]["errors"] += 1
        
        # Set up signal processing thread
        def signal_processing_thread():
            while not self.stop_event.is_set():
                try:
                    # Get enabled trading pairs
                    trading_pairs = self.config.get_enabled_trading_pairs()
                    
                    # Process each trading pair
                    for pair_config in trading_pairs:
                        symbol = pair_config["symbol"]
                        
                        # Get recent signals
                        signals = self.signal_generator.get_recent_signals(10, symbol)
                        
                        if signals:
                            # Make trading decision
                            decision = self.signal_generator.make_trading_decision(symbol, signals)
                            
                            # Handle decision
                            if decision:
                                handle_decision(symbol, decision)
                    
                    # Sleep briefly
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in signal processing thread: {str(e)}")
                    self.metrics["system_metrics"]["errors"] += 1
                    time.sleep(1)
        
        # Start signal processing thread
        signal_thread = Thread(target=signal_processing_thread)
        signal_thread.daemon = True
        signal_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            import psutil
            
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Collect CPU and memory usage
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Add to metrics
            self.metrics["system_metrics"]["cpu_usage"].append(cpu_percent)
            self.metrics["system_metrics"]["memory_usage"].append(memory_mb)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    @handle_api_error
    def _collect_hourly_metrics(self):
        """Collect and save hourly metrics"""
        try:
            # Get current balances with validation using error handling utilities
            try:
                if hasattr(self, 'paper_trading') and hasattr(self.paper_trading, 'get_all_balances'):
                    balances = self.paper_trading.get_all_balances()
                    is_valid, error_msg = validate_api_response(balances, dict)
                    if not is_valid:
                        logger.error(f"Invalid balances response: {error_msg}")
                        balances = {}
                else:
                    logger.error("Paper trading system not accessible")
                    balances = {}
            except Exception as e:
                log_exception(e, "Error getting balances")
                balances = {}
            
            # Calculate total equity in USDC with validation using error handling utilities
            total_equity = parse_float_safely(safe_get(balances, "USDC"), 0.0)
            
            # Add BTC value with robust validation using error handling utilities
            btc_balance = parse_float_safely(safe_get(balances, "BTC"), 0.0)
            if btc_balance > 0:
                try:
                    btc_price_data = self.client.get_ticker_price("BTCUSDC")
                    
                    # Validate price data using error handling utilities
                    is_valid, error_msg = validate_api_response(btc_price_data, dict, ["price"])
                    if is_valid:
                        btc_price = parse_float_safely(safe_get(btc_price_data, "price"))
                        if btc_price > 0:
                            total_equity += btc_balance * btc_price
                        else:
                            logger.error(f"Invalid BTC price value: {btc_price}")
                    else:
                        logger.error(f"Invalid BTC price data: {error_msg}")
                except Exception as e:
                    log_exception(e, "Error getting BTC price")
            
            # Add ETH value with robust validation using error handling utilities
            eth_balance = parse_float_safely(safe_get(balances, "ETH"), 0.0)
            if eth_balance > 0:
                try:
                    eth_price_data = self.client.get_ticker_price("ETHUSDC")
                    
                    # Validate price data using error handling utilities
                    is_valid, error_msg = validate_api_response(eth_price_data, dict, ["price"])
                    if is_valid:
                        eth_price = parse_float_safely(safe_get(eth_price_data, "price"))
                        if eth_price > 0:
                            total_equity += eth_balance * eth_price
                        else:
                            logger.error(f"Invalid ETH price value: {eth_price}")
                    else:
                        logger.error(f"Invalid ETH price data: {error_msg}")
                except Exception as e:
                    log_exception(e, "Error getting ETH price")
        
        # Get current session with validation using error handling utilities
        try:
            current_session = "UNKNOWN"
            if hasattr(self, 'session_manager') and hasattr(self.session_manager, 'get_current_session_name'):
                session = self.session_manager.get_current_session_name()
                if session and isinstance(session, str):
                    current_session = session
            else:
                logger.error("Session manager not accessible")
        except Exception as e:
            log_exception(e, "Error getting current session")
            current_session = "UNKNOWN"
        
        # Create hourly metrics with validation using error handling utilities
        try:
            hourly_metric = {
                "timestamp": int(time.time() * 1000),
                "session": current_session,
                "balances": balances,
                "total_equity": total_equity
            }
            
            # Add metrics with validation using error handling utilities
            try:
                # Validate metrics structure
                is_valid, error_msg = validate_api_response(self.metrics, dict, [
                    "signals_by_session", "trades_by_session", 
                    "wins_by_session", "losses_by_session", "pnl_by_session"
                ])
                
                if is_valid:
                    # Add signals with validation
                    hourly_metric["signals"] = dict(safe_get(self.metrics, "signals_by_session", 
                                                   {"ASIA": 0, "EUROPE": 0, "US": 0}))
                    
                    # Add trades with validation
                    hourly_metric["trades"] = dict(safe_get(self.metrics, "trades_by_session", 
                                                  {"ASIA": 0, "EUROPE": 0, "US": 0}))
                    
                    # Add wins with validation
                    hourly_metric["wins"] = dict(safe_get(self.metrics, "wins_by_session", 
                                                {"ASIA": 0, "EUROPE": 0, "US": 0}))
                    
                    # Add losses with validation
                    hourly_metric["losses"] = dict(safe_get(self.metrics, "losses_by_session", 
                                                  {"ASIA": 0, "EUROPE": 0, "US": 0}))
                    
                    # Add pnl with validation
                    hourly_metric["pnl"] = dict(safe_get(self.metrics, "pnl_by_session", 
                                               {"ASIA": 0.0, "EUROPE": 0.0, "US": 0.0}))
                else:
                    logger.error(f"Metrics object is invalid: {error_msg}")
                    hourly_metric["signals"] = {"ASIA": 0, "EUROPE": 0, "US": 0}
                    hourly_metric["trades"] = {"ASIA": 0, "EUROPE": 0, "US": 0}
                    hourly_metric["wins"] = {"ASIA": 0, "EUROPE": 0, "US": 0}
                    hourly_metric["losses"] = {"ASIA": 0, "EUROPE": 0, "US": 0}
                    hourly_metric["pnl"] = {"ASIA": 0.0, "EUROPE": 0.0, "US": 0.0}
            except Exception as e:
                log_exception(e, "Error adding metrics to hourly data")
                hourly_metric["error"] = str(e)
        except Exception as e:
            log_exception(e, "Error creating hourly metric")
            hourly_metric = {
                "timestamp": int(time.time() * 1000),
                "error": str(e)
            }
        
        # Add to metrics with validation using error handling utilities
        try:
            is_valid, error_msg = validate_api_response(self.metrics, dict, ["hourly_metrics"])
            if is_valid:
                hourly_metrics = safe_get(self.metrics, "hourly_metrics")
                if not isinstance(hourly_metrics, list):
                    self.metrics["hourly_metrics"] = []
                self.metrics["hourly_metrics"].append(hourly_metric)
            else:
                logger.error(f"Cannot add hourly metric: {error_msg}")
        except Exception as e:
            log_exception(e, "Error adding hourly metric to metrics")
        
        # Log hourly update with validation
        try:
            # Log metrics count with validation
            try:
                if isinstance(self.metrics, dict) and "hourly_metrics" in self.metrics and isinstance(self.metrics["hourly_metrics"], list):
                    logger.info(f"Hour {len(self.metrics['hourly_metrics'])} metrics collected")
                else:
                    logger.info("Hourly metrics collected (count unknown)")
            except Exception as e:
                logger.error(f"Error logging metrics count: {str(e)}")
            
            # Log session with validation
            logger.info(f"Current session: {current_session}")
            
            # Log equity with validation
            if isinstance(total_equity, (int, float)):
                logger.info(f"Total equity: {total_equity:.2f} USDC")
            else:
                logger.info("Total equity: Unknown")
            
            # Log trades with validation
            try:
                if isinstance(self.metrics, dict) and "trades_by_session" in self.metrics and isinstance(self.metrics["trades_by_session"], dict):
                    logger.info(f"Trades by session: {self.metrics['trades_by_session']}")
                else:
                    logger.info("Trades by session: Data unavailable")
            except Exception as e:
                logger.error(f"Error logging trades data: {str(e)}")
        except Exception as e:
            logger.error(f"Error in hourly update logging: {str(e)}")
            logger.info("Hourly metrics collection completed with errors")
    
    def run(self):
        """Run the long duration test"""
        if self.running:
            logger.warning("Test already running")
            return False
        
        try:
            # Set up test environment
            if not self.setup():
                logger.error("Failed to set up test environment")
                return False
            
            # Set running flag
            self.running = True
            self.stop_event.clear()
            
            # Record start time
            self.metrics["start_time"] = int(time.time() * 1000)
            
            # Start signal generator
            trading_pairs = [pair["symbol"] for pair in self.config.get_enabled_trading_pairs()]
            self.signal_generator.start(trading_pairs)
            
            # Log test start
            logger.info(f"Starting {self.duration_hours}-hour test at {datetime.now()}")
            logger.info(f"Testing trading pairs: {trading_pairs}")
            logger.info(f"Initial balances: {self.paper_trading.get_all_balances()}")
            
            # Calculate end time
            start_time = time.time()
            end_time = start_time + self.duration_seconds
            
            # Set up metric collection intervals
            system_metric_interval = 60  # Collect system metrics every minute
            hourly_metric_interval = 3600  # Collect hourly metrics
            
            last_system_metric_time = start_time
            last_hourly_metric_time = start_time
            
            # Main test loop
            while time.time() < end_time and not self.stop_event.is_set():
                current_time = time.time()
                
                # Collect system metrics
                if current_time - last_system_metric_time >= system_metric_interval:
                    self._collect_system_metrics()
                    last_system_metric_time = current_time
                
                # Collect hourly metrics
                if current_time - last_hourly_metric_time >= hourly_metric_interval:
                    self._collect_hourly_metrics()
                    last_hourly_metric_time = current_time
                
                # Process paper trading orders
                self.paper_trading.process_open_orders()
                
                # Update API call count
                self.metrics["system_metrics"]["api_calls"] = self.client.get_request_count()
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(1)
                
                # Log progress every hour
                elapsed_hours = (current_time - start_time) / 3600
                if int(elapsed_hours) > int((current_time - 10 - start_time) / 3600):
                    logger.info(f"Test running for {int(elapsed_hours)} hours, {self.duration_hours - int(elapsed_hours)} hours remaining")
            
            # Record end time
            self.metrics["end_time"] = int(time.time() * 1000)
            
            # Stop signal generator
            self.signal_generator.stop()
            
            # Collect final metrics
            self._collect_system_metrics()
            self._collect_hourly_metrics()
            
            # Generate test report
            self._generate_report()
            
            # Reset running flag
            self.running = False
            
            logger.info(f"Test completed after {self.duration_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the running test"""
        if not self.running:
            logger.warning("No test running")
            return False
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for components to stop
        time.sleep(2)
        
        # Stop signal generator if running
        if self.signal_generator and self.signal_generator.running:
            self.signal_generator.stop()
        
        # Reset running flag
        self.running = False
        
        logger.info("Test stopped")
        return True
    
    def _generate_report(self):
        """Generate test report"""
        try:
            # Calculate test duration
            duration_ms = self.metrics["end_time"] - self.metrics["start_time"]
            duration_hours = duration_ms / (1000 * 3600)
            
            # Calculate average latencies
            avg_latencies = {}
            for session, latencies in self.metrics["latency_by_session"].items():
                if latencies:
                    avg_latencies[session] = sum(latencies) / len(latencies)
                else:
                    avg_latencies[session] = 0
            
            # Calculate win rates
            win_rates = {}
            for session in self.metrics["trades_by_session"].keys():
                total_trades = self.metrics["wins_by_session"][session] + self.metrics["losses_by_session"][session]
                if total_trades > 0:
                    win_rates[session] = (self.metrics["wins_by_session"][session] / total_trades) * 100
                else:
                    win_rates[session] = 0
            
            # Get final balances
            final_balances = self.paper_trading.get_all_balances()
            
            # Create report
            report = {
                "test_duration_hours": duration_hours,
                "start_time": datetime.fromtimestamp(self.metrics["start_time"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.fromtimestamp(self.metrics["end_time"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "signals_by_session": self.metrics["signals_by_session"],
                "trades_by_session": self.metrics["trades_by_session"],
                "win_rates": win_rates,
                "pnl_by_session": self.metrics["pnl_by_session"],
                "avg_latencies_ms": avg_latencies,
                "final_balances": final_balances,
                "system_metrics": {
                    "avg_cpu_usage": sum(self.metrics["system_metrics"]["cpu_usage"]) / len(self.metrics["system_metrics"]["cpu_usage"]) if self.metrics["system_metrics"]["cpu_usage"] else 0,
                    "max_cpu_usage": max(self.metrics["system_metrics"]["cpu_usage"]) if self.metrics["system_metrics"]["cpu_usage"] else 0,
                    "avg_memory_mb": sum(self.metrics["system_metrics"]["memory_usage"]) / len(self.metrics["system_metrics"]["memory_usage"]) if self.metrics["system_metrics"]["memory_usage"] else 0,
                    "max_memory_mb": max(self.metrics["system_metrics"]["memory_usage"]) if self.metrics["system_metrics"]["memory_usage"] else 0,
                    "total_api_calls": self.metrics["system_metrics"]["api_calls"],
                    "total_errors": self.metrics["system_metrics"]["errors"]
                }
            }
            
            # Save report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"test_results/long_duration_test_{timestamp}.json"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Test report saved to {report_file}")
            
            # Generate plots
            self._generate_plots(timestamp)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
    
    def _generate_plots(self, timestamp):
        """Generate plots from test data"""
        try:
            # Create plots directory if it doesn't exist
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract hourly equity data
            if self.metrics["hourly_metrics"]:
                times = [datetime.fromtimestamp(m["timestamp"] / 1000) for m in self.metrics["hourly_metrics"]]
                equity = [m["total_equity"] for m in self.metrics["hourly_metrics"]]
                sessions = [m["session"] for m in self.metrics["hourly_metrics"]]
                
                # Plot equity curve
                plt.figure(figsize=(12, 6))
                plt.plot(times, equity, marker='o')
                plt.title('Equity Curve Over Test Duration')
                plt.xlabel('Time')
                plt.ylabel('Total Equity (USDC)')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                equity_plot_file = f"{plots_dir}/equity_curve_{timestamp}.png"
                plt.savefig(equity_plot_file)
                plt.close()
                
                logger.info(f"Equity curve plot saved to {equity_plot_file}")
                
                # Plot trades by session
                sessions_list = ["ASIA", "EUROPE", "US"]
                trades_by_session = [self.metrics["trades_by_session"][s] for s in sessions_list]
                
                plt.figure(figsize=(10, 6))
                plt.bar(sessions_list, trades_by_session)
                plt.title('Trades by Trading Session')
                plt.xlabel('Session')
                plt.ylabel('Number of Trades')
                plt.grid(True, axis='y')
                session_plot_file = f"{plots_dir}/trades_by_session_{timestamp}.png"
                plt.savefig(session_plot_file)
                plt.close()
                
                logger.info(f"Session performance plot saved to {session_plot_file}")
                
                # Plot system metrics
                if self.metrics["system_metrics"]["cpu_usage"] and self.metrics["system_metrics"]["memory_usage"]:
                    # Assuming we collected metrics at regular intervals
                    metric_times = [datetime.fromtimestamp(self.metrics["start_time"] / 1000) + timedelta(minutes=i) 
                                   for i in range(len(self.metrics["system_metrics"]["cpu_usage"]))]
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    # CPU usage plot
                    ax1.plot(metric_times, self.metrics["system_metrics"]["cpu_usage"], color='blue')
                    ax1.set_title('CPU Usage Over Test Duration')
                    ax1.set_ylabel('CPU Usage (%)')
                    ax1.grid(True)
                    
                    # Memory usage plot
                    ax2.plot(metric_times, self.metrics["system_metrics"]["memory_usage"], color='green')
                    ax2.set_title('Memory Usage Over Test Duration')
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Memory Usage (MB)')
                    ax2.grid(True)
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    system_plot_file = f"{plots_dir}/system_metrics_{timestamp}.png"
                    plt.savefig(system_plot_file)
                    plt.close()
                    
                    logger.info(f"System metrics plot saved to {system_plot_file}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Long Duration Test for Flash Trading System')
    parser.add_argument('--duration', type=int, default=24, help='Test duration in hours')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default=None, help='Path to config file')
    args = parser.parse_args()
    
    # Run test
    test = LongDurationTest(
        duration_hours=args.duration,
        env_path=args.env,
        config_path=args.config
    )
    
    try:
        test.run()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        test.stop()
