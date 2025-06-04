#!/usr/bin/env python
"""
Extended Testing Framework for Flash Trading System

This module provides comprehensive testing capabilities for the flash trading system
across different market conditions, sessions, and scenarios.
"""

import time
import logging
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from threading import Thread

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_trading import FlashTradingSystem
from trading_session_manager import TradingSessionManager as SessionManager
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading import PaperTradingSystem
from flash_trading_config import FlashTradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_results/extended_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("extended_testing")

class ExtendedTestingFramework:
    """Framework for extended testing of flash trading system"""
    
    def __init__(self, env_path=None, config_path=None):
        """Initialize the extended testing framework"""
        # Ensure directories exist
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Load configuration
        self.config = FlashTradingConfig(config_path)
        
        # Create API client
        self.client = OptimizedMEXCClient(env_path=env_path)
        
        # Create session manager
        self.session_manager = SessionManager()
        
        # Create flash trading system
        self.flash_trading = FlashTradingSystem(env_path, config_path)
        
        # Test results
        self.results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "duration": 0,
            "sessions": [],
            "signals": [],
            "orders": [],
            "trades": [],
            "balances": {},
            "metrics": {},
            "errors": []
        }
        
        # Performance metrics
        self.metrics = {
            "signal_count": 0,
            "order_count": 0,
            "trade_count": 0,
            "api_requests": 0,
            "latency_ms": []
        }
        
        # Market data cache
        self.market_data = {}
        
    def run_test(self, duration_hours=24, update_interval=1.0, save_interval=300):
        """Run extended test for specified duration"""
        logger.info(f"Starting extended test for {duration_hours} hours")
        
        # Convert hours to seconds
        duration_seconds = duration_hours * 3600
        
        # Start time
        start_time = time.time()
        end_time = start_time + duration_seconds
        next_save_time = start_time + save_interval
        
        try:
            # Start flash trading system
            if not self.flash_trading.start():
                logger.error("Failed to start flash trading system")
                return False
            
            # Main test loop
            while time.time() < end_time:
                # Get current session
                current_session = self.session_manager.get_current_session()
                
                # Process signals and execute trades
                self.flash_trading.process_signals_and_execute()
                
                # Collect metrics
                self._collect_metrics()
                
                # Save results periodically
                if time.time() >= next_save_time:
                    self._save_interim_results()
                    next_save_time = time.time() + save_interval
                
                # Print status every 60 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 60 == 0:
                    self._print_status(elapsed, current_session)
                
                # Sleep for update interval
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            logger.info("Extended test interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during extended test: {str(e)}")
            self.results["errors"].append({
                "timestamp": time.time(),
                "error": str(e)
            })
        
        finally:
            # Stop flash trading system
            self.flash_trading.stop()
            
            # Calculate final duration
            self.results["duration"] = time.time() - start_time
            
            # Save final results
            self._save_final_results()
            
            # Generate performance charts
            self._generate_performance_charts()
            
            logger.info(f"Extended test completed. Duration: {self.results['duration'] / 3600:.2f} hours")
        
        return True
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        try:
            # Get signal count with validation
            if hasattr(self.flash_trading, 'signal_generator') and hasattr(self.flash_trading.signal_generator, 'stats'):
                stats = self.flash_trading.signal_generator.stats
                if isinstance(stats, dict) and "signals_generated" in stats:
                    self.metrics["signal_count"] = stats["signals_generated"]
            
            # Get order count with validation
            if hasattr(self.flash_trading, 'stats'):
                stats = self.flash_trading.stats
                if isinstance(stats, dict) and "orders_placed" in stats:
                    self.metrics["order_count"] = stats["orders_placed"]
            
            # Get API request count with validation
            try:
                request_count = self.client.get_request_count()
                if isinstance(request_count, (int, float)):
                    self.metrics["api_requests"] = request_count
            except Exception as e:
                logger.error(f"Error getting API request count: {str(e)}")
            
            # Collect latency sample with validation
            try:
                start_time = time.time()
                ticker_response = self.client.get_ticker_price("BTCUSDC")
                
                # Validate response before calculating latency
                if ticker_response is not None:
                    latency_ms = (time.time() - start_time) * 1000
                    if isinstance(latency_ms, (int, float)) and latency_ms > 0:
                        self.metrics["latency_ms"].append(latency_ms)
            except Exception as e:
                logger.error(f"Error collecting latency sample: {str(e)}")
            
            # Update results with validation
            self.results["metrics"] = self.metrics.copy()
            
            # Get current balances with validation
            try:
                account = self.flash_trading.paper_trading.get_account()
                
                # Validate account response
                if isinstance(account, dict) and "balances" in account:
                    balances_list = account["balances"]
                    
                    # Validate balances list
                    if isinstance(balances_list, list):
                        balances = {}
                        for balance in balances_list:
                            # Validate each balance entry
                            if isinstance(balance, dict) and "asset" in balance and "free" in balance:
                                asset = balance["asset"]
                                free = balance["free"]
                                
                                # Validate values
                                if isinstance(asset, str) and isinstance(free, (int, float)) and free > 0:
                                    balances[asset] = free
                        
                        self.results["balances"] = balances
            except Exception as e:
                logger.error(f"Error getting account balances: {str(e)}")
            
            # Get recent signals with validation
            try:
                if hasattr(self.flash_trading, 'signal_generator') and hasattr(self.flash_trading.signal_generator, 'get_recent_signals'):
                    signals = self.flash_trading.signal_generator.get_recent_signals(100)
                    
                    # Validate signals
                    if isinstance(signals, list):
                        self.results["signals"] = signals
            except Exception as e:
                logger.error(f"Error getting recent signals: {str(e)}")
            
            # Get order history with validation
            try:
                if hasattr(self.flash_trading, 'paper_trading') and hasattr(self.flash_trading.paper_trading, 'order_history'):
                    order_history = self.flash_trading.paper_trading.order_history
                    
                    # Validate order history
                    if isinstance(order_history, list):
                        self.results["orders"] = order_history
            except Exception as e:
                logger.error(f"Error getting order history: {str(e)}")
            
            # Get trade history with validation
            try:
                if hasattr(self.flash_trading, 'paper_trading') and hasattr(self.flash_trading.paper_trading, 'trade_history'):
                    trade_history = self.flash_trading.paper_trading.trade_history
                    
                    # Validate trade history
                    if isinstance(trade_history, list):
                        self.results["trades"] = trade_history
            except Exception as e:
                logger.error(f"Error getting trade history: {str(e)}")
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
        
        # Get current session with validation
        try:
            if hasattr(self, 'session_manager') and hasattr(self.session_manager, 'get_current_session'):
                current_session = self.session_manager.get_current_session()
                
                # Validate current session
                if current_session and isinstance(current_session, str):
                    # Check if session exists in results
                    session_exists = False
                    if isinstance(self.results.get("sessions"), list):
                        session_exists = current_session in [s.get("session") for s in self.results["sessions"] if isinstance(s, dict)]
                    
                    # Add new session if needed
                    if not session_exists:
                        if not isinstance(self.results.get("sessions"), list):
                            self.results["sessions"] = []
                            
                        self.results["sessions"].append({
                            "session": current_session,
                            "start_time": time.time(),
                            "signals": 0,
                            "orders": 0,
                            "trades": 0
                        })
                    
                    # Update session metrics with validation
                    try:
                        for session in self.results["sessions"]:
                            if isinstance(session, dict) and session.get("session") == current_session:
                                # Validate session data
                                start_time = session.get("start_time", 0)
                                if not isinstance(start_time, (int, float)):
                                    start_time = 0
                                
                                # Validate signals with safe access
                                signals_count = 0
                                if isinstance(self.results.get("signals"), list):
                                    signals_count = sum(1 for s in self.results["signals"] 
                                                      if isinstance(s, dict) 
                                                      and isinstance(s.get("timestamp"), (int, float)) 
                                                      and s.get("timestamp", 0) > start_time * 1000)
                                session["signals"] = signals_count
                                
                                # Validate orders with safe access
                                orders_count = 0
                                if isinstance(self.results.get("orders"), list):
                                    orders_count = sum(1 for o in self.results["orders"] 
                                                     if isinstance(o, dict) 
                                                     and isinstance(o.get("timestamp"), (int, float)) 
                                                     and o.get("timestamp", 0) > start_time * 1000)
                                session["orders"] = orders_count
                                
                                # Validate trades with safe access
                                trades_count = 0
                                if isinstance(self.results.get("trades"), list):
                                    trades_count = sum(1 for t in self.results["trades"] 
                                                     if isinstance(t, dict) 
                                                     and isinstance(t.get("timestamp"), (int, float)) 
                                                     and t.get("timestamp", 0) > start_time * 1000)
                                session["trades"] = trades_count
                    except Exception as e:
                        logger.error(f"Error updating session metrics: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing session information: {str(e)}")
    
    def _save_interim_results(self):
        """Save interim test results"""
        try:
            # Generate timestamp with validation
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                logger.error(f"Error generating timestamp: {str(e)}")
                timestamp = str(int(time.time()))
            
            # Create filename with validation
            filename = f"test_results/interim_{timestamp}.json"
            
            # Ensure directory exists
            os.makedirs("test_results", exist_ok=True)
            
            # Save with error handling
            try:
                with open(filename, "w") as f:
                    json.dump(self.results, f, indent=2)
                
                logger.info(f"Interim results saved to {filename}")
            except (IOError, OSError) as e:
                logger.error(f"Error saving interim results to file: {str(e)}")
                # Try alternative location
                alt_filename = f"interim_results_{timestamp}.json"
                try:
                    with open(alt_filename, "w") as f:
                        json.dump(self.results, f, indent=2)
                    logger.info(f"Interim results saved to alternative location: {alt_filename}")
                except Exception as e2:
                    logger.error(f"Failed to save interim results to alternative location: {str(e2)}")
        except Exception as e:
            logger.error(f"Unexpected error saving interim results: {str(e)}")
    
    def _save_final_results(self):
        """Save final test results"""
        try:
            # Generate timestamp with validation
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                logger.error(f"Error generating timestamp: {str(e)}")
                timestamp = str(int(time.time()))
            
            # Create filename with validation
            filename = f"test_results/extended_test_{timestamp}.json"
            
            # Ensure directory exists
            os.makedirs("test_results", exist_ok=True)
            
            # Save with error handling
            try:
                with open(filename, "w") as f:
                    json.dump(self.results, f, indent=2)
                
                logger.info(f"Final results saved to {filename}")
                return filename
            except (IOError, OSError) as e:
                logger.error(f"Error saving final results to file: {str(e)}")
                # Try alternative location
                alt_filename = f"extended_test_results_{timestamp}.json"
                try:
                    with open(alt_filename, "w") as f:
                        json.dump(self.results, f, indent=2)
                    logger.info(f"Final results saved to alternative location: {alt_filename}")
                    return alt_filename
                except Exception as e2:
                    logger.error(f"Failed to save final results to alternative location: {str(e2)}")
                    return None
        except Exception as e:
            logger.error(f"Unexpected error saving final results: {str(e)}")
            return None
    
    def _print_status(self, elapsed, current_session):
        """Print current test status"""
        try:
            # Get account information with validation
            try:
                if hasattr(self.flash_trading, 'paper_trading') and hasattr(self.flash_trading.paper_trading, 'get_account'):
                    account = self.flash_trading.paper_trading.get_account()
                else:
                    logger.error("Paper trading system not accessible")
                    account = {"balances": []}
            except Exception as e:
                logger.error(f"Error getting account information: {str(e)}")
                account = {"balances": []}
            
            # Print status with validation
            print("\n--- Extended Test Status ---")
            
            # Validate elapsed time
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                print(f"Elapsed Time: {elapsed / 3600:.2f} hours")
            else:
                print(f"Elapsed Time: Unknown")
            
            # Validate current session
            if current_session and isinstance(current_session, str):
                print(f"Current Session: {current_session}")
            else:
                print(f"Current Session: Unknown")
            
            # Print balances with validation
            print("\nBalances:")
            if isinstance(account, dict) and "balances" in account and isinstance(account["balances"], list):
                for balance in account["balances"]:
                    if isinstance(balance, dict) and "asset" in balance and "free" in balance:
                        free_balance = balance.get("free", 0)
                        if isinstance(free_balance, (int, float)) and free_balance > 0:
                            asset = balance.get("asset", "Unknown")
                            print(f"  {asset}: {free_balance}")
            else:
                print("  No balance information available")
            
            # Print metrics with validation
            print("\nMetrics:")
            
            # Validate signal count
            signal_count = self.metrics.get("signal_count", "Unknown")
            if not isinstance(signal_count, (int, float)):
                signal_count = "Unknown"
            print(f"  Signals: {signal_count}")
            
            # Validate order count
            order_count = self.metrics.get("order_count", "Unknown")
            if not isinstance(order_count, (int, float)):
                order_count = "Unknown"
            print(f"  Orders: {order_count}")
            
            # Validate API request count
            api_requests = self.metrics.get("api_requests", "Unknown")
            if not isinstance(api_requests, (int, float)):
                api_requests = "Unknown"
            print(f"  API Requests: {api_requests}")
            
            # Validate latency metrics
            latency_ms = self.metrics.get("latency_ms", [])
            if isinstance(latency_ms, list) and latency_ms:
                try:
                    avg_latency = sum(latency_ms) / len(latency_ms)
                    if isinstance(avg_latency, (int, float)):
                        print(f"  Avg Latency: {avg_latency:.2f} ms")
                    else:
                        print(f"  Avg Latency: Unknown")
                except Exception as e:
                    logger.error(f"Error calculating average latency: {str(e)}")
                    print(f"  Avg Latency: Error")
        except Exception as e:
            logger.error(f"Error printing status: {str(e)}")
            print("\n--- Extended Test Status ---")
            print("Error retrieving status information")
    
    def _generate_performance_charts(self):
        """Generate performance charts from test results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create dataframes from results
            if not self.results["trades"]:
                logger.warning("No trades to generate charts from")
                return
            
            # Convert trades to dataframe
            trades_df = pd.DataFrame(self.results["trades"])
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], unit="ms")
            trades_df.set_index("timestamp", inplace=True)
            
            # Generate balance chart
            self._generate_balance_chart(timestamp)
            
            # Generate trade volume chart
            self._generate_trade_volume_chart(trades_df, timestamp)
            
            # Generate latency chart
            self._generate_latency_chart(timestamp)
            
            # Generate session performance chart
            self._generate_session_performance_chart(timestamp)
            
            logger.info(f"Performance charts generated in plots/ directory")
        
        except Exception as e:
            logger.error(f"Error generating performance charts: {str(e)}")
    
    def _generate_balance_chart(self, timestamp):
        """Generate balance chart"""
        # Get balance history from trades
        balance_history = self._calculate_balance_history()
        
        if not balance_history:
            logger.warning("No balance history to generate chart from")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot each asset balance
        for asset, history in balance_history.items():
            times = [h["timestamp"] for h in history]
            values = [h["balance"] for h in history]
            plt.plot(times, values, label=asset)
        
        plt.title("Asset Balance History")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid(True)
        
        # Save chart
        plt.savefig(f"plots/balance_history_{timestamp}.png")
        plt.close()
    
    def _calculate_balance_history(self):
        """Calculate balance history from trades"""
        # Start with initial balances
        balance_history = {}
        initial_balances = self.config.config["paper_trading"]["initial_balance"]
        
        for asset, amount in initial_balances.items():
            balance_history[asset] = [{
                "timestamp": datetime.fromtimestamp(self.results["trades"][0]["timestamp"] / 1000 if self.results["trades"] else time.time()),
                "balance": amount
            }]
        
        # Process trades to update balances
        for trade in sorted(self.results["trades"], key=lambda x: x["timestamp"]):
            timestamp = datetime.fromtimestamp(trade["timestamp"] / 1000)
            symbol = trade["symbol"]
            side = trade["side"]
            quantity = trade["quantity"]
            quote_qty = trade["quoteQty"]
            
            # Extract assets from symbol
            base_asset = symbol[:-4]  # Assuming USDC pairs like BTCUSDC
            quote_asset = "USDC"
            
            # Ensure assets exist in history
            if base_asset not in balance_history:
                balance_history[base_asset] = [{
                    "timestamp": timestamp,
                    "balance": 0.0
                }]
            
            if quote_asset not in balance_history:
                balance_history[quote_asset] = [{
                    "timestamp": timestamp,
                    "balance": 0.0
                }]
            
            # Get current balances
            base_balance = balance_history[base_asset][-1]["balance"]
            quote_balance = balance_history[quote_asset][-1]["balance"]
            
            # Update balances based on trade
            if side == "BUY":
                base_balance += quantity
                quote_balance -= quote_qty
            else:  # SELL
                base_balance -= quantity
                quote_balance += quote_qty
            
            # Add to history
            balance_history[base_asset].append({
                "timestamp": timestamp,
                "balance": base_balance
            })
            
            balance_history[quote_asset].append({
                "timestamp": timestamp,
                "balance": quote_balance
            })
        
        return balance_history
    
    def _generate_trade_volume_chart(self, trades_df, timestamp):
        """Generate trade volume chart"""
        if trades_df.empty:
            logger.warning("No trades to generate volume chart from")
            return
        
        # Resample by hour
        hourly_volume = trades_df.resample("1H").sum()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot volume
        plt.bar(hourly_volume.index, hourly_volume["quoteQty"], width=0.02)
        
        plt.title("Hourly Trading Volume (USDC)")
        plt.xlabel("Time")
        plt.ylabel("Volume (USDC)")
        plt.grid(True)
        
        # Save chart
        plt.savefig(f"plots/trade_volume_{timestamp}.png")
        plt.close()
    
    def _generate_latency_chart(self, timestamp):
        """Generate API latency chart"""
        if not self.metrics["latency_ms"]:
            logger.warning("No latency data to generate chart from")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot latency
        plt.plot(self.metrics["latency_ms"])
        
        plt.title("API Request Latency")
        plt.xlabel("Request Number")
        plt.ylabel("Latency (ms)")
        plt.grid(True)
        
        # Save chart
        plt.savefig(f"plots/api_latency_{timestamp}.png")
        plt.close()
    
    def _generate_session_performance_chart(self, timestamp):
        """Generate session performance chart"""
        if not self.results["sessions"]:
            logger.warning("No session data to generate chart from")
            return
        
        # Extract session data
        sessions = [s["session"] for s in self.results["sessions"]]
        signals = [s["signals"] for s in self.results["sessions"]]
        orders = [s["orders"] for s in self.results["sessions"]]
        trades = [s["trades"] for s in self.results["sessions"]]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(sessions))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, signals, width, label="Signals")
        plt.bar(x, orders, width, label="Orders")
        plt.bar(x + width, trades, width, label="Trades")
        
        plt.title("Performance by Trading Session")
        plt.xlabel("Session")
        plt.ylabel("Count")
        plt.xticks(x, sessions)
        plt.legend()
        plt.grid(True)
        
        # Save chart
        plt.savefig(f"plots/session_performance_{timestamp}.png")
        plt.close()

    def simulate_market_condition(self, condition_type):
        """Simulate specific market conditions for testing"""
        logger.info(f"Simulating market condition: {condition_type}")
        
        if condition_type == "high_volatility":
            # Modify signal generator parameters for high volatility
            self.flash_trading.signal_generator.config.update({
                "volatility_threshold": 0.005,  # Lower threshold to generate more signals
                "signal_amplification": 2.0,    # Amplify signal strength
                "trend_sensitivity": 1.5        # Increase trend sensitivity
            })
            
        elif condition_type == "low_liquidity":
            # Modify paper trading parameters for low liquidity
            self.flash_trading.paper_trading.paper_config.update({
                "simulate_slippage": True,
                "slippage_bps": 50,             # Higher slippage
                "simulate_partial_fills": True,
                "partial_fill_probability": 0.7  # Higher chance of partial fills
            })
            
        elif condition_type == "news_event":
            # Simulate market reaction to news event
            # Create a background thread to inject price spikes
            def inject_price_spikes():
                logger.info("Injecting price spikes to simulate news event")
                # This is a simulation - in a real system, this would modify market data
                time.sleep(300)  # Wait 5 minutes before event
                
                # Modify signal generator to react to "news"
                self.flash_trading.signal_generator.inject_event({
                    "type": "news",
                    "impact": "high",
                    "direction": "positive",
                    "symbols": ["BTCUSDC", "ETHUSDC"]
                })
                
                # Reset after 10 minutes
                time.sleep(600)
                self.flash_trading.signal_generator.reset_events()
            
            # Start background thread
            Thread(target=inject_price_spikes, daemon=True).start()
        
        else:
            logger.warning(f"Unknown market condition type: {condition_type}")
            return False
        
        return True
    
    def run_market_condition_tests(self, duration_hours=2):
        """Run tests for different market conditions"""
        conditions = ["normal", "high_volatility", "low_liquidity", "news_event"]
        results = {}
        
        for condition in conditions:
            logger.info(f"Starting test for {condition} market condition")
            
            # Reset paper trading system
            self.flash_trading.paper_trading.reset()
            
            # Set up condition
            if condition != "normal":
                self.simulate_market_condition(condition)
            
            # Run test
            self.run_test(duration_hours=duration_hours)
            
            # Store results
            results[condition] = {
                "signals": self.metrics["signal_count"],
                "orders": self.metrics["order_count"],
                "trades": len(self.results["trades"]),
                "final_balances": self.results["balances"]
            }
            
            # Reset condition
            if condition != "normal":
                # Reset signal generator and paper trading to default
                self.flash_trading = FlashTradingSystem(
                    env_path=self.client.env_path,
                    config_path=self.config.config_path
                )
        
        # Save comparative results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/market_conditions_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Market condition test results saved to {filename}")
        
        # Generate comparative chart
        self._generate_condition_comparison_chart(results, timestamp)
        
        return results
    
    def _generate_condition_comparison_chart(self, results, timestamp):
        """Generate chart comparing performance across market conditions"""
        conditions = list(results.keys())
        signals = [results[c]["signals"] for c in conditions]
        orders = [results[c]["orders"] for c in conditions]
        trades = [results[c]["trades"] for c in conditions]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(conditions))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, signals, width, label="Signals")
        plt.bar(x, orders, width, label="Orders")
        plt.bar(x + width, trades, width, label="Trades")
        
        plt.title("Performance Across Market Conditions")
        plt.xlabel("Market Condition")
        plt.ylabel("Count")
        plt.xticks(x, conditions)
        plt.legend()
        plt.grid(True)
        
        # Save chart
        plt.savefig(f"plots/market_conditions_{timestamp}.png")
        plt.close()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extended Testing Framework')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default="flash_trading_config.json", help='Path to config file')
    parser.add_argument('--duration', type=float, default=24, help='Test duration in hours')
    parser.add_argument('--condition', choices=['normal', 'high_volatility', 'low_liquidity', 'news_event', 'all'], 
                        default='normal', help='Market condition to test')
    
    args = parser.parse_args()
    
    # Create extended testing framework
    testing = ExtendedTestingFramework(args.env, args.config)
    
    # Run test based on condition
    if args.condition == 'all':
        testing.run_market_condition_tests(args.duration / 4)  # Split duration across 4 conditions
    else:
        if args.condition != 'normal':
            testing.simulate_market_condition(args.condition)
        testing.run_test(duration_hours=args.duration)
