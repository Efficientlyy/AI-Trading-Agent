#!/usr/bin/env python
"""
Test Session Awareness

This module provides tests for the session awareness functionality in the flash trading system.
"""

import time
import logging
import json
import os
import argparse
from datetime import datetime, timezone
from threading import Thread, Event
from flash_trading_signals import SignalGenerator
from trading_session_manager import TradingSessionManager, TradingSession
from optimized_mexc_client import OptimizedMEXCClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_session_awareness")

class SessionAwarenessTest:
    """Test session awareness functionality"""
    
    def __init__(self, env_path=None):
        """Initialize test with API client"""
        self.env_path = env_path or ".env-secure/.env"
        self.client = OptimizedMEXCClient(env_path=self.env_path)
        self.session_manager = TradingSessionManager()
        self.signal_generator = None
    
    def run_tests(self, symbols=None, duration=30):
        """Run all session awareness tests"""
        symbols = symbols or ["BTCUSDC", "ETHUSDC"]
        logger.info(f"Starting session awareness tests for symbols: {symbols}")
        
        # Test session detection
        logger.info("Testing session detection...")
        self.test_session_detection()
        
        # Test session parameter loading
        logger.info("Testing session parameter loading...")
        self.test_session_parameter_loading()
        
        # Test session transitions
        logger.info("Testing session transitions...")
        self.test_session_transitions()
        
        # Test signal generation with session awareness
        logger.info("Testing signal generation with session awareness...")
        self.test_signal_generation(symbols, duration)
        
        # Test decision making with session awareness
        logger.info("Testing decision making with session awareness...")
        self.test_decision_making(symbols)
    
    def test_session_detection(self):
        """Test session detection functionality"""
        # Get current session
        current_session = self.session_manager.get_current_session_name()
        current_session_obj = self.session_manager.get_current_session()
        
        # Verify session is detected
        assert current_session is not None, "Current session should not be None"
        assert current_session in ["ASIA", "EUROPE", "US"], f"Current session should be one of ASIA, EUROPE, US, got {current_session}"
        assert current_session_obj is not None, "Current session object should not be None"
        assert current_session_obj.is_active(), "Current session should be active"
        
        logger.info(f"Session detection passed: Current session is {current_session}")
    
    def test_session_parameter_loading(self):
        """Test session parameter loading functionality"""
        # Get current session
        current_session = self.session_manager.get_current_session_name()
        
        # Get session parameters
        params = self.session_manager.get_all_session_parameters()
        
        # Verify parameters are loaded
        assert params is not None, "Session parameters should not be None"
        assert len(params) > 0, "Session parameters should not be empty"
        
        # Check for specific parameters
        expected_params = [
            "imbalance_threshold",
            "volatility_threshold",
            "momentum_threshold",
            "position_size_factor",
            "take_profit_bps",
            "stop_loss_bps"
        ]
        
        for param in expected_params:
            assert param in params, f"Session parameters should include {param}"
        
        logger.info(f"Session parameter loading passed: {len(params)} parameters loaded for {current_session}")
    
    def test_session_transitions(self):
        """Test session transitions functionality"""
        # Get current session
        original_session = self.session_manager.get_current_session_name()
        
        # Create a mock session that overlaps with current time
        # This tests the session prioritization logic
        current_hour_utc = datetime.now(timezone.utc).hour
        
        # Create a test session that is active now
        self.session_manager.add_session(
            "TEST_SESSION",
            (current_hour_utc - 1) % 24,  # Start 1 hour ago
            (current_hour_utc + 1) % 24,  # End 1 hour from now
            "Test Session"
        )
        
        # Update session parameters for test session
        self.session_manager.update_session_parameter(
            "TEST_SESSION",
            "position_size_factor",
            2.0  # Double the position size
        )
        
        # Force update current session
        self.session_manager.update_current_session()
        
        # Get new current session
        new_session = self.session_manager.get_current_session_name()
        
        # Verify session transition
        assert new_session == "TEST_SESSION", f"Session should transition to TEST_SESSION, got {new_session}"
        
        # Get parameters for new session
        params = self.session_manager.get_all_session_parameters()
        
        # Verify parameters are updated
        assert "position_size_factor" in params, "Session parameters should include position_size_factor"
        assert params["position_size_factor"] == 2.0, f"position_size_factor should be 2.0, got {params['position_size_factor']}"
        
        # Clean up
        self.session_manager.remove_session("TEST_SESSION")
        
        # Force update current session
        self.session_manager.update_current_session()
        
        # Verify session is back to original
        final_session = self.session_manager.get_current_session_name()
        assert final_session in ["ASIA", "EUROPE", "US"], f"Final session should be one of ASIA, EUROPE, US, got {final_session}"
        
        logger.info(f"Session transitions passed: {original_session} -> TEST_SESSION -> {final_session}")
    
    def test_signal_generation(self, symbols, duration):
        """Test signal generation with session awareness"""
        # Create signal generator
        self.signal_generator = SignalGenerator(
            client=self.client,
            env_path=self.env_path
        )
        
        # Start signal generator
        self.signal_generator.start(symbols)
        
        try:
            # Run for specified duration
            start_time = time.time()
            end_time = start_time + duration
            
            # Track signals by session
            signals_by_session = {}
            
            while time.time() < end_time:
                # Sleep for a bit
                time.sleep(0.5)
                
                # Print status every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    # Get recent signals
                    signals = self.signal_generator.get_recent_signals(100)
                    
                    # Group by session
                    for signal in signals:
                        session = signal.get("session")
                        if session not in signals_by_session:
                            signals_by_session[session] = 0
                        signals_by_session[session] += 1
                    
                    # Print status
                    logger.info(f"Elapsed: {elapsed:.1f}s, Total signals: {len(signals)}")
            
            # Verify signals were generated
            total_signals = sum(signals_by_session.values())
            assert total_signals > 0, "Signal generator should generate signals"
            
            # Print signals by session
            for session, count in signals_by_session.items():
                logger.info(f"  {session}: {count} signals")
            
            logger.info(f"Signal generation passed: {total_signals} signals generated")
            
        finally:
            # Stop signal generator
            if self.signal_generator:
                self.signal_generator.stop()
    
    def test_decision_making(self, symbols):
        """Test decision making with session awareness"""
        # Create signal generator if not already created
        if not self.signal_generator:
            self.signal_generator = SignalGenerator(
                client=self.client,
                env_path=self.env_path
            )
        
        # Create mock signals for testing
        mock_signals = self._create_mock_signals(symbols)
        
        # Test decision making with US session
        self._force_session("US")
        us_decisions = {}
        
        for symbol in symbols:
            symbol_signals = [s for s in mock_signals if s["symbol"] == symbol]
            decision = self.signal_generator.make_trading_decision(symbol, symbol_signals)
            if decision:
                us_decisions[symbol] = decision
        
        # Test decision making with EUROPE session
        self._force_session("EUROPE")
        europe_decisions = {}
        
        for symbol in symbols:
            symbol_signals = [s for s in mock_signals if s["symbol"] == symbol]
            decision = self.signal_generator.make_trading_decision(symbol, symbol_signals)
            if decision:
                europe_decisions[symbol] = decision
        
        # Test decision making with ASIA session
        self._force_session("ASIA")
        asia_decisions = {}
        
        for symbol in symbols:
            symbol_signals = [s for s in mock_signals if s["symbol"] == symbol]
            decision = self.signal_generator.make_trading_decision(symbol, symbol_signals)
            if decision:
                asia_decisions[symbol] = decision
        
        # Verify decisions were made for all sessions
        for symbol in symbols:
            assert symbol in us_decisions, f"US session should make decision for {symbol}"
            assert symbol in europe_decisions, f"EUROPE session should make decision for {symbol}"
            assert symbol in asia_decisions, f"ASIA session should make decision for {symbol}"
        
        # Verify position sizes differ by session
        for symbol in symbols:
            us_size = us_decisions[symbol]["size"]
            europe_size = europe_decisions[symbol]["size"]
            asia_size = asia_decisions[symbol]["size"]
            
            # Check position size factors from session parameters
            # US: 1.2, EUROPE: 1.0, ASIA: 0.8
            assert us_size > europe_size, f"US position size ({us_size}) should be larger than EUROPE ({europe_size})"
            assert europe_size > asia_size, f"EUROPE position size ({europe_size}) should be larger than ASIA ({asia_size})"
            
            # Verify position size factors are applied correctly
            us_factor = us_decisions[symbol].get("position_size_factor", 1.0)
            europe_factor = europe_decisions[symbol].get("position_size_factor", 1.0)
            asia_factor = asia_decisions[symbol].get("position_size_factor", 1.0)
            
            assert us_factor == 1.2, f"US position size factor should be 1.2, got {us_factor}"
            assert europe_factor == 1.0, f"EUROPE position size factor should be 1.0, got {europe_factor}"
            assert asia_factor == 0.8, f"ASIA position size factor should be 0.8, got {asia_factor}"
        
        logger.info("Decision making with session awareness passed")
    
    def _create_mock_signals(self, symbols):
        """Create mock signals for testing"""
        signals = []
        
        for symbol in symbols:
            # Create BUY signals
            for i in range(5):
                signals.append({
                    "type": "BUY",
                    "source": "test",
                    "strength": 0.5,
                    "timestamp": int(time.time() * 1000),
                    "price": 50000.0 if symbol == "BTCUSDC" else 3000.0,
                    "symbol": symbol,
                    "session": "TEST"
                })
            
            # Create SELL signals
            for i in range(2):
                signals.append({
                    "type": "SELL",
                    "source": "test",
                    "strength": 0.3,
                    "timestamp": int(time.time() * 1000),
                    "price": 50000.0 if symbol == "BTCUSDC" else 3000.0,
                    "symbol": symbol,
                    "session": "TEST"
                })
        
        return signals
    
    def _force_session(self, session_name):
        """Force a specific session for testing"""
        # Temporarily modify the session manager to force a specific session
        original_update = self.signal_generator.session_manager.update_current_session
        
        def mock_update():
            self.signal_generator.session_manager.current_session = session_name
            logger.info(f"Forced session: {session_name}")
            return session_name
        
        # Replace the update method
        self.signal_generator.session_manager.update_current_session = mock_update
        
        # Force update
        self.signal_generator.session_manager.update_current_session()
        
        # Update session parameters in signal generator
        self.signal_generator.session_manager.get_current_session_name()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Session Awareness')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--symbols', default="BTCUSDC,ETHUSDC", help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Create and run test
    test = SessionAwarenessTest(env_path=args.env)
    test.run_tests(symbols=symbols, duration=args.duration)
