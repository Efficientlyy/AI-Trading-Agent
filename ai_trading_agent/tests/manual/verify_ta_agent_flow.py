"""
Manual Verification Script for Technical Analysis Agent Flow

This script provides a simple way to verify the complete flow of the
Technical Analysis Agent, from data mode toggling to signal generation
and routing to the decision component.

Run this script directly to perform verification tests.
"""

import sys
import os
import time
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TAAgentVerification")

# Import project modules
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.orchestration.ta_agent_integration import TechnicalAgentOrchestrator
from ai_trading_agent.common.event_bus import EventBus, get_event_bus

# Test symbols and timeframes
TEST_SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD"]
TEST_TIMEFRAMES = ["1h", "4h", "1d"]

class SignalCapture:
    """Capture and log signals from different queues"""
    def __init__(self):
        self.decision_signals = []
        self.visualization_signals = []
        
    def capture_decision_signal(self, signal):
        """Capture signal from decision queue"""
        logger.info(f"DECISION SIGNAL: {signal}")
        self.decision_signals.append(signal)
        return signal
    
    def capture_visualization_signal(self, signal):
        """Capture signal from visualization queue"""
        logger.info(f"VISUALIZATION SIGNAL: {signal}")
        self.visualization_signals.append(signal)
        return signal
    
    def get_stats(self):
        """Get statistics on captured signals"""
        return {
            "decision_count": len(self.decision_signals),
            "visualization_count": len(self.visualization_signals),
            "signals_by_symbol": self._count_by_field("symbol"),
            "signals_by_direction": self._count_by_field("direction"),
            "signals_by_type": self._count_by_field("signal_type"),
        }
    
    def _count_by_field(self, field):
        """Count signals by a specific field value"""
        counts = {}
        for signal in self.decision_signals:
            value = getattr(signal, field, "unknown")
            if isinstance(value, object) and hasattr(value, "value"):
                value = value.value
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts


async def run_verification_test():
    """Run the verification test for the Technical Analysis Agent flow"""
    logger.info("Starting Technical Analysis Agent verification test")
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Create signal capture
    signal_capture = SignalCapture()
    
    # Create orchestrator
    orchestrator = TechnicalAgentOrchestrator(event_bus=event_bus)
    
    # Register signal consumers
    orchestrator.register_consumer('decision', signal_capture.capture_decision_signal)
    orchestrator.register_consumer('visualization', signal_capture.capture_visualization_signal)
    
    # Start the orchestrator
    logger.info(f"Starting orchestrator with symbols {TEST_SYMBOLS} and timeframes {TEST_TIMEFRAMES}")
    orchestrator.start(TEST_SYMBOLS, TEST_TIMEFRAMES)
    
    # Check initial status
    initial_status = orchestrator.get_status()
    logger.info(f"Initial status: {initial_status}")
    
    # Wait for initial signals to be generated
    logger.info("Waiting for initial signals (10 seconds)...")
    await asyncio.sleep(10)
    
    # Check metrics
    logger.info("Checking metrics after initial period")
    metrics = orchestrator.ta_agent.get_metrics()
    logger.info(f"Agent metrics: {metrics}")
    
    # Get signal statistics
    logger.info("Signal statistics after initial period:")
    stats = signal_capture.get_stats()
    logger.info(json.dumps(stats, indent=2))
    
    # Toggle to real data
    logger.info("Toggling to real data mode...")
    event_bus.publish('data_source_toggled', {'is_mock': False}, source='test_script')
    
    # Wait for change to take effect
    await asyncio.sleep(2)
    
    # Check status after toggle
    status_after_toggle = orchestrator.get_status()
    logger.info(f"Status after toggle: {status_after_toggle}")
    
    # Wait for signals with real data
    logger.info("Waiting for signals with real data (10 seconds)...")
    await asyncio.sleep(10)
    
    # Check metrics again
    logger.info("Checking metrics after real data mode")
    metrics = orchestrator.ta_agent.get_metrics()
    logger.info(f"Agent metrics: {metrics}")
    
    # Get signal statistics
    logger.info("Signal statistics after real data mode:")
    stats = signal_capture.get_stats()
    logger.info(json.dumps(stats, indent=2))
    
    # Toggle back to mock data
    logger.info("Toggling back to mock data mode...")
    event_bus.publish('data_source_toggled', {'is_mock': True}, source='test_script')
    
    # Wait for change to take effect
    await asyncio.sleep(2)
    
    # Check status after toggle back
    status_after_toggle_back = orchestrator.get_status()
    logger.info(f"Status after toggle back: {status_after_toggle_back}")
    
    # Wait for final signals
    logger.info("Waiting for final signals (10 seconds)...")
    await asyncio.sleep(10)
    
    # Final metrics
    logger.info("Final metrics:")
    metrics = orchestrator.ta_agent.get_metrics()
    logger.info(f"Agent metrics: {metrics}")
    
    # Final signal statistics
    logger.info("Final signal statistics:")
    stats = signal_capture.get_stats()
    logger.info(json.dumps(stats, indent=2))
    
    # Stop the orchestrator
    logger.info("Stopping orchestrator")
    orchestrator.stop()
    
    # Summary report
    logger.info("=== TEST SUMMARY ===")
    logger.info(f"Total decision signals: {len(signal_capture.decision_signals)}")
    logger.info(f"Total visualization signals: {len(signal_capture.visualization_signals)}")
    
    signal_flow_verified = len(signal_capture.decision_signals) > 0
    data_toggle_verified = (initial_status['data_source'] != status_after_toggle['data_source'] and 
                           status_after_toggle['data_source'] != status_after_toggle_back['data_source'])
    
    if signal_flow_verified and data_toggle_verified:
        logger.info("✅ VERIFICATION SUCCESSFUL: Technical Analysis Agent flow is working correctly")
    else:
        logger.error("❌ VERIFICATION FAILED")
        if not signal_flow_verified:
            logger.error("- No signals were routed to decision queue")
        if not data_toggle_verified:
            logger.error("- Data source toggling did not work correctly")


if __name__ == "__main__":
    asyncio.run(run_verification_test())
