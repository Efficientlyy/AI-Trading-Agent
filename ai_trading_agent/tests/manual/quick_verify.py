"""
Quick verification script for the Technical Analysis Agent's data mode toggle functionality.

This script directly tests the key components we've modified without requiring the full application.
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import time

# Add project root to Python path
script_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuickVerification")

# Import just the components we need to test
try:
    from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
    from ai_trading_agent.common.event_bus import EventBus, get_event_bus
    
    def test_data_mode_toggle():
        """Test the technical analysis agent's data mode toggle functionality"""
        logger.info("===== TESTING TECHNICAL ANALYSIS AGENT DATA MODE TOGGLE =====")
        
        # Create the technical analysis agent
        logger.info("Creating Technical Analysis Agent...")
        ta_agent = AdvancedTechnicalAnalysisAgent()
        
        # Get initial data mode
        initial_mode = ta_agent.get_data_source_type()
        logger.info(f"Initial data mode: {initial_mode}")
        
        # Toggle data mode
        logger.info("Toggling data mode...")
        new_mode = ta_agent.toggle_data_source()
        logger.info(f"Data mode toggled to: {new_mode}")
        
        # Verify toggle worked
        current_mode = ta_agent.get_data_source_type()
        logger.info(f"Current data mode: {current_mode}")
        
        if initial_mode != current_mode:
            logger.info("✅ Data mode toggle SUCCESSFUL")
        else:
            logger.error("❌ Data mode toggle FAILED")
        
        # Toggle back to original mode
        logger.info("Toggling data mode back...")
        back_mode = ta_agent.toggle_data_source()
        logger.info(f"Data mode toggled back to: {back_mode}")
        
        # Get metrics
        metrics = ta_agent.get_metrics()
        logger.info(f"Agent metrics: {json.dumps({k: v for k, v in metrics.items() if not callable(v)}, indent=2)}")
        
        logger.info("===== TECHNICAL ANALYSIS AGENT DATA MODE TOGGLE TEST COMPLETE =====")
        
    def test_event_bus_integration():
        """Test the event bus integration for data mode toggle"""
        logger.info("===== TESTING EVENT BUS INTEGRATION FOR DATA MODE TOGGLE =====")
        
        # Create event bus
        event_bus = get_event_bus()
        
        # Create test event handler
        event_received = False
        
        def test_handler(event):
            nonlocal event_received
            logger.info(f"Received event: {event}")
            logger.info(f"Event data: {event.data}")
            event_received = True
        
        # Subscribe to data_source_toggled events
        event_bus.subscribe('data_source_toggled', test_handler)
        
        # Publish test event
        logger.info("Publishing test data_source_toggled event...")
        event_bus.publish('data_source_toggled', {'is_mock': True}, source='test_script')
        
        # Wait a moment for event to be processed
        time.sleep(1)
        
        if event_received:
            logger.info("✅ Event bus integration SUCCESSFUL")
        else:
            logger.error("❌ Event bus integration FAILED")
        
        logger.info("===== EVENT BUS INTEGRATION TEST COMPLETE =====")
    
    def run_all_tests():
        """Run all verification tests"""
        test_data_mode_toggle()
        print("\n")
        test_event_bus_integration()
    
    if __name__ == "__main__":
        run_all_tests()
        
except Exception as e:
    logger.error(f"Error during verification: {str(e)}")
    import traceback
    traceback.print_exc()
