"""
Test script to verify that our fixes to the health monitoring imports work properly.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_core_definitions():
    """Test importing the core definitions module."""
    logger.info("Testing import of core_definitions...")
    try:
        from ai_trading_agent.common.health_monitoring.core_definitions import (
            HealthStatus, AlertSeverity, ThresholdType
        )
        logger.info("Successfully imported core definitions")
        return True
    except Exception as e:
        logger.error(f"Failed to import core definitions: {e}")
        return False

def test_health_monitoring_package():
    """Test importing the entire health monitoring package."""
    logger.info("Testing import of health_monitoring package...")
    try:
        import ai_trading_agent.common.health_monitoring
        logger.info("Successfully imported health_monitoring package")
        return True
    except Exception as e:
        logger.error(f"Failed to import health_monitoring package: {e}")
        return False

def test_component_imports():
    """Test importing each component individually."""
    components = [
        "ai_trading_agent.common.health_monitoring.component_health",
        "ai_trading_agent.common.health_monitoring.health_status",
        "ai_trading_agent.common.health_monitoring.alert_manager",
        "ai_trading_agent.common.health_monitoring.heartbeat_manager",
        "ai_trading_agent.common.health_monitoring.health_metrics",
        "ai_trading_agent.common.health_monitoring.recovery_coordinator",
        "ai_trading_agent.common.health_monitoring.health_monitor"
    ]
    
    success = True
    for component in components:
        logger.info(f"Testing import of {component}...")
        try:
            module = __import__(component, fromlist=['*'])
            logger.info(f"Successfully imported {component}")
        except Exception as e:
            logger.error(f"Failed to import {component}: {e}")
            success = False
    
    return success

def test_health_integrated_orchestrator():
    """Test importing the health integrated orchestrator."""
    logger.info("Testing import of health_integrated_orchestrator...")
    try:
        from ai_trading_agent.agent.health_integrated_orchestrator import (
            HealthIntegratedOrchestrator
        )
        logger.info("Successfully imported HealthIntegratedOrchestrator")
        return True
    except Exception as e:
        logger.error(f"Failed to import HealthIntegratedOrchestrator: {e}")
        return False

def run_tests():
    """Run all tests."""
    tests = [
        ("Core Definitions", test_core_definitions),
        ("Health Monitoring Package", test_health_monitoring_package),
        ("Component Imports", test_component_imports),
        ("Health Integrated Orchestrator", test_health_integrated_orchestrator)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n=== Testing {name} ===")
        success = test_func()
        results.append((name, success))
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name} test: {status}")
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = all(success for _, success in results)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")
    
    overall = "PASSED" if all_passed else "FAILED"
    logger.info(f"\nOverall result: {overall}")
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
