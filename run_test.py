"""
Simple script to run the health integrated orchestrator tests.
This is a workaround to see clearer error messages.
"""

import os
import sys
import unittest

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the test case
from tests.unit.agent.test_health_integrated_orchestrator import TestHealthIntegratedOrchestrator

if __name__ == "__main__":
    # Run the test
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealthIntegratedOrchestrator)
    unittest.TextTestRunner(verbosity=2).run(suite)
