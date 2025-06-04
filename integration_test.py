#!/usr/bin/env python
"""
Integration Test Script for Trading-Agent System

This script tests the integration of all components in the Trading-Agent system,
including visualization, risk management, performance optimization, and monitoring.
"""

import os
import sys
import time
import logging
import unittest
import threading
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_test")

# Import system components
try:
    # Import visualization components
    from multi_asset_data_service import MultiAssetDataService
    from advanced_chart_component import AdvancedChartComponent
    from dashboard_ui import DashboardUI
    
    # Import risk management components
    from risk_management import RiskManager, RiskDashboard, RiskParameters
    
    # Import performance optimization components
    from performance_optimization import (
        DataStreamManager, OptimizedDataCache, BatchProcessor,
        DataAggregator, OptimizedChartDataProcessor, UIUpdateOptimizer,
        WebSocketOptimizer
    )
    
    # Import error handling and logging components
    from error_handling_and_logging import (
        LoggerFactory, ErrorHandler, retry, log_execution_time,
        PerformanceMonitor, WebSocketLogger, APILogger
    )
    
    # Import monitoring components
    from monitoring_dashboard_service import MonitoringDashboardService
    
    # Import trading components
    from enhanced_dl_integration_fixed import EnhancedPatternRecognitionIntegration
    from enhanced_flash_trading_signals import EnhancedFlashTradingSignals
    from rl_agent_fixed_v4 import TradingRLAgent
    from optimized_mexc_client import OptimizedMEXCClient
    from execution_optimization import OrderRouter
    
    COMPONENTS_IMPORTED = True
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    COMPONENTS_IMPORTED = False

class IntegrationTest(unittest.TestCase):
    """Integration tests for Trading-Agent system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Setting up integration test environment")
        
        # Skip tests if components not imported
        if not COMPONENTS_IMPORTED:
            logger.warning("Skipping tests due to import errors")
            return
        
        # Create mock exchange client
        cls.exchange_client = MagicMock(spec=OptimizedMEXCClient)
        cls.exchange_client.get_ticker.return_value = {"price": 35000.0}
        cls.exchange_client.get_klines.return_value = [
            {
                "time": time.time() * 1000 - i * 60000,
                "open": 35000.0 - i * 10,
                "high": 35100.0 - i * 10,
                "low": 34900.0 - i * 10,
                "close": 35050.0 - i * 10,
                "volume": 10.0 + i
            }
            for i in range(100)
        ]
        
        # Initialize components
        cls._initialize_components()
    
    @classmethod
    def _initialize_components(cls):
        """Initialize system components"""
        try:
            # Initialize data service
            cls.data_service = MultiAssetDataService()
            cls.data_service.exchange_client = cls.exchange_client
            
            # Initialize chart component
            cls.chart_component = AdvancedChartComponent(cls.data_service)
            
            # Initialize risk manager
            cls.risk_manager = RiskManager(portfolio_value=10000.0)
            
            # Initialize risk dashboard
            cls.risk_dashboard = RiskDashboard(cls.risk_manager)
            
            # Initialize performance components
            cls.data_stream_manager = DataStreamManager()
            cls.data_cache = OptimizedDataCache()
            cls.batch_processor = BatchProcessor(lambda x: None)
            cls.data_aggregator = DataAggregator()
            cls.chart_processor = OptimizedChartDataProcessor()
            cls.ui_optimizer = UIUpdateOptimizer()
            cls.ws_optimizer = WebSocketOptimizer()
            
            # Initialize trading components
            cls.pattern_recognition = EnhancedPatternRecognitionIntegration()
            cls.trading_signals = EnhancedFlashTradingSignals(
                pattern_recognition=cls.pattern_recognition
            )
            cls.trading_agent = TradingRLAgent(state_dim=10, action_dim=3)
            cls.order_router = OrderRouter()
            
            # Initialize monitoring dashboard
            cls.monitoring_dashboard = MonitoringDashboardService(debug=False)
            
            logger.info("All components initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def test_data_service_initialization(self):
        """Test data service initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.data_service)
        self.assertEqual(len(self.data_service.get_supported_assets()), 3)
        self.assertIn("BTC/USDC", self.data_service.get_supported_assets())
        self.assertIn("ETH/USDC", self.data_service.get_supported_assets())
        self.assertIn("SOL/USDC", self.data_service.get_supported_assets())
    
    def test_chart_component_initialization(self):
        """Test chart component initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.chart_component)
        self.assertGreater(len(self.chart_component.get_available_indicators()), 0)
        
        # Test adding indicator
        result = self.chart_component.add_indicator("RSI")
        self.assertTrue(result)
        self.assertIn("RSI", self.chart_component.get_indicators())
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.risk_manager)
        
        # Test position sizing
        position_size = self.risk_manager.calculate_position_size("BTC/USDC", 35000.0)
        self.assertGreater(position_size, 0)
        
        # Test stop-loss calculation
        stop_loss = self.risk_manager.calculate_stop_loss("BTC/USDC", 35000.0, "buy")
        self.assertLess(stop_loss, 35000.0)
        
        # Test take-profit calculation
        take_profit = self.risk_manager.calculate_take_profit("BTC/USDC", 35000.0, "buy")
        self.assertGreater(take_profit, 35000.0)
    
    def test_risk_dashboard_initialization(self):
        """Test risk dashboard initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.risk_dashboard)
        
        # Test dashboard data
        dashboard_data = self.risk_dashboard.get_dashboard_data()
        self.assertIn("risk_indicators", dashboard_data)
        self.assertIn("positions", dashboard_data)
        self.assertIn("trade_history", dashboard_data)
    
    def test_performance_components_initialization(self):
        """Test performance components initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.data_stream_manager)
        self.assertIsNotNone(self.data_cache)
        self.assertIsNotNone(self.batch_processor)
        self.assertIsNotNone(self.data_aggregator)
        self.assertIsNotNone(self.chart_processor)
        self.assertIsNotNone(self.ui_optimizer)
        self.assertIsNotNone(self.ws_optimizer)
        
        # Test data stream manager
        self.data_stream_manager.create_stream("test_stream")
        self.data_stream_manager.push_data("test_stream", {"value": 100})
        data = self.data_stream_manager.get_data("test_stream")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["value"], 100)
        
        # Test data cache
        self.data_cache.set("test_key", "test_value")
        value = self.data_cache.get("test_key")
        self.assertEqual(value, "test_value")
    
    def test_trading_components_initialization(self):
        """Test trading components initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.pattern_recognition)
        self.assertIsNotNone(self.trading_signals)
        self.assertIsNotNone(self.trading_agent)
        self.assertIsNotNone(self.order_router)
    
    def test_monitoring_dashboard_initialization(self):
        """Test monitoring dashboard initialization"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        self.assertIsNotNone(self.monitoring_dashboard)
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow"""
        if not COMPONENTS_IMPORTED:
            self.skipTest("Components not imported")
        
        try:
            # 1. Update price data
            self.risk_manager.update_price_data("BTC/USDC", 35000.0, 10.0)
            
            # 2. Open position
            position = self.risk_manager.open_position("BTC/USDC", "buy", 0.1, 35000.0)
            self.assertIsNotNone(position)
            self.assertEqual(position["symbol"], "BTC/USDC")
            self.assertEqual(position["side"], "buy")
            self.assertEqual(position["quantity"], 0.1)
            self.assertEqual(position["entry_price"], 35000.0)
            
            # 3. Update trailing stop
            trailing_stop = self.risk_manager.update_trailing_stop("BTC/USDC", 36000.0, "buy")
            self.assertIsNotNone(trailing_stop)
            self.assertLess(trailing_stop, 36000.0)
            
            # 4. Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            self.assertIn("portfolio_value", risk_metrics)
            self.assertIn("daily_pnl", risk_metrics)
            self.assertIn("risk_level", risk_metrics)
            self.assertIn("trading_status", risk_metrics)
            
            # 5. Get dashboard data
            dashboard_data = self.risk_dashboard.get_dashboard_data()
            self.assertIn("risk_indicators", dashboard_data)
            self.assertIn("positions", dashboard_data)
            self.assertEqual(len(dashboard_data["positions"]), 1)
            
            # 6. Process chart data
            klines = self.exchange_client.get_klines()
            processed_klines = self.chart_processor.process_klines(klines)
            self.assertEqual(len(processed_klines), len(klines))
            
            # 7. Close position
            self.risk_manager._close_position("BTC/USDC", 36000.0, "take_profit")
            positions = self.risk_manager.get_all_positions()
            self.assertEqual(len(positions), 0)
            
            # 8. Get trade history
            trade_history = self.risk_manager.get_trade_history()
            self.assertEqual(len(trade_history), 1)
            self.assertEqual(trade_history[0]["symbol"], "BTC/USDC")
            self.assertEqual(trade_history[0]["reason"], "take_profit")
            
            logger.info("End-to-end workflow test completed successfully")
        
        except Exception as e:
            logger.error(f"Error in end-to-end workflow test: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logger.info("Cleaning up integration test environment")

def run_tests():
    """Run integration tests"""
    logger.info("Starting integration tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Integration tests completed")

if __name__ == "__main__":
    run_tests()
