"""
Integration Test for Adaptive Health Orchestrator

This tests the full integration between Market Regime Classification, 
Adaptive Response System, and Health Monitoring within the main
trading agent architecture.
"""

import os
import sys
import unittest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
from ai_trading_agent.agent.adaptive_manager import AdaptiveStrategyManager
from ai_trading_agent.agent.meta_strategy import DynamicAggregationMetaStrategy
from ai_trading_agent.market_regime import MarketRegimeClassifier, MarketRegimeConfig
from ai_trading_agent.market_regime import MarketRegimeType, VolatilityRegimeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockMarketDataAgent:
    """Mock market data agent for testing."""
    
    def __init__(self, use_test_data=True):
        self.use_test_data = use_test_data
        self.test_data = self._create_test_data()
        
    def _create_test_data(self):
        """Create synthetic test data for different market regimes."""
        data = {}
        
        # Create date range for last 2 years of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create synthetic price data for SPY
        n = len(dates)
        
        # Base trend
        base_trend = np.linspace(300, 450, n)
        
        # Add cycles and noise
        cycle1 = 20 * np.sin(np.linspace(0, 15 * np.pi, n))  # Long cycle
        cycle2 = 10 * np.sin(np.linspace(0, 50 * np.pi, n))  # Short cycle
        noise = np.random.normal(0, 5, n)
        
        # COVID crash effect (sharp drop and recovery)
        covid_effect = np.zeros(n)
        crash_start = int(n * 0.25)  # 25% into the series
        crash_end = crash_start + 40
        recovery_end = crash_end + 80
        
        # Generate crash and recovery
        for i in range(crash_start, crash_end):
            covid_effect[i] = -120 * (i - crash_start) / (crash_end - crash_start)
        
        for i in range(crash_end, recovery_end):
            covid_effect[i] = -120 * (1 - (i - crash_end) / (recovery_end - crash_end))
        
        # Create final price series
        prices = base_trend + cycle1 + cycle2 + noise + covid_effect
        
        # Create synthetic volume data (higher in volatile periods)
        base_volume = 5000000 * np.ones(n)
        volume_cycle = 2000000 * np.sin(np.linspace(0, 25 * np.pi, n)) ** 2
        volume_noise = np.random.normal(0, 500000, n)
        
        # Volume spikes during crash
        volume_spikes = np.zeros(n)
        for i in range(crash_start, crash_end + 20):
            spike_factor = 1 + 3 * np.exp(-0.1 * abs(i - crash_start - 10))
            volume_spikes[i] = 5000000 * spike_factor
        
        volumes = base_volume + volume_cycle + volume_noise + volume_spikes
        volumes = np.maximum(volumes, 1000000)  # Ensure minimum volume
        
        # Calculate returns
        returns = np.zeros(n)
        returns[1:] = np.diff(prices) / prices[:-1]
        
        # Calculate high and low prices (simple approximation)
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        
        # Store in SPY data dictionary
        spy_data = {
            'prices': pd.Series(prices, index=dates),
            'volume': pd.Series(volumes, index=dates),
            'high': pd.Series(highs, index=dates),
            'low': pd.Series(lows, index=dates),
            'returns': pd.Series(returns, index=dates)
        }
        
        # Add to overall data dictionary
        data['SPY'] = spy_data
        
        # Also add a few more assets with correlations to SPY
        tickers = ['QQQ', 'IWM', 'TLT', 'GLD']
        correlations = [0.9, 0.85, -0.4, -0.2]  # Correlation with SPY
        
        for ticker, corr in zip(tickers, correlations):
            # Create correlated returns
            correlated_noise = np.random.normal(0, 0.015, n)
            ticker_returns = corr * returns + np.sqrt(1 - corr**2) * correlated_noise
            
            # Convert to prices
            ticker_prices = np.zeros(n)
            ticker_prices[0] = 100  # Starting price
            for i in range(1, n):
                ticker_prices[i] = ticker_prices[i-1] * (1 + ticker_returns[i])
            
            # Add to data dictionary
            data[ticker] = {
                'prices': pd.Series(ticker_prices, index=dates),
                'returns': pd.Series(ticker_returns, index=dates)
            }
        
        return data
    
    def get_market_data(self):
        """Return market data for testing."""
        if self.use_test_data:
            return self.test_data
        else:
            # In a real implementation, this would fetch data from an API
            logger.warning("Real market data fetching not implemented in mock")
            return self.test_data


class MockAdaptiveStrategy(AdaptiveStrategyManager):
    """Mock adaptive strategy for testing the integration."""
    
    def __init__(self):
        # Create mock dependencies
        mock_strategy_manager = type('MockStrategyManager', (), {'current_strategy': 'trend_following'})
        mock_performance_history = [{'sharpe_ratio': 1.2, 'max_drawdown': 0.05, 'win_rate': 0.6}]
        available_strategies = ['momentum', 'mean_reversion', 'trend_following', 'volatility_breakout']
        
        # Initialize with mock values
        super().__init__(
            strategy_manager=mock_strategy_manager,
            performance_history=mock_performance_history,
            available_strategies=available_strategies,
            enable_temporal_adaptation=True
        )
        self.adaptations = []
    
    def evaluate_and_adapt(self, metrics, market_regime=None):
        """Record adaptations for testing."""
        adaptation = f"Adapted to {market_regime or metrics.get('market_regime', 'unknown')} regime"
        self.adaptations.append({
            'time': datetime.now(),
            'regime': market_regime or metrics.get('market_regime', 'unknown'),
            'volatility': metrics.get('volatility_regime', 'unknown'),
            'adaptation': adaptation
        })
        return adaptation


class MockMetaStrategy:
    """Mock meta strategy for testing the integration."""
    
    def __init__(self):
        # Create a simple methods dictionary with lambda placeholders
        self.methods = {
            'majority_vote': lambda x: x,
            'weighted_average': lambda x: x,
            'reinforcement_learning': lambda x: x
        }
        self.current_method = 'majority_vote'
        self.method_selections = []
        self.performance_history = []
        self.method_performance = {
            'majority_vote': {'sharpe': 0.8, 'win_rate': 0.6},
            'weighted_average': {'sharpe': 1.2, 'win_rate': 0.7},
            'reinforcement_learning': {'sharpe': 1.0, 'win_rate': 0.65}
        }
    
    def select_best_method(self, market_conditions):
        """Select method based on market conditions."""
        regime = market_conditions.get('regime', 'unknown')
        
        # Simple mapping for testing
        method_map = {
            'BULL': 'weighted_average',
            'BEAR': 'majority_vote',
            'VOLATILE': 'reinforcement_learning',
            'SIDEWAYS': 'weighted_average',
            'TRENDING': 'weighted_average',
            'CHOPPY': 'majority_vote',
            'BREAKDOWN': 'reinforcement_learning',
            'RECOVERY': 'weighted_average',
        }
        
        selected = method_map.get(regime, 'majority_vote')
        self.current_method = selected
        
        # Record selection
        self.method_selections.append({
            'time': datetime.now(),
            'regime': regime,
            'method': selected
        })
        
        return selected


class AdaptiveOrchestrationIntegrationTest(unittest.TestCase):
    """Integration test for the Adaptive Health Orchestrator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create orchestrator
        self.orchestrator = AdaptiveHealthOrchestrator(
            regime_config=MarketRegimeConfig(),
            temporal_pattern_enabled=True,
            adaptation_interval_minutes=0.01  # Low value for testing
        )
        
        # Create and register agents
        self.data_agent = MockMarketDataAgent()
        self.adaptive_strategy = MockAdaptiveStrategy()
        self.meta_strategy = MockMetaStrategy()
        
        # Register agents
        self.orchestrator.register_agent('market_data', self.data_agent)
        self.orchestrator.register_agent('adaptive_strategy', self.adaptive_strategy)
        self.orchestrator.register_agent('meta_strategy', self.meta_strategy)
        
        # Register market data source
        self.orchestrator.register_market_data_source('market_data', self.data_agent)
        
    def test_market_regime_detection(self):
        """Test market regime detection integration."""
        # Run a cycle to trigger market regime detection and adaptation
        results = self.orchestrator.run_cycle()
        
        # Verify that we detected a market regime
        self.assertIn('current_regime', results)
        self.assertIsNotNone(self.orchestrator.current_regime.get('global'))
        self.assertIn('regime_type', self.orchestrator.current_regime.get('global', {}))
        
        # Get the global regime
        global_regime = self.orchestrator.get_current_regime('global')
        self.assertIsNotNone(global_regime.get('regime_type'))
        self.assertIsNotNone(global_regime.get('volatility_type'))
        self.assertIsNotNone(global_regime.get('confidence'))
        
        logger.info(f"Detected regime: {global_regime.get('regime_type')} with "
                  f"{global_regime.get('volatility_type')} volatility")
    
    def test_adaptive_response(self):
        """Test adaptive response integration."""
        # Run a cycle to trigger market regime detection and adaptation
        self.orchestrator.run_cycle()
        
        # Check that the adaptive strategy was updated
        self.assertTrue(len(self.adaptive_strategy.adaptations) > 0)
        
        # Get the latest adaptation
        latest_adaptation = self.adaptive_strategy.adaptations[-1]
        self.assertIsNotNone(latest_adaptation.get('regime'))
        
        logger.info(f"Adaptation: {latest_adaptation.get('adaptation')}")
        
    def test_meta_strategy_selection(self):
        """Test meta-strategy method selection based on regime."""
        # Run a cycle to trigger market regime detection and adaptation
        self.orchestrator.run_cycle()
        
        # Check that the meta strategy method was selected
        self.assertTrue(len(self.meta_strategy.method_selections) > 0)
        
        # Get the latest method selection
        latest_selection = self.meta_strategy.method_selections[-1]
        self.assertIsNotNone(latest_selection.get('method'))
        self.assertIsNotNone(latest_selection.get('regime'))
        
        logger.info(f"Selected method: {latest_selection.get('method')} for "
                  f"regime: {latest_selection.get('regime')}")
    
    def test_temporal_pattern_integration(self):
        """Test temporal pattern recognition integration."""
        # Run a cycle to trigger market regime detection
        self.orchestrator.run_cycle()
        
        # Get regime history
        history = self.orchestrator.get_regime_history()
        self.assertIsNotNone(history)
        
        # Get regime statistics
        stats = self.orchestrator.get_regime_statistics()
        self.assertIsNotNone(stats)
        
        if 'average_duration' in stats:
            for regime, duration in stats['average_duration'].items():
                logger.info(f"Average duration for {regime}: {duration:.1f} days")
    
    def test_full_integration_cycle(self):
        """Test a full integration cycle with all components."""
        # Run multiple cycles to see adaptation over time
        for i in range(3):
            results = self.orchestrator.run_cycle()
            
            # Check cycle results
            self.assertIsNotNone(results)
            self.assertIn('cycle_duration', results)
            self.assertIn('current_regime', results)
            
            # Log results
            logger.info(f"Cycle {i+1}: Regime={results['current_regime']}, "
                      f"Duration={results['cycle_duration']:.4f}s")
            
            # Wait briefly between cycles
            import time
            time.sleep(0.1)


if __name__ == '__main__':
    unittest.main()
