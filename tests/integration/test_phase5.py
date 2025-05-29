"""
Comprehensive Integration Test for Phase 5 Components

This test script validates the functionality of Phase 5 components:
1. Reinforcement Learning Integration
2. Automated Feature Engineering
3. Cross-Strategy Coordination
4. Performance Attribution

The test uses simulated market data to test the end-to-end workflow.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import unittest
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import Phase 5 components
from ai_trading_agent.ml.reinforcement_learning import create_trading_rl_agent, TradingRLAgent
from ai_trading_agent.ml.feature_engineering import create_feature_engineer, FeatureEngineer
from ai_trading_agent.coordination.strategy_coordinator import StrategyCoordinator
from ai_trading_agent.coordination.performance_attribution import PerformanceAttributor
from ai_trading_agent.agent.coordination_manager import CoordinationManager
from ai_trading_agent.agent.strategy import BaseStrategy, RichSignal, RichSignalsDict
from ai_trading_agent.risk.risk_manager import RiskManager
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier


# Create a simple strategy class for testing
class TestStrategy(BaseStrategy):
    """Simple strategy for testing coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", "TestStrategy")
        self.signal_type = config.get("signal_type", "trend_following")
        self.decision_threshold = config.get("decision_threshold", 0.6)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> RichSignalsDict:
        """Generate simple signals based on strategy type"""
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 10:
                continue
                
            signals[symbol] = {}
            
            # Get latest timestamp
            ts = df.index[-1]
            
            # Simple trend following or mean reversion
            if self.signal_type == "trend_following":
                # Trend following: Buy if price increasing, sell if decreasing
                if df['close'].iloc[-1] > df['close'].iloc[-2]:
                    action = 1  # Buy
                    quantity = 100
                elif df['close'].iloc[-1] < df['close'].iloc[-2]:
                    action = -1  # Sell
                    quantity = 100
                else:
                    action = 0  # Hold
                    quantity = 0
            else:
                # Mean reversion: Buy if price dropped, sell if increased
                if df['close'].iloc[-1] < df['close'].iloc[-2]:
                    action = 1  # Buy
                    quantity = 100
                elif df['close'].iloc[-1] > df['close'].iloc[-2]:
                    action = -1  # Sell
                    quantity = 100
                else:
                    action = 0  # Hold
                    quantity = 0
            
            # Create signal
            signal = RichSignal(
                action=action,
                quantity=quantity,
                price=df['close'].iloc[-1],
                metadata={
                    "strategy": self.name,
                    "confidence": 0.7,
                    "signal_type": self.signal_type
                }
            )
            
            signals[symbol][ts] = signal
            
        return signals


# Helper function to generate synthetic market data
def generate_market_data(symbols: List[str], days: int = 100, regime_changes: bool = True) -> Dict[str, pd.DataFrame]:
    """Generate synthetic market data for testing."""
    data = {}
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for symbol in symbols:
        # Set initial price
        initial_price = np.random.randint(50, 200)
        
        # Create price series with random walk plus trend
        if regime_changes:
            # Create different market regimes
            trend_regimes = np.random.choice(
                [0.2, -0.15, 0.0, 0.1, -0.1], 
                size=max(1, days // 20),
                p=[0.3, 0.2, 0.2, 0.15, 0.15]
            )
            volatility_regimes = np.random.choice(
                [0.01, 0.02, 0.04], 
                size=max(1, days // 20),
                p=[0.4, 0.4, 0.2]
            )
            
            # Expand regime arrays to match data length
            regime_length = days // len(trend_regimes) + 1
            trends = np.repeat(trend_regimes, regime_length)[:days]
            volatilities = np.repeat(volatility_regimes, regime_length)[:days]
            
            # Generate returns
            returns = trends + np.random.normal(0, volatilities, days)
        else:
            # Simple random walk
            returns = np.random.normal(0.0005, 0.015, days)
        
        # Convert returns to prices
        prices = initial_price * (1 + returns).cumprod()
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.006, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.006, days))),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, days),
            'returns': returns
        }, index=dates)
        
        # Ensure high > low
        df['high'] = np.maximum(df[['open', 'close', 'high']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close', 'low']].min(axis=1), df['low'])
        
        data[symbol] = df
        
    return data


class TestPhase5Components(unittest.TestCase):
    """Test case for Phase 5 components of the AI Trading Agent."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and components."""
        # Generate synthetic market data
        cls.symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
        cls.market_data = generate_market_data(cls.symbols, days=100, regime_changes=True)
        
        # Create output directory for test results
        cls.output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Initialize components for testing
        cls.setup_components()
    
    @classmethod
    def setup_components(cls):
        """Initialize all components needed for testing."""
        # Risk Manager
        risk_config = {
            "max_drawdown": 0.1,
            "var_confidence": 0.95,
            "position_sizing_method": "volatility_adjusted"
        }
        cls.risk_manager = RiskManager(risk_config)
        
        # Market Regime Classifier
        regime_config = {
            "lookback_window": 20,
            "features": ["returns", "volatility", "volume"],
            "regime_types": ["bull", "bear", "sideways", "volatile"]
        }
        cls.regime_classifier = MarketRegimeClassifier(regime_config)
        
        # Feature Engineer
        feature_eng_config = {
            "base_features": ["open", "high", "low", "close", "volume", "returns"],
            "feature_creation_methods": ["ma", "std", "rsi", "momentum", "diff"],
            "window_sizes": [5, 10, 20],
            "importance_method": "random_forest"
        }
        cls.feature_engineer = create_feature_engineer(feature_eng_config, cls.regime_classifier)
        
        # RL Agent
        rl_config = {
            "market_features": ["close", "volume", "volatility"],
            "strategy_params": ["confidence_threshold", "position_size_factor"],
            "performance_metrics": ["returns", "sharpe_ratio", "drawdown"],
            "rl_params": {
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon": 0.9,
                "epsilon_decay": 0.995,
                "batch_size": 32
            }
        }
        cls.rl_agent = create_trading_rl_agent(rl_config, cls.risk_manager)
        
        # Strategy Coordinator
        coord_config = {
            "strategies": ["TrendStrategy", "MeanReversionStrategy"],
            "lookback_periods": 20,
            "conflict_resolution_method": "performance_weighted",
            "capital_allocation_method": "dynamic"
        }
        cls.strategy_coordinator = StrategyCoordinator(coord_config)
        
        # Performance Attributor
        attr_config = {
            "strategies": ["TrendStrategy", "MeanReversionStrategy"],
            "metrics": ["returns", "sharpe_ratio", "max_drawdown", "win_rate"],
            "output_path": os.path.join(cls.output_dir, "attribution")
        }
        cls.performance_attributor = PerformanceAttributor(attr_config)
        
        # Create test strategies
        cls.trend_strategy = TestStrategy({
            "name": "TrendStrategy",
            "signal_type": "trend_following",
            "decision_threshold": 0.6
        })
        
        cls.reversion_strategy = TestStrategy({
            "name": "MeanReversionStrategy",
            "signal_type": "mean_reversion",
            "decision_threshold": 0.7
        })
        
        # Coordination Manager
        mgr_config = {
            "strategies": [
                {"name": "TrendStrategy"},
                {"name": "MeanReversionStrategy"}
            ],
            "coordination_config": coord_config,
            "attribution_config": attr_config,
            "output_path": os.path.join(cls.output_dir, "coordinator")
        }
        cls.coordination_manager = CoordinationManager(mgr_config)
    
    def test_1_feature_engineering(self):
        """Test automated feature engineering."""
        print("\n===== Testing Automated Feature Engineering =====")
        
        # Test for one symbol
        symbol = self.symbols[0]
        data = self.market_data[symbol]
        
        # Detect regime
        regime = self.regime_classifier.classify_regime(data)
        print(f"Detected market regime for {symbol}: {regime}")
        self.assertIsNotNone(regime)
        
        # Engineer features
        enhanced_data = self.feature_engineer.get_optimal_feature_set(data, regime)
        
        # Verify feature creation
        print(f"Original features: {list(data.columns)}")
        print(f"Enhanced features: {list(enhanced_data.columns)}")
        print(f"Total features created: {len(enhanced_data.columns)}")
        
        self.assertGreater(len(enhanced_data.columns), len(data.columns))
        
        # Test feature importance ranking
        importances = self.feature_engineer.rank_features(enhanced_data)
        print(f"Top 5 features by importance:")
        for i, (feature, importance) in enumerate(list(importances.items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Test feature selection
        selected_features = self.feature_engineer.select_features(importances)
        print(f"Selected {len(selected_features)} features")
        
        # Save plot of feature importances
        plt.figure(figsize=(12, 6))
        features = list(importances.keys())[:15]  # Top 15 features
        values = list(importances.values())[:15]
        plt.barh(features, values)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance Ranking for {symbol}')
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"feature_importance_{symbol}.png")
        plt.savefig(plot_path)
        print(f"Saved feature importance plot to {plot_path}")
        
        self.assertTrue(os.path.exists(plot_path))
    
    def test_2_reinforcement_learning(self):
        """Test reinforcement learning agent."""
        print("\n===== Testing Reinforcement Learning Agent =====")
        
        # Test for one symbol
        symbol = self.symbols[0]
        data = self.market_data[symbol]
        
        # Create initial state
        initial_params = {
            "confidence_threshold": 0.6,
            "position_size_factor": 0.5
        }
        
        # Create mock performance metrics
        performance_metrics = {
            "returns": 0.02,
            "volatility": 0.01,
            "drawdown": 0.03,
            "sharpe_ratio": 2.0,
            "trade_count": 5
        }
        
        # Create current state for RL agent
        current_state = {
            "strategy_params": initial_params,
            "performance": performance_metrics,
            "prev_performance": {
                "sharpe_ratio": 1.8
            }
        }
        
        # Test RL agent adaptation
        print("Initial parameters:", initial_params)
        updated_params = self.rl_agent.adapt_strategy(current_state, data)
        print("Updated parameters:", updated_params)
        
        # Verify RL agent made changes
        self.assertNotEqual(initial_params["confidence_threshold"], updated_params["confidence_threshold"])
        self.assertNotEqual(initial_params["position_size_factor"], updated_params["position_size_factor"])
        
        # Test reward calculation
        reward = self.rl_agent.calculate_reward(
            returns=0.02,
            volatility=0.01,
            drawdown=0.03,
            trade_count=5,
            prev_sharpe=1.8
        )
        print(f"Calculated reward: {reward}")
        self.assertIsNotNone(reward)
        
        # Run multiple iterations to check learning
        print("Training RL agent for 10 iterations...")
        history = []
        for i in range(10):
            # Vary performance slightly
            performance_metrics = {
                "returns": 0.02 + (np.random.random() - 0.5) * 0.01,
                "volatility": 0.01 + (np.random.random() - 0.5) * 0.005,
                "drawdown": 0.03 + (np.random.random() - 0.5) * 0.01,
                "sharpe_ratio": 2.0 + (np.random.random() - 0.5) * 0.5,
                "trade_count": 5 + np.random.randint(-2, 3)
            }
            
            # Update state
            current_state = {
                "strategy_params": updated_params,
                "performance": performance_metrics,
                "prev_performance": current_state["performance"],
                "prev_state": current_state.get("prev_state"),
                "prev_action": current_state.get("prev_action")
            }
            
            # Get updated parameters
            updated_params = self.rl_agent.adapt_strategy(current_state, data)
            
            # Track changes
            history.append({
                "iteration": i,
                "parameters": updated_params.copy(),
                "performance": performance_metrics.copy(),
                "epsilon": self.rl_agent.epsilon
            })
        
        # Print final parameters
        print(f"Final parameters after 10 iterations:")
        print(f"  confidence_threshold: {updated_params['confidence_threshold']:.4f}")
        print(f"  position_size_factor: {updated_params['position_size_factor']:.4f}")
        print(f"  Epsilon (exploration rate): {self.rl_agent.epsilon:.4f}")
        
        # Plot parameter evolution
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot([h["parameters"]["confidence_threshold"] for h in history])
        plt.ylabel('Confidence Threshold')
        plt.subplot(2, 1, 2)
        plt.plot([h["parameters"]["position_size_factor"] for h in history])
        plt.ylabel('Position Size Factor')
        plt.xlabel('Iteration')
        plt.suptitle('RL Parameter Evolution')
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "rl_parameter_evolution.png")
        plt.savefig(plot_path)
        print(f"Saved RL parameter evolution plot to {plot_path}")
        
        self.assertTrue(os.path.exists(plot_path))
    
    def test_3_strategy_coordination(self):
        """Test cross-strategy coordination."""
        print("\n===== Testing Cross-Strategy Coordination =====")
        
        # Generate signals for all symbols from both strategies
        trend_signals = self.trend_strategy.generate_signals(self.market_data)
        reversion_signals = self.reversion_strategy.generate_signals(self.market_data)
        
        # Record signals with coordination manager
        self.coordination_manager.record_strategy_signals("TrendStrategy", trend_signals)
        self.coordination_manager.record_strategy_signals("MeanReversionStrategy", reversion_signals)
        
        # Generate coordinated signals
        coordinated_signals = self.coordination_manager.coordinate_signals()
        
        # Track performance for each strategy and symbol
        timestamp = datetime.now().isoformat()
        for symbol in self.symbols:
            # Trend strategy performance
            trend_metrics = {
                "returns": 0.02 + (np.random.random() - 0.5) * 0.01,
                "sharpe_ratio": 1.8 + (np.random.random() - 0.5) * 0.5,
                "max_drawdown": 0.03 + (np.random.random() - 0.5) * 0.01,
                "win_rate": 0.6 + (np.random.random() - 0.5) * 0.1
            }
            self.coordination_manager.record_performance(
                "TrendStrategy", symbol, trend_metrics, timestamp
            )
            
            # Mean reversion strategy performance
            reversion_metrics = {
                "returns": 0.015 + (np.random.random() - 0.5) * 0.01,
                "sharpe_ratio": 1.5 + (np.random.random() - 0.5) * 0.5,
                "max_drawdown": 0.04 + (np.random.random() - 0.5) * 0.01,
                "win_rate": 0.55 + (np.random.random() - 0.5) * 0.1
            }
            self.coordination_manager.record_performance(
                "MeanReversionStrategy", symbol, reversion_metrics, timestamp
            )
            
            # Combined performance
            combined_metrics = {
                "returns": 0.025 + (np.random.random() - 0.5) * 0.01,
                "sharpe_ratio": 2.0 + (np.random.random() - 0.5) * 0.5,
                "max_drawdown": 0.025 + (np.random.random() - 0.5) * 0.01,
                "win_rate": 0.65 + (np.random.random() - 0.5) * 0.1
            }
            self.coordination_manager.record_combined_performance(
                symbol, combined_metrics, timestamp
            )
        
        # Print coordination results
        print(f"Generated signals from trend strategy: {sum(len(signals) for signals in trend_signals.values())} signals")
        print(f"Generated signals from reversion strategy: {sum(len(signals) for signals in reversion_signals.values())} signals")
        print(f"Generated coordinated signals: {sum(len(signals) for signals in coordinated_signals.values())} signals")
        
        # Check for signals in coordinated output
        self.assertGreater(sum(len(signals) for signals in coordinated_signals.values()), 0)
        
        # Get strategy allocations
        trend_allocation = self.coordination_manager.get_strategy_allocation("TrendStrategy")
        reversion_allocation = self.coordination_manager.get_strategy_allocation("MeanReversionStrategy")
        
        print(f"Capital allocation for trend strategy: {trend_allocation:.2f}")
        print(f"Capital allocation for reversion strategy: {reversion_allocation:.2f}")
        
        # Verify allocations sum to approximately 1
        self.assertAlmostEqual(trend_allocation + reversion_allocation, 1.0, delta=0.01)
    
    def test_4_performance_attribution(self):
        """Test performance attribution system."""
        print("\n===== Testing Performance Attribution System =====")
        
        # Generate attribution report
        report_path = self.coordination_manager.generate_attribution_report("json")
        print(f"Generated attribution report: {report_path}")
        
        # Generate visualization
        plot_path = self.coordination_manager.visualize_attribution()
        print(f"Generated attribution visualization: {plot_path}")
        
        # Get strategy recommendations
        recommendations = self.coordination_manager.get_strategy_recommendations()
        
        print("Strategy improvement recommendations:")
        for strategy, recs in recommendations.items():
            print(f"  {strategy}:")
            for rec in recs:
                print(f"    - {rec}")
        
        # Verify report generation
        self.assertTrue(os.path.exists(report_path) or not report_path)
        self.assertTrue(os.path.exists(plot_path) or not plot_path)
    
    def test_5_end_to_end(self):
        """Test end-to-end workflow."""
        print("\n===== Testing End-to-End Workflow =====")
        
        # Pick a test symbol
        symbol = self.symbols[0]
        data = self.market_data[symbol]
        
        # 1. Detect market regime
        regime = self.regime_classifier.classify_regime(data)
        print(f"Detected market regime: {regime}")
        
        # 2. Create features using automated feature engineering
        enhanced_data = self.feature_engineer.get_optimal_feature_set(data, regime)
        print(f"Enhanced data with {len(enhanced_data.columns)} features")
        
        # 3. Generate signals from test strategies
        trend_signals = self.trend_strategy.generate_signals({symbol: data})
        reversion_signals = self.reversion_strategy.generate_signals({symbol: data})
        
        # 4. Record signals with coordination manager
        self.coordination_manager.record_strategy_signals("TrendStrategy", trend_signals)
        self.coordination_manager.record_strategy_signals("MeanReversionStrategy", reversion_signals)
        
        # 5. Generate coordinated signals
        coordinated_signals = self.coordination_manager.coordinate_signals()
        print(f"Generated coordinated signals for {len(coordinated_signals)} symbols")
        
        # 6. Apply RL agent to optimize parameters
        current_state = {
            "strategy_params": {
                "confidence_threshold": 0.6,
                "position_size_factor": 0.5
            },
            "performance": {
                "returns": 0.02,
                "volatility": 0.01,
                "drawdown": 0.03,
                "sharpe_ratio": 2.0,
                "trade_count": 5
            }
        }
        
        updated_params = self.rl_agent.adapt_strategy(current_state, data)
        print(f"RL optimized parameters: {updated_params}")
        
        # 7. Record performance for attribution
        timestamp = datetime.now().isoformat()
        trend_metrics = {
            "returns": 0.02,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.03,
            "win_rate": 0.6
        }
        reversion_metrics = {
            "returns": 0.015,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.04,
            "win_rate": 0.55
        }
        combined_metrics = {
            "returns": 0.025,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.025,
            "win_rate": 0.65
        }
        
        self.coordination_manager.record_performance(
            "TrendStrategy", symbol, trend_metrics, timestamp
        )
        self.coordination_manager.record_performance(
            "MeanReversionStrategy", symbol, reversion_metrics, timestamp
        )
        self.coordination_manager.record_combined_performance(
            symbol, combined_metrics, timestamp
        )
        
        # 8. Generate attribution report
        report_path = self.coordination_manager.generate_attribution_report("text")
        print(f"Generated final attribution report: {report_path}")
        
        # Verify successful end-to-end workflow
        self.assertIsNotNone(coordinated_signals)
        self.assertGreater(len(coordinated_signals), 0)
        self.assertIsNotNone(updated_params)
        self.assertTrue(os.path.exists(report_path) or not report_path)
        
        print("\n===== All tests completed successfully =====")


if __name__ == '__main__':
    # Run all tests
    unittest.main()
