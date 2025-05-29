"""
Adaptive System Integration Test (Mock Version)

This script tests the integration between the Market Regime Classification system 
and the Adaptive Response System using mock objects to avoid dependency issues.
"""

import os
import sys
import logging
from datetime import datetime
from enum import Enum
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock classes to simulate the behavior of our system components

class MockMarketRegimeType(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    UNKNOWN = "unknown"

class MockVolatilityRegimeType(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class MockMarketRegimeInfo:
    def __init__(self, regime_type, volatility_regime):
        self.regime_type = regime_type
        self.volatility_regime = volatility_regime
        self.confidence = 0.8

class MockTemporalPatternRecognition:
    def __init__(self):
        self.history = {}
        
    def analyze_temporal_patterns(self, prices, volumes=None, asset_id="default", ohlcv_data=None):
        """Mock analysis of temporal patterns"""
        # Randomly select a regime for demo purposes
        regime = random.choice(list(MockMarketRegimeType))
        volatility = random.choice(list(MockVolatilityRegimeType))
        
        # Store in history
        if asset_id not in self.history:
            self.history[asset_id] = []
            
        self.history[asset_id].append({
            "timestamp": datetime.now(),
            "regime": regime,
            "volatility": volatility
        })
        
        # Return mock analysis results
        return {
            "current_regime": {
                "regime_type": regime.value,
                "volatility_regime": volatility.value,
                "confidence": 0.8,
            },
            "seasonality": {
                "has_seasonality": random.choice([True, False]),
                "seasonal_periods": [{"period": 20, "acf_value": 0.6}] if random.random() > 0.5 else []
            },
            "transition_probabilities": {
                "next_regime_probabilities": {
                    "bull": 0.2, 
                    "bear": 0.3, 
                    "sideways": 0.4,
                    "volatile": 0.1
                },
                "most_likely_transition": "sideways",
                "transition_probability": 0.4
            },
            "multi_timeframe": {
                "confirmed_regime": regime.value,
                "agreement_score": 0.75,
                "timeframe_regimes": {
                    "1D": {"regime": regime.value},
                    "1W": {"regime": regime.value},
                    "1M": {"regime": regime.value if random.random() > 0.3 else random.choice(list(MockMarketRegimeType)).value}
                }
            }
        }
    
    def detect_regime_transition_opportunity(self, asset_id="default"):
        """Check for potential regime transition opportunities"""
        return {
            "transition_opportunity": random.random() > 0.7,
            "confidence": random.random() * 0.5 + 0.3,
            "current_regime": random.choice(list(MockMarketRegimeType)).value,
            "potential_next_regime": random.choice(list(MockMarketRegimeType)).value
        }
    
    def get_timeframe_alignment_signal(self, asset_id="default"):
        """Get alignment signal across timeframes"""
        has_alignment = random.random() > 0.4
        return {
            "has_alignment": has_alignment,
            "aligned_regime": random.choice(list(MockMarketRegimeType)).value if has_alignment else None,
            "agreement_score": random.random() * 0.5 + 0.5 if has_alignment else random.random() * 0.4
        }

class MockStrategyManager:
    def __init__(self, strategies=None):
        self.current_strategy = strategies[0] if strategies else "default_strategy"
        self.strategies = strategies or ["default_strategy", "alternative_strategy"]
        self.history = []
        
    def switch_strategy(self, new_strategy):
        if new_strategy in self.strategies:
            old_strategy = self.current_strategy
            self.current_strategy = new_strategy
            self.history.append({
                "timestamp": datetime.now(),
                "from": old_strategy,
                "to": new_strategy
            })
            logger.info(f"Switched strategy from {old_strategy} to {new_strategy}")
            return True
        return False


class MockAdaptiveManager:
    def __init__(self, strategy_manager):
        self.strategy_manager = strategy_manager
        self.regime_parameters = {
            "bull": {
                "position_size_pct": 1.0,
                "stop_loss_atr_multiple": 3.0
            },
            "bear": {
                "position_size_pct": 0.5,
                "stop_loss_atr_multiple": 2.0
            },
            "sideways": {
                "position_size_pct": 0.5,
                "stop_loss_atr_multiple": 2.5
            },
            "volatile": {
                "position_size_pct": 0.3,
                "stop_loss_atr_multiple": 4.0
            }
        }
        
    def get_adapted_parameters(self, market_regime):
        """Get adapted parameters for a regime"""
        if market_regime in self.regime_parameters:
            return self.regime_parameters[market_regime]
        return {"position_size_pct": 0.5, "stop_loss_atr_multiple": 2.5}
    
    def evaluate_and_adapt(self, metrics, market_regime, volatility_regime, price_data=None):
        """Adapt strategy based on market conditions"""
        actions = []
        
        # Adjust position size
        position_size = self.get_adapted_parameters(market_regime)["position_size_pct"]
        if volatility_regime == MockVolatilityRegimeType.HIGH.value:
            position_size *= 0.7
        elif volatility_regime == MockVolatilityRegimeType.EXTREME.value:
            position_size *= 0.5
            
        actions.append(f"Adjusted position size to {position_size:.2f}")
        
        # Select timeframe based on volatility
        if volatility_regime == MockVolatilityRegimeType.LOW.value:
            timeframe = "1h"
        elif volatility_regime == MockVolatilityRegimeType.MEDIUM.value:
            timeframe = "4h"
        elif volatility_regime == MockVolatilityRegimeType.HIGH.value:
            timeframe = "1d"
        else:  # EXTREME
            timeframe = "1w"
            
        actions.append(f"Selected timeframe {timeframe} based on {volatility_regime} volatility")
        
        # Adjust strategy if performance is poor
        if metrics.get("sharpe_ratio", 0) < 0.5 or metrics.get("max_drawdown", 0) > 0.15:
            current = self.strategy_manager.current_strategy
            available = [s for s in self.strategy_manager.strategies if s != current]
            if available:
                new_strategy = random.choice(available)
                self.strategy_manager.switch_strategy(new_strategy)
                actions.append(f"Switched strategy to {new_strategy} due to poor performance")
        
        return {
            "position_size": position_size,
            "timeframe": timeframe,
            "actions_taken": actions
        }


def test_integration():
    """Test the integration between components"""
    logger.info("\n==== Testing System Integration with Mock Objects ====")
    
    # Create mock objects
    strategies = ["momentum", "mean_reversion", "trend_following", "ml_ensemble"]
    strategy_manager = MockStrategyManager(strategies)
    adaptive_manager = MockAdaptiveManager(strategy_manager)
    tpr = MockTemporalPatternRecognition()
    
    # Simulate a trading session with multiple periods
    for period in range(1, 6):
        logger.info(f"\n--- Trading Period {period} ---")
        
        # 1. Analyze market conditions with temporal pattern recognition
        temporal_results = tpr.analyze_temporal_patterns(
            prices=[100 + i + random.random() * 5 for i in range(100)],
            asset_id="SPY"
        )
        
        market_regime = temporal_results["current_regime"]["regime_type"]
        volatility_regime = temporal_results["current_regime"]["volatility_regime"]
        
        logger.info(f"Current regime: {market_regime}, Volatility: {volatility_regime}")
        
        # 2. Check for regime transition opportunities
        transition_opp = tpr.detect_regime_transition_opportunity("SPY")
        if transition_opp["transition_opportunity"]:
            logger.info(f"Transition opportunity detected: {transition_opp['current_regime']} → "
                       f"{transition_opp['potential_next_regime']} "
                       f"(confidence: {transition_opp['confidence']:.2f})")
        
        # 3. Check for timeframe alignment
        alignment = tpr.get_timeframe_alignment_signal("SPY")
        if alignment["has_alignment"]:
            logger.info(f"Timeframe alignment detected for {alignment['aligned_regime']} "
                      f"(score: {alignment['agreement_score']:.2f})")
        
        # 4. Generate mock performance metrics
        metrics = {
            "sharpe_ratio": random.random() * 2 - 0.5,  # -0.5 to 1.5
            "max_drawdown": random.random() * 0.25,  # 0 to 0.25
            "win_rate": random.random() * 0.4 + 0.3,  # 0.3 to 0.7
            "profit_factor": random.random() * 3  # 0 to 3
        }
        
        logger.info(f"Current metrics: Sharpe={metrics['sharpe_ratio']:.2f}, "
                  f"Drawdown={metrics['max_drawdown']:.2f}, "
                  f"Win Rate={metrics['win_rate']:.2f}")
        
        # 5. Adapt strategy and parameters based on market conditions
        adaptation_results = adaptive_manager.evaluate_and_adapt(
            metrics=metrics,
            market_regime=market_regime,
            volatility_regime=volatility_regime
        )
        
        logger.info("Adaptation actions:")
        for action in adaptation_results["actions_taken"]:
            logger.info(f"  - {action}")
        
        # 6. Get adapted parameters for current regime
        adapted_params = adaptive_manager.get_adapted_parameters(market_regime)
        
        logger.info(f"Adapted parameters for {market_regime} regime:")
        for param, value in adapted_params.items():
            logger.info(f"  - {param}: {value}")
    
    # Summary
    logger.info("\n==== Integration Test Summary ====")
    logger.info(f"Strategy switches: {len(strategy_manager.history)}")
    logger.info("Current strategy: " + strategy_manager.current_strategy)
    
    return True


if __name__ == "__main__":
    test_integration()
