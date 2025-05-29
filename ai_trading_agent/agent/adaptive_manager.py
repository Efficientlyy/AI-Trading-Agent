import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from ai_trading_agent.optimization.strategy_evaluator import StrategyEvaluator
from ai_trading_agent.optimization.ga_optimizer import GAOptimizer
from ai_trading_agent.market_regime import (
    MarketRegimeType,
    VolatilityRegimeType,
    TemporalPatternRecognition
)

logger = logging.getLogger(__name__)

class AdaptiveStrategyManager:
    """
    Monitors agent performance and market regime, and adaptively switches strategies or 
    adjusts strategy parameters based on market conditions.
    
    Features include:
    - Dynamic timeframe selection based on volatility
    - Adaptive position sizing based on regime classification
    - Autonomous indicator threshold adjustment
    - Predictive strategy switching
    - Reinforcement learning for strategy selection (when enabled)
    """
    def __init__(self, strategy_manager, performance_history: List[Dict[str, Any]], 
                 available_strategies: List[str], optimizer: Optional[GAOptimizer] = None,
                 enable_temporal_adaptation: bool = True,
                 enable_reinforcement_learning: bool = False):
        """
        Initialize the Adaptive Strategy Manager.
        
        Args:
            strategy_manager: Reference to the strategy manager instance
            performance_history: Historical performance metrics
            available_strategies: List of available strategy names
            optimizer: Optional genetic algorithm optimizer for parameter tuning
            enable_temporal_adaptation: Whether to use temporal pattern recognition for adaptation
            enable_reinforcement_learning: Whether to use RL for strategy selection
        """
        self.strategy_manager = strategy_manager
        self.performance_history = performance_history  # List of dicts with metrics per period
        self.available_strategies = available_strategies
        self.optimizer = optimizer
        self.last_switch_reason = None
        self.current_strategy = strategy_manager.current_strategy if hasattr(strategy_manager, 'current_strategy') else None
        
        # New adaptive features
        self.enable_temporal_adaptation = enable_temporal_adaptation
        self.enable_reinforcement_learning = enable_reinforcement_learning
        self.temporal_pattern_recognizer = TemporalPatternRecognition() if enable_temporal_adaptation else None
        
        # Strategy performance tracking by regime
        self.regime_strategy_performance = {
            regime.value: {strat: [] for strat in available_strategies} 
            for regime in MarketRegimeType
        }
        
        # Default parameters by regime (these can be optimized over time)
        self._initialize_regime_parameters()
        
        # Timeframe adaptation settings
        self.available_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        self.current_timeframe = "1h"  # Default
        
        # Indicator threshold adaptation
        self.indicator_thresholds = self._initialize_indicator_thresholds()
        self.threshold_update_frequency = 20  # Update every N periods
        self.adaptation_count = 0

    def _initialize_regime_parameters(self):
        """Initialize default parameter sets for different market regimes."""
        self.regime_parameters = {
            MarketRegimeType.BULL.value: {
                "position_size_pct": 1.0,  # Full position size in bull markets
                "stop_loss_atr_multiple": 3.0,
                "take_profit_atr_multiple": 6.0,
                "entry_momentum_threshold": 0.2,
                "trailing_stop_enabled": True,
                "timeframe_preference": "1h"
            },
            MarketRegimeType.BEAR.value: {
                "position_size_pct": 0.5,  # Reduced position size in bear markets
                "stop_loss_atr_multiple": 2.0,
                "take_profit_atr_multiple": 4.0,
                "entry_momentum_threshold": 0.3,
                "trailing_stop_enabled": True,
                "timeframe_preference": "1d"  # Longer timeframe in bear markets
            },
            MarketRegimeType.SIDEWAYS.value: {
                "position_size_pct": 0.5,
                "stop_loss_atr_multiple": 2.5,
                "take_profit_atr_multiple": 2.5,  # Equal take-profit and stop-loss for range markets
                "entry_momentum_threshold": 0.4,
                "trailing_stop_enabled": False,  # No trailing stop in sideways markets
                "timeframe_preference": "1h"
            },
            MarketRegimeType.VOLATILE.value: {
                "position_size_pct": 0.3,  # Lowest position size in volatile markets
                "stop_loss_atr_multiple": 4.0,  # Wider stops in volatile markets
                "take_profit_atr_multiple": 3.0,
                "entry_momentum_threshold": 0.5,  # Higher threshold for entry
                "trailing_stop_enabled": True,
                "timeframe_preference": "4h"  # Longer timeframe to filter noise
            },
            MarketRegimeType.TRENDING.value: {
                "position_size_pct": 0.8,
                "stop_loss_atr_multiple": 3.0,
                "take_profit_atr_multiple": 5.0,
                "entry_momentum_threshold": 0.3,
                "trailing_stop_enabled": True,
                "timeframe_preference": "4h"
            }
        }
        
        # Default unknown regime parameters
        self.regime_parameters[MarketRegimeType.UNKNOWN.value] = {
            "position_size_pct": 0.5,
            "stop_loss_atr_multiple": 2.5,
            "take_profit_atr_multiple": 3.0,
            "entry_momentum_threshold": 0.3,
            "trailing_stop_enabled": True,
            "timeframe_preference": "1h"
        }
    
    def _initialize_indicator_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default indicator thresholds that can be adapted over time."""
        return {
            "rsi": {
                "oversold": 30.0,
                "overbought": 70.0,
                "neutral": 50.0
            },
            "macd": {
                "signal_threshold": 0.0
            },
            "bollinger": {
                "band_width": 2.0
            },
            "atr": {
                "volatility_threshold": 0.02
            },
            "adx": {
                "trend_strength": 25.0,
                "strong_trend": 40.0
            },
            "sentiment": {
                "positive_threshold": 0.6,
                "negative_threshold": 0.4
            }
        }

    def evaluate_and_adapt(self, metrics: Dict[str, Any], market_regime: Optional[str] = None, 
                          volatility_regime: Optional[str] = None, 
                          price_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate current strategy performance and decide whether to adapt parameters,
        switch strategies, or optimize based on market conditions.
        
        Args:
            metrics: Performance metrics for current strategy
            market_regime: Current market regime if known
            volatility_regime: Current volatility regime if known
            price_data: Recent price data for temporal pattern analysis
            
        Returns:
            Dictionary with results of adaptation actions
        """
        # Increment adaptation counter
        self.adaptation_count += 1
        adaptation_results = {"actions_taken": []}
        
        # Get basic metrics
        sharpe = metrics.get('sharpe_ratio')
        drawdown = metrics.get('max_drawdown')
        win_rate = metrics.get('win_rate')
        strategy_name = self.current_strategy
        
        # Detect market regime if not provided
        if market_regime is None and price_data and self.enable_temporal_adaptation:
            prices = price_data.get('close', [])
            volumes = price_data.get('volume', [])
            
            # Use temporal pattern recognition to detect regime
            if len(prices) > 60 and self.temporal_pattern_recognizer:
                result = self.temporal_pattern_recognizer.analyze_temporal_patterns(
                    prices=prices, 
                    volumes=volumes if len(volumes) > 0 else None
                )
                market_regime = result['current_regime']['regime_type']
                volatility_regime = result['current_regime']['volatility_regime']
                
                # Check for regime transition opportunity
                transition_opportunity = self.temporal_pattern_recognizer.detect_regime_transition_opportunity()
                if transition_opportunity.get('transition_opportunity', False):
                    adaptation_results["regime_transition_opportunity"] = transition_opportunity
                    
                    # Log transition opportunity
                    logger.info(f"Detected regime transition opportunity from {transition_opportunity['current_regime']} "
                               f"to {transition_opportunity['potential_next_regime']} "
                               f"with confidence {transition_opportunity['confidence']:.2f}")
        
        # Store performance metrics for this regime if known
        if market_regime and strategy_name:
            if market_regime in self.regime_strategy_performance:
                if strategy_name in self.regime_strategy_performance[market_regime]:
                    self.regime_strategy_performance[market_regime][strategy_name].append({
                        'timestamp': datetime.now(),
                        'sharpe': sharpe,
                        'drawdown': drawdown,
                        'win_rate': win_rate,
                        'profit_factor': metrics.get('profit_factor', 0)
                    })
                
        # 1. ADAPTIVE POSITION SIZING based on regime
        if market_regime:
            position_size = self.get_adaptive_position_size(market_regime, volatility_regime, metrics)
            adaptation_results["position_size"] = position_size
            adaptation_results["actions_taken"].append(f"Adjusted position size to {position_size:.2f} based on {market_regime} regime")
        
        # 2. DYNAMIC TIMEFRAME SELECTION based on volatility
        if volatility_regime:
            new_timeframe = self.select_optimal_timeframe(volatility_regime, market_regime)
            if new_timeframe != self.current_timeframe:
                self.current_timeframe = new_timeframe
                adaptation_results["timeframe"] = new_timeframe
                adaptation_results["actions_taken"].append(f"Adjusted timeframe to {new_timeframe} based on {volatility_regime} volatility")
        
        # 3. INDICATOR THRESHOLD ADJUSTMENT
        if self.adaptation_count % self.threshold_update_frequency == 0:
            self.adjust_indicator_thresholds(metrics, market_regime)
            adaptation_results["indicator_thresholds"] = self.indicator_thresholds
            adaptation_results["actions_taken"].append("Updated indicator thresholds based on performance")
        
        # 4. PREDICTIVE STRATEGY SWITCHING
        # Only consider switching if strategy is underperforming
        if (sharpe is not None and sharpe < 0.5) or (drawdown is not None and drawdown > 0.15):
            # Try to find a better strategy based on regime and historical performance
            new_strategy = self.select_optimal_strategy(market_regime)
            
            if new_strategy and new_strategy != strategy_name:
                self.strategy_manager.switch_strategy(new_strategy)
                self.current_strategy = new_strategy
                reason = (f"Predictively switched strategy from {strategy_name} to {new_strategy} "
                         f"based on {market_regime} regime performance history")
                logger.info(reason)
                self.last_switch_reason = reason
                adaptation_results["strategy_switch"] = {
                    "from": strategy_name,
                    "to": new_strategy,
                    "reason": reason
                }
                adaptation_results["actions_taken"].append(f"Switched strategy to {new_strategy}")
            
            # If no better strategy is found or we're already using the best one,
            # optimize the current strategy if optimizer is available
            elif self.optimizer is not None:
                self.optimizer.optimize(self.current_strategy)
                reason = f"Triggered optimization for {strategy_name} due to poor performance"
                logger.info(reason)
                self.last_switch_reason = reason
                adaptation_results["optimization"] = {
                    "strategy": strategy_name,
                    "reason": reason
                }
                adaptation_results["actions_taken"].append(f"Optimized parameters for {strategy_name}")
                
        # Return all adaptation results
        return adaptation_results

    def get_adaptive_position_size(self, market_regime: str, volatility_regime: Optional[str] = None, 
                            metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate adaptive position size based on market regime and current volatility.
        
        Args:
            market_regime: Current market regime
            volatility_regime: Current volatility regime if available
            metrics: Performance metrics
            
        Returns:
            Adjusted position size as a percentage (0.0-1.0)
        """
        # Start with base position size for the regime
        base_size = self.regime_parameters.get(market_regime, {}).get("position_size_pct", 0.5)
        
        # Adjust for volatility if available
        if volatility_regime:
            if volatility_regime == VolatilityRegimeType.HIGH.value:
                vol_factor = 0.7  # Reduce by 30% in high volatility
            elif volatility_regime == VolatilityRegimeType.EXTREME.value:
                vol_factor = 0.5  # Reduce by 50% in extreme volatility
            else:
                vol_factor = 1.0  # No change in normal/low volatility
                
            base_size *= vol_factor
        
        # Adjust for recent performance if metrics available
        if metrics:
            # If recent performance is poor, reduce position size
            sharpe = metrics.get('sharpe_ratio')
            if sharpe is not None:
                if sharpe < 0:
                    base_size *= 0.7  # Reduce by 30% for negative Sharpe
                elif sharpe < 1.0:
                    base_size *= 0.85  # Reduce by 15% for suboptimal Sharpe
            
            # If drawdown is significant, reduce position size
            drawdown = metrics.get('max_drawdown')
            if drawdown is not None and drawdown > 0.05:  # 5% drawdown
                # Scale down position size proportionally to drawdown
                # Maximum reduction of 50% at 20% drawdown
                drawdown_factor = max(0.5, 1.0 - drawdown * 2.5)
                base_size *= drawdown_factor
        
        # Ensure position size stays within reasonable bounds
        return max(0.1, min(1.0, base_size))  # Between 10% and 100%
    
    def select_optimal_timeframe(self, volatility_regime: str, market_regime: Optional[str] = None) -> str:
        """
        Select the optimal timeframe based on current market volatility and regime.
        
        Args:
            volatility_regime: Current volatility regime
            market_regime: Optional market regime
            
        Returns:
            Selected timeframe
        """
        # If we have a preferred timeframe for this regime, use it as a base
        base_timeframe = self.current_timeframe
        if market_regime:
            base_timeframe = self.regime_parameters.get(market_regime, {}).get(
                "timeframe_preference", self.current_timeframe)
        
        # Adjust timeframe based on volatility
        if volatility_regime == VolatilityRegimeType.LOW.value:
            # In low volatility, can use shorter timeframes for more trading opportunities
            timeframe_index = max(0, self.available_timeframes.index(base_timeframe) - 1)
            return self.available_timeframes[timeframe_index]
        
        elif volatility_regime == VolatilityRegimeType.HIGH.value:
            # In high volatility, use longer timeframes to filter noise
            timeframe_index = min(len(self.available_timeframes) - 1, 
                                self.available_timeframes.index(base_timeframe) + 1)
            return self.available_timeframes[timeframe_index]
            
        elif volatility_regime == VolatilityRegimeType.EXTREME.value:
            # In extreme volatility, use longest timeframes
            timeframe_index = min(len(self.available_timeframes) - 1,
                                self.available_timeframes.index(base_timeframe) + 2)
            return self.available_timeframes[timeframe_index]
            
        # Default: return the base timeframe
        return base_timeframe
    
    def adjust_indicator_thresholds(self, metrics: Dict[str, Any], market_regime: Optional[str] = None) -> None:
        """
        Autonomously adjust indicator thresholds based on performance metrics and market regime.
        
        Args:
            metrics: Performance metrics from recent trading
            market_regime: Current market regime if available
        """
        # Only adjust if we have meaningful performance metrics
        if not metrics:
            return
            
        # Get key metrics
        win_rate = metrics.get('win_rate')
        false_signals = metrics.get('false_signals', 0)
        profit_factor = metrics.get('profit_factor', 1.0)
        
        # RSI adjustments
        if 'rsi' in self.indicator_thresholds:
            rsi_thresholds = self.indicator_thresholds['rsi']
            
            # If win rate is low and false signals are high, make thresholds more extreme
            if win_rate and win_rate < 0.4 and false_signals > 5:
                # Make oversold more extreme (lower)
                rsi_thresholds['oversold'] = max(20, rsi_thresholds['oversold'] - 2)
                # Make overbought more extreme (higher)
                rsi_thresholds['overbought'] = min(80, rsi_thresholds['overbought'] + 2)
                
            # If profit factor is good but win rate is low, make thresholds more balanced
            elif profit_factor > 1.5 and win_rate and win_rate < 0.5:
                # Adjust oversold/overbought to be more moderate
                rsi_thresholds['oversold'] = min(35, rsi_thresholds['oversold'] + 1)
                rsi_thresholds['overbought'] = max(65, rsi_thresholds['overbought'] - 1)
        
        # ADX adjustments - threshold for what's considered a trend
        if 'adx' in self.indicator_thresholds:
            adx_thresholds = self.indicator_thresholds['adx']
            
            # In trending markets, we can use lower ADX threshold
            if market_regime in [MarketRegimeType.TRENDING.value, MarketRegimeType.BULL.value, MarketRegimeType.BEAR.value]:
                adx_thresholds['trend_strength'] = max(20, adx_thresholds['trend_strength'] - 1)
            
            # In sideways markets, we need higher ADX for confirmed trends
            elif market_regime == MarketRegimeType.SIDEWAYS.value:
                adx_thresholds['trend_strength'] = min(30, adx_thresholds['trend_strength'] + 1)
        
        # Bollinger Band width adjustment based on volatility
        if 'bollinger' in self.indicator_thresholds and market_regime:
            if market_regime == MarketRegimeType.VOLATILE.value:
                # Wider bands in volatile markets
                self.indicator_thresholds['bollinger']['band_width'] = min(2.5, 
                    self.indicator_thresholds['bollinger']['band_width'] + 0.1)
            elif market_regime in [MarketRegimeType.SIDEWAYS.value, MarketRegimeType.BULL.value]:
                # Tighter bands in sideways or bull markets
                self.indicator_thresholds['bollinger']['band_width'] = max(1.5, 
                    self.indicator_thresholds['bollinger']['band_width'] - 0.1)
    
    def select_optimal_strategy(self, market_regime: Optional[str] = None) -> Optional[str]:
        """
        Select the optimal strategy based on historical performance in the current market regime.
        
        Args:
            market_regime: Current market regime if known
            
        Returns:
            Name of the selected strategy or None if current strategy is optimal
        """
        if not market_regime or market_regime not in self.regime_strategy_performance:
            # No regime information, fall back to simple selection logic
            return self._simple_strategy_selection()
            
        # Get performance data for this regime
        regime_performance = self.regime_strategy_performance[market_regime]
        
        # Calculate average performance metrics for each strategy in this regime
        strategy_scores = {}
        
        for strategy, history in regime_performance.items():
            if not history:  # Skip strategies with no history in this regime
                continue
                
            # Take the most recent 5 performance records or all if fewer
            recent_history = history[-5:] if len(history) > 5 else history
            
            # Calculate average metrics
            avg_sharpe = np.mean([record.get('sharpe', 0) for record in recent_history if 'sharpe' in record])
            avg_drawdown = np.mean([record.get('drawdown', 1) for record in recent_history if 'drawdown' in record])
            avg_win_rate = np.mean([record.get('win_rate', 0) for record in recent_history if 'win_rate' in record])
            avg_profit_factor = np.mean([record.get('profit_factor', 0) for record in recent_history if 'profit_factor' in record])
            
            # Calculate a composite score (higher is better)
            # This formula weights metrics differently based on importance
            score = (
                0.4 * avg_sharpe +  # 40% weight on Sharpe ratio
                0.3 * (1 - avg_drawdown) +  # 30% weight on drawdown (inverted)
                0.2 * avg_win_rate +  # 20% weight on win rate
                0.1 * min(3, avg_profit_factor)  # 10% weight on profit factor (capped)
            )
            
            strategy_scores[strategy] = score
        
        # If current strategy is performing well enough, stick with it
        current_score = strategy_scores.get(self.current_strategy, -float('inf'))
        
        # Find best strategy
        best_strategy = None
        best_score = -float('inf')
        
        for strategy, score in strategy_scores.items():
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        # Only switch if the best strategy is significantly better
        # (prevents frequent switching due to small differences)
        if best_strategy and best_score > current_score * 1.2:  # 20% better
            return best_strategy
        
        # Otherwise stick with current strategy
        return None
    
    def _simple_strategy_selection(self) -> Optional[str]:
        """
        Simple strategy selection when regime-specific data isn't available.
        
        Returns:
            Name of the selected strategy or None if current strategy is optimal
        """
        # Extract performance metrics from overall history
        if not self.performance_history:
            return None
            
        # Get the most recent metrics
        recent_metrics = self.performance_history[-1]
        sharpe = recent_metrics.get('sharpe_ratio')
        drawdown = recent_metrics.get('max_drawdown')
        
        # Simple rule-based switching logic
        if sharpe is not None and sharpe < 0.5:
            # Try another strategy if available
            for strat in self.available_strategies:
                if strat != self.current_strategy:
                    return strat
                    
        return None
        
    def get_last_reason(self):
        """
        Get the reason for the last strategy switch or parameter change.
        
        Returns:
            Explanation for the last adaptation
        """
        return self.last_switch_reason
        
    def get_adapted_parameters(self, market_regime: str) -> Dict[str, Any]:
        """
        Get the adapted parameter set for the current market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary of adapted strategy parameters
        """
        if market_regime in self.regime_parameters:
            return self.regime_parameters[market_regime].copy()
        
        # Default to unknown regime parameters
        return self.regime_parameters[MarketRegimeType.UNKNOWN.value].copy()
