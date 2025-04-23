import logging
from typing import List, Dict, Any, Optional
from ai_trading_agent.optimization.strategy_evaluator import StrategyEvaluator
from ai_trading_agent.optimization.ga_optimizer import GAOptimizer

logger = logging.getLogger(__name__)

class AdaptiveStrategyManager:
    """
    Monitors agent performance and market regime, and adaptively switches strategies or triggers optimization.
    """
    def __init__(self, strategy_manager, performance_history: List[Dict[str, Any]], available_strategies: List[str], optimizer: Optional[GAOptimizer] = None):
        self.strategy_manager = strategy_manager
        self.performance_history = performance_history  # List of dicts with metrics per period
        self.available_strategies = available_strategies
        self.optimizer = optimizer
        self.last_switch_reason = None
        self.current_strategy = strategy_manager.current_strategy if hasattr(strategy_manager, 'current_strategy') else None

    def evaluate_and_adapt(self, metrics: Dict[str, Any], market_regime: Optional[str] = None) -> Optional[str]:
        """
        Evaluate current strategy performance and decide whether to switch or optimize.
        Returns a string describing the action taken (for logging/UI), or None if no action.
        """
        sharpe = metrics.get('sharpe_ratio')
        drawdown = metrics.get('max_drawdown')
        win_rate = metrics.get('win_rate')
        strategy_name = self.current_strategy

        # Simple rule-based switching logic (can expand later)
        if sharpe is not None and sharpe < 1.0:
            # Try another strategy if available
            for strat in self.available_strategies:
                if strat != strategy_name:
                    self.strategy_manager.switch_strategy(strat)
                    self.current_strategy = strat
                    reason = f"Switched strategy from {strategy_name} to {strat} due to low Sharpe ratio ({sharpe:.2f})"
                    logger.info(reason)
                    self.last_switch_reason = reason
                    return reason
        if drawdown is not None and drawdown > 0.10:
            # Trigger parameter optimization for current strategy
            if self.optimizer is not None:
                self.optimizer.optimize(self.current_strategy)
                reason = f"Triggered optimization for {strategy_name} due to high drawdown ({drawdown:.2%})"
                logger.info(reason)
                self.last_switch_reason = reason
                return reason
        # Add additional rules (market regime, win rate, etc.) here
        return None

    def get_last_reason(self):
        return self.last_switch_reason
