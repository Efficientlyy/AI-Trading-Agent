"""
Portfolio-level risk management and allocation logic.
"""

from typing import Dict, List, Any
import numpy as np

class PortfolioRiskManager:
    """
    Handles portfolio-level risk management and allocation.
    """

    def __init__(self, max_position_pct=0.2, max_correlation=0.7):
        self.max_position_pct = max_position_pct
        self.max_correlation = max_correlation

    def check_position_size(self, portfolio_value: float, symbol: str, current_positions: Dict[str, float], proposed_qty: float, price: float) -> bool:
        """
        Check if the proposed position size is within allowed limits.

        Returns:
            True if allowed, False otherwise.
        """
        position_value = abs(proposed_qty * price)
        if position_value > portfolio_value * self.max_position_pct:
            return False
        return True

    def check_correlation(self, symbol: str, existing_symbols: List[str], correlation_matrix: Dict[str, Dict[str, float]]) -> bool:
        """
        Check if adding a new position exceeds correlation limits.

        Returns:
            True if allowed, False otherwise.
        """
        for existing in existing_symbols:
            corr = correlation_matrix.get(symbol, {}).get(existing, 0)
            if abs(corr) > self.max_correlation:
                return False
        return True

    def generate_rebalance_signals(self, current_allocations: Dict[str, float], target_allocations: Dict[str, float], threshold: float = 0.05) -> Dict[str, float]:
        """
        Generate rebalance signals based on current vs. target allocations.

        Returns:
            Dict of symbol to target position size adjustments.
        """
        signals = {}
        for symbol, target_weight in target_allocations.items():
            current_weight = current_allocations.get(symbol, 0)
            diff = target_weight - current_weight
            if abs(diff) > threshold:
                signals[symbol] = diff
        return signals