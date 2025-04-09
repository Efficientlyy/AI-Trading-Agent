from typing import Dict, List, Any, Optional
import pandas as pd
from src.trading_engine.models import Order
from src.strategies.base_strategy import BaseStrategy

class SentimentStrategy(BaseStrategy):
    """
    Abstract base class for sentiment-based trading strategies.
    """
    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        pass

class DummySentimentStrategy(SentimentStrategy):
    """
    Concrete dummy implementation of SentimentStrategy to fix instantiation errors.
    """
    def __init__(self, symbols: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(symbols or [], parameters or {}, "DummySentimentStrategy")
        self.source_weights = (parameters or {}).get('source_weights', {})
        self.trade_history = []

    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        # No-op dummy implementation
        pass

    def generate_orders(self, signals: pd.DataFrame, timestamp: pd.Timestamp,
                       current_positions: Dict[str, Any]) -> List[Order]:
        # Dummy implementation returns empty order list
        return []

    def generate_signals(
        self,
        data: Any = None,
        portfolio: Any = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # Dummy implementation returns empty DataFrame
        return pd.DataFrame()

    def update_trade_history(self, trade_result: Dict[str, Any]):
        # Dummy implementation appends to history
        self.trade_history.append(trade_result)
