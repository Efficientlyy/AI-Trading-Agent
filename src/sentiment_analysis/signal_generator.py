"""
Generates trading signals based on aggregated sentiment data.
"""

from typing import List, Dict, Any

class SentimentSignalGenerator:
    """
    Aggregates sentiment scores and generates trading signals.
    """

    def __init__(self, buy_threshold: float = 0.3, sell_threshold: float = -0.3):
        """
        Initialize the signal generator.

        Args:
            buy_threshold: Sentiment score above which to generate a buy signal.
            sell_threshold: Sentiment score below which to generate a sell signal.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(self, sentiment_data: List[Dict[str, Any]]) -> str:
        """
        Generate a trading signal based on sentiment data.

        Args:
            sentiment_data: List of sentiment records with 'score' key.

        Returns:
            'buy', 'sell', or 'hold'
        """
        if not sentiment_data:
            return "hold"

        avg_score = sum(d.get("score", 0) for d in sentiment_data) / len(sentiment_data)

        if avg_score >= self.buy_threshold:
            return "buy"
        elif avg_score <= self.sell_threshold:
            return "sell"
        else:
            return "hold"