"""
Generates trading signals (1 for Buy, -1 for Sell, 0 for Hold)
based on a Series of sentiment scores using simple thresholds.
"""

import pandas as pd

class SentimentSignalGenerator:
    """
    Generates trading signals (1 for Buy, -1 for Sell, 0 for Hold)
    based on a Series of sentiment scores using simple thresholds.
    """

    def __init__(self, buy_threshold: float = 0.1, sell_threshold: float = -0.1):
        """
        Initialize the signal generator with thresholds.

        Args:
            buy_threshold: The sentiment score above which a Buy signal is generated.
            sell_threshold: The sentiment score below which a Sell signal is generated.
        """
        if not isinstance(buy_threshold, (int, float)) or not isinstance(sell_threshold, (int, float)):
            raise TypeError("Thresholds must be numeric.")
        if buy_threshold <= sell_threshold:
            raise ValueError("Buy threshold must be greater than sell threshold.")

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signals_from_scores(self, sentiment_scores: pd.Series) -> pd.Series:
        """
        Generates trading signals based on a Series of sentiment scores.

        Args:
            sentiment_scores: A pandas Series of sentiment scores (e.g., VADER compound scores).
                              The index should ideally align with timestamps/data points.

        Returns:
            A pandas Series containing the trading signals (1, -1, 0), aligned with the input Series index.
            Returns an empty Series if the input is invalid or empty.
        """
        if not isinstance(sentiment_scores, pd.Series):
            print("Warning: Input 'sentiment_scores' is not a pandas Series. Returning empty Series.")
            return pd.Series(dtype=int)

        if sentiment_scores.empty:
            print("Warning: Input 'sentiment_scores' Series is empty. Returning empty Series.")
            return pd.Series(dtype=int)

        # Ensure scores are numeric
        if not pd.api.types.is_numeric_dtype(sentiment_scores):
             print("Warning: Input 'sentiment_scores' Series does not contain numeric data. Returning empty Series.")
             return pd.Series(dtype=int)

        signals = pd.Series(0, index=sentiment_scores.index, dtype=int)  # Default to Hold (0)
        signals.loc[sentiment_scores > self.buy_threshold] = 1       # Buy signal
        signals.loc[sentiment_scores < self.sell_threshold] = -1      # Sell signal
        return signals