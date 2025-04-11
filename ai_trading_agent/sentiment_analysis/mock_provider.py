"""
Mock sentiment data provider for testing and development.
"""

import random
import time
import warnings 
from datetime import datetime
from typing import List, Dict, Any, Iterator, Optional 
import pandas as pd 

from .base_provider import BaseSentimentProvider

class MockSentimentProvider(BaseSentimentProvider):
    """
    Generates synthetic sentiment data or uses predefined data for testing.
    """

    SOURCES = ["twitter", "reddit", "news", "fear_greed"]

    def __init__(self, predefined_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initializes the provider.

        Args:
            predefined_data: Optional dictionary where keys are symbols and
                             values are DataFrames indexed by datetime, with
                             'sentiment_score' and 'confidence' columns.
        """
        self.predefined_data = predefined_data
        if self.predefined_data:
            # Validate structure slightly
            for symbol, df in self.predefined_data.items():
                if not isinstance(df.index, pd.DatetimeIndex):
                    warnings.warn(f"Index for symbol '{symbol}' in predefined_data is not DatetimeIndex.")
                if 'sentiment_score' not in df.columns:
                    warnings.warn(f"'sentiment_score' column missing for symbol '{symbol}' in predefined_data.")
                # Confidence is optional in tests for now, add check later if needed
                # if 'confidence' not in df.columns:
                #     warnings.warn(f"'confidence' column missing for symbol '{symbol}' in predefined_data.")

    def get_sentiment(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Retrieves predefined sentiment for a specific symbol and timestamp.

        Args:
            symbol: The asset symbol.
            timestamp: The specific timestamp (or date) to look up.

        Returns:
            A dictionary containing 'sentiment_score' and 'confidence',
            or None if no predefined data exists for that symbol/timestamp.
        """
        if self.predefined_data and symbol in self.predefined_data:
            data_for_symbol = self.predefined_data[symbol]
            try:
                # Use asof for robustness against exact time matches
                relevant_data = data_for_symbol.asof(timestamp)
                if pd.notna(relevant_data['sentiment_score']): # Check if score is not NaN
                    score = relevant_data['sentiment_score']
                    # Use confidence if available, otherwise default to 1.0
                    confidence = relevant_data.get('confidence', 1.0) 
                    # Return None if confidence is also NaN (treat as missing)
                    if pd.isna(confidence):
                        return None 
                    return {
                        'sentiment_score': score,
                        'confidence': confidence
                    }
                else:
                    return None # Explicitly return None for NaN scores
            except KeyError: # Handles cases where timestamp is outside the index range
                return None
            except Exception as e:
                warnings.warn(f"Error accessing predefined data for {symbol} at {timestamp}: {e}")
                return None
        return None # No predefined data for this symbol

    def fetch_sentiment_data(self, count: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate a list of mock sentiment records. 
        NOTE: This method does not use predefined data.
        """
        if self.predefined_data:
            warnings.warn("fetch_sentiment_data called on MockSentimentProvider with predefined_data. Generating random data instead.")
        return [self._generate_record(**kwargs) for _ in range(count)]

    def stream_sentiment_data(self, delay: float = 1.0, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Stream mock sentiment data indefinitely.
        NOTE: This method does not use predefined data.
        """
        if self.predefined_data:
            warnings.warn("stream_sentiment_data called on MockSentimentProvider with predefined_data. Generating random data instead.")
        while True:
            yield self._generate_record(**kwargs)
            time.sleep(delay)

    def _generate_record(self, symbol: str = 'UNKNOWN', timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a single mock sentiment record.
        This is primarily a fallback if predefined data isn't used/found.
        """
        ts = timestamp or datetime.utcnow()
        return {
            "timestamp": ts.isoformat(),
            "symbol": symbol, # Add symbol
            "source": random.choice(self.SOURCES),
            "text": f"Sample sentiment text for {symbol}.",
            "score": random.uniform(-1, 1), # Original score range
            "confidence": random.uniform(0.5, 1.0), # Add confidence
            "metadata": {"mock": True, "generated": True}
        }