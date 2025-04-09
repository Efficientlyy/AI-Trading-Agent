"""
Base interface for sentiment data providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSentimentProvider(ABC):
    """
    Abstract base class for sentiment data providers.
    """

    @abstractmethod
    def fetch_sentiment_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch sentiment data.

        Returns:
            List of sentiment records, each as a dict with keys like:
            - 'timestamp'
            - 'source' (e.g., 'twitter', 'reddit')
            - 'text'
            - 'score' (raw or processed sentiment score)
            - 'metadata' (optional)
        """
        pass

    @abstractmethod
    def stream_sentiment_data(self, **kwargs):
        """
        Stream sentiment data in real-time or near real-time.

        Should yield sentiment records as dicts.
        """
        pass