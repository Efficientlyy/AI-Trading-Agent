"""
Mock sentiment data provider for testing and development.
"""

import random
import time
from datetime import datetime
from typing import List, Dict, Any, Iterator

from .base_provider import BaseSentimentProvider

class MockSentimentProvider(BaseSentimentProvider):
    """
    Generates synthetic sentiment data for testing.
    """

    SOURCES = ["twitter", "reddit", "news", "fear_greed"]

    def fetch_sentiment_data(self, count: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate a list of mock sentiment records.

        Args:
            count: Number of records to generate.

        Returns:
            List of sentiment records.
        """
        return [self._generate_record() for _ in range(count)]

    def stream_sentiment_data(self, delay: float = 1.0, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Stream mock sentiment data indefinitely.

        Args:
            delay: Delay between records in seconds.
        """
        while True:
            yield self._generate_record()
            time.sleep(delay)

    def _generate_record(self) -> Dict[str, Any]:
        """
        Generate a single mock sentiment record.
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source": random.choice(self.SOURCES),
            "text": "Sample sentiment text.",
            "score": random.uniform(-1, 1),
            "metadata": {"mock": True}
        }