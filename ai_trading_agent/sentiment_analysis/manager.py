"""
Manages sentiment data collection, processing, and signal generation.
"""

from typing import List, Dict, Any

from .mock_provider import MockSentimentProvider
from .nlp_pipeline import SentimentNLPProcessor
from .signal_generator import SentimentSignalGenerator

class SentimentManager:
    """
    Integrates sentiment data collection, NLP processing, and signal generation.
    """

    def __init__(self):
        self.provider = MockSentimentProvider()
        self.nlp = SentimentNLPProcessor()
        self.signal_generator = SentimentSignalGenerator()

    def get_sentiment_signal(self) -> str:
        """
        Fetch sentiment data, process it, and generate a trading signal.

        Returns:
            'buy', 'sell', or 'hold'
        """
        raw_data = self.provider.fetch_sentiment_data(count=20)
        processed_data = []

        for record in raw_data:
            cleaned = self.nlp.clean_text(record.get("text", ""))
            score_info = self.nlp.score_sentiment(cleaned)
            processed_data.append({
                "score": score_info["score"],
                "source": record.get("source"),
                "timestamp": record.get("timestamp"),
                "text": cleaned
            })

        signal = self.signal_generator.generate_signal(processed_data)
        return signal