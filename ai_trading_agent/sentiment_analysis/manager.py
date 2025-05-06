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

        # Convert processed data to a pandas Series of sentiment scores
        import pandas as pd
        sentiment_scores = pd.Series([item['score'] for item in processed_data])
        
        # Generate signals using the correct method
        signals = self.signal_generator.generate_signals_from_scores(sentiment_scores)
        
        # Convert numerical signal to string format
        if signals.empty or signals.iloc[-1] == 0:
            signal = 'hold'
        elif signals.iloc[-1] == 1:
            signal = 'buy'
        else:  # -1
            signal = 'sell'
        return signal