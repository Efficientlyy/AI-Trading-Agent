"""
NLP processing for sentiment data.

This module contains functions and classes to process raw sentiment data
(e.g., from Reddit) and extract sentiment scores or other relevant features.
"""

from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Import VADER

class SentimentProcessor:
    """
    Processes raw sentiment data using NLP techniques.
    """
    def __init__(self, **kwargs):
        """
        Initializes the SentimentProcessor.

        Args:
            **kwargs: Additional configuration parameters for NLP models, etc.
        """
        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of raw sentiment data entries using VADER.

        Args:
            raw_data: A list of dictionaries containing raw sentiment data.
                      Each dictionary is expected to have a 'text' key.

        Returns:
            A list of dictionaries with added 'sentiment_score' (compound score from VADER).
        """
        processed_data = []
        for entry in raw_data:
            if 'text' in entry and entry['text']:
                # Get VADER sentiment scores
                vs = self.analyzer.polarity_scores(entry['text'])
                # Add the compound score to the entry
                processed_entry = entry.copy()
                processed_entry['sentiment_score'] = vs['compound']
                processed_data.append(processed_entry)
            else:
                # Handle entries with no text or empty text
                processed_entry = entry.copy()
                processed_entry['sentiment_score'] = 0.0 # Neutral score for no text
                processed_data.append(processed_entry)
        return processed_data

    def process_stream_entry(self, raw_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single raw sentiment data entry from a stream using VADER.

        Args:
            raw_entry: A dictionary containing a single raw sentiment data entry.
                       Expected to have a 'text' key.

        Returns:
            A dictionary with added 'sentiment_score' (compound score from VADER).
        """
        processed_entry = raw_entry.copy()
        if 'text' in raw_entry and raw_entry['text']:
            # Get VADER sentiment scores
            vs = self.analyzer.polarity_scores(raw_entry['text'])
            # Add the compound score to the entry
            processed_entry['sentiment_score'] = vs['compound']
        else:
            # Handle entries with no text or empty text
            processed_entry['sentiment_score'] = 0.0 # Neutral score for no text

        return processed_entry

# No additional functions needed for VADER loading/configuration
