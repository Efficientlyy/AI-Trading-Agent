"""
NLP processing pipeline for sentiment analysis.
"""

import re
from typing import Dict, Any, Optional

from .nlp_processing import TextPreprocessor
from .nlp_processing import EntityExtractor

from textblob import TextBlob  # For initial sentiment scoring, can replace with transformers later

class SentimentNLPProcessor:
    """
    Processes raw text into cleaned text, sentiment scores, and optional entities.
    Advanced text preprocessing using NLTK-based pipeline and entity extraction for financial terms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NLP processor with optional preprocessing and entity extraction config.
        """
        if config is None:
            config = {
                "remove_numbers": True,
                "custom_stop_words": [],
                "entity_extraction": {}
            }
        self.preprocessor = TextPreprocessor(config)
        self.entity_extractor = EntityExtractor(config.get("entity_extraction", {}))

    def clean_text(self, text: str, remove_stop_words: bool = True, lemmatize: bool = True) -> str:
        """
        Clean and normalize raw text using advanced preprocessing.

        Args:
            text: Raw input text.
            remove_stop_words: Whether to remove stop words.
            lemmatize: Whether to lemmatize words.

        Returns:
            Cleaned text.
        """
        return self.preprocessor.preprocess(text, remove_stop_words=remove_stop_words, lemmatize=lemmatize)

    def extract_entities(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract entities from text using the advanced EntityExtractor.

        Args:
            text: Input text.

        Returns:
            Dictionary with extracted entities (asset_symbols, financial_terms, prices, cashtags).
        """
        return self.entity_extractor.extract_entities(text)
        return self.entity_extractor.extract_entities(text)

    def score_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Score sentiment of the input text.

        Args:
            text: Cleaned input text.

        Returns:
            Dict with sentiment score and details.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        return {
            "score": polarity,
            "subjectivity": subjectivity,
            "model": "TextBlob"
        }

    def extract_entities(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract entities from text (optional, placeholder).

        Args:
            text: Cleaned input text.

        Returns:
            Dict of entities or None.
        """
        # TODO: Implement entity recognition (e.g., spaCy, transformers)
        return None