# ai_trading_agent/sentiment_analysis/analyzer.py
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """
    Analyzes the sentiment of given text using VADER.
    """
    def __init__(self):
        """
        Initializes the SentimentAnalyzer and the VADER tool.
        Downloads necessary NLTK data if not present.
        """
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            print("Downloading VADER lexicon for NLTK...")
            nltk.download('vader_lexicon')
        except LookupError:
             print("Downloading VADER lexicon for NLTK (LookupError)...")
             nltk.download('vader_lexicon')

        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> float:
        """
        Analyzes the sentiment of the input text using VADER.

        Args:
            text: The input text to analyze.

        Returns:
            The compound sentiment score from VADER (between -1.0 and 1.0).
            Positive score indicates positive sentiment, negative score
            indicates negative sentiment, and score around 0 indicates
            neutral sentiment.
        """
        if not isinstance(text, str):
            # Handle non-string input, perhaps log a warning or return neutral
            return 0.0
            
        vs = self.analyzer.polarity_scores(text)
        # We'll use the 'compound' score as the primary sentiment indicator
        return vs['compound']
