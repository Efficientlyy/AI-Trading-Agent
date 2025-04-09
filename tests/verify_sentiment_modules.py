"""
Simple script to verify that the sentiment analysis modules can be imported correctly.
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the modules
try:
    print("Importing data_collection module...")
    from src.sentiment_analysis.data_collection import (
        BaseSentimentCollector,
        TwitterSentimentCollector,
        RedditSentimentCollector,
        NewsAPISentimentCollector,
        FearGreedIndexCollector,
        SentimentCollectionService
    )
    print("✓ Successfully imported data_collection module")
    
    print("\nImporting nlp_processing module...")
    from src.sentiment_analysis.nlp_processing import (
        TextPreprocessor,
        SentimentAnalyzer,
        EntityExtractor,
        NLPPipeline
    )
    print("✓ Successfully imported nlp_processing module")
    
    print("\nImporting strategy module...")
    from src.sentiment_analysis.strategy import SentimentStrategy
    print("✓ Successfully imported strategy module")
    
    print("\nImporting service module...")
    from src.sentiment_analysis.service import SentimentAnalysisService
    print("✓ Successfully imported service module")
    
    print("\nAll sentiment analysis modules imported successfully!")
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Try creating instances of the classes
try:
    print("\nCreating instances of the classes...")
    
    config = {
        'data_collection': {
            'collectors': {
                'twitter': {'api_key': 'dummy'},
                'reddit': {'client_id': 'dummy'},
                'news': {'api_key': 'dummy'},
                'fear_greed': {}
            }
        },
        'nlp_processing': {
            'preprocessing': {'custom_stop_words': ['crypto']},
            'sentiment_analysis': {},
            'entity_extraction': {}
        },
        'strategy': {
            'sentiment_threshold_long': 0.4,
            'sentiment_threshold_short': -0.4,
            'sentiment_window': 5,
            'risk_per_trade': 0.03,
            'max_position_size': 0.25
        },
        'output_dir': './output'
    }
    
    # Create instances
    text_preprocessor = TextPreprocessor(config['nlp_processing']['preprocessing'])
    sentiment_analyzer = SentimentAnalyzer(config['nlp_processing'])
    entity_extractor = EntityExtractor(config['nlp_processing'])
    nlp_pipeline = NLPPipeline(config['nlp_processing'])
    sentiment_strategy = SentimentStrategy(config['strategy'])
    
    # Create the collection service
    collection_service = SentimentCollectionService(config['data_collection'])
    
    # Create the sentiment analysis service
    service = SentimentAnalysisService(config)
    
    print("✓ Successfully created instances of all classes")
    
except Exception as e:
    print(f"Error creating instances: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nVerification completed successfully!")
