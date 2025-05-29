"""
Real-Time News Impact Analyzer for AI Trading Agent.

This module provides advanced analysis of financial news and its potential impact 
on market sentiment and price movements, including:
- Event classification
- Event impact scoring
- Market reaction prediction
- Automated sentiment sensitivity configuration
"""

import logging
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
from collections import defaultdict

# Import NLP libraries
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Import other sentiment modules
from ai_trading_agent.sentiment.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.common import logger
from ai_trading_agent.market_regime import MarketRegimeType


class NewsImpactAnalyzer:
    """
    Analyzes financial news and events to determine potential market impact.
    
    Features:
    - Event categorization and classification
    - Impact scoring based on event type and historical correlation
    - Temporal decay model for news relevance
    - Cross-asset impact analysis
    - Automated sensitivity adjustment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the news impact analyzer.
        
        Args:
            config: Configuration dictionary containing:
                - impact_threshold: Threshold for significant news impact (0.0-1.0)
                - event_categories: List of event categories to monitor
                - temporal_decay_factor: Factor controlling relevance decay over time
                - max_news_age_hours: Maximum age for news consideration
                - asset_keywords: Dictionary mapping assets to relevant keywords
                - event_impact_weights: Dictionary with event type impact weights
                - sentiment_adjustment_period: Days between sentiment sensitivity adjustments
                - history_file: File to store historical impact data
        """
        self.name = "NewsImpactAnalyzer"
        
        # Core configuration
        self.impact_threshold = config.get('impact_threshold', 0.6)
        self.event_categories = config.get('event_categories', [
            'earnings', 'economic_data', 'central_bank', 'political', 
            'regulatory', 'merger_acquisition', 'market_sentiment'
        ])
        self.temporal_decay_factor = config.get('temporal_decay_factor', 0.85)
        self.max_news_age_hours = config.get('max_news_age_hours', 24)
        self.sentiment_adjustment_period = config.get('sentiment_adjustment_period', 7)
        self.history_file = config.get('history_file', 'news_impact_history.json')
        
        # Asset-specific keywords
        self.asset_keywords = config.get('asset_keywords', {})
        
        # Event impact configuration
        self.event_impact_weights = config.get('event_impact_weights', {
            'earnings': 0.8,
            'economic_data': 0.7,
            'central_bank': 0.9,
            'political': 0.5,
            'regulatory': 0.6,
            'merger_acquisition': 0.7,
            'market_sentiment': 0.4
        })
        
        # Initialize sentiment analysis tools
        self.nltk_analyzer = SentimentIntensityAnalyzer() if NLTK_AVAILABLE else None
        self.sentiment_analyzer = SentimentAnalyzer(config)
        
        # News and event tracking
        self.active_news_events = []
        self.historical_impacts = self._load_historical_impacts()
        self.last_sensitivity_adjustment = datetime.now()
        
        # Cached sentiment scores
        self.sentiment_cache = {}
        self.cache_expiry = config.get('cache_expiry_hours', 2)
        
        # Performance tracking
        self.prediction_accuracy = {
            'total': 0,
            'correct': 0,
            'by_category': defaultdict(lambda: {'total': 0, 'correct': 0})
        }
        
        logger.info(f"Initialized {self.name} with {len(self.event_categories)} event categories")
    
    def analyze_news(self, news_items: List[Dict[str, Any]], 
                    price_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Analyze a batch of news items for potential market impact.
        
        Args:
            news_items: List of news item dictionaries with:
                - title: News headline
                - content: Full news content
                - source: News source
                - published_at: Publication timestamp
                - url: Source URL
                - related_assets: Optional list of asset identifiers
            price_data: Optional dictionary of price data for impact correlation
            
        Returns:
            Dictionary with impact analysis results:
                - impacts: List of impact assessments per news item
                - aggregated_sentiment: Overall sentiment score per asset
                - critical_events: List of high-impact events
                - predicted_reactions: Predicted market reactions per asset
        """
        # Process each news item
        impacts = []
        
        for item in news_items:
            # Skip if missing required fields
            if 'title' not in item or 'published_at' not in item:
                continue
                
            # Check if news is too old
            if self._is_news_too_old(item):
                continue
            
            # Calculate basic sentiment
            sentiment = self._calculate_sentiment(item)
            
            # Classify event type
            event_type = self._classify_event_type(item)
            
            # Identify affected assets
            affected_assets = item.get('related_assets', [])
            if not affected_assets:
                affected_assets = self._identify_affected_assets(item)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(item, event_type, sentiment)
            
            # Apply temporal decay for older news
            impact_score = self._apply_temporal_decay(impact_score, item['published_at'])
            
            # Predict market reaction
            market_reaction = self._predict_market_reaction(
                event_type, sentiment, impact_score, affected_assets
            )
            
            # Create impact assessment
            impact = {
                'news_id': item.get('id', str(hash(item['title']))),
                'headline': item['title'],
                'event_type': event_type,
                'sentiment': sentiment,
                'impact_score': impact_score,
                'affected_assets': affected_assets,
                'published_at': item['published_at'],
                'predicted_reaction': market_reaction,
                'source': item.get('source', 'unknown')
            }
            
            impacts.append(impact)
            
            # Add to active news events if significant impact
            if impact_score >= self.impact_threshold:
                self.active_news_events.append(impact)
        
        # Prune old events
        self._prune_old_events()
        
        # Aggregate sentiment by asset
        aggregated_sentiment = self._aggregate_sentiment_by_asset(impacts)
        
        # Identify critical (high-impact) events
        critical_events = [
            event for event in impacts 
            if event['impact_score'] >= self.impact_threshold
        ]
        
        # Predict market reactions for affected assets
        predicted_reactions = self._aggregate_predicted_reactions(impacts)
        
        # Update historical impacts
        if price_data:
            self._update_historical_impacts(impacts, price_data)
        
        # Check if we should adjust sensitivity
        self._check_sensitivity_adjustment()
        
        # Return complete analysis
        return {
            'impacts': impacts,
            'aggregated_sentiment': aggregated_sentiment,
            'critical_events': critical_events,
            'predicted_reactions': predicted_reactions,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _is_news_too_old(self, news_item: Dict[str, Any]) -> bool:
        """Check if a news item is too old to be relevant."""
        # Parse published timestamp
        if isinstance(news_item['published_at'], str):
            published_time = datetime.fromisoformat(news_item['published_at'].replace('Z', '+00:00'))
        else:
            published_time = news_item['published_at']
        
        age_hours = (datetime.now() - published_time).total_seconds() / 3600
        return age_hours > self.max_news_age_hours
    
    def _calculate_sentiment(self, news_item: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sentiment scores for a news item."""
        # Check cache first
        cache_key = news_item.get('id', news_item['title'])
        if cache_key in self.sentiment_cache:
            cache_entry = self.sentiment_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_expiry * 3600:
                return cache_entry['sentiment']
        
        # Combine title and content for analysis
        text = news_item['title']
        if 'content' in news_item and news_item['content']:
            text += " " + news_item['content']
        
        # Use our main sentiment analyzer
        sentiment_scores = self.sentiment_analyzer.analyze_text(text)
        
        # If NLTK is available, combine with its scores
        if self.nltk_analyzer:
            nltk_scores = self.nltk_analyzer.polarity_scores(text)
            
            # Convert NLTK scores to our format
            nltk_compound = nltk_scores['compound']
            
            # Use compound score to adjust our sentiment
            if nltk_compound > 0.05:
                nltk_sentiment = 'positive'
                nltk_value = nltk_compound
            elif nltk_compound < -0.05:
                nltk_sentiment = 'negative'
                nltk_value = -nltk_compound
            else:
                nltk_sentiment = 'neutral'
                nltk_value = 1 - abs(nltk_compound)
            
            # Blend NLTK and our sentiment (0.3/0.7 weighting)
            for key in sentiment_scores:
                if key == nltk_sentiment:
                    sentiment_scores[key] = 0.7 * sentiment_scores[key] + 0.3 * nltk_value
                else:
                    sentiment_scores[key] = sentiment_scores[key] * 0.7
        
        # Cache the result
        self.sentiment_cache[cache_key] = {
            'sentiment': sentiment_scores,
            'timestamp': datetime.now()
        }
        
        return sentiment_scores
    
    def _classify_event_type(self, news_item: Dict[str, Any]) -> str:
        """Classify the news item into an event type."""
        text = news_item['title'].lower()
        if 'content' in news_item and news_item['content']:
            text += " " + news_item['content'].lower()
        
        # Classification rules
        if re.search(r'earnings|revenue|profit|eps|income statement|quarterly', text):
            return 'earnings'
        elif re.search(r'fed|central bank|rate|monetary|powell|ecb|boj|rba|boe', text):
            return 'central_bank'
        elif re.search(r'gdp|inflation|cpi|unemployment|payroll|economic|retail sales|pmi', text):
            return 'economic_data'
        elif re.search(r'merger|acquisition|takeover|bid|buyout', text):
            return 'merger_acquisition'
        elif re.search(r'regulation|compliance|sec|regulator|fine|settlement', text):
            return 'regulatory'
        elif re.search(r'election|president|congress|senate|house|government|vote|bill|law', text):
            return 'political'
        else:
            return 'market_sentiment'
    
    def _identify_affected_assets(self, news_item: Dict[str, Any]) -> List[str]:
        """Identify assets that might be affected by the news item."""
        affected_assets = []
        
        # Combine title and content
        text = news_item['title'].lower()
        if 'content' in news_item and news_item['content']:
            text += " " + news_item['content'].lower()
        
        # Check each asset's keywords
        for asset, keywords in self.asset_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    affected_assets.append(asset)
                    break
        
        return affected_assets
    
    def _calculate_impact_score(self, news_item: Dict[str, Any], 
                               event_type: str, sentiment: Dict[str, float]) -> float:
        """Calculate the potential market impact score of a news item."""
        # Base impact from event type
        base_impact = self.event_impact_weights.get(event_type, 0.5)
        
        # Sentiment intensity factor
        sentiment_intensity = max(
            sentiment.get('positive', 0), 
            sentiment.get('negative', 0)
        )
        
        # Source credibility factor
        source = news_item.get('source', 'unknown').lower()
        credibility_factor = 1.0
        
        # Adjust credibility based on source
        major_sources = ['bloomberg', 'reuters', 'ft', 'wsj', 'cnbc', 'financial times']
        if any(s in source for s in major_sources):
            credibility_factor = 1.2
        elif source == 'unknown':
            credibility_factor = 0.8
        
        # Calculate final impact score
        impact_score = base_impact * sentiment_intensity * credibility_factor
        
        # Cap at 1.0
        return min(impact_score, 1.0)
    
    def _apply_temporal_decay(self, impact_score: float, published_at: str) -> float:
        """Apply temporal decay to impact score based on news age."""
        if isinstance(published_at, str):
            published_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        else:
            published_time = published_at
        
        # Calculate hours since publication
        hours_old = max(0, (datetime.now() - published_time).total_seconds() / 3600)
        
        # Apply exponential decay
        decay_factor = self.temporal_decay_factor ** (hours_old / 6)  # Half-life of ~6 hours
        
        return impact_score * decay_factor
    
    def _predict_market_reaction(self, event_type: str, sentiment: Dict[str, float],
                               impact_score: float, affected_assets: List[str]) -> Dict[str, Any]:
        """Predict potential market reaction based on news impact."""
        # Determine sentiment direction
        if sentiment.get('positive', 0) > sentiment.get('negative', 0):
            direction = 'positive'
            strength = sentiment.get('positive', 0)
        elif sentiment.get('negative', 0) > sentiment.get('positive', 0):
            direction = 'negative'
            strength = sentiment.get('negative', 0)
        else:
            direction = 'neutral'
            strength = sentiment.get('neutral', 0)
        
        # Adjust reaction based on event type
        reaction_modifier = 1.0
        event_volatility = 1.0
        
        if event_type == 'central_bank':
            reaction_modifier = 1.5
            event_volatility = 1.3
        elif event_type == 'earnings':
            reaction_modifier = 1.2
            event_volatility = 1.4
        elif event_type == 'economic_data':
            reaction_modifier = 1.3
            event_volatility = 1.2
        
        # Calculate expected price movement
        expected_movement = strength * impact_score * reaction_modifier
        
        # Cap movement at reasonable values
        expected_movement = min(expected_movement, 0.1)  # Max 10% move
        
        # Apply sign based on direction
        if direction == 'negative':
            expected_movement = -expected_movement
        elif direction == 'neutral':
            expected_movement = 0.0
        
        # Calculate potential volatility impact
        volatility_impact = impact_score * event_volatility
        
        # Create reaction prediction
        reaction = {
            'direction': direction,
            'expected_movement': expected_movement,
            'confidence': impact_score,
            'volatility_impact': volatility_impact,
            'affected_assets': affected_assets
        }
        
        return reaction
    
    def _aggregate_sentiment_by_asset(self, impacts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate sentiment scores by asset."""
        asset_sentiment = {}
        
        for impact in impacts:
            assets = impact['affected_assets']
            sentiment = impact['sentiment']
            impact_score = impact['impact_score']
            
            for asset in assets:
                if asset not in asset_sentiment:
                    asset_sentiment[asset] = {
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 0.0,
                        'weighted_score': 0.0,
                        'impact_count': 0
                    }
                
                # Weighted sentiment contribution
                for sentiment_type, score in sentiment.items():
                    if sentiment_type in ['positive', 'negative', 'neutral']:
                        asset_sentiment[asset][sentiment_type] += score * impact_score
                
                # Calculate weighted sentiment score (-1 to +1 range)
                pos = asset_sentiment[asset]['positive']
                neg = asset_sentiment[asset]['negative']
                weighted_score = (pos - neg) / (pos + neg + 0.0001)  # Avoid division by zero
                
                asset_sentiment[asset]['weighted_score'] = weighted_score
                asset_sentiment[asset]['impact_count'] += 1
        
        return asset_sentiment
    
    def _aggregate_predicted_reactions(self, impacts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate predicted market reactions by asset."""
        asset_reactions = {}
        
        for impact in impacts:
            reaction = impact['predicted_reaction']
            assets = reaction['affected_assets']
            
            for asset in assets:
                if asset not in asset_reactions:
                    asset_reactions[asset] = {
                        'expected_movement': 0.0,
                        'confidence': 0.0,
                        'volatility_impact': 0.0,
                        'critical_events': [],
                        'impact_count': 0
                    }
                
                # Accumulate expected movement
                asset_reactions[asset]['expected_movement'] += reaction['expected_movement']
                
                # Use max for volatility impact
                asset_reactions[asset]['volatility_impact'] = max(
                    asset_reactions[asset]['volatility_impact'],
                    reaction['volatility_impact']
                )
                
                # Track confidence as average
                current_count = asset_reactions[asset]['impact_count']
                current_confidence = asset_reactions[asset]['confidence']
                new_confidence = (current_confidence * current_count + reaction['confidence']) / (current_count + 1)
                asset_reactions[asset]['confidence'] = new_confidence
                
                # Add critical events
                if impact['impact_score'] >= self.impact_threshold:
                    asset_reactions[asset]['critical_events'].append({
                        'news_id': impact['news_id'],
                        'headline': impact['headline'],
                        'impact_score': impact['impact_score']
                    })
                
                asset_reactions[asset]['impact_count'] += 1
        
        return asset_reactions
    
    def _prune_old_events(self):
        """Remove old events from the active events list."""
        current_time = datetime.now()
        self.active_news_events = [
            event for event in self.active_news_events
            if not self._is_news_too_old(event)
        ]
    
    def _load_historical_impacts(self) -> Dict[str, Any]:
        """Load historical impact data from file."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty structure if file doesn't exist or is invalid
            return {
                'events': [],
                'asset_correlations': {},
                'accuracy': {
                    'total': 0,
                    'correct': 0,
                    'by_category': {}
                }
            }
    
    def _save_historical_impacts(self):
        """Save historical impact data to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.historical_impacts, f)
        except Exception as e:
            logger.error(f"Failed to save historical impacts: {e}")
    
    def _update_historical_impacts(self, impacts: List[Dict[str, Any]], 
                                  price_data: Dict[str, pd.DataFrame]):
        """Update historical impact data with actual market reactions."""
        for impact in impacts:
            # Only process impacts from before
            if impact['news_id'] in [e['news_id'] for e in self.historical_impacts['events']]:
                continue
                
            event_time = datetime.fromisoformat(impact['published_at'].replace('Z', '+00:00'))
            
            # Check reactions for each affected asset
            for asset in impact['affected_assets']:
                if asset not in price_data:
                    continue
                    
                # Get price data for asset
                asset_prices = price_data[asset]
                
                # Find price before and after event
                pre_event_idx = asset_prices.index.get_loc(event_time, method='pad')
                
                # Look for price 24 hours after event
                post_event_time = event_time + timedelta(hours=24)
                try:
                    post_event_idx = asset_prices.index.get_loc(post_event_time, method='pad')
                    
                    # Calculate actual price change
                    pre_price = asset_prices.iloc[pre_event_idx]['close']
                    post_price = asset_prices.iloc[post_event_idx]['close']
                    actual_change = (post_price - pre_price) / pre_price
                    
                    # Compare with predicted change
                    predicted_change = impact['predicted_reaction']['expected_movement']
                    
                    # Consider prediction correct if the direction matches
                    correct = (predicted_change > 0 and actual_change > 0) or \
                              (predicted_change < 0 and actual_change < 0) or \
                              (abs(predicted_change) < 0.001 and abs(actual_change) < 0.01)
                    
                    # Update accuracy statistics
                    self.prediction_accuracy['total'] += 1
                    if correct:
                        self.prediction_accuracy['correct'] += 1
                    
                    event_type = impact['event_type']
                    self.prediction_accuracy['by_category'][event_type]['total'] += 1
                    if correct:
                        self.prediction_accuracy['by_category'][event_type]['correct'] += 1
                    
                    # Save event with actual outcome
                    impact_copy = impact.copy()
                    impact_copy['actual_change'] = actual_change
                    impact_copy['prediction_correct'] = correct
                    
                    self.historical_impacts['events'].append(impact_copy)
                    
                    # Update asset correlations
                    if asset not in self.historical_impacts['asset_correlations']:
                        self.historical_impacts['asset_correlations'][asset] = {
                            'events': 0,
                            'correct': 0,
                            'sensitivity': 1.0
                        }
                    
                    self.historical_impacts['asset_correlations'][asset]['events'] += 1
                    if correct:
                        self.historical_impacts['asset_correlations'][asset]['correct'] += 1
                    
                except (KeyError, IndexError):
                    # Couldn't find post-event price data
                    pass
        
        # Update and save historical data
        self.historical_impacts['accuracy'] = self.prediction_accuracy
        self._save_historical_impacts()
    
    def _check_sensitivity_adjustment(self):
        """Check if we should adjust sentiment sensitivity."""
        days_since_adjustment = (datetime.now() - self.last_sensitivity_adjustment).days
        
        if days_since_adjustment >= self.sentiment_adjustment_period:
            self._adjust_sentiment_sensitivity()
            self.last_sensitivity_adjustment = datetime.now()
    
    def _adjust_sentiment_sensitivity(self):
        """Automatically adjust sentiment sensitivity based on prediction accuracy."""
        # Only adjust if we have enough data
        if self.prediction_accuracy['total'] < 20:
            return
            
        overall_accuracy = self.prediction_accuracy['correct'] / self.prediction_accuracy['total'] \
            if self.prediction_accuracy['total'] > 0 else 0.5
            
        # For each asset, adjust sensitivity
        for asset, stats in self.historical_impacts['asset_correlations'].items():
            if stats['events'] < 5:
                continue
                
            asset_accuracy = stats['correct'] / stats['events'] if stats['events'] > 0 else 0.5
            
            # If accuracy is below 0.5, reduce sensitivity
            if asset_accuracy < 0.5:
                adjustment_factor = 0.9
            # If accuracy is high, increase sensitivity
            elif asset_accuracy > 0.7:
                adjustment_factor = 1.1
            else:
                adjustment_factor = 1.0
            
            # Apply adjustment
            stats['sensitivity'] *= adjustment_factor
            
            # Keep within reasonable bounds
            stats['sensitivity'] = max(0.5, min(stats['sensitivity'], 2.0))
            
            logger.info(f"Adjusted sentiment sensitivity for {asset} to {stats['sensitivity']:.2f} "
                       f"based on accuracy {asset_accuracy:.2f}")
    
    def get_current_sentiment(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current sentiment for a specific asset or all assets.
        
        Args:
            asset: Optional asset identifier (get all if None)
            
        Returns:
            Dictionary with current sentiment state
        """
        # Calculate sentiment from active news events
        asset_sentiment = self._aggregate_sentiment_by_asset(self.active_news_events)
        
        if asset:
            # Return sentiment for specific asset
            if asset in asset_sentiment:
                return {
                    'asset': asset,
                    'sentiment': asset_sentiment[asset],
                    'active_events': [
                        event for event in self.active_news_events
                        if asset in event['affected_assets']
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'asset': asset,
                    'sentiment': {
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 1.0,
                        'weighted_score': 0.0,
                        'impact_count': 0
                    },
                    'active_events': [],
                    'timestamp': datetime.now().isoformat()
                }
        else:
            # Return sentiment for all assets
            return {
                'sentiment_by_asset': asset_sentiment,
                'active_events_count': len(self.active_news_events),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_sentiment_accuracy(self) -> Dict[str, Any]:
        """
        Get accuracy statistics for sentiment-based predictions.
        
        Returns:
            Dictionary with accuracy statistics
        """
        # Overall accuracy
        overall_accuracy = self.prediction_accuracy['correct'] / self.prediction_accuracy['total'] \
            if self.prediction_accuracy['total'] > 0 else 0.0
            
        # Accuracy by category
        category_accuracy = {}
        for category, stats in self.prediction_accuracy['by_category'].items():
            category_accuracy[category] = stats['correct'] / stats['total'] \
                if stats['total'] > 0 else 0.0
        
        # Accuracy by asset
        asset_accuracy = {}
        for asset, stats in self.historical_impacts['asset_correlations'].items():
            asset_accuracy[asset] = stats['correct'] / stats['events'] \
                if stats['events'] > 0 else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': self.prediction_accuracy['total'],
            'category_accuracy': category_accuracy,
            'asset_accuracy': asset_accuracy
        }
    
    def get_critical_events(self) -> List[Dict[str, Any]]:
        """
        Get currently active critical (high-impact) news events.
        
        Returns:
            List of critical news events
        """
        return [
            event for event in self.active_news_events
            if event['impact_score'] >= self.impact_threshold
        ]
