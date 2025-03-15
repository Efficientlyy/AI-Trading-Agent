"""Enhanced price prediction strategy using ML with sentiment and correlation analysis."""

from typing import Dict, List, Optional, Tuple, Union, Set, cast, Any, TypedDict, Protocol, NotRequired
from typing_extensions import TypeAlias
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from numpy.typing import NDArray
from collections import deque, defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Type aliases for numeric types
NumericType = Union[int, float]
ArrayType = NDArray[np.float64]

class FeatureVector(TypedDict):
    """Type definition for feature vectors."""
    technical: ArrayType
    sentiment: ArrayType
    market: ArrayType

class ModelPrediction(TypedDict):
    """Type definition for model predictions."""
    prediction: float
    confidence: float
    direction: int
    timestamp: float
    features: FeatureVector

class TechnicalFeatures:
    """Technical feature calculation with type safety."""
    
    @staticmethod
    def calculate_rsi(prices: ArrayType, period: int = 14) -> float:
        """Calculate RSI with type safety."""
        if len(prices) < period + 1:
            return 50.0  # Neutral value for insufficient data
            
        try:
            deltas = np.diff(prices)
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            
            if avg_loss == 0:
                return 100.0
                
            rs = float(avg_gain) / float(avg_loss)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return float(rsi)
            
        except Exception:
            return 50.0  # Return neutral value on error
    
    @staticmethod
    def calculate_bb_position(prices: ArrayType, current_price: float) -> float:
        """Calculate Bollinger Band position with type safety."""
        window = 20
        if len(prices) < window:
            return 0.5  # Neutral position
            
        try:
            rolling_mean = np.mean(prices[-window:])
            rolling_std = np.std(prices[-window:])
            
            upper_band = rolling_mean + (2 * rolling_std)
            lower_band = rolling_mean - (2 * rolling_std)
            
            band_range = upper_band - lower_band
            if band_range == 0:
                return 0.5
                
            position = (current_price - lower_band) / band_range
            return float(np.clip(position, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    @staticmethod
    def calculate_trend_strength(prices: ArrayType, period: int = 14) -> float:
        """Calculate trend strength with type safety."""
        if len(prices) < period + 1:
            return 0.0
            
        try:
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate trend metrics
            trend_direction = np.mean(returns[-period:])
            trend_volatility = np.std(returns[-period:])
            
            if trend_volatility == 0:
                return 0.0
                
            # Normalize trend strength
            strength = abs(trend_direction) / trend_volatility
            return float(np.clip(strength, 0.0, 1.0))
            
        except Exception:
            return 0.0

class FeatureExtractor:
    """Feature extraction with type safety."""
    
    def __init__(self):
        self.technical = TechnicalFeatures()
    
    def extract_features(
        self,
        prices: ArrayType,
        sentiment_data: Dict[str, float],
        market_data: Dict[str, float]
    ) -> FeatureVector:
        """Extract all features with type safety."""
        # Technical features
        technical = np.array([
            self.technical.calculate_rsi(prices),
            self.technical.calculate_bb_position(prices, prices[-1]),
            self.technical.calculate_trend_strength(prices)
        ], dtype=np.float64)
        
        # Sentiment features
        sentiment = np.array([
            float(sentiment_data.get("social_sentiment", 0.0)),
            float(sentiment_data.get("news_sentiment", 0.0)),
            float(sentiment_data.get("order_flow_sentiment", 0.0)),
            float(sentiment_data.get("fear_greed_index", 50.0)) / 100.0
        ], dtype=np.float64)
        
        # Market features
        market = np.array([
            float(market_data.get("liquidity_score", 0.0)),
            float(market_data.get("volatility", 0.0)),
            float(market_data.get("correlation_score", 0.0))
        ], dtype=np.float64)
        
        return {
            "technical": technical,
            "sentiment": sentiment,
            "market": market
        }
    
    def combine_features(self, features: FeatureVector) -> ArrayType:
        """Combine feature vectors safely."""
        return np.concatenate([
            features["technical"],
            features["sentiment"],
            features["market"]
        ])

class ModelPredictor:
    """Model prediction with type safety."""
    
    def __init__(self, model: Any, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, features: ArrayType, min_price_move: float = 0.0001) -> ModelPrediction:
        """Generate prediction with type safety."""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            prediction = float(self.model.predict(features_scaled)[0])
            probas = self.model.predict_proba(features_scaled)[0]
            confidence = float(max(probas))
            
            # Calculate direction
            direction = int(np.sign(prediction)) if abs(prediction) > min_price_move else 0
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": direction,
                "timestamp": float(time.time()),
                "features": {
                    "technical": features[:3],
                    "sentiment": features[3:7],
                    "market": features[7:]
                }
            }
            
        except Exception as e:
            # Return safe default values on error
            return {
                "prediction": 0.0,
                "confidence": 0.0,
                "direction": 0,
                "timestamp": float(time.time()),
                "features": {
                    "technical": np.zeros(3, dtype=np.float64),
                    "sentiment": np.zeros(4, dtype=np.float64),
                    "market": np.zeros(3, dtype=np.float64)
                }
            } 