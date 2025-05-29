"""
Enhanced Market Regime Classifier Module

This module implements advanced market regime classification methods using
multiple technical approaches including:
- Machine learning-based classification
- Statistical change point detection
- Hidden Markov Models
- Feature importance analysis
- Ensemble methods for improved accuracy
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import hmmlearn.hmm as hmm
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
import joblib
import os

from ai_trading_agent.agent.market_regime import MarketRegimeType, MarketRegimeClassifier

# Set up logger
logger = logging.getLogger(__name__)


class FeatureImportance:
    """Store feature importance data with metadata."""
    
    def __init__(self, feature_names: List[str], importance_values: List[float]):
        """Initialize with feature names and their importance values."""
        self.feature_names = feature_names
        self.importance_values = importance_values
        self.timestamp = datetime.now()
        
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N most important features."""
        paired = list(zip(self.feature_names, self.importance_values))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        return sorted_pairs[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_names": self.feature_names,
            "importance_values": self.importance_values,
            "timestamp": self.timestamp.isoformat()
        }


class EnhancedMarketRegimeClassifier:
    """
    Advanced market regime classifier with multiple detection methods and ensemble capabilities.
    
    This classifier extends the base MarketRegimeClassifier with:
    1. Multiple detection methods (ML, HMM, statistical)
    2. Ensemble approach for improved accuracy
    3. Feature importance analysis
    4. Changepoint detection for regime transitions
    5. Confidence scoring for classifications
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced market regime classifier.
        
        Args:
            config: Configuration dictionary with parameters
                - lookback_window: Window for feature calculation (default: 30)
                - ensemble_method: How to combine classifier results (default: "weighted_vote")
                - min_confidence: Minimum confidence level required (default: 0.65)
                - models_dir: Directory to save/load models (default: "./models")
                - use_hmm: Whether to use Hidden Markov Models (default: True)
                - feature_sets: List of feature sets to use (default: ["technical", "statistical", "volatility"])
                - ensemble_weights: Weights for different classifiers in ensemble
        """
        # Default configuration
        default_config = {
            "lookback_window": 30,
            "ensemble_method": "weighted_vote",
            "min_confidence": 0.65,
            "models_dir": "./models",
            "use_hmm": True,
            "feature_sets": ["technical", "statistical", "volatility"],
            "ensemble_weights": {
                "random_forest": 0.4,
                "gradient_boosting": 0.3,
                "hmm": 0.2,
                "statistical": 0.1
            }
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Create base classifier for compatibility
        self.base_classifier = MarketRegimeClassifier(self.config)
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.hmm_model = hmm.GaussianHMM(
            n_components=len(MarketRegimeType),
            covariance_type="full",
            n_iter=100,
            random_state=42
        ) if self.config["use_hmm"] else None
        
        # Initialize scaler for normalizing features
        self.scaler = StandardScaler()
        
        # Track feature importance
        self.feature_importance = None
        
        # Model persistence
        self.models_dir = self.config["models_dir"]
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Dataset for continuous learning
        self.training_data = []
        self.recent_accuracy = []
        
        # Regime history
        self.regime_history = {}
        
        logger.info(f"Enhanced Market Regime Classifier initialized with {len(self.config['feature_sets'])} feature sets")
        
    def classify_regime(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Classify the current market regime using ensemble methods.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Optional symbol identifier
            
        Returns:
            Dictionary with classification results including:
            - regime: The detected MarketRegimeType
            - confidence: Confidence score for the classification
            - feature_importance: Top contributing features
            - ensemble_votes: Individual classifier votes
            - transition_probability: Likelihood of regime change
        """
        if data is None or len(data) < self.config["lookback_window"]:
            logger.warning(f"Insufficient data for regime classification: {len(data) if data is not None else 0} rows")
            return {
                "regime": MarketRegimeType.UNKNOWN.value,
                "confidence": 0.0,
                "ensemble_votes": {},
                "transition_probability": 0.0
            }
        
        # Extract features
        features = self._extract_features(data)
        if features.empty:
            logger.warning("Feature extraction failed")
            return {
                "regime": MarketRegimeType.UNKNOWN.value,
                "confidence": 0.0,
                "ensemble_votes": {},
                "transition_probability": 0.0
            }
        
        # Get individual classifier predictions
        ensemble_results = self._run_ensemble_classification(features)
        
        # Determine final regime and confidence
        final_regime, confidence, votes = self._combine_ensemble_results(ensemble_results)
        
        # Detect potential regime transitions
        transition_prob = self._estimate_transition_probability(data, final_regime)
        
        # Store in history if symbol provided
        if symbol:
            self.regime_history[symbol] = {
                "regime": final_regime,
                "confidence": confidence,
                "timestamp": pd.Timestamp.now(),
                "transition_probability": transition_prob
            }
        
        return {
            "regime": final_regime.value,
            "confidence": confidence,
            "ensemble_votes": {k: v.value for k, v in votes.items()},
            "feature_importance": self.feature_importance.get_top_features(5) if self.feature_importance else [],
            "transition_probability": transition_prob
        }
        
    def classify_multiple(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Classify market regimes for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbols to their OHLCV DataFrames
            
        Returns:
            Dictionary mapping symbols to their classification results
        """
        results = {}
        for symbol, data in data_dict.items():
            try:
                results[symbol] = self.classify_regime(data, symbol)
                logger.debug(f"Classified {symbol}: {results[symbol]['regime']} (conf: {results[symbol]['confidence']:.2f})")
            except Exception as e:
                logger.error(f"Error classifying regime for {symbol}: {e}")
                results[symbol] = {
                    "regime": MarketRegimeType.UNKNOWN.value,
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        return results
        
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification.
        
        Implements multiple feature sets based on configuration:
        - Technical indicators
        - Statistical measures
        - Volatility metrics
        - Pattern-based features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            # Ensure we have enough data
            if len(data) < self.config["lookback_window"] * 2:
                return pd.DataFrame()
                
            features = pd.DataFrame(index=[0])
            
            # Technical indicators
            if "technical" in self.config["feature_sets"]:
                # Moving averages and crossovers
                data['ma20'] = data['close'].rolling(window=20).mean()
                data['ma50'] = data['close'].rolling(window=50).mean()
                data['ma200'] = data['close'].rolling(window=200).mean()
                
                features['ma_cross_20_50'] = 1 if data['ma20'].iloc[-1] > data['ma50'].iloc[-1] else 0
                features['ma_cross_50_200'] = 1 if data['ma50'].iloc[-1] > data['ma200'].iloc[-1] else 0
                
                # RSI
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                features['rsi'] = data['rsi'].iloc[-1]
                features['rsi_avg'] = data['rsi'].rolling(window=10).mean().iloc[-1]
                
                # MACD
                data['ema12'] = data['close'].ewm(span=12).mean()
                data['ema26'] = data['close'].ewm(span=26).mean()
                data['macd'] = data['ema12'] - data['ema26']
                data['macd_signal'] = data['macd'].ewm(span=9).mean()
                features['macd_hist'] = data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]
                
                # ADX (Average Directional Index) for trend strength
                plus_dm = data['high'].diff()
                minus_dm = data['low'].diff(-1).abs()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0
                tr = pd.DataFrame([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift(1)).abs(),
                    (data['low'] - data['close'].shift(1)).abs()
                ]).max()
                atr = tr.rolling(window=14).mean()
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                adx = dx.rolling(window=14).mean()
                features['adx'] = adx.iloc[-1]
            
            # Statistical features
            if "statistical" in self.config["feature_sets"]:
                # Returns and distributions
                returns = data['close'].pct_change()
                features['return_mean'] = returns.rolling(window=20).mean().iloc[-1]
                features['return_std'] = returns.rolling(window=20).std().iloc[-1]
                features['return_skew'] = returns.rolling(window=20).skew().iloc[-1]
                features['return_kurt'] = returns.rolling(window=20).kurt().iloc[-1]
                
                # Autocorrelation (for mean reversion vs trending)
                features['autocorr_1'] = returns.autocorr(lag=1)
                features['autocorr_5'] = returns.autocorr(lag=5)
                
                # Stationarity measure using ADF test
                adf_result = adfuller(data['close'].iloc[-30:])
                features['adf_pvalue'] = adf_result[1]
            
            # Volatility features
            if "volatility" in self.config["feature_sets"]:
                # Historical volatility measures
                price_volatility = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                features['hist_vol_20'] = price_volatility.iloc[-1]
                
                # GARCH-based volatility (simplified approximation)
                returns_squared = returns ** 2
                features['garch_proxy'] = returns_squared.ewm(alpha=0.1).mean().iloc[-1]
                
                # Volatility of volatility
                vol_of_vol = price_volatility.rolling(window=20).std()
                features['vol_of_vol'] = vol_of_vol.iloc[-1]
                
                # ATR ratio
                atr_20 = tr.rolling(window=20).mean().iloc[-1]
                atr_100 = tr.rolling(window=100).mean().iloc[-1]
                features['atr_ratio'] = atr_20 / atr_100 if atr_100 != 0 else 1.0
            
            # Price pattern features
            if "patterns" in self.config["feature_sets"]:
                # Detect recent peaks and troughs
                price_series = data['close'].values
                peaks, _ = find_peaks(price_series, distance=5)
                troughs, _ = find_peaks(-price_series, distance=5)
                
                # Count recent peaks/troughs as pattern indicators
                recent_peaks = sum(1 for p in peaks if p >= len(price_series) - 20)
                recent_troughs = sum(1 for t in troughs if t >= len(price_series) - 20)
                
                features['peak_trough_ratio'] = recent_peaks / recent_troughs if recent_troughs > 0 else recent_peaks
                features['pattern_count'] = recent_peaks + recent_troughs
            
            # Handle NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _run_ensemble_classification(self, features: pd.DataFrame) -> Dict[str, Tuple[MarketRegimeType, float]]:
        """
        Run all classifiers in the ensemble and collect their predictions.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Dictionary mapping classifier names to tuples of (predicted_regime, confidence)
        """
        ensemble_results = {}
        scaled_features = self.scaler.fit_transform(features)
        
        # Standard sklearn models
        if hasattr(self.rf_model, 'classes_'):
            rf_pred_proba = self.rf_model.predict_proba(scaled_features)[0]
            rf_pred_idx = np.argmax(rf_pred_proba)
            rf_confidence = rf_pred_proba[rf_pred_idx]
            rf_regime = self._map_index_to_regime(rf_pred_idx)
            ensemble_results['random_forest'] = (rf_regime, rf_confidence)
            
            # Update feature importance
            feature_names = features.columns.tolist()
            importance_values = self.rf_model.feature_importances_
            self.feature_importance = FeatureImportance(feature_names, importance_values.tolist())
        else:
            # Model not yet trained, use fallback (base classifier)
            base_result = self.base_classifier.classify_regime(None)  # Placeholder call
            regime = MarketRegimeType.UNKNOWN
            for rt in MarketRegimeType:
                if rt.value == base_result:
                    regime = rt
                    break
            ensemble_results['base_classifier'] = (regime, 0.5)
        
        if hasattr(self.gb_model, 'classes_'):
            gb_pred_proba = self.gb_model.predict_proba(scaled_features)[0]
            gb_pred_idx = np.argmax(gb_pred_proba)
            gb_confidence = gb_pred_proba[gb_pred_idx]
            gb_regime = self._map_index_to_regime(gb_pred_idx)
            ensemble_results['gradient_boosting'] = (gb_regime, gb_confidence)
        
        # HMM-based classification
        if self.hmm_model and hasattr(self.hmm_model, 'transmat_'):
            hmm_logprob, hidden_states = self.hmm_model.decode(scaled_features)
            hmm_regime = self._map_index_to_regime(hidden_states[0])
            # Confidence is harder to determine with HMMs
            hmm_confidence = 0.7  # Fixed confidence for now
            ensemble_results['hmm'] = (hmm_regime, hmm_confidence)
        
        # Statistical analysis approach (rule-based)
        stat_regime, stat_confidence = self._statistical_classification(features)
        ensemble_results['statistical'] = (stat_regime, stat_confidence)
        
        return ensemble_results
    
    def _statistical_classification(self, features: pd.DataFrame) -> Tuple[MarketRegimeType, float]:
        """
        Perform rule-based classification using statistical indicators.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Tuple of (MarketRegimeType, confidence_score)
        """
        # Simple rule-based logic based on features
        confidence = 0.6  # Default confidence
        
        # Check for trending market
        if 'adx' in features and features['adx'].iloc[0] > 25:
            # Strong trend detected
            if 'ma_cross_20_50' in features and features['ma_cross_20_50'].iloc[0] == 1:
                return MarketRegimeType.TRENDING_UP, min(0.5 + (features['adx'].iloc[0] - 25) / 50, 0.9)
            else:
                return MarketRegimeType.TRENDING_DOWN, min(0.5 + (features['adx'].iloc[0] - 25) / 50, 0.9)
        
        # Check for volatile market
        if 'hist_vol_20' in features and features['hist_vol_20'].iloc[0] > 0.25:
            return MarketRegimeType.VOLATILE, min(0.5 + features['hist_vol_20'].iloc[0], 0.9)
        
        # Check for ranging market
        if 'autocorr_1' in features and features['autocorr_1'].iloc[0] < -0.2:
            # Negative autocorrelation suggests mean reversion (ranging)
            return MarketRegimeType.RANGING, min(0.5 + abs(features['autocorr_1'].iloc[0]), 0.85)
        
        # Check for breakout
        if 'pattern_count' in features and features['pattern_count'].iloc[0] < 2:
            if 'vol_of_vol' in features and features['vol_of_vol'].iloc[0] > 0.1:
                return MarketRegimeType.BREAKOUT, 0.7
        
        # Default to calm if no other patterns detected
        if 'hist_vol_20' in features and features['hist_vol_20'].iloc[0] < 0.15:
            return MarketRegimeType.CALM, 0.6
        
        # Fallback to unknown
        return MarketRegimeType.UNKNOWN, 0.5
    
    def _combine_ensemble_results(self, ensemble_results: Dict[str, Tuple[MarketRegimeType, float]]) -> Tuple[MarketRegimeType, float, Dict[str, MarketRegimeType]]:
        """
        Combine results from multiple classifiers into a final prediction.
        
        Args:
            ensemble_results: Dict mapping classifier names to (regime, confidence) tuples
            
        Returns:
            Tuple of (final_regime, confidence_score, votes_dict)
        """
        if not ensemble_results:
            return MarketRegimeType.UNKNOWN, 0.0, {}
        
        method = self.config["ensemble_method"]
        weights = self.config["ensemble_weights"]
        
        # Record votes for each classifier
        votes = {k: v[0] for k, v in ensemble_results.items()}
        
        # Simple majority vote
        if method == "majority_vote":
            regime_counts = {}
            for classifier, (regime, _) in ensemble_results.items():
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
            most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            vote_confidence = regime_counts[most_common_regime] / len(ensemble_results)
            
            return most_common_regime, vote_confidence, votes
        
        # Weighted vote based on classifier importance and prediction confidence
        elif method == "weighted_vote":
            regime_scores = {}
            total_weight = 0
            
            for classifier, (regime, confidence) in ensemble_results.items():
                classifier_weight = weights.get(classifier, 1.0)
                score = classifier_weight * confidence
                regime_scores[regime] = regime_scores.get(regime, 0) + score
                total_weight += classifier_weight
            
            if total_weight > 0:
                # Normalize scores
                regime_scores = {k: v / total_weight for k, v in regime_scores.items()}
                
            # Select regime with highest weighted score
            if regime_scores:
                best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                confidence = regime_scores[best_regime]
                return best_regime, confidence, votes
        
        # Highest confidence method
        elif method == "highest_confidence":
            best_classifier = max(ensemble_results.items(), key=lambda x: x[1][1])
            return best_classifier[1][0], best_classifier[1][1], votes
        
        # Default fallback
        return MarketRegimeType.UNKNOWN, 0.0, votes
    
    def _map_index_to_regime(self, index: int) -> MarketRegimeType:
        """Map a class index to a MarketRegimeType."""
        regimes = list(MarketRegimeType)
        if 0 <= index < len(regimes):
            return regimes[index]
        return MarketRegimeType.UNKNOWN
    
    def _estimate_transition_probability(self, data: pd.DataFrame, current_regime: MarketRegimeType) -> float:
        """
        Estimate probability of an imminent regime transition.
        
        Args:
            data: OHLCV data
            current_regime: Current detected regime
            
        Returns:
            Probability of regime transition (0.0 to 1.0)
        """
        # Simple indicators of potential transitions
        volatility_change = data['close'].pct_change().rolling(window=5).std().diff(5).iloc[-1]
        volume_spike = data['volume'].iloc[-5:].mean() / data['volume'].iloc[-20:-5].mean() if 'volume' in data else 1.0
        
        # Significant changes in key metrics suggest higher transition probability
        base_probability = 0.1  # Base probability of transition
        
        # Volatility changes often precede regime transitions
        if abs(volatility_change) > 0.05:
            base_probability += min(abs(volatility_change) * 5, 0.3)
            
        # Volume spikes can signal regime transitions
        if volume_spike > 1.5:
            base_probability += min((volume_spike - 1.5) * 0.2, 0.3)
            
        # Limit final probability to 0.0-1.0 range
        return min(max(base_probability, 0.0), 1.0)
    
    def train(self, data_dict: Dict[str, pd.DataFrame], regimes: Dict[str, MarketRegimeType] = None) -> float:
        """
        Train the classifier models on historical data.
        
        Args:
            data_dict: Dictionary mapping symbols to their OHLCV DataFrames
            regimes: Optional known regimes for supervised learning
            
        Returns:
            Accuracy score of the training
        """
        # Extract features from all datasets
        all_features = []
        all_labels = []
        
        for symbol, data in data_dict.items():
            try:
                features = self._extract_features(data)
                if features.empty:
                    continue
                    
                # Get label (either supplied or from base classifier)
                if regimes and symbol in regimes:
                    label = regimes[symbol]
                    if isinstance(label, str):
                        # Convert string to enum
                        for rt in MarketRegimeType:
                            if rt.value == label:
                                label = rt
                                break
                else:
                    # Use base classifier if available or current classifier in self-training mode
                    base_result = self.base_classifier.classify_regime(data)
                    label = None
                    for rt in MarketRegimeType:
                        if rt.value == base_result:
                            label = rt
                            break
                    if label is None:
                        label = MarketRegimeType.UNKNOWN
                
                # Convert label to index
                label_idx = list(MarketRegimeType).index(label)
                
                all_features.append(features)
                all_labels.append(label_idx)
                
            except Exception as e:
                logger.error(f"Error processing training data for {symbol}: {e}")
        
        if not all_features or not all_labels:
            logger.warning("No valid training data available")
            return 0.0
            
        # Combine all features and labels
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_val_acc = accuracy_score(y_val, self.rf_model.predict(X_val_scaled))
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train_scaled, y_train)
        gb_val_acc = accuracy_score(y_val, self.gb_model.predict(X_val_scaled))
        
        # Train HMM if enabled
        if self.hmm_model:
            try:
                self.hmm_model.fit(X_train_scaled)
                # HMM doesn't have a straightforward accuracy calculation
            except Exception as e:
                logger.error(f"Error training HMM model: {e}")
        
        # Save models
        try:
            joblib.dump(self.rf_model, os.path.join(self.models_dir, 'rf_regime_model.pkl'))
            joblib.dump(self.gb_model, os.path.join(self.models_dir, 'gb_regime_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'regime_scaler.pkl'))
            if self.hmm_model:
                joblib.dump(self.hmm_model, os.path.join(self.models_dir, 'hmm_regime_model.pkl'))
        except Exception as e:
            logger.error(f"Error saving models: {e}")
        
        # Average accuracy for reporting
        avg_accuracy = (rf_val_acc + gb_val_acc) / 2
        self.recent_accuracy.append(avg_accuracy)
        if len(self.recent_accuracy) > 5:
            self.recent_accuracy.pop(0)
            
        logger.info(f"Regime classifier training completed with validation accuracy: {avg_accuracy:.4f}")
        return avg_accuracy
    
    def load_models(self) -> bool:
        """
        Load pretrained models from disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            rf_path = os.path.join(self.models_dir, 'rf_regime_model.pkl')
            gb_path = os.path.join(self.models_dir, 'gb_regime_model.pkl')
            scaler_path = os.path.join(self.models_dir, 'regime_scaler.pkl')
            hmm_path = os.path.join(self.models_dir, 'hmm_regime_model.pkl')
            
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                
            if os.path.exists(gb_path):
                self.gb_model = joblib.load(gb_path)
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            if os.path.exists(hmm_path) and self.config["use_hmm"]:
                self.hmm_model = joblib.load(hmm_path)
                
            logger.info("Successfully loaded pretrained regime classification models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
            return False
    
    def reset_calibration(self) -> None:
        """Reset classifier calibration for retraining."""
        # Reinitialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        if self.config["use_hmm"]:
            self.hmm_model = hmm.GaussianHMM(
                n_components=len(MarketRegimeType),
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
        
        # Reset scaler
        self.scaler = StandardScaler()
        
        # Clear training data
        self.training_data = []
        self.recent_accuracy = []
        
        logger.info("Regime classifier calibration reset")
