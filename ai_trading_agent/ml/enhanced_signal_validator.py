"""
Enhanced Signal Validator Module

This module provides an advanced machine learning-based validation system for trading signals
with market regime adaptation, ensemble models, and sophisticated feature engineering.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import os
from pathlib import Path
import json
import time
from functools import lru_cache
import concurrent.futures

# Optional imports for ML models
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.cluster import KMeans, DBSCAN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Local imports
from ..common.utils import get_logger
from .feature_engineering import AdvancedFeatureEngineering
from .market_regime import MarketRegimeClassifier
from .signal_clustering import SignalClusterAnalyzer

class EnhancedSignalValidator:
    """
    Advanced ML-based signal validator with market regime adaptation and ensemble models.
    
    Features:
    - Ensemble of specialized models for different market conditions
    - Advanced feature engineering with pattern recognition
    - Market regime-specific validation rules
    - Confidence scoring with uncertainty quantification
    - Continuous performance monitoring and adaptation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced signal validator.
        
        Args:
            config: Configuration dictionary with parameters
                - model_path: Path to pre-trained model files (optional)
                - ensemble_models: List of model types to include in ensemble
                - min_confidence: Minimum confidence threshold to accept signals
                - features: List of features to use for prediction
                - enable_regime_adaptation: Whether to adapt to market regimes
                - performance_tracking: Enable continuous performance tracking
                - parallel_processing: Enable parallel processing for validation
        """
        self.logger = get_logger("EnhancedSignalValidator")
        self.config = config or {}
        
        # Extract configuration
        self.model_path = self.config.get("model_path")
        self.ensemble_models = self.config.get("ensemble_models", [
            "random_forest", "gradient_boosting", "neural_network"
        ])
        self.min_confidence = self.config.get("min_confidence", 0.65)
        self.regime_adaptation = self.config.get("enable_regime_adaptation", True)
        self.performance_tracking = self.config.get("performance_tracking", True)
        self.parallel_processing = self.config.get("parallel_processing", True)
        self.max_workers = self.config.get("max_workers", 4)
        
        # Initialize components
        self.models = {}
        self.ensemble = None
        self.scaler = None
        self.feature_importance = {}
        self.validation_history = []
        self.regime_thresholds = {
            "trending": 0.60,
            "ranging": 0.70,
            "volatile": 0.75,
            "unknown": 0.65
        }
        
        # Initialize sub-components
        self.feature_engineer = AdvancedFeatureEngineering(self.config.get("feature_config", {}))
        self.regime_classifier = MarketRegimeClassifier(self.config.get("regime_config", {}))
        self.cluster_analyzer = SignalClusterAnalyzer(self.config.get("cluster_config", {}))
        
        # Check for ML libraries
        if not ML_AVAILABLE:
            self.logger.warning(
                "Machine learning libraries not available. "
                "Install scikit-learn for ML-based signal validation."
            )
            return
        
        # Load or initialize model
        if self.model_path and os.path.exists(self.model_path):
            self._load_models(self.model_path)
        else:
            self._init_models()
    
    def _init_models(self):
        """Initialize ensemble of models for different market regimes."""
        if not ML_AVAILABLE:
            return
            
        # Create specialized models for different regimes
        self.models = {
            "trending": self._create_model("random_forest", "trending"),
            "ranging": self._create_model("gradient_boosting", "ranging"),
            "volatile": self._create_model("neural_network", "volatile"),
            "general": self._create_model("voting", "general")
        }
            
        self.scaler = StandardScaler()
        self.logger.info(f"Initialized ensemble of models for different regimes")
    
    def _create_model(self, model_type: str, regime: str):
        """Create a model specialized for a specific market regime."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                random_state=42
            )
        elif model_type == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        elif model_type == "svm":
            return SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == "voting":
            # Create an ensemble of different models
            estimators = []
            if "random_forest" in self.ensemble_models:
                estimators.append(('rf', self._create_model("random_forest", regime)))
            if "gradient_boosting" in self.ensemble_models:
                estimators.append(('gb', self._create_model("gradient_boosting", regime)))
            if "neural_network" in self.ensemble_models:
                estimators.append(('nn', self._create_model("neural_network", regime)))
            if "svm" in self.ensemble_models:
                estimators.append(('svm', self._create_model("svm", regime)))
                
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[1] * len(estimators)
            )
        else:
            # Default to Random Forest
            return RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
    
    def _load_models(self, model_path: str):
        """Load pre-trained models from files."""
        try:
            # Load base model data
            with open(os.path.join(model_path, "ensemble_data.pkl"), 'rb') as f:
                model_data = pickle.load(f)
                
            self.scaler = model_data.get('scaler')
            self.feature_importance = model_data.get('feature_importance', {})
            
            # Load specialized models for different regimes
            for regime in ['trending', 'ranging', 'volatile', 'general']:
                regime_model_path = os.path.join(model_path, f"{regime}_model.pkl")
                if os.path.exists(regime_model_path):
                    with open(regime_model_path, 'rb') as f:
                        self.models[regime] = pickle.load(f)
                        self.logger.info(f"Loaded {regime} model from {regime_model_path}")
                else:
                    self.models[regime] = self._create_model("random_forest", regime)
                    self.logger.warning(f"No model found for {regime} regime, created new one")
            
            self.logger.info(f"Successfully loaded model ensemble from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self._init_models()
    
    def save_models(self, save_path: str):
        """Save all models to files."""
        if not ML_AVAILABLE or not self.models:
            self.logger.warning("No models to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save base model data
            model_data = {
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'config': self.config
            }
            
            with open(os.path.join(save_path, "ensemble_data.pkl"), 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save specialized models for different regimes
            for regime, model in self.models.items():
                with open(os.path.join(save_path, f"{regime}_model.pkl"), 'wb') as f:
                    pickle.dump(model, f)
                    
            self.logger.info(f"Successfully saved model ensemble to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            return False
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'success', 
              regime_column: str = None):
        """
        Train the model ensemble on historical signal data.
        
        Args:
            training_data: DataFrame with signal features and outcomes
            target_column: Name of the column indicating signal success (1=success, 0=failure)
            regime_column: Optional column indicating market regime for specialized training
            
        Returns:
            Dictionary with training metrics
        """
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available, cannot train models")
            return {"error": "ML libraries not available"}
            
        if training_data.empty:
            self.logger.warning("Empty training data provided")
            return {"error": "Empty training data"}
            
        try:
            # Prepare features and target
            X = training_data.drop([target_column], axis=1, errors='ignore')
            y = training_data[target_column]
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            metrics = {}
            
            # If regime column is provided, train specialized models
            if regime_column and regime_column in training_data.columns:
                for regime in self.models:
                    if regime != 'general':
                        # Filter data for this regime
                        regime_mask = training_data[regime_column] == regime
                        if regime_mask.sum() > 20:  # Ensure enough samples
                            X_regime = X_scaled[regime_mask]
                            y_regime = y[regime_mask]
                            
                            # Train regime-specific model
                            self.models[regime].fit(X_regime, y_regime)
                            
                            # Calculate metrics
                            y_pred = self.models[regime].predict(X_regime)
                            metrics[regime] = {
                                'accuracy': accuracy_score(y_regime, y_pred),
                                'precision': precision_score(y_regime, y_pred, zero_division=0),
                                'recall': recall_score(y_regime, y_pred, zero_division=0),
                                'f1': f1_score(y_regime, y_pred, zero_division=0),
                                'samples': len(y_regime)
                            }
                            
                            self.logger.info(f"Trained {regime} model with {len(y_regime)} samples")
            
            # Train general model on all data
            self.models['general'].fit(X_train, y_train)
            
            # Evaluate general model
            y_pred = self.models['general'].predict(X_test)
            y_prob = self.models['general'].predict_proba(X_test)[:,1]
            
            # Calculate performance metrics
            metrics['general'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'samples': len(y_test)
            }
            
            # Extract feature importance if available
            if hasattr(self.models['general'], 'feature_importances_'):
                self.feature_importance = dict(zip(
                    X.columns, 
                    self.models['general'].feature_importances_
                ))
            
            self.logger.info(f"Model training complete with accuracy: {metrics['general']['accuracy']:.4f}")
            return {
                'metrics': metrics,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return {"error": str(e)}
    
    def validate_signal(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate a trading signal using the enhanced validation system.
        
        Args:
            signal: Signal dictionary to validate
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary of technical indicators
            
        Returns:
            Tuple of (is_valid, confidence_score, validation_metadata)
        """
        start_time = time.time()
        
        try:
            # Extract symbol from signal
            symbol = signal.get('symbol', '')
            if not symbol or symbol not in market_data:
                return False, 0.0, {"error": f"Invalid symbol: {symbol}"}
            
            # Determine market regime
            regime = self._detect_market_regime(market_data[symbol], indicators.get(symbol, {}))
            
            # Extract features for validation
            features = self.feature_engineer.extract_features(signal, market_data, indicators)
            
            # If no ML libraries or model not trained, fall back to heuristics
            if not ML_AVAILABLE or not self.models:
                is_valid, confidence, metadata = self._validate_with_heuristics(
                    signal, market_data, indicators, regime
                )
                
                # Track validation for performance monitoring
                if self.performance_tracking:
                    self._track_validation(signal, features, is_valid, confidence, regime)
                
                return is_valid, confidence, metadata
            
            # Find similar historical signals
            similar_signals = self.cluster_analyzer.find_similar_signals(
                features, self.validation_history
            )
            
            # Select appropriate model based on regime
            model = self.models.get(regime, self.models['general'])
            
            # Prepare features for prediction
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and confidence
            confidence = float(model.predict_proba(X_scaled)[0, 1])
            
            # Apply regime-specific threshold
            threshold = self.regime_thresholds.get(regime, self.min_confidence)
            is_valid = confidence >= threshold
            
            # Calculate uncertainty estimate
            uncertainty = self._calculate_uncertainty(features, similar_signals)
            
            # Apply additional validation rules based on regime
            is_valid, rule_adjustments = self._apply_regime_rules(
                is_valid, confidence, regime, signal, market_data, indicators
            )
            
            # Prepare metadata
            execution_time = time.time() - start_time
            metadata = {
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "threshold": float(threshold),
                "regime": regime,
                "similar_signals_count": len(similar_signals),
                "rule_adjustments": rule_adjustments,
                "execution_time_ms": int(execution_time * 1000),
                "validation_method": "ml_ensemble"
            }
            
            # Track validation for performance monitoring
            if self.performance_tracking:
                self._track_validation(signal, features, is_valid, confidence, regime)
            
            return is_valid, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error in signal validation: {str(e)}")
            return False, 0.0, {"error": str(e)}
    
    def _detect_market_regime(self, market_data: pd.DataFrame, indicators: Dict) -> str:
        """Detect the current market regime based on price action and indicators."""
        try:
            return self.regime_classifier.classify(market_data, indicators)
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
    
    def _calculate_uncertainty(self, features: Dict, similar_signals: List[Dict]) -> float:
        """Calculate uncertainty estimate based on similar historical signals."""
        if not similar_signals:
            return 0.5  # High uncertainty when no similar signals
            
        # Calculate variance in outcomes of similar signals
        outcomes = [s.get('outcome', 0.5) for s in similar_signals]
        return float(np.std(outcomes))
    
    def _apply_regime_rules(
        self, 
        is_valid: bool, 
        confidence: float,
        regime: str,
        signal: Dict,
        market_data: Dict[str, pd.DataFrame],
        indicators: Dict
    ) -> Tuple[bool, Dict]:
        """Apply additional validation rules based on market regime."""
        adjustments = {}
        
        # Apply regime-specific rules
        if regime == "trending":
            # In trending markets, favor signals aligned with the trend
            trend_direction = signal.get('metadata', {}).get('trend_direction', 0)
            signal_direction = 1 if signal.get('direction', '') == 'buy' else -1
            
            if trend_direction * signal_direction < 0:  # Signal against trend
                confidence *= 0.8
                adjustments["trend_alignment"] = "reduced_confidence"
                
        elif regime == "ranging":
            # In ranging markets, favor signals near support/resistance levels
            near_level = signal.get('metadata', {}).get('near_support_resistance', False)
            if near_level:
                confidence *= 1.2
                confidence = min(confidence, 1.0)
                adjustments["support_resistance"] = "increased_confidence"
                
        elif regime == "volatile":
            # In volatile markets, require stronger confirmation
            confirmation = signal.get('metadata', {}).get('confirmation_count', 0)
            if confirmation < 2:
                confidence *= 0.7
                adjustments["low_confirmation"] = "reduced_confidence"
        
        # Re-evaluate validity based on adjusted confidence
        threshold = self.regime_thresholds.get(regime, self.min_confidence)
        is_valid = confidence >= threshold
        
        return is_valid, adjustments
    
    def _track_validation(
        self, 
        signal: Dict, 
        features: Dict, 
        is_valid: bool, 
        confidence: float, 
        regime: str
    ):
        """Track validation for performance monitoring."""
        validation_entry = {
            "signal_id": signal.get("id", str(hash(json.dumps(signal, default=str)))),
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "is_valid": is_valid,
            "confidence": confidence,
            "regime": regime,
            "outcome": None  # To be updated later
        }
        
        self.validation_history.append(validation_entry)
        
        # Limit history size
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def update_validation_outcome(self, signal_id: str, outcome: bool):
        """
        Update the outcome of a previously validated signal.
        
        Args:
            signal_id: ID of the signal
            outcome: Whether the signal was successful (True) or not (False)
        """
        for entry in self.validation_history:
            if entry["signal_id"] == signal_id:
                entry["outcome"] = outcome
                break
    
    def optimize_validation_parameters(self):
        """
        Optimize validation parameters based on historical performance.
        
        This adapts thresholds and weights based on recent validation outcomes.
        """
        if not self.validation_history or not self.performance_tracking:
            return
            
        # Filter entries with known outcomes
        entries_with_outcomes = [
            entry for entry in self.validation_history 
            if entry["outcome"] is not None
        ]
        
        if len(entries_with_outcomes) < 20:
            return  # Not enough data to optimize
            
        # Calculate performance by regime
        regime_performance = {}
        for regime in set(entry["regime"] for entry in entries_with_outcomes):
            regime_entries = [e for e in entries_with_outcomes if e["regime"] == regime]
            if len(regime_entries) < 5:
                continue
                
            # Calculate optimal threshold for this regime
            confidences = np.array([e["confidence"] for e in regime_entries])
            outcomes = np.array([e["outcome"] for e in regime_entries])
            
            # Find threshold that maximizes F1 score
            best_f1 = 0
            best_threshold = self.regime_thresholds.get(regime, self.min_confidence)
            
            for threshold in np.arange(0.5, 0.95, 0.05):
                predictions = confidences >= threshold
                if sum(predictions) == 0:
                    continue
                    
                precision = np.sum(predictions & outcomes) / np.sum(predictions)
                recall = np.sum(predictions & outcomes) / np.sum(outcomes)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            # Update regime threshold
            self.regime_thresholds[regime] = best_threshold
            self.logger.info(f"Optimized {regime} threshold to {best_threshold:.2f} (F1: {best_f1:.2f})")
    
    def _validate_with_heuristics(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict],
        regime: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate a signal using enhanced heuristics when ML is not available.
        
        This provides a sophisticated fallback method with regime-specific rules.
        """
        try:
            symbol = signal.get('symbol', '')
            if not symbol or symbol not in market_data:
                return False, 0.0, {"error": f"Invalid symbol: {symbol}"}
                
            data = market_data[symbol]
            indicator_data = indicators.get(symbol, {})
            
            # Get signal metadata
            metadata = signal.get('metadata', {})
            
            # Calculate basic signal strength
            signal_strength = abs(signal.get('signal', 0))
            
            # Initialize confidence and reasons
            confidence = 0.5  # Start with neutral confidence
            reasons = []
            
            # Check signal strength
            if signal_strength > 0.8:
                confidence += 0.15
                reasons.append("Strong signal intensity")
            elif signal_strength > 0.5:
                confidence += 0.05
                reasons.append("Moderate signal intensity")
                
            # Check volume confirmation if available
            volume_ratio = metadata.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                confidence += 0.1
                reasons.append("Above average volume")
            elif volume_ratio < 0.5:
                confidence -= 0.1
                reasons.append("Below average volume")
                
            # Check volatility
            volatility = metadata.get('volatility_percent', 0.0)
            if volatility > 3.0:
                confidence -= 0.1
                reasons.append("Excessive volatility")
                
            # Check confirmation count
            confirmation_count = metadata.get('confirmation_count', 0)
            if confirmation_count >= 3:
                confidence += 0.15
                reasons.append("Multiple confirmations")
            elif confirmation_count >= 1:
                confidence += 0.05
                reasons.append("Some confirmation")
                
            # Apply regime-specific adjustments
            if regime == "trending":
                trend_direction = metadata.get('trend_direction', 0)
                signal_direction = 1 if signal.get('direction', '') == 'buy' else -1
                
                if trend_direction * signal_direction > 0:
                    confidence += 0.1
                    reasons.append("Aligned with trend")
                else:
                    confidence -= 0.1
                    reasons.append("Against prevailing trend")
                    
            elif regime == "ranging":
                near_level = metadata.get('near_support_resistance', False)
                if near_level:
                    confidence += 0.15
                    reasons.append("Near key support/resistance")
                    
            elif regime == "volatile":
                # Require stronger confirmation in volatile markets
                if confirmation_count < 2:
                    confidence -= 0.15
                    reasons.append("Insufficient confirmation for volatile market")
            
            # Ensure confidence is in [0, 1] range
            confidence = max(0.0, min(1.0, confidence))
            
            # Get threshold for this regime
            threshold = self.regime_thresholds.get(regime, self.min_confidence)
            
            # Determine validity based on threshold
            is_valid = confidence >= threshold
            
            metadata = {
                "confidence": float(confidence),
                "threshold": float(threshold),
                "validation_method": "enhanced_heuristic",
                "regime": regime,
                "reasons": reasons
            }
            
            return is_valid, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error in heuristic validation: {str(e)}")
            return False, 0.0, {"error": str(e)}
    
    def set_minimum_confidence(self, threshold: float):
        """Set the minimum confidence threshold for signal validation."""
        self.min_confidence = max(0.0, min(1.0, threshold))
        # Update regime-specific thresholds based on new base threshold
        self.regime_thresholds = {
            "trending": max(0.5, self.min_confidence - 0.05),
            "ranging": max(0.55, self.min_confidence + 0.05),
            "volatile": max(0.6, self.min_confidence + 0.1),
            "unknown": self.min_confidence
        }
        self.logger.info(f"Set minimum confidence threshold to {self.min_confidence}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the enhanced signal validator."""
        # Calculate performance metrics from validation history
        validation_performance = self._calculate_validation_performance()
        
        metrics = {
            "models": list(self.models.keys()),
            "regime_thresholds": self.regime_thresholds,
            "min_confidence": self.min_confidence,
            "feature_importance": self.feature_importance,
            "validation_performance": validation_performance,
            "ml_available": ML_AVAILABLE
        }
        
        return metrics
    
    def _calculate_validation_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics from validation history."""
        if not self.validation_history:
            return {}
            
        # Filter entries with known outcomes
        entries_with_outcomes = [
            entry for entry in self.validation_history 
            if entry["outcome"] is not None
        ]
        
        if not entries_with_outcomes:
            return {}
            
        # Calculate overall metrics
        total = len(entries_with_outcomes)
        true_positives = sum(1 for e in entries_with_outcomes if e["is_valid"] and e["outcome"])
        false_positives = sum(1 for e in entries_with_outcomes if e["is_valid"] and not e["outcome"])
        true_negatives = sum(1 for e in entries_with_outcomes if not e["is_valid"] and not e["outcome"])
        false_negatives = sum(1 for e in entries_with_outcomes if not e["is_valid"] and e["outcome"])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate metrics by regime
        regime_metrics = {}
        for regime in set(entry["regime"] for entry in entries_with_outcomes):
            regime_entries = [e for e in entries_with_outcomes if e["regime"] == regime]
            if not regime_entries:
                continue
                
            r_total = len(regime_entries)
            r_true_positives = sum(1 for e in regime_entries if e["is_valid"] and e["outcome"])
            r_false_positives = sum(1 for e in regime_entries if e["is_valid"] and not e["outcome"])
            r_precision = r_true_positives / (r_true_positives + r_false_positives) if (r_true_positives + r_false_positives) > 0 else 0
            
            regime_metrics[regime] = {
                "count": r_total,
                "precision": r_precision,
                "threshold": self.regime_thresholds.get(regime, self.min_confidence)
            }
        
        return {
            "total_validations": total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "by_regime": regime_metrics
        }
