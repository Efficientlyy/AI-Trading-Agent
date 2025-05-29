"""
Signal Validator Module

This module provides machine learning-based validation for trading signals
to help filter out false positives and improve signal quality.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import os
from pathlib import Path
import json

# Optional imports for ML models
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..common.utils import get_logger


class MLSignalValidator:
    """
    Uses machine learning to validate technical signals and filter false positives.
    
    This class trains a model on historical signal data and uses it to assign
    a confidence score to new signals.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal validator.
        
        Args:
            config: Configuration dictionary with parameters
                - model_path: Path to a pre-trained model file (optional)
                - model_type: Type of model to use (random_forest, gradient_boosting)
                - min_confidence: Minimum confidence threshold to accept signals
                - features: List of features to use for prediction
        """
        self.logger = get_logger("MLSignalValidator")
        self.config = config or {}
        
        # Extract configuration
        self.model_path = self.config.get("model_path")
        self.model_type = self.config.get("model_type", "random_forest")
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.feature_list = self.config.get("features", [
            "signal_strength", 
            "volatility_percent", 
            "volume_ratio", 
            "price_change_percent",
            "rsi_value",
            "ma_diff_percent",
            "confirmation_count"
        ])
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        
        # Check for ML libraries
        if not ML_AVAILABLE:
            self.logger.warning(
                "Machine learning libraries not available. "
                "Install scikit-learn for ML-based signal validation."
            )
            return
        
        # Load or initialize model
        if self.model_path and os.path.exists(self.model_path):
            self._load_model(self.model_path)
        else:
            self._init_model()
    
    def _init_model(self):
        """Initialize a new model based on the configuration."""
        if not ML_AVAILABLE:
            return
            
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            
        self.scaler = StandardScaler()
        self.logger.info(f"Initialized new {self.model_type} model")
    
    def _load_model(self, model_path: str):
        """Load a pre-trained model from a file."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_importance = model_data.get('feature_importance', {})
            
            self.logger.info(f"Loaded model from {model_path}")
            
            # Verify model type
            if self.model is None:
                self._init_model()
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            self._init_model()
    
    def save_model(self, save_path: str):
        """Save the trained model to a file."""
        if self.model is None:
            self.logger.warning("No model to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'success'):
        """
        Train the model on historical signal data.
        
        Args:
            training_data: DataFrame with signal features and outcomes
            target_column: Name of the column indicating signal success (1=success, 0=failure)
        
        Returns:
            Dictionary with training metrics
        """
        if not ML_AVAILABLE or training_data.empty:
            return {"error": "ML libraries not available or empty training data"}
            
        # Ensure required columns are present
        required_columns = self.feature_list + [target_column]
        missing_columns = [col for col in required_columns if col not in training_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return {"error": f"Missing columns: {missing_columns}"}
            
        try:
            # Prepare features and target
            X = training_data[self.feature_list]
            y = training_data[target_column].astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_list, 
                    self.model.feature_importances_
                ))
                metrics["feature_importance"] = self.feature_importance
            
            self.logger.info(
                f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return {"error": str(e)}
    
    def validate_signal(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate a trading signal and assign a confidence score.
        
        Args:
            signal: Signal dictionary to validate
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary with calculated indicator values
            
        Returns:
            Tuple of (is_valid, confidence_score, validation_metadata)
        """
        if not ML_AVAILABLE or self.model is None:
            # No ML validation available, use simple heuristics
            return self._validate_with_heuristics(signal, market_data, indicators)
            
        try:
            # Extract features from the signal
            features = self._extract_features(signal, market_data, indicators)
            
            if features is None:
                return False, 0.0, {"error": "Failed to extract features"}
                
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            confidence = self.model.predict_proba(features_scaled)[0][1]
            
            # Determine if signal is valid based on confidence threshold
            is_valid = confidence >= self.min_confidence
            
            # Create metadata
            metadata = {
                "confidence": float(confidence),
                "threshold": float(self.min_confidence),
                "model_type": self.model_type,
                "features_used": self.feature_list,
                "validation_method": "machine_learning"
            }
            
            if self.feature_importance:
                metadata["feature_importance"] = {
                    k: float(v) for k, v in self.feature_importance.items()
                }
            
            return is_valid, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {str(e)}")
            # Fall back to heuristics
            return self._validate_with_heuristics(signal, market_data, indicators)
    
    def _extract_features(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict]
    ) -> np.ndarray:
        """Extract features from a signal for model prediction."""
        try:
            features = []
            symbol = signal["payload"]["symbol"]
            
            # Basic signal features
            features.append(abs(signal["payload"]["signal"]))  # signal_strength
            
            # Volatility
            volatility_percent = 0.0
            if "metadata" in signal["payload"] and "volatility_percent" in signal["payload"]["metadata"]:
                volatility_percent = signal["payload"]["metadata"]["volatility_percent"]
            features.append(volatility_percent)
            
            # Volume ratio
            volume_ratio = 1.0
            if "metadata" in signal["payload"] and "volume_ratio" in signal["payload"]["metadata"]:
                volume_ratio = signal["payload"]["metadata"]["volume_ratio"]
            features.append(volume_ratio)
            
            # Price change percent (past N periods)
            price_change_percent = 0.0
            if symbol in market_data and not market_data[symbol].empty:
                df = market_data[symbol]
                if len(df) >= 5:
                    current_price = df['close'].iloc[-1]
                    past_price = df['close'].iloc[-5]
                    price_change_percent = (current_price - past_price) / past_price * 100
            features.append(price_change_percent)
            
            # RSI value
            rsi_value = 50.0  # Default neutral
            if symbol in indicators and "rsi" in indicators[symbol]:
                rsi = indicators[symbol]["rsi"]
                if hasattr(rsi, 'iloc'):
                    rsi_value = rsi.iloc[-1]
                else:
                    rsi_value = rsi[-1]
            features.append(rsi_value)
            
            # MA difference percent
            ma_diff_percent = 0.0
            if symbol in indicators:
                ind = indicators[symbol]
                if "ema" in ind and "9" in ind["ema"] and "sma" in ind and "21" in ind["sma"]:
                    ema = ind["ema"]["9"]
                    sma = ind["sma"]["21"]
                    
                    if hasattr(ema, 'iloc') and hasattr(sma, 'iloc'):
                        ema_val = ema.iloc[-1]
                        sma_val = sma.iloc[-1]
                    else:
                        ema_val = ema[-1]
                        sma_val = sma[-1]
                        
                    ma_diff_percent = (ema_val - sma_val) / sma_val * 100
            features.append(ma_diff_percent)
            
            # Confirmation count from multi-timeframe analysis
            confirmation_count = 0
            if "metadata" in signal["payload"] and "confirmation_count" in signal["payload"]["metadata"]:
                confirmation_count = signal["payload"]["metadata"]["confirmation_count"]
            features.append(confirmation_count)
            
            # Return as numpy array
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def _validate_with_heuristics(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate a signal using simple heuristics when ML is not available.
        
        This is a fallback method that uses rule-based checks to validate signals.
        """
        confidence = 0.5  # Start with neutral confidence
        is_valid = True
        reasons = []
        
        try:
            # Check signal strength
            signal_strength = abs(signal["payload"]["signal"])
            if signal_strength < 0.3:
                confidence -= 0.2
                reasons.append("Low signal strength")
                
            if signal_strength > 0.7:
                confidence += 0.2
                reasons.append("High signal strength")
            
            # Check volume confirmation
            if "metadata" in signal["payload"] and "volume_confirmation" in signal["payload"]["metadata"]:
                vol_conf = signal["payload"]["metadata"]["volume_confirmation"]
                if vol_conf == "high":
                    confidence += 0.15
                    reasons.append("Volume confirmation")
                elif vol_conf == "low":
                    confidence -= 0.1
                    reasons.append("Low volume")
            
            # Check volatility
            if "metadata" in signal["payload"] and "volatility_status" in signal["payload"]["metadata"]:
                vol_status = signal["payload"]["metadata"]["volatility_status"]
                if vol_status == "excessive":
                    confidence -= 0.15
                    reasons.append("Excessive volatility")
            
            # Check confirmation in multi-timeframe strategies
            if "metadata" in signal["payload"] and "confirmation_count" in signal["payload"]["metadata"]:
                count = signal["payload"]["metadata"]["confirmation_count"]
                min_conf = signal["payload"]["metadata"].get("min_confirmations", 1)
                
                if count > min_conf:
                    confidence += 0.1 * (count - min_conf)
                    reasons.append(f"Strong multi-timeframe confirmation ({count})")
            
            # Cap confidence between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            # Determine validity based on threshold
            is_valid = confidence >= self.min_confidence
            
            metadata = {
                "confidence": float(confidence),
                "threshold": float(self.min_confidence),
                "validation_method": "heuristic",
                "reasons": reasons
            }
            
            return is_valid, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error in heuristic validation: {str(e)}")
            return False, 0.0, {"error": str(e)}
    
    def generate_training_data(
        self, 
        signals: List[Dict[str, Any]], 
        outcomes: Dict[str, bool]
    ) -> pd.DataFrame:
        """
        Generate training data from historical signals and their outcomes.
        
        Args:
            signals: List of historical signal dictionaries
            outcomes: Dictionary mapping signal IDs to outcomes (True=success, False=failure)
            
        Returns:
            DataFrame with features and outcomes for training
        """
        data = []
        
        for signal in signals:
            try:
                # Get signal ID and outcome
                signal_id = signal.get("id", str(hash(json.dumps(signal, default=str))))
                
                # Skip if we don't have an outcome for this signal
                if signal_id not in outcomes:
                    continue
                
                # Create feature dictionary
                features = {}
                
                # Add success/failure outcome
                features["success"] = 1 if outcomes[signal_id] else 0
                
                # Add signal strength
                features["signal_strength"] = abs(signal["payload"]["signal"])
                
                # Add metadata features if available
                if "metadata" in signal["payload"]:
                    metadata = signal["payload"]["metadata"]
                    
                    # Volatility
                    features["volatility_percent"] = metadata.get("volatility_percent", 0.0)
                    
                    # Volume
                    features["volume_ratio"] = metadata.get("volume_ratio", 1.0)
                    
                    # Confirmation count
                    features["confirmation_count"] = metadata.get("confirmation_count", 0)
                else:
                    features["volatility_percent"] = 0.0
                    features["volume_ratio"] = 1.0
                    features["confirmation_count"] = 0
                
                # Add placeholder values for features we can't extract from the signal alone
                features["price_change_percent"] = 0.0
                features["rsi_value"] = 50.0
                features["ma_diff_percent"] = 0.0
                
                # Add to data list
                data.append(features)
                
            except Exception as e:
                self.logger.error(f"Error processing signal {signal.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Convert to DataFrame
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=self.feature_list + ["success"])
    
    def set_minimum_confidence(self, threshold: float):
        """Set the minimum confidence threshold for signal validation."""
        self.min_confidence = max(0.0, min(1.0, threshold))
        self.logger.info(f"Set minimum confidence threshold to {self.min_confidence}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the signal validator."""
        metrics = {
            "model_type": self.model_type,
            "min_confidence": self.min_confidence,
            "feature_count": len(self.feature_list),
            "features": self.feature_list,
            "feature_importance": self.feature_importance,
            "ml_available": ML_AVAILABLE
        }
        
        return metrics
