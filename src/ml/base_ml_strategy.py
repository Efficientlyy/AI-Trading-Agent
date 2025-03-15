"""Base ML strategy for the AI Crypto Trading System.

This module defines the base class for all machine learning based trading strategies.
It extends the base Strategy class with ML-specific functionality including:
- Model management (loading, saving, versioning)
- Feature engineering pipeline integration
- Prediction generation and signal conversion
- Online learning and model updates
"""

import os
from abc import abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None
    print("Warning: joblib not installed. Model persistence will not be available.")
    print("Install with: pip install joblib")

from src.strategy.base_strategy import Strategy
from src.models.market_data import CandleData, OrderBookData, TradeData, TimeFrame
from src.models.signals import Signal, SignalType
from src.common.logging import get_logger


class BaseMLStrategy(Strategy):
    """Base class for machine learning based trading strategies.
    
    This class extends the base Strategy class with functionality specific to
    ML-based strategies, including model management, feature engineering,
    and prediction generation.
    """
    
    def __init__(self, strategy_id: str):
        """Initialize the ML strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        
        # ML-specific configuration
        self.model_dir = os.path.join("models", strategy_id)
        self.feature_config = self._load_feature_config()
        self.model_config = self._load_model_config()
        
        # Model state
        self.models: Dict[str, Any] = {}  # symbol -> model
        self.feature_pipeline = None
        self.feature_buffer: Dict[str, List[Dict[str, float]]] = {}  # symbol -> feature history
        self.prediction_history: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> prediction history
        
        # Performance tracking
        self.prediction_metrics: Dict[str, Dict[str, float]] = {}  # symbol -> metrics
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _load_feature_config(self) -> Dict[str, Any]:
        """Load feature engineering configuration.
        
        Returns:
            Dictionary containing feature configuration
        """
        return {
            "window_size": 100,  # Number of candles to keep in feature buffer
            "features": [
                "returns",  # Price returns
                "volume",   # Volume
                "volatility",  # Price volatility
                "rsi",     # Relative Strength Index
                "macd",    # MACD
                "bb",      # Bollinger Bands
                # Add more features as needed
            ],
            "normalization": "standard",  # or "minmax", "robust", etc.
            "missing_data": "forward_fill"  # or "backward_fill", "zero", etc.
        }
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            "type": "classifier",  # or "regressor"
            "prediction_threshold": 0.6,  # Minimum confidence for signal generation
            "update_frequency": 1000,  # Candles between model updates
            "max_history": 10000,  # Maximum predictions to keep in history
            "validation": {
                "window_size": 500,  # Samples for validation
                "min_accuracy": 0.55,  # Minimum accuracy to keep model
                "min_samples": 1000,  # Minimum samples before training
            }
        }
    
    async def _strategy_initialize(self) -> None:
        """Initialize the ML strategy."""
        # Load saved models if they exist
        for symbol in self.symbols:
            model_path = self._get_model_path(symbol)
            if os.path.exists(model_path):
                try:
                    self.models[symbol] = self._load_model(model_path)
                    self.logger.info(f"Loaded model for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error loading model for {symbol}: {e}")
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = self._create_feature_pipeline()
        
        # Initialize buffers
        for symbol in self.symbols:
            self.feature_buffer[symbol] = []
            self.prediction_history[symbol] = []
            self.prediction_metrics[symbol] = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle.
        
        This method handles the core ML strategy logic:
        1. Update feature buffer
        2. Generate features
        3. Make predictions
        4. Generate signals
        5. Update model if needed
        
        Args:
            candle: The new candle data
        """
        symbol = candle.symbol
        
        # Update feature buffer
        features = self._extract_features(candle)
        self._update_feature_buffer(symbol, features)
        
        # Generate prediction if we have enough data
        if len(self.feature_buffer[symbol]) >= self.feature_config["window_size"]:
            # Prepare features for prediction
            X = self._prepare_features(symbol)
            
            # Get prediction
            if symbol in self.models:
                prediction = self._generate_prediction(symbol, X)
                self._update_prediction_history(symbol, prediction)
                
                # Generate signal if confidence is high enough
                if abs(prediction["confidence"]) >= self.model_config["prediction_threshold"]:
                    await self._generate_signal(symbol, prediction)
                
                # Update model if needed
                await self._update_model_if_needed(symbol)
    
    @abstractmethod
    def _extract_features(self, candle: CandleData) -> Dict[str, float]:
        """Extract features from candle data.
        
        Args:
            candle: The candle data
            
        Returns:
            Dictionary of feature names to values
        """
        pass
    
    @abstractmethod
    def _create_feature_pipeline(self) -> Any:
        """Create the feature engineering pipeline.
        
        Returns:
            The feature pipeline object
        """
        pass
    
    @abstractmethod
    def _prepare_features(self, symbol: str) -> np.ndarray:
        """Prepare features for prediction.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            numpy array of features
        """
        pass
    
    @abstractmethod
    def _generate_prediction(self, symbol: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction from features.
        
        Args:
            symbol: The trading pair symbol
            features: The prepared features
            
        Returns:
            Dictionary containing prediction details
        """
        pass
    
    def _update_feature_buffer(self, symbol: str, features: Dict[str, float]) -> None:
        """Update the feature buffer for a symbol.
        
        Args:
            symbol: The trading pair symbol
            features: The new features to add
        """
        self.feature_buffer[symbol].append(features)
        
        # Maintain buffer size
        if len(self.feature_buffer[symbol]) > self.feature_config["window_size"]:
            self.feature_buffer[symbol].pop(0)
    
    def _update_prediction_history(self, symbol: str, prediction: Dict[str, Any]) -> None:
        """Update prediction history for a symbol.
        
        Args:
            symbol: The trading pair symbol
            prediction: The new prediction to add
        """
        self.prediction_history[symbol].append(prediction)
        
        # Maintain history size
        if len(self.prediction_history[symbol]) > self.model_config["max_history"]:
            self.prediction_history[symbol].pop(0)
    
    async def _generate_signal(self, symbol: str, prediction: Dict[str, Any]) -> None:
        """Generate a trading signal from a prediction.
        
        Args:
            symbol: The trading pair symbol
            prediction: The prediction details
        """
        # Determine signal direction
        if prediction["direction"] > 0:
            signal_type = SignalType.ENTRY
            direction = "long"
        else:
            signal_type = SignalType.ENTRY
            direction = "short"
        
        # Calculate take profit and stop loss
        current_price = prediction["price"]
        take_profit = current_price * (1 + prediction["target_return"]) if direction == "long" else \
                     current_price * (1 - prediction["target_return"])
        stop_loss = current_price * (1 - prediction["max_loss"]) if direction == "long" else \
                   current_price * (1 + prediction["max_loss"])
        
        # Publish the signal
        await self.publish_signal(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            timeframe=TimeFrame.ONE_HOUR,  # Default timeframe
            price=current_price,
            confidence=prediction["confidence"],
            reason=f"ML prediction: {prediction['reason']}",
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata={
                "model_version": self.models[symbol].get("version", "unknown"),
                "features_used": list(prediction["feature_importance"].keys()),
                "prediction_details": prediction
            }
        )
    
    async def _update_model_if_needed(self, symbol: str) -> None:
        """Update the model if conditions are met.
        
        Args:
            symbol: The trading pair symbol
        """
        # Check if we have enough new data
        if len(self.prediction_history[symbol]) < self.model_config["update_frequency"]:
            return
        
        # Get validation data
        validation_window = self.prediction_history[symbol][-self.model_config["validation"]["window_size"]:]
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(validation_window)
        self.prediction_metrics[symbol].update(metrics)
        
        # Check if model needs updating
        if metrics["accuracy"] < self.model_config["validation"]["min_accuracy"]:
            self.logger.warning(f"Model performance below threshold for {symbol}, updating model")
            await self._retrain_model(symbol)
    
    def _calculate_performance_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics for a set of predictions.
        
        Args:
            predictions: List of prediction records
            
        Returns:
            Dictionary of metric names to values
        """
        # This is a simple implementation - extend based on needs
        total = len(predictions)
        if total == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / total
        
        # Add more sophisticated metrics as needed
        return {
            "accuracy": accuracy,
            "precision": accuracy,  # Simplified
            "recall": accuracy,     # Simplified
            "f1": accuracy         # Simplified
        }
    
    async def _retrain_model(self, symbol: str) -> None:
        """Retrain the model for a symbol.
        
        Args:
            symbol: The trading pair symbol
        """
        try:
            # Collect training data
            X_train, y_train = self._prepare_training_data(symbol)
            
            if len(X_train) < self.model_config["validation"]["min_samples"]:
                self.logger.warning(f"Insufficient training data for {symbol}")
                return
            
            # Train new model
            new_model = await self._train_model(X_train, y_train)
            
            # Validate new model
            validation_score = self._validate_model(new_model, symbol)
            
            if validation_score >= self.model_config["validation"]["min_accuracy"]:
                # Save and update model
                self._save_model(new_model, symbol)
                self.models[symbol] = new_model
                self.logger.info(f"Updated model for {symbol}, validation score: {validation_score:.3f}")
            else:
                self.logger.warning(f"New model for {symbol} failed validation")
                
        except Exception as e:
            self.logger.error(f"Error retraining model for {symbol}: {e}")
    
    @abstractmethod
    async def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a new model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            The trained model
        """
        pass
    
    @abstractmethod
    def _validate_model(self, model: Any, symbol: str) -> float:
        """Validate a model's performance.
        
        Args:
            model: The model to validate
            symbol: The trading pair symbol
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        pass
    
    def _get_model_path(self, symbol: str) -> str:
        """Get the path for a model file.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Path to the model file
        """
        # Convert symbol to filename-safe format
        safe_symbol = symbol.replace("/", "_").lower()
        return os.path.join(self.model_dir, f"{safe_symbol}_model.joblib")
    
    def _save_model(self, model: Any, symbol: str) -> None:
        """Save a model to disk.
        
        Args:
            model: The model to save
            symbol: The trading pair symbol
        """
        if joblib is None:
            self.logger.error("Cannot save model: joblib not installed")
            return
            
        try:
            model_path = self._get_model_path(symbol)
            joblib.dump(model, model_path)
            self.logger.info(f"Saved model for {symbol}")
        except Exception as e:
            self.logger.error(f"Error saving model for {symbol}: {e}")
    
    def _load_model(self, model_path: str) -> Any:
        """Load a model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            The loaded model or None if loading fails
        """
        if joblib is None:
            self.logger.error("Cannot load model: joblib not installed")
            return None
            
        try:
            return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def _prepare_training_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical predictions.
        
        This method collects historical predictions and their outcomes
        to create a training dataset for model updates.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Tuple of (features array, labels array)
        """
        # Get historical predictions
        history = self.prediction_history[symbol]
        
        if not history:
            return np.array([]), np.array([])
        
        # Extract features and labels
        features = []
        labels = []
        
        for pred in history:
            if "features" in pred and "actual_outcome" in pred:
                features.append(pred["features"])
                labels.append(pred["actual_outcome"])
        
        if not features:
            return np.array([]), np.array([])
        
        return np.array(features), np.array(labels) 