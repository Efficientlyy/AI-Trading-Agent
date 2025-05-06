"""
Machine Learning Strategy Module

This module implements trading strategies based on machine learning models
that predict price movements using technical indicators and other features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .strategy import BaseStrategy, RichSignal, RichSignalsDict
from ..common import logger
from ..indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands


class MLStrategy(BaseStrategy):
    """
    Trading strategy based on machine learning predictions.
    
    This strategy:
    1. Extracts features from price data using technical indicators
    2. Uses a trained ML model to predict price direction
    3. Generates trading signals based on prediction confidence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MLStrategy.
        
        Args:
            config: Configuration dictionary with parameters for the strategy.
                - name: Name of the strategy
                - model_type: Type of ML model to use ('random_forest' or 'gradient_boosting')
                - prediction_horizon: How many periods ahead to predict
                - training_lookback: How many periods of data to use for training
                - retrain_frequency: How often to retrain the model (in periods)
                - confidence_threshold: Minimum prediction probability to generate a signal
                - features: List of features/indicators to use
                - model_path: Optional path to save/load trained models
        """
        super().__init__(config)
        self.name = config.get("name", "MLStrategy")
        self.model_type = config.get("model_type", "random_forest")
        self.prediction_horizon = config.get("prediction_horizon", 1)
        self.training_lookback = config.get("training_lookback", 500)
        self.retrain_frequency = config.get("retrain_frequency", 20)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.feature_list = config.get("features", [
            "rsi", "macd", "bb_position", "price_change", "volume_change", 
            "volatility", "trend_strength"
        ])
        self.model_path = config.get("model_path", None)
        
        # Initialize models dictionary (one model per symbol)
        self.models = {}
        self.scalers = {}
        self.last_train_time = {}
        self.period_counter = {}
        
        logger.info(f"{self.name} initialized with {self.model_type} model and {len(self.feature_list)} features")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> RichSignalsDict:
        """
        Generate trading signals based on ML model predictions.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            **kwargs: Additional keyword arguments
                - current_portfolio: Current portfolio state
                - timestamp: Current timestamp
                
        Returns:
            Dictionary mapping symbols to their signal dictionaries
        """
        if not data:
            logger.warning(f"{self.name}: No data provided for signal generation")
            return {}
        
        signals = {}
        timestamp = kwargs.get("timestamp", pd.Timestamp.now())
        
        for symbol, df in data.items():
            # Skip if not enough data
            if len(df) < 30:  # Need minimum data for feature calculation
                logger.warning(f"{self.name}: Not enough data for {symbol}, skipping")
                continue
            
            # Initialize period counter for this symbol if needed
            if symbol not in self.period_counter:
                self.period_counter[symbol] = 0
            else:
                self.period_counter[symbol] += 1
            
            # Check if we need to train/retrain the model
            if (symbol not in self.models or 
                self.period_counter[symbol] >= self.retrain_frequency):
                self._train_model(symbol, df)
                self.period_counter[symbol] = 0
            
            # Extract features for prediction
            features = self._extract_features(df)
            if features is None or features.empty:
                logger.warning(f"{self.name}: Could not extract features for {symbol}")
                continue
            
            # Make prediction
            signal_data = self._predict_and_generate_signal(symbol, features, timestamp)
            if signal_data:
                signals[symbol] = signal_data
        
        return signals
    
    def _train_model(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Train or retrain the ML model for a symbol.
        
        Args:
            symbol: The trading symbol
            data: Historical price data for the symbol
        """
        logger.info(f"{self.name}: Training model for {symbol}")
        
        # Prepare data for training
        features, target = self._prepare_training_data(data)
        if features is None or target is None:
            logger.warning(f"{self.name}: Could not prepare training data for {symbol}")
            return
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )
        
        # Create and train the model
        if self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            logger.error(f"{self.name}: Unknown model type {self.model_type}")
            return
        
        # Create a pipeline with scaling
        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        # Train the model
        try:
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"{self.name}: Model for {symbol} trained with "
                      f"accuracy={accuracy:.4f}, precision={precision:.4f}, "
                      f"recall={recall:.4f}, f1={f1:.4f}")
            
            # Save the model
            self.models[symbol] = pipeline
            self.last_train_time[symbol] = pd.Timestamp.now()
            
            # Save model to disk if path is provided
            if self.model_path:
                model_file = f"{self.model_path}/{symbol}_{self.name}_model.pkl"
                joblib.dump(pipeline, model_file)
                logger.info(f"{self.name}: Model for {symbol} saved to {model_file}")
                
        except Exception as e:
            logger.error(f"{self.name}: Error training model for {symbol}: {e}")
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features and target variable for model training.
        
        Args:
            data: Historical price data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Extract features
            features = self._extract_features(data)
            if features is None or features.empty:
                return None, None
            
            # Create target variable (price direction after prediction_horizon periods)
            if 'close' in data.columns:
                price_col = 'close'
            else:
                price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
                if not price_cols:
                    logger.warning(f"{self.name}: No suitable price column found")
                    return None, None
                price_col = price_cols[0]
            
            # Calculate future returns
            future_returns = data[price_col].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            
            # Create classification target (1 for up, 0 for down)
            target = (future_returns > 0).astype(int)
            
            # Align features and target and drop NaNs
            aligned_data = pd.concat([features, target], axis=1).dropna()
            if len(aligned_data) < 30:  # Need minimum samples for training
                logger.warning(f"{self.name}: Not enough aligned data for training")
                return None, None
                
            X = aligned_data.iloc[:, :-1]
            y = aligned_data.iloc[:, -1]
            
            return X, y
            
        except Exception as e:
            logger.error(f"{self.name}: Error preparing training data: {e}")
            return None, None
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract technical indicators and other features from price data.
        
        Args:
            data: Historical price data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            # Identify price and volume columns
            if 'close' in data.columns:
                price_col = 'close'
            else:
                price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
                if not price_cols:
                    logger.warning(f"{self.name}: No suitable price column found")
                    return None
                price_col = price_cols[0]
                
            if 'volume' in data.columns:
                volume_col = 'volume'
            else:
                volume_cols = [col for col in data.columns if 'volume' in col.lower()]
                volume_col = volume_cols[0] if volume_cols else None
            
            features = pd.DataFrame(index=data.index)
            
            # Calculate requested features
            for feature in self.feature_list:
                if feature == "rsi" and "rsi" in self.feature_list:
                    features['rsi'] = calculate_rsi(data[price_col])
                    
                elif feature == "macd" and "macd" in self.feature_list:
                    macd, signal, hist = calculate_macd(data[price_col])
                    features['macd'] = macd
                    features['macd_signal'] = signal
                    features['macd_hist'] = hist
                    
                elif feature == "bb_position" and "bb_position" in self.feature_list:
                    upper, middle, lower = calculate_bollinger_bands(data[price_col])
                    # Calculate position within Bollinger Bands (0 to 1)
                    features['bb_position'] = (data[price_col] - lower) / (upper - lower)
                    
                elif feature == "price_change" and "price_change" in self.feature_list:
                    # Price changes over different periods
                    features['price_change_1d'] = data[price_col].pct_change(1)
                    features['price_change_5d'] = data[price_col].pct_change(5)
                    features['price_change_10d'] = data[price_col].pct_change(10)
                    
                elif feature == "volume_change" and "volume_change" in self.feature_list and volume_col:
                    # Volume changes
                    features['volume_change'] = data[volume_col].pct_change(1)
                    features['volume_ma_ratio'] = data[volume_col] / data[volume_col].rolling(10).mean()
                    
                elif feature == "volatility" and "volatility" in self.feature_list:
                    # Volatility (standard deviation of returns)
                    features['volatility_5d'] = data[price_col].pct_change().rolling(5).std()
                    features['volatility_10d'] = data[price_col].pct_change().rolling(10).std()
                    
                elif feature == "trend_strength" and "trend_strength" in self.feature_list:
                    # Simple trend strength indicator
                    prices = data[price_col].values
                    windows = [10, 20, 30]
                    for window in windows:
                        if len(prices) >= window:
                            x = np.arange(window)
                            y = prices[-window:]
                            slope, _ = np.polyfit(x, y, 1)
                            features[f'trend_strength_{window}d'] = slope / np.mean(y) * 100  # Normalized slope
            
            # Drop NaN values that might have been introduced
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Return only the last row for prediction or all rows for training
            return features
            
        except Exception as e:
            logger.error(f"{self.name}: Error extracting features: {e}")
            return None
    
    def _predict_and_generate_signal(self, symbol: str, features: pd.DataFrame, 
                                    timestamp: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """
        Make a prediction using the trained model and generate a trading signal.
        
        Args:
            symbol: The trading symbol
            features: Extracted features for prediction
            timestamp: Current timestamp
            
        Returns:
            Signal dictionary or None if prediction fails
        """
        try:
            if symbol not in self.models:
                logger.warning(f"{self.name}: No trained model for {symbol}")
                return None
            
            # Get the latest feature set for prediction
            latest_features = features.iloc[-1:].copy()
            
            # Make prediction
            model = self.models[symbol]
            prediction_proba = model.predict_proba(latest_features)[0]
            
            # Get probability of price increase (class 1)
            if len(prediction_proba) >= 2:
                up_probability = prediction_proba[1]
            else:
                logger.warning(f"{self.name}: Unexpected prediction format for {symbol}")
                return None
            
            # Convert probability to signal strength (-1 to 1)
            # 0.5 probability means neutral (0), higher means buy, lower means sell
            signal_strength = (up_probability - 0.5) * 2
            
            # Calculate confidence based on distance from 0.5
            confidence_score = abs(up_probability - 0.5) * 2
            
            # Only generate signal if confidence exceeds threshold
            if confidence_score < self.confidence_threshold:
                signal_strength = 0.0  # Neutral signal if not confident enough
            
            # Create rich signal
            return {
                "signal_strength": signal_strength,
                "confidence_score": confidence_score,
                "signal_type": self.name,
                "metadata": {
                    "prediction_probability": up_probability,
                    "prediction_horizon": self.prediction_horizon,
                    "model_type": self.model_type,
                    "last_trained": self.last_train_time.get(symbol, "never"),
                    "timestamp": timestamp
                }
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error making prediction for {symbol}: {e}")
            return None
            
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the strategy's configuration parameters dynamically.
        
        Args:
            config_updates: A dictionary containing parameters to update.
        """
        # Update main config
        self.config.update(config_updates)
        
        # Update specific parameters
        if "model_type" in config_updates:
            self.model_type = config_updates["model_type"]
            # Reset models if model type changes
            self.models = {}
            self.last_train_time = {}
            
        if "prediction_horizon" in config_updates:
            self.prediction_horizon = config_updates["prediction_horizon"]
            # Reset models if prediction horizon changes
            self.models = {}
            self.last_train_time = {}
            
        if "training_lookback" in config_updates:
            self.training_lookback = config_updates["training_lookback"]
            
        if "retrain_frequency" in config_updates:
            self.retrain_frequency = config_updates["retrain_frequency"]
            
        if "confidence_threshold" in config_updates:
            self.confidence_threshold = config_updates["confidence_threshold"]
            
        if "features" in config_updates:
            self.feature_list = config_updates["features"]
            # Reset models if feature list changes
            self.models = {}
            self.last_train_time = {}
            
        if "model_path" in config_updates:
            self.model_path = config_updates["model_path"]
            
        logger.info(f"{self.name} configuration updated")
