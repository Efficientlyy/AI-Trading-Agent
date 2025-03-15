"""Price prediction strategy using machine learning.

This strategy uses a simple ML model to predict price movements based on
technical indicators and market data.
"""

import numpy as np
from typing import Dict, Any, cast, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.ml.base_ml_strategy import BaseMLStrategy
from src.models.market_data import CandleData, TradeData, OrderBookData
from src.common.logging import get_logger


class PricePredictionStrategy(BaseMLStrategy):
    """ML-based price prediction strategy.
    
    This strategy uses a Random Forest classifier to predict price movements
    based on technical indicators and recent price action.
    """
    
    def __init__(self, strategy_id: str = "price_prediction"):
        """Initialize the strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", strategy_id)
        
        # Feature engineering components
        self.scaler = StandardScaler()
        self.feature_names = [
            "returns_1",      # 1-period returns
            "returns_5",      # 5-period returns
            "returns_20",     # 20-period returns
            "volume_ma_ratio",  # Volume / MA ratio
            "rsi_14",        # 14-period RSI
            "bb_position",   # Position within Bollinger Bands
            "macd_hist"      # MACD histogram
        ]
        
        # Strategy state
        self._is_running = False
    
    async def _strategy_start(self) -> bool:
        """Start the strategy.
        
        Returns:
            True if started successfully, False otherwise
        """
        self._is_running = True
        self.logger.info("Price prediction strategy started")
        return True
    
    async def _strategy_stop(self) -> bool:
        """Stop the strategy.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        self._is_running = False
        self.logger.info("Price prediction strategy stopped")
        return True
    
    async def _strategy_tick(self) -> None:
        """Execute strategy logic on each tick."""
        # This strategy is event-driven via on_candle
        pass
    
    async def process_trade(self, trade: TradeData) -> None:
        """Process new trade data.
        
        Args:
            trade: The trade data
        """
        # This strategy only uses candle data
        pass
    
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process new orderbook data.
        
        Args:
            orderbook: The orderbook data
        """
        # This strategy only uses candle data
        pass
    
    async def process_indicator(self, indicator_name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Process new indicator value.
        
        Args:
            indicator_name: Name of the indicator
            value: The indicator value
            metadata: Optional metadata about the indicator
        """
        # This strategy calculates its own indicators
        pass
    
    async def process_pattern(self, pattern_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Process new pattern detection.
        
        Args:
            pattern_name: Name of the pattern
            metadata: Optional metadata about the pattern
        """
        # This strategy doesn't use pattern detection
        pass
    
    async def _strategy_on_trade(self, trade: TradeData) -> None:
        """Process new trade data.
        
        Args:
            trade: The trade data
        """
        await self.process_trade(trade)
    
    async def _strategy_on_orderbook(self, orderbook: OrderBookData) -> None:
        """Process new orderbook data.
        
        Args:
            orderbook: The orderbook data
        """
        await self.process_orderbook(orderbook)
    
    async def _strategy_on_candle(self, candle: CandleData) -> Optional[Dict[str, Any]]:
        """Process new candle data.
        
        Args:
            candle: The candle data
            
        Returns:
            Optional prediction results
        """
        if not self._is_running:
            return None
            
        # Extract features from new candle
        features = self._extract_features(candle)
        
        # Store features for training
        symbol = candle.symbol
        if symbol not in self.feature_buffer:
            self.feature_buffer[symbol] = []
        self.feature_buffer[symbol].append(features)
        
        # Generate prediction if we have enough data
        X = self._prepare_features(symbol)
        if X.size > 0:
            return self._generate_prediction(symbol, X)
        
        return None
    
    def _extract_features(self, candle: CandleData) -> Dict[str, float]:
        """Extract features from candle data.
        
        Args:
            candle: The candle data
            
        Returns:
            Dictionary of feature names to values
        """
        # Get historical data from buffer
        symbol = candle.symbol
        history = self.feature_buffer.get(symbol, [])
        
        # Calculate basic features
        features = {
            "returns_1": 0.0,
            "returns_5": 0.0,
            "returns_20": 0.0,
            "volume_ma_ratio": 1.0,
            "rsi_14": 50.0,
            "bb_position": 0.0,
            "macd_hist": 0.0
        }
        
        if history:
            # Price returns
            prev_close = float(history[-1].get("close", candle.close))
            curr_close = float(candle.close)
            features["returns_1"] = (curr_close / prev_close) - 1
            
            if len(history) >= 5:
                prev_close_5 = float(history[-5].get("close", candle.close))
                features["returns_5"] = (curr_close / prev_close_5) - 1
            
            if len(history) >= 20:
                prev_close_20 = float(history[-20].get("close", candle.close))
                features["returns_20"] = (curr_close / prev_close_20) - 1
            
            # Volume features
            volumes = [float(h.get("volume", 0)) for h in history[-20:]]
            if volumes:
                volume_ma = sum(volumes) / len(volumes)
                curr_volume = float(candle.volume)
                features["volume_ma_ratio"] = curr_volume / volume_ma if volume_ma > 0 else 1.0
            
            # Technical indicators (simplified calculations)
            closes = [float(h.get("close", 0)) for h in history[-14:]] + [float(candle.close)]
            if len(closes) >= 14:
                # RSI
                changes = np.diff(closes)
                gains = np.where(changes > 0, changes, 0).mean()
                losses = -np.where(changes < 0, changes, 0).mean()
                rs = gains / losses if losses != 0 else 0
                features["rsi_14"] = float(100 - (100 / (1 + rs)))
                
                # Bollinger Bands
                ma20 = np.mean(closes[-20:])
                std20 = np.std(closes[-20:])
                upper_band = ma20 + 2 * std20
                lower_band = ma20 - 2 * std20
                band_range = upper_band - lower_band
                if band_range > 0:
                    features["bb_position"] = float((curr_close - lower_band) / band_range)
                
                # MACD (12,26,9)
                ema12 = np.mean(closes[-12:])  # Simplified EMA
                ema26 = np.mean(closes[-26:])  # Simplified EMA
                macd_line = ema12 - ema26
                signal_line = np.mean([macd_line])  # Simplified signal line
                features["macd_hist"] = float(macd_line - signal_line)
        
        return features
    
    def _create_feature_pipeline(self) -> Any:
        """Create the feature engineering pipeline.
        
        Returns:
            The feature pipeline object
        """
        return self.scaler
    
    def _prepare_features(self, symbol: str) -> np.ndarray:
        """Prepare features for prediction.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            numpy array of features
        """
        # Get recent features from buffer
        features = self.feature_buffer[symbol]
        if not features:
            return np.array([])
        
        # Extract feature values in consistent order
        X = np.array([[feat[name] for name in self.feature_names] 
                      for feat in features])
        
        # Scale features if we have a fitted scaler
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        
        return cast(np.ndarray, X)
    
    def _generate_prediction(self, symbol: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction from features.
        
        Args:
            symbol: The trading pair symbol
            features: The prepared features
            
        Returns:
            Dictionary containing prediction details
        """
        if not features.size or symbol not in self.models:
            return {
                "direction": 0,
                "confidence": 0.0,
                "price": 0.0,
                "target_return": 0.0,
                "max_loss": 0.0,
                "reason": "Insufficient data",
                "feature_importance": {}
            }
        
        # Get latest feature values
        latest_features = features[-1:]
        
        # Make prediction
        model = self.models[symbol]
        probabilities = model.predict_proba(latest_features)[0]
        prediction = model.predict(latest_features)[0]
        
        # Get feature importance
        importance = dict(zip(self.feature_names, 
                            model.feature_importances_))
        
        # Calculate confidence and target metrics
        confidence = max(probabilities)
        direction = 1 if prediction == 1 else -1
        
        # Get current price from latest candle
        current_price = float(self.feature_buffer[symbol][-1].get("close", 0))
        
        # Calculate target return and max loss based on confidence
        target_return = 0.02 * confidence  # 2% target at max confidence
        max_loss = 0.01 * (1 + confidence)  # 1-2% stop loss based on confidence
        
        return {
            "direction": direction,
            "confidence": float(confidence),
            "price": current_price,
            "target_return": float(target_return),
            "max_loss": float(max_loss),
            "reason": f"ML prediction with {confidence:.1%} confidence",
            "feature_importance": importance,
            "features": latest_features[0].tolist()  # Store for training
        }
    
    async def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a new model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            The trained model
        """
        # Create and train a new Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        
        # Fit the model and scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        return model
    
    def _validate_model(self, model: Any, symbol: str) -> float:
        """Validate a model's performance.
        
        Args:
            model: The model to validate
            symbol: The trading pair symbol
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        # Get validation data
        X_val, y_val = self._prepare_training_data(symbol)
        
        if not X_val.size:
            return 0.0
        
        # Scale features
        X_val_scaled = self.scaler.transform(X_val)
        
        # Calculate validation score
        score = model.score(X_val_scaled, y_val)
        
        return float(score) 