"""Enhanced price prediction strategy using ML with sentiment and correlation analysis."""

from typing import Dict, List, Optional, Tuple, Union, Set, cast, Any, TypedDict, Protocol, NotRequired
from typing_extensions import TypeAlias
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from numpy.typing import NDArray, ArrayLike
from collections import deque, defaultdict
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.ml.base_ml_strategy import BaseMLStrategy
from src.models.market_data import CandleData, OrderBookData, TradeData, TimeFrame
from src.indicators.advanced_indicators import (
    calculate_heikin_ashi,
    calculate_keltner_channels,
    calculate_atr,
    detect_market_regime,
    calculate_stoch_rsi,
    calculate_cmf
)
from src.indicators.types import (
    HeikinAshiCandles,
    KeltnerChannels,
    MarketRegime,
    PriceData,
    VolumeData,
    IndicatorOutput
)
from src.sentiment.market_sentiment import MarketSentiment
from src.analysis.correlation_tracker import CorrelationTracker
from src.common.logging import get_logger
from src.models.signals import SignalType

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Type aliases for clarity
PriceType: TypeAlias = float
VolumeType: TypeAlias = float
TimestampType: TypeAlias = datetime
ScoreType: TypeAlias = float

# TypedDict definitions for structured data
class OrderLevel(TypedDict):
    price: PriceType
    amount: VolumeType

class PredictionMetrics(TypedDict):
    prediction: float
    confidence: float
    direction: int
    timestamp: float

class ModelMetrics(TypedDict):
    mae: float
    rmse: float
    accuracy: float
    f1_score: float
    timestamp: NotRequired[float]

# Protocol for sentiment analysis
class SentimentAnalyzer(Protocol):
    def update_social_sentiment(self, symbol: str, score: float) -> None: ...
    def update_news_sentiment(self, symbol: str, score: float) -> None: ...
    def update_fear_greed(self, score: float) -> None: ...
    def update_order_flow_sentiment(self, symbol: str, buy_volume: float, sell_volume: float) -> None: ...
    def get_aggregate_sentiment(self, symbol: str) -> Dict[str, float]: ...

class EnhancedPricePredictionStrategy(BaseMLStrategy):
    """Enhanced price prediction strategy using ML with sentiment and correlation analysis."""
    
    def __init__(
        self,
        strategy_id: str,
        trading_symbols: List[str],
        lookback_window: int = 100,
        prediction_horizon: int = 24,
        confidence_threshold: float = 0.7,
        correlation_threshold: float = 0.7,
        timeframe: TimeFrame = TimeFrame.ONE_HOUR,
        max_buffer_size: int = 1000,
        max_workers: int = 4,  # Number of threads for parallel processing
        cache_size: int = 128,  # Size of LRU cache for calculations
        config: Dict[str, Any] = {}
    ):
        """Initialize the strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            trading_symbols: List of trading symbols to track
            lookback_window: Number of candles to use for feature calculation
            prediction_horizon: Number of periods to predict ahead
            confidence_threshold: Minimum confidence for trade signals
            correlation_threshold: Threshold for correlation clustering
            timeframe: The time frame for the strategy
            max_buffer_size: Maximum size for data buffers
            max_workers: Number of threads for parallel processing
            cache_size: Size of LRU cache for calculations
            config: Strategy configuration
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", strategy_id)
        
        self.trading_symbols = trading_symbols
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        self.correlation_threshold = correlation_threshold
        self.timeframe = timeframe
        self.max_buffer_size = max_buffer_size
        
        # Performance monitoring
        self.performance_metrics = {
            "prediction_time": deque(maxlen=100),
            "feature_extraction_time": deque(maxlen=100),
            "signal_generation_time": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "prediction_accuracy": deque(maxlen=100)
        }
        
        # Initialize components
        self.sentiment_analyzer = MarketSentiment()
        self.correlation_tracker = CorrelationTracker()
        
        # ML Models
        self.rf_models: Dict[str, RandomForestClassifier] = {}
        self.gb_models: Dict[str, GradientBoostingClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Use deque for efficient memory management
        self.candle_buffer: Dict[str, deque] = {}
        self.trade_buffer: Dict[str, deque] = {}
        self.orderbook_buffer: Dict[str, deque] = {}
        self.pattern_signals: Dict[str, Dict[str, Any]] = {}
        self.feature_buffer: Dict[str, List[Dict[str, float]]] = {}
        self.atr_values: Dict[str, deque] = {}
        self.prediction_history: Dict[str, deque] = {}
        
        # Initialize buffers for each symbol
        for symbol in trading_symbols:
            self.candle_buffer[symbol] = deque(maxlen=max_buffer_size)
            self.trade_buffer[symbol] = deque(maxlen=max_buffer_size)
            self.orderbook_buffer[symbol] = deque(maxlen=max_buffer_size)
            self.atr_values[symbol] = deque(maxlen=100)
            self.prediction_history[symbol] = deque(maxlen=100)
            
            # Initialize ML models
            self.rf_models[symbol] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
            self.gb_models[symbol] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.scalers[symbol] = StandardScaler()
        
        # Feature names
        self.feature_names = [
            "returns_1h", "returns_4h", "returns_24h",
            "volume_ma_ratio", "volatility",
            "rsi_14", "bb_position", "macd_hist",
            "sentiment_score", "correlation_score",
            "market_regime", "liquidity_score"
        ]
        
        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.calculation_cache: Dict[str, Any] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Model performance tracking
        self.model_metrics = {
            "accuracy": defaultdict(list),
            "precision": defaultdict(list),
            "recall": defaultdict(list),
            "f1_score": defaultdict(list)
        }
        
        # Circuit breakers
        self.volatility_threshold = 3.0  # Standard deviations
        self.max_drawdown_threshold = 0.1  # 10% drawdown
        self.circuit_breakers: Dict[str, bool] = {symbol: False for symbol in trading_symbols}
        
        # Recovery tracking
        self.data_gaps: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.max_recovery_attempts = 3
        
        # Configuration
        self.min_price_move = config.get("min_price_move", 0.0001)  # Minimum price movement to consider
        self.min_samples = config.get("min_samples", 30)  # Minimum samples needed for calculations
    
    async def _strategy_initialize(self) -> None:
        """Initialize the strategy."""
        self.logger.info(
            f"Initializing enhanced price prediction strategy for symbols: {self.trading_symbols}"
        )
        
        # Initialize buffers for each symbol
        for symbol in self.trading_symbols:
            # Initialize ML models
            self.rf_models[symbol] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
            self.gb_models[symbol] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.scalers[symbol] = StandardScaler()
    
    async def _strategy_start(self) -> None:
        """Start strategy execution."""
        self.logger.info("Starting enhanced price prediction strategy")
        # Initialize data for each symbol
        for symbol in self.trading_symbols:
            self._update_sentiment(symbol)
            
    async def _strategy_stop(self) -> None:
        """Stop strategy execution."""
        self.logger.info("Stopping enhanced price prediction strategy")
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle and generate predictions."""
        if candle.symbol in self.trading_symbols:
            # Update data buffers
            self.candle_buffer[candle.symbol].append(candle)
            if len(self.candle_buffer[candle.symbol]) > self.lookback_window:
                self.candle_buffer[candle.symbol].popleft()  # Use popleft() for deque
            
            # Update correlation tracker
            self.correlation_tracker.update_data(
                candle.symbol,
                float(candle.close),
                float(candle.volume),
                candle.timestamp
            )
            
            # Update sentiment
            self._update_sentiment(candle.symbol)
            
            # Generate predictions if we have enough data
            if len(self.candle_buffer[candle.symbol]) >= self.lookback_window:
                await self._generate_predictions(candle.symbol)
    
    async def process_trade(self, trade: TradeData) -> None:
        """Process a new trade."""
        if trade.symbol in self.trading_symbols:
            self.trade_buffer[trade.symbol].append(trade)
            if len(self.trade_buffer[trade.symbol]) > self.lookback_window:
                self.trade_buffer[trade.symbol].popleft()  # Use popleft() for deque
    
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process a new orderbook update."""
        if orderbook.symbol in self.trading_symbols:
            self.orderbook_buffer[orderbook.symbol].append(orderbook)
            if len(self.orderbook_buffer[orderbook.symbol]) > self.lookback_window:
                self.orderbook_buffer[orderbook.symbol].popleft()  # Use popleft() for deque
    
    async def process_indicator(self, symbol: str, timeframe: TimeFrame, indicator_name: str, values: Dict) -> None:
        """Process a technical indicator update."""
        if symbol not in self.trading_symbols:
            return
            
        # Update relevant indicator values in the feature set
        if indicator_name in self.feature_names:
            self.feature_buffer[symbol].append({indicator_name: values.get("value", 0.0)})
    
    async def process_pattern(self, symbol: str, timeframe: TimeFrame, pattern_name: str, confidence: float, target_price: Optional[float], invalidation_price: Optional[float]) -> None:
        """Process a pattern detection."""
        if symbol not in self.trading_symbols:
            return
            
        # Use pattern information to adjust prediction confidence
        if confidence > self.confidence_threshold:
            self.pattern_signals[symbol] = {
                "pattern": pattern_name,
                "confidence": confidence,
                "target_price": target_price,
                "invalidation_price": invalidation_price
            }
    
    def _update_sentiment(self, symbol: str) -> None:
        """Update sentiment metrics for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Update social media sentiment (default to neutral)
            self.sentiment_analyzer.update_social_sentiment(symbol, 0.0)
            
            # Update news sentiment (default to neutral)
            self.sentiment_analyzer.update_news_sentiment(symbol, 0.0)
            
            # Update fear/greed index (default to neutral)
            self.sentiment_analyzer.update_fear_greed(50.0)
            
            # Calculate order flow sentiment
            if self.trade_buffer[symbol]:
                buy_volume = sum(
                    float(t.price * t.size) for t in self.trade_buffer[symbol] if t.side == "buy"
                )
                sell_volume = sum(
                    float(t.price * t.size) for t in self.trade_buffer[symbol] if t.side == "sell"
                )
                # Update order flow sentiment
                self.sentiment_analyzer.update_order_flow_sentiment(symbol, buy_volume, sell_volume)
                
        except Exception as e:
            self.logger.error(f"Error updating sentiment for {symbol}: {str(e)}")
    
    def _extract_features(self, candle: CandleData) -> Dict[str, float]:
        """Extract features from market data."""
        symbol = candle.symbol
        features = {}
        
        if len(self.candle_buffer[symbol]) < self.lookback_window:
            return features
            
        # Calculate price returns
        current_price = float(candle.close)
        hist_prices = [float(c.close) for c in self.candle_buffer[symbol]]
        
        features["returns_1h"] = (current_price / hist_prices[-1]) - 1 if hist_prices else 0
        features["returns_4h"] = (current_price / hist_prices[-4]) - 1 if len(hist_prices) >= 4 else 0
        features["returns_24h"] = (current_price / hist_prices[-24]) - 1 if len(hist_prices) >= 24 else 0
        
        # Volume and volatility
        volumes = [float(c.volume) for c in self.candle_buffer[symbol]]
        features["volume_ma_ratio"] = float(candle.volume) / (sum(volumes) / len(volumes)) if volumes else 1.0
        features["volatility"] = np.std(hist_prices) if len(hist_prices) > 1 else 0
        
        # Technical indicators (simplified calculations)
        features["rsi_14"] = self._calculate_rsi(np.array(hist_prices[-14:]), 14)
        features["bb_position"] = self._calculate_bb_position(np.array(hist_prices[-20:]), current_price)
        features["macd_hist"] = self._calculate_macd_histogram(np.array(hist_prices[-26:]))
        
        # Market sentiment and correlation
        sentiment_data = self.sentiment_analyzer.get_aggregate_sentiment(symbol)
        features["sentiment_score"] = float(sentiment_data["sentiment_score"])
        
        # Get correlations for the symbol
        correlations = self.correlation_tracker.get_correlations(symbol, window=24)  # Use 24h window
        features["correlation_score"] = float(np.mean([
            corr for corr in correlations["price"].values() if not np.isnan(corr)
        ]))
        
        # Market regime and liquidity
        features["market_regime"] = float(self._detect_market_regime(np.array(hist_prices[-20:])))
        features["liquidity_score"] = float(self._calculate_liquidity_score(symbol))
        
        return features
    
    def _get_feature_importance(self, symbol: str, features: NDArray) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            symbol: Trading symbol
            features: Feature vector
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if symbol not in self.models:
            return {}
            
        # Get feature importance from both models
        rf_importance = self.models[symbol]["random_forest"].feature_importances_
        gb_importance = self.models[symbol]["gradient_boost"].feature_importances_
        
        # Average the importance scores
        importance = (rf_importance + gb_importance) / 2
        
        # Map scores to feature names
        feature_names = [
            "price_trend",
            "volume_profile",
            "volatility",
            "sentiment",
            "correlation"
        ]
        
        return {name: float(score) for name, score in zip(feature_names, importance)}

    def _monitor_performance(self, operation: str, start_time: float) -> None:
        """Monitor and log performance metrics."""
        end_time = time.time()
        duration = end_time - start_time
        
        if operation in self.performance_metrics:
            self.performance_metrics[operation].append(duration)
            
        # Monitor memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.performance_metrics["memory_usage"].append(memory_info.rss / 1024 / 1024)  # MB
        
        # Log performance metrics periodically
        if len(self.performance_metrics[operation]) == self.performance_metrics[operation].maxlen:
            avg_duration = np.mean(self.performance_metrics[operation])
            avg_memory = np.mean(self.performance_metrics["memory_usage"])
            self.logger.info(
                f"Performance metrics - {operation}: {avg_duration:.3f}s, "
                f"Memory: {avg_memory:.2f}MB"
            )

    def _validate_prediction(self, symbol: str, prediction: Dict[str, Any]) -> bool:
        """Validate prediction output."""
        try:
            # Check required fields
            required_fields = ["direction", "confidence", "price", "target_return", "max_loss"]
            if not all(field in prediction for field in required_fields):
                self.logger.warning(f"Missing required fields in prediction for {symbol}")
                return False
                
            # Validate numeric values
            if not isinstance(prediction["confidence"], (int, float)):
                self.logger.warning(f"Invalid confidence value for {symbol}")
                return False
                
            if not 0 <= prediction["confidence"] <= 1:
                self.logger.warning(f"Confidence value out of range for {symbol}")
                return False
                
            if not isinstance(prediction["price"], (int, float)) or prediction["price"] <= 0:
                self.logger.warning(f"Invalid price value for {symbol}")
                return False
                
            # Check for extreme predictions
            if abs(prediction["target_return"]) > 0.1:  # More than 10% target
                self.logger.warning(f"Unusually large target return for {symbol}")
                return False
                
            if abs(prediction["max_loss"]) > 0.05:  # More than 5% stop loss
                self.logger.warning(f"Unusually large max loss for {symbol}")
                return False
                
            # Store prediction for historical analysis
            self.prediction_history[symbol].append(prediction)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating prediction for {symbol}: {str(e)}")
            return False

    async def train_models(self, symbol: str, historical_data: pd.DataFrame) -> None:
        """Train ML models with historical data.
        
        Args:
            symbol: Trading symbol
            historical_data: DataFrame with historical price and feature data
        """
        try:
            # Prepare features and labels
            X = historical_data[self.feature_names].values
            y = self._prepare_labels(historical_data)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            rf_metrics = []
            gb_metrics = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                X_train_scaled = self.scalers[symbol].fit_transform(X_train)
                X_test_scaled = self.scalers[symbol].transform(X_test)
                
                # Train Random Forest
                self.rf_models[symbol].fit(X_train_scaled, y_train)
                rf_pred = self.rf_models[symbol].predict(X_test_scaled)
                rf_metrics.append(self._calculate_metrics(y_test, rf_pred))
                
                # Train Gradient Boosting
                self.gb_models[symbol].fit(X_train_scaled, y_train)
                gb_pred = self.gb_models[symbol].predict(X_test_scaled)
                gb_metrics.append(self._calculate_metrics(y_test, gb_pred))
            
            # Update model metrics
            self._update_model_metrics(symbol, "random_forest", rf_metrics)
            self._update_model_metrics(symbol, "gradient_boost", gb_metrics)
            
            self.logger.info(f"Models trained successfully for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {str(e)}")
            raise

    def _prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare labels for model training."""
        # Calculate future returns
        future_returns = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Create binary labels (1 for positive returns, 0 for negative)
        labels = (future_returns > 0).astype(int)
        
        return labels.values[:-self.prediction_horizon]  # Remove last rows where we don't have future data

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred))
        }

    def _update_model_metrics(self, symbol: str, model_type: str, metrics: List[Dict[str, float]]) -> None:
        """Update model performance metrics."""
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            avg_metric = np.mean([m[metric_name] for m in metrics])
            self.model_metrics[metric_name][f"{symbol}_{model_type}"].append(avg_metric)

    @lru_cache(maxsize=128)
    def _calculate_technical_indicators(self, symbol: str, prices: Tuple[float, ...]) -> Dict[str, float]:
        """Calculate technical indicators with caching."""
        price_array = np.array(prices)
        return {
            "rsi": self._calculate_rsi(price_array),
            "bb_position": self._calculate_bb_position(price_array, price_array[-1]),
            "macd_hist": self._calculate_macd_histogram(price_array)
        }

    def _check_circuit_breakers(self, symbol: str) -> bool:
        """Check if circuit breakers should be activated."""
        if not self.candle_buffer[symbol]:
            return False
            
        try:
            # Calculate recent volatility
            prices = [float(c.close) for c in self.candle_buffer[symbol]]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)
            
            # Calculate drawdown
            peak = max(prices)
            current = prices[-1]
            drawdown = (peak - current) / peak
            
            # Activate circuit breakers if thresholds are exceeded
            if volatility > self.volatility_threshold or drawdown > self.max_drawdown_threshold:
                self.circuit_breakers[symbol] = True
                self.logger.warning(
                    f"Circuit breakers activated for {symbol}. "
                    f"Volatility: {volatility:.4f}, Drawdown: {drawdown:.4f}"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breakers for {symbol}: {str(e)}")
            return True  # Activate circuit breakers on error

    def _handle_data_gap(self, symbol: str, current_time: datetime, last_update: datetime) -> None:
        """Handle data gaps in market data."""
        gap_duration = current_time - last_update
        
        if gap_duration > timedelta(minutes=5):  # Threshold for gap detection
            self.data_gaps[symbol].append((last_update, current_time))
            self.logger.warning(f"Data gap detected for {symbol}: {gap_duration}")
            
            # Attempt recovery
            if self.recovery_attempts[symbol] < self.max_recovery_attempts:
                self.recovery_attempts[symbol] += 1
                self._attempt_data_recovery(symbol, last_update, current_time)
            else:
                self.logger.error(f"Max recovery attempts reached for {symbol}")

    def _attempt_data_recovery(self, symbol: str, start_time: datetime, end_time: datetime) -> None:
        """Attempt to recover missing data."""
        try:
            # Implement data recovery logic here
            # This could involve:
            # 1. Requesting historical data from exchange
            # 2. Interpolating missing values
            # 3. Using alternative data sources
            pass
            
        except Exception as e:
            self.logger.error(f"Error recovering data for {symbol}: {str(e)}")

    def _calculate_sentiment_features(self, symbol: str) -> NDArray[np.float64]:
        """Calculate sentiment-based features.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Array of sentiment features
        """
        try:
            sentiment_data = self.sentiment_analyzer.get_aggregate_sentiment(symbol)
            
            # Extract and normalize sentiment scores
            social_score = float(sentiment_data["social_sentiment"])
            news_score = float(sentiment_data["news_sentiment"])
            order_flow_score = float(sentiment_data["order_flow_sentiment"])
            fear_greed = float(sentiment_data["fear_greed_index"]) / 100.0  # Normalize to 0-1
            
            return np.array([
                social_score,
                news_score,
                order_flow_score,
                fear_greed
            ], dtype=np.float64)
            
        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Error calculating sentiment features: {e}")
            return np.zeros(4, dtype=np.float64)

    def _calculate_trend_strength(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate trend strength using ADX.
        
        Args:
            prices: Array of prices
            period: Calculation period
            
        Returns:
            Trend strength value
        """
        try:
            if len(prices) < period:
                return 0.0
                
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate directional movement
            pos_dm = np.where(price_changes > 0, price_changes, 0)
            neg_dm = np.where(price_changes < 0, -price_changes, 0)
            
            # Calculate smoothed DM
            pos_dm_smooth = np.mean(pos_dm[-period:])
            neg_dm_smooth = np.mean(neg_dm[-period:])
            
            # Calculate ADX
            if pos_dm_smooth + neg_dm_smooth == 0:
                return 0.0
                
            dx = abs(pos_dm_smooth - neg_dm_smooth) / (pos_dm_smooth + neg_dm_smooth)
            # Convert dx to array before indexing
            dx_array = np.array([dx])
            adx = float(np.mean(dx_array))
            
            return min(max(adx, 0.0), 1.0)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
            
    def _calculate_volatility(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate price volatility.
        
        Args:
            prices: Array of prices
            period: Calculation period
            
        Returns:
            Volatility value
        """
        try:
            if len(prices) < period:
                return 0.0
                
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate volatility (standard deviation of returns)
            volatility = float(np.std(returns[-period:]))
            
            # Normalize to 0-1 using typical volatility range
            normalized = min(max(volatility / 0.02, 0.0), 1.0)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        # Convert numpy array to list for compatibility
        price_list = prices.tolist()
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _calculate_bb_position(self, prices: np.ndarray, current_price: Optional[float] = None) -> float:
        """Calculate position within Bollinger Bands."""
        if current_price is None:
            if len(prices) == 0:
                return 0.0
            current_price = float(prices[-1])
        
        window = 20
        num_std = 2
        
        if len(prices) < window:
            return 0.0
        
        rolling_mean = np.mean(prices[-window:])
        rolling_std = np.std(prices[-window:])
        
        upper_band = rolling_mean + (num_std * rolling_std)
        lower_band = rolling_mean - (num_std * rolling_std)
        
        # Calculate position as percentage between lower and upper bands
        band_range = upper_band - lower_band
        if band_range == 0:
            return 0.0
        
        position = (float(current_price) - float(lower_band)) / float(band_range)
        return float(position)

    def _calculate_macd_histogram(self, prices: np.ndarray) -> float:
        """Calculate MACD histogram value."""
        # Convert numpy array to list for compatibility
        price_list = prices.tolist()
        if len(price_list) < 26:
            return 0.0
        
        # Calculate EMAs
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        
        # Calculate MACD line
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-period EMA of MACD line)
        signal_line = np.mean(prices[-9:])
        
        # Calculate histogram
        histogram = macd_line - signal_line
        return float(histogram)

    def _detect_market_regime(self, prices: np.ndarray) -> float:
        """Detect market regime (trending vs ranging)."""
        # Convert numpy array to list for compatibility
        price_list = prices.tolist()
        if len(price_list) < 20:
            return 0.0
        
        # Calculate directional movement
        returns = np.diff(prices) / prices[:-1]
        trend_strength = np.abs(np.mean(returns)) / np.std(returns)
        return float(trend_strength)

    def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate market liquidity score based on order book."""
        if symbol not in self.orderbook_buffer or not self.orderbook_buffer[symbol]:
            return 0.0
        
        ob = self.orderbook_buffer[symbol][-1]
        if not ob.bids or not ob.asks:
            return 0.0
        
        try:
            # Get bid and ask data
            bids = list(ob.bids)
            asks = list(ob.asks)
            
            if not bids or not asks:
                return 0.0
            
            # Get best bid/ask prices using tuple unpacking
            try:
                first_bid = bids[0]
                first_ask = asks[0]
                
                if not isinstance(first_bid, (list, tuple)) or not isinstance(first_ask, (list, tuple)):
                    return 0.0
                    
                # Safely unpack price and quantity
                if len(first_bid) >= 2 and len(first_ask) >= 2:
                    bid_price, bid_size = first_bid[0], first_bid[1]
                    ask_price, ask_size = first_ask[0], first_ask[1]
                else:
                    return 0.0
                
                # Convert to float
                bid_price = float(bid_price)
                ask_price = float(ask_price)
                
                if bid_price <= 0.0 or ask_price <= 0.0:
                    return 0.0
                
                spread = (ask_price - bid_price) / bid_price
                
                # Calculate depth using tuple unpacking
                bid_depth = 0.0
                ask_depth = 0.0
                
                # Process first 5 bid levels
                for level in bids[:5]:
                    if isinstance(level, (list, tuple)) and len(level) >= 2:
                        try:
                            _, quantity = level[0], level[1]  # Unpack tuple
                            bid_depth += float(quantity)
                        except (TypeError, ValueError):
                            continue
                
                # Process first 5 ask levels
                for level in asks[:5]:
                    if isinstance(level, (list, tuple)) and len(level) >= 2:
                        try:
                            _, quantity = level[0], level[1]  # Unpack tuple
                            ask_depth += float(quantity)
                        except (TypeError, ValueError):
                            continue
                
                # Calculate liquidity score
                if spread > 0 and (bid_depth > 0 or ask_depth > 0):
                    liquidity_score = (bid_depth + ask_depth) / (1 + spread)
                    return float(liquidity_score)
                
                return 0.0
                
            except (IndexError, TypeError, ValueError):
                return 0.0
            
        except Exception:
            return 0.0

    async def _handle_prediction(self, symbol: str, prediction: Dict[str, Any]) -> None:
        """Handle a new prediction and generate signals."""
        try:
            current_price = float(prediction["price"])
            
            # Calculate ATR for dynamic position sizing and stop loss
            if symbol not in self.atr_values or not self.atr_values[symbol]:
                return
            
            atr = self.atr_values[symbol]
            
            # Calculate stop loss and take profit distances based on ATR
            atr_multiple = 2.0
            stop_distance = float(atr[-1]) * atr_multiple
            take_profit_distance = float(atr[-1]) * atr_multiple * 1.5  # 1.5x the stop distance
            
            # Calculate position size based on risk
            risk_per_trade = 0.02  # 2% risk per trade
            account_value = 10000  # Example account value
            risk_amount = account_value * risk_per_trade
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Adjust position size based on confidence and correlation
            confidence = float(prediction["confidence"])
            position_size *= confidence
            
            # Get correlation metrics
            correlation_metrics = self.correlation_tracker.get_correlations(symbol, window=24)
            correlation_score = 1.0  # Default neutral value
            
            if isinstance(correlation_metrics, dict):
                try:
                    # Safely access dictionary and convert value
                    corr_val = correlation_metrics.get("price_correlation", 1.0)
                    if isinstance(corr_val, (int, float)):
                        correlation_score = float(corr_val)
                except (TypeError, ValueError):
                    pass
            
            # Adjust position size based on correlation
            position_size *= (1 + correlation_score) / 2  # Scale between 0.5x and 1.5x
            
            # Calculate take profit and stop loss levels
            if prediction["direction"] == "long":
                stop_loss = current_price - stop_distance
                take_profit = current_price + take_profit_distance
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - take_profit_distance
            
            # Prepare metadata
            metadata: Dict[str, Any] = {
                "atr": float(atr[-1]),
                "position_size": float(position_size)
            }
            
            # Add feature importance if available
            if "feature_importance" in prediction:
                feature_importance = prediction["feature_importance"]
                if isinstance(feature_importance, dict):
                    feature_dict = {}
                    for k, v in feature_importance.items():
                        try:
                            feature_dict[str(k)] = float(v) if isinstance(v, (int, float)) else str(v)
                        except (TypeError, ValueError):
                            feature_dict[str(k)] = str(v)
                    metadata["model_features"] = feature_dict
            
            # Generate signal
            await self.publish_signal(
                symbol=symbol,
                signal_type=SignalType.ENTRY,
                direction=prediction["direction"],
                timeframe=self.timeframe,
                price=current_price,
                confidence=confidence,
                reason=f"ML prediction with {confidence:.2f} confidence",
                take_profit=take_profit,
                stop_loss=stop_loss,
                metadata=metadata
            )
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Error handling prediction for {symbol}: {str(e)}")
    
    async def _generate_signal(self, symbol: str, prediction: Dict[str, Any]) -> None:
        """Generate a trading signal from a prediction.
        
        Args:
            symbol: The trading pair symbol
            prediction: The prediction details
        """
        # Determine signal direction
        if prediction["direction"] > 0:
            direction = "long"
        else:
            direction = "short"
        
        # Calculate take profit and stop loss
        current_price = float(prediction["price"])
        take_profit = current_price * (1 + float(prediction["target_return"])) if direction == "long" else \
                     current_price * (1 - float(prediction["target_return"]))
        stop_loss = current_price * (1 - float(prediction["max_loss"])) if direction == "long" else \
                   current_price * (1 + float(prediction["max_loss"]))
        
        # Generate signal reason
        reason = f"ML prediction: {prediction['reason']} with confidence {float(prediction['confidence']):.2f}"
        
        # Publish the signal
        await self.publish_signal(
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            timeframe=self.timeframe,
            price=current_price,
            confidence=float(prediction["confidence"]),
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata={
                "model_version": self.models[symbol].get("version", "unknown"),
                "features_used": list(prediction["feature_importance"].keys()),
                "prediction_details": prediction
            }
        )

    async def _create_feature_pipeline(self) -> None:
        """Create the feature pipeline for the strategy."""
        # Initialize feature pipeline components
        self.feature_pipeline = {
            "technical": {
                "rsi": self._calculate_rsi,
                "bb_position": self._calculate_bb_position,
                "macd": self._calculate_macd_histogram
            },
            "market": {
                "regime": self._detect_market_regime,
                "liquidity": self._calculate_liquidity_score
            },
            "sentiment": {
                "social": self.sentiment_analyzer.get_social_sentiment,
                "news": self.sentiment_analyzer.get_news_sentiment,
                "order_flow": self.sentiment_analyzer.get_order_flow_sentiment
            }
        }

    async def _generate_prediction(self, symbol: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction from features.
        
        Args:
            symbol: Trading symbol
            features: Feature vector
            
        Returns:
            Dictionary containing prediction details
        """
        try:
            # Scale features
            features_scaled = self.scalers[symbol].transform(features.reshape(1, -1))
            
            # Get predictions from both models
            rf_pred = self.rf_models[symbol].predict_proba(features_scaled)
            gb_pred = self.gb_models[symbol].predict_proba(features_scaled)
            
            # Ensemble predictions
            pred_proba = (rf_pred + gb_pred) / 2
            direction = 1 if float(pred_proba[0][1]) > float(pred_proba[0][0]) else -1
            confidence = float(max(pred_proba[0]))
            
            # Get current price
            current_price = float(self.candle_buffer[symbol][-1].close)
            
            return {
                "direction": direction,
                "confidence": confidence,
                "price": current_price,
                "target_return": 0.02 * confidence,  # Scale target with confidence
                "max_loss": 0.01,  # Fixed stop loss
                "reason": "ML prediction based on technical and sentiment analysis",
                "feature_importance": self._get_feature_importance(symbol, features)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            return {
                "direction": 0,
                "confidence": 0.0,
                "price": 0.0,
                "target_return": 0.0,
                "max_loss": 0.0,
                "reason": f"Error: {str(e)}",
                "feature_importance": {}
            }

    async def _train_model(self, symbol: str, training_data: pd.DataFrame) -> None:
        """Train the ML models for a symbol.
        
        Args:
            symbol: Trading symbol
            training_data: DataFrame with historical data for training
        """
        try:
            # Prepare features and labels
            X = training_data[self.feature_names].values
            y = self._prepare_labels(training_data)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            rf_metrics = []
            gb_metrics = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                X_train_scaled = self.scalers[symbol].fit_transform(X_train)
                X_test_scaled = self.scalers[symbol].transform(X_test)
                
                # Train Random Forest
                self.rf_models[symbol].fit(X_train_scaled, y_train)
                rf_pred = self.rf_models[symbol].predict(X_test_scaled)
                rf_metrics.append(self._calculate_metrics(y_test, rf_pred))
                
                # Train Gradient Boosting
                self.gb_models[symbol].fit(X_train_scaled, y_train)
                gb_pred = self.gb_models[symbol].predict(X_test_scaled)
                gb_metrics.append(self._calculate_metrics(y_test, gb_pred))
            
            # Update model metrics
            self._update_model_metrics(symbol, "random_forest", rf_metrics)
            self._update_model_metrics(symbol, "gradient_boost", gb_metrics)
            
            self.logger.info(f"Models trained successfully for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {str(e)}")
            raise

    async def _validate_model(self, symbol: str, validation_data: pd.DataFrame) -> Dict[str, float]:
        """Validate the ML models for a symbol.
        
        Args:
            symbol: Trading symbol
            validation_data: DataFrame with validation data
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Prepare features and labels
            X = validation_data[self.feature_names].values
            y = self._prepare_labels(validation_data)
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(X)
            
            # Get predictions from both models
            rf_pred = self.rf_models[symbol].predict(X_scaled)
            gb_pred = self.gb_models[symbol].predict(X_scaled)
            
            # Calculate metrics for both models
            rf_metrics = self._calculate_metrics(y, rf_pred)
            gb_metrics = self._calculate_metrics(y, gb_pred)
            
            # Average metrics
            metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                metrics[metric] = (rf_metrics[metric] + gb_metrics[metric]) / 2
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error validating models for {symbol}: {str(e)}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }

    async def _generate_predictions(self, symbol: str) -> Optional[PredictionMetrics]:
        """Generate predictions for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing prediction metrics
        """
        try:
            # Prepare feature vector
            features = self._prepare_features(symbol)
            if features is None:
                return None
            
            # Generate prediction using the model
            if symbol not in self.models:
                self.logger.warning(f"No model available for {symbol}")
                return None
            
            model = self.models[symbol]
            prediction = float(model.predict(features.reshape(1, -1))[0])
            probas = model.predict_proba(features.reshape(1, -1))[0]
            confidence = float(max(probas))
            
            # Calculate direction (-1 for down, 0 for neutral, 1 for up)
            direction = int(np.sign(prediction)) if abs(prediction) > self.min_price_move else 0
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": direction,
                "timestamp": float(time.time())
            }
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return None

    def _prepare_features(self, symbol: str) -> Optional[NDArray[np.float64]]:
        """Prepare feature vector for prediction.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Get technical indicators
            tech_features = self._calculate_technical_indicators(symbol)
            if tech_features is None:
                return None
            
            # Convert technical features to numpy array
            tech_array = np.array([
                float(tech_features["rsi"]),
                float(tech_features["bb_position"]),
                float(tech_features["macd_hist"])
            ], dtype=np.float64)
            
            # Get sentiment features (already a numpy array)
            sentiment_features = self._calculate_sentiment_features(symbol)
            
            # Get liquidity features
            liquidity_score = self._calculate_liquidity_score(symbol)
            liquidity_array = np.array([liquidity_score], dtype=np.float64)
            
            # All arrays are now properly typed NDArray[np.float64]
            return np.concatenate([
                tech_array,
                sentiment_features,
                liquidity_array
            ])
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None 