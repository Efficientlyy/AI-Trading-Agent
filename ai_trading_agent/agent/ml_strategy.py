"""
Machine Learning Strategy Module

This module implements trading strategies based on machine learning models
that predict price movements using technical indicators and other features.

Enhanced with reinforcement learning for meta-parameter optimization and
automated feature engineering for dynamic feature selection and creation.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .strategy import BaseStrategy, RichSignal, RichSignalsDict
from .market_regime import MarketRegimeClassifier
from ..common import logger
from ..indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
from ..ml.reinforcement_learning import create_trading_rl_agent, TradingRLAgent
from ..ml.feature_engineering import create_feature_engineer, FeatureEngineer
from ..risk.risk_manager import RiskManager
from ..coordination.strategy_coordinator import StrategyCoordinator
from ..coordination.performance_attribution import PerformanceAttributor


class MLStrategy(BaseStrategy):
    """
    Trading strategy based on machine learning predictions with reinforcement learning optimization.
    
    This enhanced strategy:
    1. Uses automated feature engineering to dynamically select and create features
    2. Employs ML models to predict price direction and generate base signals
    3. Optimizes strategy parameters using reinforcement learning
    4. Adapts to different market regimes for improved performance
    5. Self-tunes its features and parameters based on performance feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced MLStrategy.
        
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
                - enable_rl: Whether to enable reinforcement learning optimization
                - enable_feature_engineering: Whether to enable automated feature engineering
                - rl_config: Configuration for the reinforcement learning agent
                - feature_engineering_config: Configuration for the feature engineering system
                - market_regime_config: Configuration for market regime detection
        """
        super().__init__(config)
        self.name = config.get("name", "EnhancedMLStrategy")
        
        # Core ML strategy parameters
        self.model_type = config.get("model_type", "random_forest")
        self.prediction_horizon = config.get("prediction_horizon", 1)
        self.training_lookback = config.get("training_lookback", 500)
        self.retrain_frequency = config.get("retrain_frequency", 20)
        self.confidence_threshold = config.get("confidence_threshold", 0.65)
        self.feature_list = config.get("features", ["rsi", "macd", "bb", "price_momentum", "volume_momentum"])
        self.model_path = config.get("model_path", None)
        
        # Enhanced capabilities flags
        self.enable_rl = config.get("enable_rl", True)
        self.enable_feature_engineering = config.get("enable_feature_engineering", True)
        self.enable_regime_adaptation = config.get("enable_regime_adaptation", True)
        
        # Initialize models and tracking
        self.models = {}
        self.last_train_time = {}
        self.performance_history = {}
        self.current_params = {}
        self.feature_sets = {}
        self.current_regimes = {}
        
        # Initialize risk manager for risk-adjusted rewards
        self.risk_manager = RiskManager(config.get("risk_config", {}))
        
        # Initialize market regime classifier if enabled
        if self.enable_regime_adaptation:
            regime_config = config.get("market_regime_config", {})
            self.regime_classifier = MarketRegimeClassifier(regime_config)
        else:
            self.regime_classifier = None
        
        # Initialize feature engineering system if enabled
        if self.enable_feature_engineering:
            feature_eng_config = config.get("feature_engineering_config", {})
            self.feature_engineer = create_feature_engineer(
                feature_eng_config, 
                self.regime_classifier
            )
        else:
            self.feature_engineer = None
            
        # Initialize reinforcement learning agent if enabled
        if self.enable_rl:
            rl_config = config.get("rl_config", {})
            self.rl_agent = create_trading_rl_agent(
                rl_config,
                self.risk_manager
            )
        else:
            self.rl_agent = None
            
        # Initialize cross-strategy coordination if enabled
        self.enable_coordination = config.get("enable_coordination", True)
        if self.enable_coordination:
            coord_config = config.get("coordination_config", {})
            self.strategy_coordinator = StrategyCoordinator(coord_config)
        else:
            self.strategy_coordinator = None
            
        # Initialize performance attribution if enabled
        self.enable_attribution = config.get("enable_attribution", True)
        if self.enable_attribution:
            attr_config = config.get("attribution_config", {})
            self.performance_attributor = PerformanceAttributor(attr_config)
        else:
            self.performance_attributor = None
            
        # Strategy parameter ranges for RL optimization
        self.param_ranges = {
            "confidence_threshold": (0.55, 0.95),
            "prediction_horizon": (1, 10),
            "training_window_scale": (0.5, 2.0),  # Scaling factor for training_lookback
            "position_size_factor": (0.1, 1.0)
        }
        
        logger.info(f"{self.name} initialized with RL: {self.enable_rl}, "
                   f"Feature Engineering: {self.enable_feature_engineering}, "
                   f"Regime Adaptation: {self.enable_regime_adaptation}")
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
        Generate trading signals based on ML model predictions with reinforcement learning optimization.
        
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
            
            try:
                # 1. Detect market regime if enabled
                if self.enable_regime_adaptation and self.regime_classifier is not None:
                    regime = self.detect_market_regime(symbol, df)
                    self.current_regimes[symbol] = regime
                    logger.info(f"{self.name}: Current regime for {symbol}: {regime}")
                
                # Initialize period counter for this symbol if needed
                if symbol not in self.period_counter:
                    self.period_counter[symbol] = 0
                else:
                    self.period_counter[symbol] += 1
                
                # 2. Apply automated feature engineering if enabled
                if self.enable_feature_engineering and self.feature_engineer is not None:
                    features = self.engineer_features(symbol, df)
                else:
                    # Use traditional feature extraction
                    features = self._extract_features(df)
                
                if features is None or features.empty:
                    logger.warning(f"{self.name}: Could not extract features for {symbol}")
                    continue
                
                # 3. Apply reinforcement learning for parameter optimization if enabled
                if self.enable_rl and self.rl_agent is not None and symbol in self.performance_history:
                    # Get recent performance data
                    recent_performance = self.performance_history[symbol][-1] if self.performance_history[symbol] else {}
                    
                    # Optimize parameters using RL
                    optimized_params = self.optimize_parameters(symbol, df, recent_performance)
                    
                    # Apply the optimized parameters
                    self.apply_optimized_parameters(symbol)
                
                # 4. Check if we need to train/retrain the model
                if (symbol not in self.models or 
                    self.period_counter[symbol] >= self.retrain_frequency):
                    # Train with potentially new features from feature engineering
                    self._train_model(symbol, df)
                    self.period_counter[symbol] = 0
                
                # 5. Make prediction with optimized parameters
                signal_data = self._predict_and_generate_signal(symbol, features, timestamp)
                
                if signal_data:
                    # Apply position sizing from RL if available
                    if (self.enable_rl and self.rl_agent is not None and
                        symbol in self.current_params and "position_size_factor" in self.current_params[symbol]):
                        position_size_factor = self.current_params[symbol]["position_size_factor"]
                        
                        # Apply position sizing to signal while preserving direction
                        for ts, signal in signal_data.items():
                            if signal.action != 0:  # If not a HOLD signal
                                original_quantity = signal.quantity
                                signal.quantity = original_quantity * position_size_factor
                                if signal.metadata is None:
                                    signal.metadata = {}
                                signal.metadata["position_size_factor"] = position_size_factor
                                logger.info(f"{self.name}: Applied position sizing factor {position_size_factor} "
                                           f"to {symbol} signal: {original_quantity} -> {signal.quantity}")
                    
                    signals[symbol] = signal_data
                    
                    # Calculate returns and drawdown for RL feedback if we have sufficient future data
                    current_price_idx = df.index.get_indexer([timestamp], method='pad')[0]
                    if current_price_idx >= 0 and current_price_idx + self.prediction_horizon < len(df):
                        current_price = df['close'].iloc[current_price_idx]
                        future_price = df['close'].iloc[current_price_idx + self.prediction_horizon]
                        
                        # Calculate returns aligned with signal direction
                        raw_return = (future_price - current_price) / current_price
                        
                        # Get the main signal (using the first one in the dictionary)
                        main_signal = next(iter(signal_data.values()))
                        
                        # Align return with signal direction (positive for correct prediction)
                        if main_signal.action < 0:  # Short position
                            period_return = -raw_return  # Invert for short positions
                        else:
                            period_return = raw_return
                        
                        # Calculate drawdown
                        price_window = df['close'].iloc[current_price_idx:current_price_idx + self.prediction_horizon + 1]
                        if len(price_window) > 1:
                            if main_signal.action > 0:  # For long positions
                                peak = price_window.max()
                                trough = price_window.min()
                                drawdown = (peak - trough) / peak if peak > 0 else 0
                            else:  # For short positions (drawdown is when price goes up)
                                trough = price_window.min()
                                peak = price_window.max()
                                drawdown = (peak - trough) / trough if trough > 0 else 0
                            
                            # Update performance history for RL feedback
                            self.update_performance_history(symbol, signal_data, period_return, drawdown)
            
            except Exception as e:
                logger.error(f"{self.name}: Error generating signals for {symbol}: {e}")
                continue
        
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
            
    def detect_market_regime(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Detect the current market regime for a symbol.
        
        Args:
            symbol: The trading symbol
            data: Historical price data
            
        Returns:
            String identifying the market regime
        """
        if not self.enable_regime_adaptation or self.regime_classifier is None:
            return "default"
            
        try:
            regime = self.regime_classifier.classify_regime(data)
            self.current_regimes[symbol] = regime
            logger.info(f"{self.name}: Detected {regime} regime for {symbol}")
            return regime
        except Exception as e:
            logger.error(f"{self.name}: Error detecting market regime for {symbol}: {e}")
            return "default"
    
    def engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply automated feature engineering to improve prediction quality.
        
        Args:
            symbol: The trading symbol
            data: Historical price data
            
        Returns:
            DataFrame with engineered features
        """
        if not self.enable_feature_engineering or self.feature_engineer is None:
            # Use basic feature extraction if automated feature engineering is disabled
            return self._extract_features(data)
            
        try:
            # Detect current market regime
            regime = self.current_regimes.get(symbol, self.detect_market_regime(symbol, data))
            
            # Get optimal feature set for this regime
            enhanced_data = self.feature_engineer.get_optimal_feature_set(data, regime)
            
            # Save the feature set for this symbol
            self.feature_sets[symbol] = enhanced_data.columns.tolist()
            
            logger.info(f"{self.name}: Generated {len(enhanced_data.columns)} features for {symbol} in {regime} regime")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"{self.name}: Error in feature engineering for {symbol}: {e}")
            # Fall back to basic feature extraction
            return self._extract_features(data)
    
    def optimize_parameters(self, symbol: str, data: pd.DataFrame, performance: Dict[str, float]) -> Dict[str, float]:
        """
        Use reinforcement learning to optimize strategy parameters.
        
        Args:
            symbol: The trading symbol
            data: Historical price data
            performance: Dictionary of performance metrics
            
        Returns:
            Dictionary of optimized parameters
        """
        if not self.enable_rl or self.rl_agent is None:
            # Return current parameters if RL is disabled
            return self.current_params.get(symbol, {})
            
        try:
            # Prepare current state for RL agent
            regime = self.current_regimes.get(symbol, self.detect_market_regime(symbol, data))
            
            # Normalize strategy parameters to [0,1] range for RL
            normalized_params = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                if param == "confidence_threshold":
                    current_val = self.confidence_threshold
                elif param == "prediction_horizon":
                    current_val = self.prediction_horizon
                elif param == "training_window_scale":
                    current_val = 1.0  # Default scale factor
                elif param == "position_size_factor":
                    # Get from performance data or use default
                    current_val = performance.get("position_size_factor", 0.5)
                else:
                    continue
                    
                # Normalize to [0,1]
                param_range = max_val - min_val
                normalized_params[param] = (current_val - min_val) / param_range
            
            # Create state dict for RL agent
            current_state = {
                "strategy_params": normalized_params,
                "performance": performance,
                "market_regime": regime
            }
            
            # Get optimized parameters from RL agent
            updated_normalized_params = self.rl_agent.adapt_strategy(
                current_state,
                data
            )
            
            # Convert back to actual parameter values
            updated_params = {}
            for param, norm_val in updated_normalized_params.items():
                min_val, max_val = self.param_ranges[param]
                param_range = max_val - min_val
                actual_val = min_val + (norm_val * param_range)
                updated_params[param] = actual_val
            
            # Save the updated parameters
            self.current_params[symbol] = updated_params
            
            # Log the parameter updates
            logger.info(f"{self.name}: RL optimized parameters for {symbol} in {regime} regime: {updated_params}")
            
            return updated_params
            
        except Exception as e:
            logger.error(f"{self.name}: Error in RL parameter optimization for {symbol}: {e}")
            # Return current parameters as fallback
            return self.current_params.get(symbol, {})
    
    def apply_optimized_parameters(self, symbol: str) -> None:
        """
        Apply optimized parameters from RL to the strategy.
        
        Args:
            symbol: The trading symbol
        """
        if symbol not in self.current_params:
            return
            
        params = self.current_params[symbol]
        
        # Apply parameters selectively
        if "confidence_threshold" in params:
            self.confidence_threshold = params["confidence_threshold"]
            
        if "prediction_horizon" in params:
            # Ensure prediction_horizon is an integer
            self.prediction_horizon = int(round(params["prediction_horizon"]))
            
        if "training_window_scale" in params:
            # Apply scaling to training_lookback
            scale = params["training_window_scale"]
            self.training_lookback = int(self.config.get("training_lookback", 500) * scale)
            
        logger.info(f"{self.name}: Applied optimized parameters for {symbol}: "
                   f"confidence_threshold={self.confidence_threshold}, "
                   f"prediction_horizon={self.prediction_horizon}, "
                   f"training_lookback={self.training_lookback}")
    
    def update_performance_history(self, symbol: str, signals: RichSignalsDict, 
                                 returns: float, drawdown: float) -> None:
        """
        Update performance history for reinforcement learning feedback.
        
        Args:
            symbol: The trading symbol
            signals: Generated trading signals
            returns: Period returns
            drawdown: Maximum drawdown
        """
        # Calculate performance metrics
        signal_count = len([s for s in signals.values() if s.action != 0])
        
        # Calculate volatility from returns
        if symbol in self.performance_history and len(self.performance_history[symbol]) > 0:
            prev_returns = [p["returns"] for p in self.performance_history[symbol][-20:]]
            volatility = np.std(prev_returns + [returns])
        else:
            volatility = 0.01  # Default value
        
        # Create performance record
        performance = {
            "timestamp": datetime.now().isoformat(),
            "returns": returns,
            "volatility": volatility,
            "drawdown": drawdown,
            "sharpe_ratio": returns / max(volatility, 1e-8),
            "trade_count": signal_count,
            "market_regime": self.current_regimes.get(symbol, "default")
        }
        
        # Initialize performance history for this symbol if needed
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []
            
        # Add to history
        self.performance_history[symbol].append(performance)
        
        # Trim history to avoid memory bloat
        max_history = 100
        if len(self.performance_history[symbol]) > max_history:
            self.performance_history[symbol] = self.performance_history[symbol][-max_history:]
            
        # Update feature engineering with performance feedback if enabled
        if self.enable_feature_engineering and self.feature_engineer is not None:
            try:
                regime = self.current_regimes.get(symbol, "default")
                self.feature_engineer.adapt_features_to_regime(
                    regime, 
                    performance["sharpe_ratio"]
                )
            except Exception as e:
                logger.error(f"{self.name}: Error updating feature engineering: {e}")
    
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
            
        # Update enhanced capabilities flags
        if "enable_rl" in config_updates:
            self.enable_rl = config_updates["enable_rl"]
            
        if "enable_feature_engineering" in config_updates:
            self.enable_feature_engineering = config_updates["enable_feature_engineering"]
            
        if "enable_regime_adaptation" in config_updates:
            self.enable_regime_adaptation = config_updates["enable_regime_adaptation"]
            
        logger.info(f"{self.name} configuration updated")
