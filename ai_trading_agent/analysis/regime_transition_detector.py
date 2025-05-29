"""
Market Regime Transition Detector Module

This module implements advanced detection of market regime transitions and
provides early warning signals when the market is likely to shift from one
regime to another.

Key capabilities:
- Detect early signs of regime transitions
- Calculate transition probabilities
- Monitor leading indicators for regime shifts
- Provide confidence scores for transition predictions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
from scipy import signal

from ai_trading_agent.agent.market_regime import MarketRegimeType

# Set up logger
logger = logging.getLogger(__name__)


class TransitionDirection(Enum):
    """Direction of market regime transition."""
    IMPROVING = "improving"  # Moving toward more favorable regimes (e.g., calm, trending up)
    DETERIORATING = "deteriorating"  # Moving toward less favorable regimes (e.g., volatile, trending down)
    NEUTRAL = "neutral"  # Lateral transition with no clear positive/negative implication


class TransitionSignal:
    """Represents a detected signal of a potential regime transition."""
    
    def __init__(
        self,
        from_regime: MarketRegimeType,
        to_regime: MarketRegimeType,
        probability: float,
        indicators: Dict[str, float],
        direction: TransitionDirection,
        estimated_timeframe: str
    ):
        """
        Initialize a transition signal.
        
        Args:
            from_regime: Current market regime
            to_regime: Target market regime (predicted)
            probability: Transition probability (0.0 to 1.0)
            indicators: Dictionary of indicator values supporting the prediction
            direction: Transition direction (improving, deteriorating, neutral)
            estimated_timeframe: Estimated timeframe for transition (e.g., "1-3 days")
        """
        self.from_regime = from_regime
        self.to_regime = to_regime
        self.probability = probability
        self.indicators = indicators
        self.direction = direction
        self.estimated_timeframe = estimated_timeframe
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "probability": self.probability,
            "indicators": self.indicators,
            "direction": self.direction.value,
            "estimated_timeframe": self.estimated_timeframe,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a TransitionSignal from a dictionary."""
        # Convert string regime values to enum
        from_regime = MarketRegimeType.UNKNOWN
        to_regime = MarketRegimeType.UNKNOWN
        
        for rt in MarketRegimeType:
            if rt.value == data.get("from_regime"):
                from_regime = rt
            if rt.value == data.get("to_regime"):
                to_regime = rt
        
        # Convert string direction to enum
        direction = TransitionDirection.NEUTRAL
        for td in TransitionDirection:
            if td.value == data.get("direction"):
                direction = td
        
        return cls(
            from_regime=from_regime,
            to_regime=to_regime,
            probability=data.get("probability", 0.0),
            indicators=data.get("indicators", {}),
            direction=direction,
            estimated_timeframe=data.get("estimated_timeframe", "unknown")
        )


class RegimeTransitionDetector:
    """
    Detects market regime transitions and provides early warning signals.
    
    This detector uses multiple approaches to identify potential regime transitions:
    1. Leading indicator analysis
    2. Divergence detection
    3. Statistical changepoint detection
    4. Entropy and complexity measures
    5. Hidden patterns in market structure
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the regime transition detector.
        
        Args:
            config: Configuration dictionary with parameters
                - lookback_window: Window for analyzing transition patterns
                - min_probability: Minimum probability to report a transition
                - early_warning_window: How many periods ahead to predict
                - indicator_weights: Weights for different transition indicators
                - models_dir: Directory for transition model storage
        """
        # Default configuration
        default_config = {
            "lookback_window": 60,
            "min_probability": 0.6,
            "early_warning_window": 5,
            "indicator_weights": {
                "volatility_change": 0.2,
                "correlation_breakdown": 0.15,
                "volume_pattern": 0.15,
                "breadth_change": 0.1,
                "momentum_divergence": 0.2,
                "statistical_measure": 0.1,
                "sentiment_shift": 0.1
            },
            "models_dir": "./models/transitions"
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize models directory
        os.makedirs(self.config["models_dir"], exist_ok=True)
        
        # Transition matrix (probabilities between regimes)
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Transition history
        self.transition_history = []
        
        # Active signals
        self.active_signals = []
        
        # Pattern recognition components
        self.kmeans = KMeans(n_clusters=5, random_state=42)  # For pattern clustering
        self.pca = PCA(n_components=2)  # For dimension reduction
        
        logger.info(f"Regime Transition Detector initialized with {self.config['early_warning_window']} period early warning")
    
    def _initialize_transition_matrix(self) -> pd.DataFrame:
        """
        Initialize transition probability matrix between regimes.
        
        Returns:
            DataFrame with transition probabilities
        """
        regimes = [r.value for r in MarketRegimeType]
        
        # Start with uniform transition probabilities
        matrix = pd.DataFrame(
            data=np.ones((len(regimes), len(regimes))) / len(regimes),
            index=regimes,
            columns=regimes
        )
        
        # Some transitions are more likely than others
        # Example: Adjust based on typical market behavior
        
        # Trending up can go to ranging (correction) or volatile (euphoria)
        matrix.loc[MarketRegimeType.TRENDING_UP.value, MarketRegimeType.RANGING.value] = 0.3
        matrix.loc[MarketRegimeType.TRENDING_UP.value, MarketRegimeType.VOLATILE.value] = 0.2
        
        # Trending down can resolve to ranging or continue to volatile
        matrix.loc[MarketRegimeType.TRENDING_DOWN.value, MarketRegimeType.RANGING.value] = 0.25
        matrix.loc[MarketRegimeType.TRENDING_DOWN.value, MarketRegimeType.VOLATILE.value] = 0.25
        
        # Volatile often leads to trending down or reversal
        matrix.loc[MarketRegimeType.VOLATILE.value, MarketRegimeType.TRENDING_DOWN.value] = 0.3
        matrix.loc[MarketRegimeType.VOLATILE.value, MarketRegimeType.REVERSAL.value] = 0.25
        
        # Normalize to ensure rows sum to 1
        for regime in regimes:
            matrix.loc[regime] = matrix.loc[regime] / matrix.loc[regime].sum()
        
        return matrix
    
    def detect_transition_signals(
        self, 
        data: pd.DataFrame, 
        current_regime: MarketRegimeType
    ) -> List[TransitionSignal]:
        """
        Detect potential regime transition signals in the market data.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            current_regime: Current market regime classification
            
        Returns:
            List of TransitionSignal objects
        """
        if data is None or len(data) < self.config["lookback_window"]:
            logger.warning(f"Insufficient data for transition detection: {len(data) if data is not None else 0} rows")
            return []
        
        try:
            signals = []
            
            # Extract features relevant to transitions
            transition_features = self._extract_transition_features(data)
            
            # Check for specific transition patterns from the current regime
            for target_regime in MarketRegimeType:
                if target_regime == current_regime or target_regime == MarketRegimeType.UNKNOWN:
                    continue
                
                # Calculate probability of transition to this regime
                probability = self._calculate_transition_probability(
                    transition_features, current_regime, target_regime)
                
                # Only keep signals above the minimum probability threshold
                if probability >= self.config["min_probability"]:
                    # Determine transition direction
                    direction = self._determine_transition_direction(current_regime, target_regime)
                    
                    # Estimate timeframe for transition
                    timeframe = self._estimate_transition_timeframe(
                        transition_features, current_regime, target_regime)
                    
                    # Create signal with supporting indicators
                    signal = TransitionSignal(
                        from_regime=current_regime,
                        to_regime=target_regime,
                        probability=probability,
                        indicators=transition_features,
                        direction=direction,
                        estimated_timeframe=timeframe
                    )
                    
                    signals.append(signal)
                    
                    # Update transition history
                    self.transition_history.append(signal.to_dict())
            
            # Keep history from growing too large (keep last 1000 signals)
            if len(self.transition_history) > 1000:
                self.transition_history = self.transition_history[-1000:]
            
            # Update active signals list
            self.active_signals = signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting transition signals: {e}")
            return []
    
    def _extract_transition_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features that are useful for predicting regime transitions.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Dictionary of transition features
        """
        features = {}
        
        try:
            # Recent price/return behavior
            returns = data['close'].pct_change().dropna()
            recent_returns = returns.iloc[-self.config["lookback_window"]:]
            
            # Volatility changes - increasing volatility often precedes regime shifts
            vol_short = returns.iloc[-10:].std() * np.sqrt(252)
            vol_long = returns.iloc[-30:].std() * np.sqrt(252)
            vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0
            features["volatility_change"] = vol_ratio - 1.0  # Percent change in volatility
            
            # Volume patterns - regime shifts often accompanied by volume changes
            if 'volume' in data.columns:
                vol_change = data['volume'].iloc[-5:].mean() / data['volume'].iloc[-20:-5].mean()
                features["volume_pattern"] = vol_change - 1.0
            else:
                features["volume_pattern"] = 0.0
            
            # Autocorrelation changes - regime shifts often change market efficiency
            acf_recent = acf(recent_returns, nlags=5)
            features["autocorrelation_change"] = abs(acf_recent[1])
            
            # Momentum changes and divergences
            if len(data) >= 30:
                rsi_values = self._calculate_rsi(data['close'])
                if len(rsi_values) > 0:
                    price_momentum = data['close'].iloc[-1] - data['close'].iloc[-20]
                    rsi_momentum = rsi_values[-1] - rsi_values[-20] if len(rsi_values) >= 20 else 0
                    
                    # Divergence: Price up but RSI down, or vice versa
                    if (price_momentum > 0 and rsi_momentum < 0) or (price_momentum < 0 and rsi_momentum > 0):
                        features["momentum_divergence"] = abs(price_momentum) * abs(rsi_momentum)
                    else:
                        features["momentum_divergence"] = 0.0
                else:
                    features["momentum_divergence"] = 0.0
            else:
                features["momentum_divergence"] = 0.0
            
            # Entropy measure - increasing entropy can signal regime shifts
            if len(returns) > 20:
                # Bin the returns and calculate shannon entropy
                hist, _ = np.histogram(returns.iloc[-20:], bins=10, density=True)
                hist = hist[hist > 0]  # Remove zeros before log
                entropy_val = entropy(hist)
                features["return_entropy"] = entropy_val
            else:
                features["return_entropy"] = 0.0
            
            # Change in skewness and kurtosis - regime shifts often change return distribution
            if len(returns) > 20:
                recent_skew = returns.iloc[-10:].skew()
                older_skew = returns.iloc[-20:-10].skew()
                features["skew_change"] = abs(recent_skew - older_skew)
                
                recent_kurt = returns.iloc[-10:].kurt()
                older_kurt = returns.iloc[-20:-10].kurt()
                features["kurt_change"] = abs(recent_kurt - older_kurt)
            else:
                features["skew_change"] = 0.0
                features["kurt_change"] = 0.0
            
            # Detect outlier behavior in indicators
            if 'ma20' in data.columns and 'ma50' in data.columns:
                ma_spread = (data['ma20'] - data['ma50']) / data['ma50']
                recent_spread = ma_spread.iloc[-5:].mean()
                historical_spread = ma_spread.iloc[-30:-5].mean()
                features["ma_spread_change"] = abs(recent_spread - historical_spread)
            else:
                features["ma_spread_change"] = 0.0
            
            # Break in trend structure
            if len(data) > 50:
                # Simple trend deviation measure
                ma50 = data['close'].rolling(window=50).mean()
                if not ma50.isnull().all():
                    recent_dev = abs(data['close'].iloc[-10:] - ma50.iloc[-10:]).mean()
                    historical_dev = abs(data['close'].iloc[-30:-10] - ma50.iloc[-30:-10]).mean()
                    features["trend_structure_break"] = recent_dev / historical_dev if historical_dev > 0 else 1.0
                else:
                    features["trend_structure_break"] = 1.0
            else:
                features["trend_structure_break"] = 1.0
            
            # Detect pattern changes using signal processing
            if len(returns) >= 30:
                # Find peaks in returns
                peaks, _ = signal.find_peaks(returns.iloc[-30:])
                troughs, _ = signal.find_peaks(-returns.iloc[-30:])
                
                # Pattern density change
                recent_pattern_count = len(peaks[-10:]) + len(troughs[-10:]) if len(peaks) + len(troughs) > 0 else 0
                older_pattern_count = len(peaks[:-10]) + len(troughs[:-10]) if len(peaks) + len(troughs) > 0 else 0
                
                features["pattern_density_change"] = recent_pattern_count / max(1, older_pattern_count)
            else:
                features["pattern_density_change"] = 1.0
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting transition features: {e}")
            return {"error": str(e)}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index."""
        try:
            delta = prices.diff().dropna()
            
            # Make two series: one for gains and one for losses
            gain = delta.copy()
            loss = delta.copy()
            
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi.dropna()
        except:
            return pd.Series()
    
    def _calculate_transition_probability(
        self, 
        features: Dict[str, float], 
        current_regime: MarketRegimeType,
        target_regime: MarketRegimeType
    ) -> float:
        """
        Calculate probability of transitioning from current to target regime.
        
        Args:
            features: Dictionary of transition features
            current_regime: Current market regime
            target_regime: Potential target regime
            
        Returns:
            Transition probability (0.0 to 1.0)
        """
        # Start with base probability from transition matrix
        base_prob = self.transition_matrix.loc[current_regime.value, target_regime.value]
        
        # Adjust based on detected features
        prob_adjustment = 0.0
        weights = self.config["indicator_weights"]
        
        # Volatility changes strongly indicate regime transitions
        if "volatility_change" in features:
            vol_change = features["volatility_change"]
            # Higher volatility tends to lead to certain regimes
            if target_regime in [MarketRegimeType.VOLATILE, MarketRegimeType.TRENDING_DOWN]:
                # Increasing volatility makes volatile/down regimes more likely
                if vol_change > 0.1:
                    prob_adjustment += vol_change * weights.get("volatility_change", 0.2)
            elif target_regime in [MarketRegimeType.CALM, MarketRegimeType.RANGING]:
                # Decreasing volatility makes calm/ranging regimes more likely
                if vol_change < -0.1:
                    prob_adjustment += abs(vol_change) * weights.get("volatility_change", 0.2)
        
        # Volume patterns are useful for breakouts and trending regimes
        if "volume_pattern" in features:
            vol_pattern = features["volume_pattern"]
            if target_regime in [MarketRegimeType.BREAKOUT, MarketRegimeType.TRENDING_UP]:
                # Increasing volume supports breakouts and new trends
                if vol_pattern > 0.2:
                    prob_adjustment += vol_pattern * weights.get("volume_pattern", 0.15)
        
        # Momentum divergences often precede reversals
        if "momentum_divergence" in features:
            divergence = features["momentum_divergence"]
            if target_regime == MarketRegimeType.REVERSAL:
                prob_adjustment += min(divergence * weights.get("momentum_divergence", 0.2), 0.3)
        
        # Pattern density changes can indicate shifts between trending and ranging
        if "pattern_density_change" in features:
            density_change = features["pattern_density_change"]
            if target_regime == MarketRegimeType.RANGING and density_change > 1.3:
                prob_adjustment += 0.15
            elif target_regime in [MarketRegimeType.TRENDING_UP, MarketRegimeType.TRENDING_DOWN] and density_change < 0.7:
                prob_adjustment += 0.15
        
        # Entropy increases often precede volatile regimes
        if "return_entropy" in features:
            entropy_val = features["return_entropy"]
            if target_regime == MarketRegimeType.VOLATILE and entropy_val > 1.5:
                prob_adjustment += 0.1
        
        # Trend structure breaks often precede reversals or regime changes
        if "trend_structure_break" in features:
            structure_break = features["trend_structure_break"]
            if structure_break > 1.5:
                prob_adjustment += 0.1
        
        # Combine base probability with adjustments
        final_prob = min(max(base_prob + prob_adjustment, 0.0), 1.0)
        
        return final_prob
    
    def _determine_transition_direction(
        self,
        current_regime: MarketRegimeType,
        target_regime: MarketRegimeType
    ) -> TransitionDirection:
        """
        Determine if a regime transition is improving or deteriorating.
        
        Args:
            current_regime: Current market regime
            target_regime: Target market regime
            
        Returns:
            TransitionDirection enumeration
        """
        # Define regime favorability ranking (higher is better)
        regime_ranking = {
            MarketRegimeType.TRENDING_UP: 5,
            MarketRegimeType.CALM: 4,
            MarketRegimeType.RANGING: 3,
            MarketRegimeType.BREAKOUT: 3,  # Neutral but potentially profitable
            MarketRegimeType.REVERSAL: 2,  # Neutral but uncertain
            MarketRegimeType.TRENDING_DOWN: 1,
            MarketRegimeType.VOLATILE: 0,
            MarketRegimeType.UNKNOWN: -1
        }
        
        current_rank = regime_ranking.get(current_regime, -1)
        target_rank = regime_ranking.get(target_regime, -1)
        
        if target_rank > current_rank:
            return TransitionDirection.IMPROVING
        elif target_rank < current_rank:
            return TransitionDirection.DETERIORATING
        else:
            return TransitionDirection.NEUTRAL
    
    def _estimate_transition_timeframe(
        self,
        features: Dict[str, float],
        current_regime: MarketRegimeType,
        target_regime: MarketRegimeType
    ) -> str:
        """
        Estimate the timeframe for a potential regime transition.
        
        Args:
            features: Dictionary of transition features
            current_regime: Current market regime
            target_regime: Target market regime
            
        Returns:
            String description of timeframe (e.g., "1-3 days")
        """
        # Base timeframe estimates (conservative defaults)
        base_timeframes = {
            MarketRegimeType.TRENDING_UP: "5-10 days",
            MarketRegimeType.TRENDING_DOWN: "3-7 days",
            MarketRegimeType.VOLATILE: "1-3 days",
            MarketRegimeType.CALM: "7-14 days",
            MarketRegimeType.RANGING: "5-10 days",
            MarketRegimeType.BREAKOUT: "1-2 days",
            MarketRegimeType.REVERSAL: "2-5 days",
            MarketRegimeType.UNKNOWN: "unknown"
        }
        
        # Adjust based on current features
        
        # Volatility suggests faster transitions
        volatility_factor = features.get("volatility_change", 0)
        
        # Volume suggests momentum for faster change
        volume_factor = features.get("volume_pattern", 0)
        
        # Pattern density changes suggest structural shifts
        pattern_factor = abs(features.get("pattern_density_change", 1) - 1)
        
        # Combined speed factor (higher means faster transition)
        speed_factor = volatility_factor + volume_factor + pattern_factor
        
        # Translate to timeframe
        if speed_factor > 0.5:
            return "1-3 days"  # Very fast transition
        elif speed_factor > 0.3:
            return "2-5 days"  # Fast transition
        elif speed_factor > 0.1:
            return "5-10 days"  # Moderate pace
        else:
            return base_timeframes.get(target_regime, "7-14 days")  # Default or slow transition
    
    def update_transition_matrix(self, actual_transitions: List[Tuple[MarketRegimeType, MarketRegimeType]], learning_rate: float = 0.1) -> None:
        """
        Update the transition matrix based on observed regime transitions.
        
        Args:
            actual_transitions: List of (from_regime, to_regime) tuples
            learning_rate: Rate at which to update probabilities
        """
        if not actual_transitions:
            return
            
        # Count transitions
        transition_counts = {}
        for from_regime, to_regime in actual_transitions:
            key = (from_regime.value, to_regime.value)
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Calculate new probabilities
        for (from_value, to_value), count in transition_counts.items():
            # Get current probability
            current_prob = self.transition_matrix.loc[from_value, to_value]
            
            # Calculate proportion of transitions from this regime
            total_from_this_regime = sum(
                count for (f, _), count in transition_counts.items() if f == from_value)
            
            # New probability based on observed transitions
            new_prob = count / total_from_this_regime if total_from_this_regime > 0 else 0
            
            # Update using learning rate
            updated_prob = (1 - learning_rate) * current_prob + learning_rate * new_prob
            
            # Set new probability
            self.transition_matrix.loc[from_value, to_value] = updated_prob
        
        # Normalize to ensure rows sum to 1
        for regime in self.transition_matrix.index:
            row_sum = self.transition_matrix.loc[regime].sum()
            if row_sum > 0:
                self.transition_matrix.loc[regime] = self.transition_matrix.loc[regime] / row_sum
        
        logger.info(f"Updated transition matrix with {len(actual_transitions)} transitions")
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """
        Get currently active transition signals.
        
        Returns:
            List of active signals as dictionaries
        """
        return [signal.to_dict() for signal in self.active_signals]
    
    def save_state(self) -> bool:
        """
        Save the current state of the transition detector.
        
        Returns:
            Boolean indicating success
        """
        try:
            state_path = os.path.join(self.config["models_dir"], "transition_detector_state.pkl")
            
            state = {
                "transition_matrix": self.transition_matrix,
                "transition_history": self.transition_history[-100:],  # Save last 100 entries
                "active_signals": [s.to_dict() for s in self.active_signals],
                "config": self.config
            }
            
            joblib.dump(state, state_path)
            logger.info("Saved transition detector state")
            return True
            
        except Exception as e:
            logger.error(f"Error saving transition detector state: {e}")
            return False
    
    def load_state(self) -> bool:
        """
        Load a previously saved state.
        
        Returns:
            Boolean indicating success
        """
        try:
            state_path = os.path.join(self.config["models_dir"], "transition_detector_state.pkl")
            
            if not os.path.exists(state_path):
                logger.warning("No saved state found for transition detector")
                return False
                
            state = joblib.load(state_path)
            
            self.transition_matrix = state.get("transition_matrix", self._initialize_transition_matrix())
            self.transition_history = state.get("transition_history", [])
            
            # Convert dictionaries back to TransitionSignal objects
            self.active_signals = [
                TransitionSignal.from_dict(s) for s in state.get("active_signals", [])
            ]
            
            # Update config while preserving any new keys
            saved_config = state.get("config", {})
            self.config.update({k: v for k, v in saved_config.items() if k in self.config})
            
            logger.info("Loaded transition detector state")
            return True
            
        except Exception as e:
            logger.error(f"Error loading transition detector state: {e}")
            return False
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current transition probability matrix.
        
        Returns:
            Nested dictionary with transition probabilities
        """
        # Convert DataFrame to nested dict for easier serialization
        matrix_dict = {}
        
        for from_regime in self.transition_matrix.index:
            matrix_dict[from_regime] = {}
            for to_regime in self.transition_matrix.columns:
                matrix_dict[from_regime][to_regime] = float(self.transition_matrix.loc[from_regime, to_regime])
        
        return matrix_dict
    
    def get_transition_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent transition history.
        
        Args:
            limit: Maximum number of historical entries to return
            
        Returns:
            List of recent transition signals
        """
        return self.transition_history[-limit:][::-1] if self.transition_history else []
