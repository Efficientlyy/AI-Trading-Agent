"""
Market Regime Classifier

This module provides a comprehensive market regime classification system that 
integrates multiple analysis factors including volatility clustering, momentum,
correlation patterns, and market liquidity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import json
from datetime import datetime

from .core_definitions import (
    MarketRegimeType,
    VolatilityRegimeType,
    LiquidityRegimeType,
    CorrelationRegimeType,
    MarketRegimeConfig,
    RegimeChangeSignificance,
    RegimeDetectionMethod,
    MarketRegimeInfo
)
from .volatility_clustering import VolatilityClusteringDetector
from .momentum_analysis import MomentumFactorAnalyzer
from .correlation_analysis import CorrelationAnalyzer
from .liquidity_analysis import LiquidityAnalyzer

# Set up logger
logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Multi-factor market regime classifier that integrates volatility, momentum,
    correlation, and liquidity analysis to identify market regimes.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the market regime classifier.
        
        Args:
            config: Configuration for regime classification
        """
        self.config = config or MarketRegimeConfig()
        
        # Initialize component analyzers
        self.volatility_detector = VolatilityClusteringDetector(config)
        self.momentum_analyzer = MomentumFactorAnalyzer(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.liquidity_analyzer = LiquidityAnalyzer(config)
        
        # History storage
        self.regime_history = []
        self.last_regime_info = {}  # Store last regime by asset_id
    
    def classify_regime(self,
                       prices: Union[pd.Series, Dict[str, pd.Series]],
                       volumes: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
                       high_prices: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
                       low_prices: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
                       returns: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
                       asset_id: str = "default",
                       related_assets: Optional[Dict[str, pd.DataFrame]] = None) -> MarketRegimeInfo:
        """
        Classify market regime based on multiple factors.
        
        Args:
            prices: Price data series or dict mapping asset IDs to price series
            volumes: Optional volume data series or dict
            high_prices: Optional high price series or dict
            low_prices: Optional low price series or dict
            returns: Optional pre-calculated returns or dict
            asset_id: Identifier for the primary asset
            related_assets: Optional related assets data for correlation analysis
            
        Returns:
            MarketRegimeInfo object with regime classification
        """
        # Handle single asset case
        if isinstance(prices, pd.Series):
            primary_prices = prices
            primary_volumes = volumes
            primary_high = high_prices
            primary_low = low_prices
            primary_returns = returns
        # Handle multi-asset case
        elif isinstance(prices, dict) and asset_id in prices:
            primary_prices = prices[asset_id]
            primary_volumes = volumes.get(asset_id) if volumes else None
            primary_high = high_prices.get(asset_id) if high_prices else None
            primary_low = low_prices.get(asset_id) if low_prices else None
            primary_returns = returns.get(asset_id) if returns else None
        else:
            logger.error(f"Invalid price data format or missing primary asset {asset_id}")
            return MarketRegimeInfo()  # Return default unknown regime
        
        # Ensure we have enough data
        if primary_prices is None or len(primary_prices) < 30:
            logger.warning(f"Insufficient data for regime classification for {asset_id}")
            return MarketRegimeInfo()  # Return default unknown regime
        
        # Calculate returns if needed
        if primary_returns is None:
            primary_returns = primary_prices.pct_change().dropna()
        
        # 1. Volatility analysis
        vol_result = self.volatility_detector.detect_volatility_clustering(
            prices=primary_prices,
            returns=primary_returns,
            asset_id=asset_id
        )
        volatility_regime = VolatilityRegimeType(vol_result["volatility_regime"])
        
        # 2. Momentum analysis
        mom_result = self.momentum_analyzer.analyze_momentum(
            prices=primary_prices,
            volume=primary_volumes,
            asset_id=asset_id
        )
        market_regime = MarketRegimeType(mom_result["market_regime"])
        
        # 3. Liquidity analysis
        if primary_volumes is not None:
            liq_result = self.liquidity_analyzer.analyze_liquidity(
                prices=primary_prices,
                volume=primary_volumes,
                high_prices=primary_high,
                low_prices=primary_low,
                asset_id=asset_id
            )
            liquidity_regime = LiquidityRegimeType(liq_result["liquidity_regime"])
        else:
            liq_result = {"liquidity_score": None}
            liquidity_regime = LiquidityRegimeType.UNKNOWN
        
        # 4. Correlation analysis (if we have related assets)
        if related_assets and len(related_assets) >= 3:
            # Prepare returns data
            returns_df = pd.DataFrame()
            
            # Add primary asset returns
            returns_df[asset_id] = primary_returns
            
            # Add related asset returns
            for related_id, related_data in related_assets.items():
                if 'returns' in related_data and not related_data['returns'].empty:
                    returns_df[related_id] = related_data['returns']
                elif 'prices' in related_data and not related_data['prices'].empty:
                    returns_df[related_id] = related_data['prices'].pct_change().dropna()
            
            if len(returns_df.columns) >= 3:
                corr_result = self.correlation_analyzer.analyze_correlations(
                    returns_df=returns_df
                )
                correlation_regime = CorrelationRegimeType(corr_result["correlation_regime"])
            else:
                corr_result = {"avg_correlation": None}
                correlation_regime = CorrelationRegimeType.UNKNOWN
        else:
            corr_result = {"avg_correlation": None}
            correlation_regime = CorrelationRegimeType.UNKNOWN
        
        # Detect regime changes
        regime_change = self._detect_regime_change(
            asset_id=asset_id,
            current_regime=market_regime,
            current_vol_regime=volatility_regime,
            current_liq_regime=liquidity_regime
        )
        
        # Calculate confidence based on available data and consistency
        confidence = self._calculate_confidence(
            vol_result=vol_result,
            mom_result=mom_result,
            liq_result=liq_result,
            corr_result=corr_result
        )
        
        # Compile metrics from all analyzers
        metrics = {
            "volatility": {
                "current_volatility": vol_result.get("current_volatility"),
                "clustering_score": vol_result.get("clustering_score"),
                "garch_persistence": vol_result.get("garch_persistence")
            },
            "momentum": {
                "momentum_score": mom_result.get("momentum_score"),
                "trend_strength": mom_result.get("trend_strength")
            },
            "liquidity": {
                "liquidity_score": liq_result.get("liquidity_score"),
                "relative_volume": liq_result.get("relative_volume")
            },
            "correlation": {
                "avg_correlation": corr_result.get("avg_correlation"),
                "risk_on_off_score": corr_result.get("risk_on_off_score")
            }
        }
        
        # Create the regime info object
        regime_info = MarketRegimeInfo(
            regime_type=market_regime,
            volatility_regime=volatility_regime,
            liquidity_regime=liquidity_regime,
            correlation_regime=correlation_regime,
            confidence=confidence,
            detection_method=RegimeDetectionMethod.COMBINED,
            regime_change=regime_change,
            metrics=metrics,
            timestamp=primary_prices.index[-1] if hasattr(primary_prices.index[-1], 'timestamp') else pd.Timestamp.now()
        )
        
        # Store in history
        self.regime_history.append(regime_info.to_dict())
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        # Update last regime info
        self.last_regime_info[asset_id] = regime_info
        
        return regime_info
    
    def _detect_regime_change(self,
                             asset_id: str,
                             current_regime: MarketRegimeType,
                             current_vol_regime: VolatilityRegimeType,
                             current_liq_regime: LiquidityRegimeType) -> RegimeChangeSignificance:
        """
        Detect if there has been a regime change and its significance.
        
        Args:
            asset_id: Asset identifier
            current_regime: Current market regime
            current_vol_regime: Current volatility regime
            current_liq_regime: Current liquidity regime
            
        Returns:
            RegimeChangeSignificance enum
        """
        if asset_id not in self.last_regime_info:
            return RegimeChangeSignificance.NONE
        
        last_info = self.last_regime_info[asset_id]
        changes = 0
        
        # Check market regime change
        if last_info.regime_type != current_regime:
            changes += 2
        
        # Check volatility regime change
        if last_info.volatility_regime != current_vol_regime:
            # Moving to higher volatility is more significant
            if current_vol_regime.value > last_info.volatility_regime.value:
                changes += 2
            else:
                changes += 1
        
        # Check liquidity regime change
        if last_info.liquidity_regime != current_liq_regime:
            # Moving to lower liquidity is more significant
            if current_liq_regime.value > last_info.liquidity_regime.value:
                changes += 2
            else:
                changes += 1
        
        # Determine significance level
        if changes == 0:
            return RegimeChangeSignificance.NONE
        elif changes <= 2:
            return RegimeChangeSignificance.MINOR
        elif changes <= 4:
            return RegimeChangeSignificance.MODERATE
        elif changes <= 6:
            return RegimeChangeSignificance.SIGNIFICANT
        else:
            return RegimeChangeSignificance.MAJOR
    
    def _calculate_confidence(self,
                             vol_result: Dict[str, any],
                             mom_result: Dict[str, any],
                             liq_result: Dict[str, any],
                             corr_result: Dict[str, any]) -> float:
        """
        Calculate confidence level in the regime classification.
        
        Args:
            vol_result: Volatility analysis results
            mom_result: Momentum analysis results
            liq_result: Liquidity analysis results
            corr_result: Correlation analysis results
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence level
        confidence = 0.5
        
        # Add confidence based on data availability and quality
        components = 0
        
        # Volatility component
        if vol_result.get("current_volatility") is not None:
            confidence += 0.1
            components += 1
        
        # Momentum component with strong trend
        if mom_result.get("trend_strength", 0) > 0.3:
            confidence += 0.1
            components += 1
        
        # Liquidity component
        if liq_result.get("liquidity_score") is not None:
            confidence += 0.05
            components += 1
        
        # Correlation component
        if corr_result.get("avg_correlation") is not None:
            confidence += 0.05
            components += 1
        
        # Adjust for consistent signals across components
        if components >= 3:
            consistency_bonus = 0.1
            confidence += consistency_bonus
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    def get_regime_transitions(self, 
                              asset_id: str = "default",
                              lookback_days: int = 30) -> List[Dict[str, any]]:
        """
        Get a list of regime transitions for the specified asset.
        
        Args:
            asset_id: Asset identifier
            lookback_days: Number of days to look back
            
        Returns:
            List of regime transition events
        """
        transitions = []
        prev_regime = None
        
        # Filter history for the specified asset
        asset_history = [entry for entry in self.regime_history 
                        if entry.get('asset_id', 'default') == asset_id]
        
        # Sort by timestamp
        asset_history.sort(key=lambda x: pd.Timestamp(x['timestamp']))
        
        # Find transitions
        for entry in asset_history:
            current_regime = entry['regime_type']
            
            if prev_regime is not None and current_regime != prev_regime:
                transitions.append({
                    'from_regime': prev_regime,
                    'to_regime': current_regime,
                    'timestamp': entry['timestamp'],
                    'significance': entry.get('regime_change', 'none')
                })
                
            prev_regime = current_regime
        
        return transitions
    
    def get_regime_statistics(self, 
                             asset_id: str = "default") -> Dict[str, any]:
        """
        Get statistics about regime occurrences and durations.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Dictionary with regime statistics
        """
        # Filter history for the specified asset
        asset_history = [entry for entry in self.regime_history 
                        if entry.get('asset_id', 'default') == asset_id]
        
        if not asset_history:
            return {
                "regime_counts": {},
                "average_duration": {},
                "volatility_regimes": {},
                "liquidity_regimes": {}
            }
        
        # Count regime occurrences
        regime_counts = {}
        volatility_counts = {}
        liquidity_counts = {}
        
        for entry in asset_history:
            regime = entry['regime_type']
            vol_regime = entry.get('volatility_regime')
            liq_regime = entry.get('liquidity_regime')
            
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            if vol_regime:
                volatility_counts[vol_regime] = volatility_counts.get(vol_regime, 0) + 1
                
            if liq_regime:
                liquidity_counts[liq_regime] = liquidity_counts.get(liq_regime, 0) + 1
        
        # Calculate average durations (requires timestamps)
        durations = {}
        current_regime = None
        regime_start = None
        
        # Sort by timestamp
        asset_history.sort(key=lambda x: pd.Timestamp(x['timestamp']))
        
        for entry in asset_history:
            timestamp = pd.Timestamp(entry['timestamp'])
            regime = entry['regime_type']
            
            if regime != current_regime:
                if current_regime is not None and regime_start is not None:
                    duration = (timestamp - regime_start).total_seconds() / 86400  # days
                    if current_regime in durations:
                        durations[current_regime].append(duration)
                    else:
                        durations[current_regime] = [duration]
                
                current_regime = regime
                regime_start = timestamp
        
        # Calculate average durations
        avg_durations = {regime: sum(days) / len(days) for regime, days in durations.items()}
        
        return {
            "regime_counts": regime_counts,
            "average_duration": avg_durations,
            "volatility_regimes": volatility_counts,
            "liquidity_regimes": liquidity_counts
        }
    
    def save_history(self, filepath: str) -> None:
        """
        Save regime history to a JSON file.
        
        Args:
            filepath: Path to save the history file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "regime_history": self.regime_history,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "lookback_period": self.config.lookback_period,
                        "volatility_window": self.config.volatility_window,
                        "regime_change_threshold": self.config.regime_change_threshold
                    }
                }, f, indent=2)
            logger.info(f"Regime history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save regime history: {str(e)}")
    
    def load_history(self, filepath: str) -> bool:
        """
        Load regime history from a JSON file.
        
        Args:
            filepath: Path to the history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.regime_history = data.get("regime_history", [])
                
                # Reconstruct last regime info
                for entry in self.regime_history:
                    if 'asset_id' in entry:
                        asset_id = entry['asset_id']
                        self.last_regime_info[asset_id] = MarketRegimeInfo.from_dict(entry)
            
            logger.info(f"Loaded {len(self.regime_history)} regime history entries from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load regime history: {str(e)}")
            return False
