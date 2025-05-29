"""
Pattern Breakout Strategy for the AI Trading Agent

This strategy generates trading signals based on detected chart patterns
and their breakout levels, adapting to different pattern types.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

from ..pattern_detector import PatternType


class PatternBreakoutStrategy:
    """
    Trading strategy that generates signals based on pattern breakouts.
    
    This strategy looks for price breakouts from detected chart patterns and
    generates buy/sell signals with appropriate stop loss and take profit levels.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pattern breakout strategy.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.default_params = {
            "min_confidence": 65,          # Minimum pattern confidence (0-100)
            "breakout_confirmation_bars": 1,  # Bars needed to confirm breakout
            "stop_loss_atr_multiplier": 1.5,  # ATR multiplier for stop loss
            "take_profit_atr_multiplier": 3.0,  # ATR multiplier for take profit
            "max_trades_per_symbol": 2,    # Maximum active trades per symbol
            "volume_confirmation": True,   # Require above-average volume for breakout
            "atr_period": 14               # Period for ATR calculation
        }
        
        # Merge defaults with provided config
        self.params = {**self.default_params, **self.config.get("parameters", {})}
        
        # Initialize metrics
        self.metrics = {
            "signals_generated": 0,
            "patterns_evaluated": 0,
            "successful_breakouts": 0,
            "failed_breakouts": 0
        }
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                         technical_state: Dict[str, Any], symbols: List[str]) -> List[Dict]:
        """
        Generate trading signals based on pattern breakouts.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            technical_state: Dictionary with technical analysis data including patterns
            symbols: List of symbols to generate signals for
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Extract patterns from technical state
        patterns = technical_state.get("patterns", {})
        
        for symbol in symbols:
            if symbol not in market_data or symbol not in patterns:
                continue
            
            df = market_data[symbol]
            symbol_patterns = patterns[symbol]
            
            # Skip if no patterns detected
            if not symbol_patterns:
                continue
            
            # Calculate ATR for position sizing
            atr = self._calculate_atr(df, self.params["atr_period"])
            
            # Filter patterns by confidence threshold
            high_confidence_patterns = [p for p in symbol_patterns 
                                        if p.get("confidence", 0) >= self.params["min_confidence"]]
            
            self.metrics["patterns_evaluated"] += len(high_confidence_patterns)
            
            # Process each pattern
            for pattern in high_confidence_patterns:
                pattern_type = pattern.get("pattern_type")
                if not pattern_type:
                    continue
                
                # Generate signal based on pattern type
                signal = self._generate_pattern_signal(pattern, df, symbol, atr)
                
                if signal:
                    signals.append(signal)
                    self.metrics["signals_generated"] += 1
        
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """
        Calculate Average True Range (ATR) for volatility-based position sizing.
        
        Args:
            df: OHLCV DataFrame
            period: ATR period
            
        Returns:
            ATR value
        """
        # True Range calculation
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        
        # Simple moving average of true range
        atr = np.mean(tr[-period:])
        
        return atr
    
    def _generate_pattern_signal(self, pattern: Dict, df: pd.DataFrame, 
                                 symbol: str, atr: float) -> Optional[Dict]:
        """
        Generate signal based on specific pattern type.
        
        Args:
            pattern: Pattern detection result
            df: OHLCV DataFrame
            symbol: Trading symbol
            atr: Average True Range value
            
        Returns:
            Signal dictionary or None if no signal
        """
        pattern_type = pattern.get("pattern_type")
        if not pattern_type:
            return None
        
        # Get current price (latest close)
        current_price = df['close'].iloc[-1]
        
        # Pattern price levels and target
        price_level = pattern.get("price_level")
        target_price = pattern.get("target_price")
        
        if not price_level:
            return None
        
        # Default signal attributes
        signal = {
            "symbol": symbol,
            "strategy": "pattern_breakout",
            "timestamp": datetime.now().isoformat(),
            "confidence": pattern.get("confidence", 50),
            "pattern_type": pattern_type
        }
        
        # Generate signal based on pattern type
        if pattern_type == PatternType.HEAD_AND_SHOULDERS.name:
            # Bearish pattern - sell on breakdown below neckline
            if current_price < price_level:
                signal.update({
                    "signal": "sell",
                    "entry_price": current_price,
                    "stop_loss": current_price + (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price - (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakdown below Head & Shoulders neckline at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS.name:
            # Bullish pattern - buy on breakout above neckline
            if current_price > price_level:
                signal.update({
                    "signal": "buy",
                    "entry_price": current_price,
                    "stop_loss": current_price - (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price + (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakout above Inverse Head & Shoulders neckline at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.DOUBLE_TOP.name:
            # Bearish pattern - sell on breakdown below neckline
            if current_price < price_level:
                signal.update({
                    "signal": "sell",
                    "entry_price": current_price,
                    "stop_loss": current_price + (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price - (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakdown below Double Top neckline at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.DOUBLE_BOTTOM.name:
            # Bullish pattern - buy on breakout above neckline
            if current_price > price_level:
                signal.update({
                    "signal": "buy",
                    "entry_price": current_price,
                    "stop_loss": current_price - (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price + (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakout above Double Bottom neckline at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.TRIANGLE_ASCENDING.name:
            # Bullish pattern - buy on breakout above upper trendline
            if current_price > price_level:
                signal.update({
                    "signal": "buy",
                    "entry_price": current_price,
                    "stop_loss": current_price - (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price + (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakout above Ascending Triangle resistance at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.TRIANGLE_DESCENDING.name:
            # Bearish pattern - sell on breakdown below lower trendline
            if current_price < price_level:
                signal.update({
                    "signal": "sell",
                    "entry_price": current_price,
                    "stop_loss": current_price + (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price - (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakdown below Descending Triangle support at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.FLAG_BULLISH.name:
            # Bullish pattern - buy on breakout above upper flag trendline
            if current_price > price_level:
                signal.update({
                    "signal": "buy",
                    "entry_price": current_price,
                    "stop_loss": current_price - (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price + (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakout from Bullish Flag pattern at {price_level:.2f}"
                })
                return signal
                
        elif pattern_type == PatternType.FLAG_BEARISH.name:
            # Bearish pattern - sell on breakdown below lower flag trendline
            if current_price < price_level:
                signal.update({
                    "signal": "sell",
                    "entry_price": current_price,
                    "stop_loss": current_price + (atr * self.params["stop_loss_atr_multiplier"]),
                    "take_profit": target_price if target_price else current_price - (atr * self.params["take_profit_atr_multiplier"]),
                    "reason": f"Breakdown from Bearish Flag pattern at {price_level:.2f}"
                })
                return signal
        
        # If we get here, no signal was generated
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        return self.metrics