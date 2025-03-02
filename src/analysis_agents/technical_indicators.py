"""Technical indicator analysis agent.

This module provides an analysis agent that calculates common technical indicators
for market data, including moving averages, RSI, MACD, and Bollinger Bands.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib  # Technical Analysis Library

from src.analysis_agents.base_agent import AnalysisAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class TechnicalIndicatorAgent(AnalysisAgent):
    """Analysis agent for technical indicators.
    
    This agent calculates common technical indicators from market data and
    publishes them as events for other system components to use.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the technical indicator agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "technical_indicators")
        
        # List of indicators to calculate
        self.indicators = config.get(
            f"analysis_agents.{agent_id}.indicators", 
            ["SMA", "EMA", "RSI", "MACD", "BBANDS"]
        )
        
        # Indicator parameters
        self.params = config.get(f"analysis_agents.{agent_id}.parameters", {})
        
        # Cached data for incremental calculations
        self.candle_cache: Dict[Tuple[str, str, TimeFrame], List[CandleData]] = {}
        self.max_cache_size = config.get(f"analysis_agents.{agent_id}.max_cache_size", 500)
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # Add to cache
        key = (candle.exchange, candle.symbol, candle.timeframe)
        if key not in self.candle_cache:
            self.candle_cache[key] = []
        
        # Check if we already have this candle or a candle with same timestamp
        for i, existing_candle in enumerate(self.candle_cache[key]):
            if existing_candle.timestamp == candle.timestamp:
                # Replace the existing candle with updated data
                self.candle_cache[key][i] = candle
                return
        
        # Add new candle and enforce cache size limit
        self.candle_cache[key].append(candle)
        if len(self.candle_cache[key]) > self.max_cache_size:
            self.candle_cache[key].pop(0)
        
        # Calculate indicators on new data if we have enough candles
        if len(self.candle_cache[key]) >= 30:  # Need at least 30 candles for most indicators
            # Process indicators for this candle
            await self._calculate_indicators(
                candle.exchange, 
                candle.symbol, 
                candle.timeframe, 
                self.candle_cache[key]
            )
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        if not candles:
            return
        
        # Update the cache with these candles
        key = (exchange, symbol, timeframe)
        
        # If we don't have anything in the cache yet, just use these candles
        if key not in self.candle_cache or not self.candle_cache[key]:
            # Initialize cache with a copy of candles (up to max cache size)
            self.candle_cache[key] = candles[-self.max_cache_size:]
        else:
            # Merge new candles with existing cache
            existing_timestamps = {c.timestamp for c in self.candle_cache[key]}
            for candle in candles:
                if candle.timestamp not in existing_timestamps:
                    self.candle_cache[key].append(candle)
                    existing_timestamps.add(candle.timestamp)
            
            # Sort by timestamp and limit size
            self.candle_cache[key].sort(key=lambda c: c.timestamp)
            if len(self.candle_cache[key]) > self.max_cache_size:
                self.candle_cache[key] = self.candle_cache[key][-self.max_cache_size:]
        
        # Calculate indicators on complete dataset
        await self._calculate_indicators(exchange, symbol, timeframe, self.candle_cache[key])
    
    async def _calculate_indicators(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Calculate technical indicators for a set of candles.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        if not candles:
            return
        
        # Convert candles to pandas DataFrame for easier manipulation
        df = self._candles_to_dataframe(candles)
        if df.empty:
            return
        
        # Calculate requested indicators
        for indicator in self.indicators:
            try:
                if indicator == "SMA":
                    await self._calculate_sma(exchange, symbol, timeframe, df)
                elif indicator == "EMA":
                    await self._calculate_ema(exchange, symbol, timeframe, df)
                elif indicator == "RSI":
                    await self._calculate_rsi(exchange, symbol, timeframe, df)
                elif indicator == "MACD":
                    await self._calculate_macd(exchange, symbol, timeframe, df)
                elif indicator == "BBANDS":
                    await self._calculate_bbands(exchange, symbol, timeframe, df)
                # Add more indicators as needed
            
            except Exception as e:
                self.logger.error("Error calculating indicator", 
                               indicator=indicator,
                               symbol=symbol,
                               exchange=exchange,
                               timeframe=timeframe.value,
                               error=str(e))
    
    async def _calculate_sma(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame
    ) -> None:
        """Calculate Simple Moving Average.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of candle data
        """
        # Get periods from config or use defaults
        periods = self.params.get("SMA", {}).get("periods", [20, 50, 200])
        
        # Calculate SMA for each period
        results = {}
        
        for period in periods:
            sma = talib.SMA(df['close'].values, timeperiod=period)
            
            # Convert to dict for the event
            for i, timestamp in enumerate(df.index):
                if np.isnan(sma[i]):
                    continue
                
                if timestamp not in results:
                    results[timestamp] = {}
                
                results[timestamp][f"SMA{period}"] = float(sma[i])
        
        # Publish indicator event
        if results:
            await self.publish_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="SMA",
                values=results
            )
    
    async def _calculate_ema(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame
    ) -> None:
        """Calculate Exponential Moving Average.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of candle data
        """
        # Get periods from config or use defaults
        periods = self.params.get("EMA", {}).get("periods", [12, 26, 50])
        
        # Calculate EMA for each period
        results = {}
        
        for period in periods:
            ema = talib.EMA(df['close'].values, timeperiod=period)
            
            # Convert to dict for the event
            for i, timestamp in enumerate(df.index):
                if np.isnan(ema[i]):
                    continue
                
                if timestamp not in results:
                    results[timestamp] = {}
                
                results[timestamp][f"EMA{period}"] = float(ema[i])
        
        # Publish indicator event
        if results:
            await self.publish_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="EMA",
                values=results
            )
    
    async def _calculate_rsi(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame
    ) -> None:
        """Calculate Relative Strength Index.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of candle data
        """
        # Get period from config or use default
        period = self.params.get("RSI", {}).get("period", 14)
        
        # Calculate RSI
        rsi = talib.RSI(df['close'].values, timeperiod=period)
        
        # Convert to dict for the event
        results = {}
        for i, timestamp in enumerate(df.index):
            if np.isnan(rsi[i]):
                continue
            
            results[timestamp] = float(rsi[i])
        
        # Publish indicator event
        if results:
            await self.publish_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="RSI",
                values=results
            )
    
    async def _calculate_macd(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame
    ) -> None:
        """Calculate Moving Average Convergence Divergence.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of candle data
        """
        # Get parameters from config or use defaults
        params = self.params.get("MACD", {})
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        
        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(
            df['close'].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        # Convert to dict for the event
        results = {}
        for i, timestamp in enumerate(df.index):
            if np.isnan(macd[i]) or np.isnan(macdsignal[i]) or np.isnan(macdhist[i]):
                continue
            
            results[timestamp] = {
                'macd': float(macd[i]),
                'signal': float(macdsignal[i]),
                'histogram': float(macdhist[i])
            }
        
        # Publish indicator event
        if results:
            await self.publish_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="MACD",
                values=results
            )
    
    async def _calculate_bbands(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame
    ) -> None:
        """Calculate Bollinger Bands.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of candle data
        """
        # Get parameters from config or use defaults
        params = self.params.get("BBANDS", {})
        period = params.get("period", 20)
        nbdevup = params.get("nbdevup", 2)
        nbdevdn = params.get("nbdevdn", 2)
        
        # Calculate Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(
            df['close'].values,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
        
        # Convert to dict for the event
        results = {}
        for i, timestamp in enumerate(df.index):
            if np.isnan(upperband[i]) or np.isnan(middleband[i]) or np.isnan(lowerband[i]):
                continue
            
            results[timestamp] = {
                'upper': float(upperband[i]),
                'middle': float(middleband[i]),
                'lower': float(lowerband[i])
            }
        
        # Publish indicator event
        if results:
            await self.publish_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name="BBANDS",
                values=results
            )
    
    def _candles_to_dataframe(self, candles: List[CandleData]) -> pd.DataFrame:
        """Convert a list of candles to a pandas DataFrame.
        
        Args:
            candles: The list of candles to convert
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if not candles:
            return pd.DataFrame()
        
        # Extract OHLCV data from candles
        data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'timestamp': []
        }
        
        for candle in candles:
            data['open'].append(candle.open)
            data['high'].append(candle.high)
            data['low'].append(candle.low)
            data['close'].append(candle.close)
            data['volume'].append(candle.volume)
            data['timestamp'].append(candle.timestamp)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set timestamp as index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df 