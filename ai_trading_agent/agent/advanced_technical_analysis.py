"""
Advanced Technical Analysis Integration Module

This module integrates multi-timeframe analysis, machine learning signal validation, 
and adaptive parameters into the Technical Analysis Agent framework.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os
import time
from pathlib import Path
import concurrent.futures
from multiprocessing import cpu_count
from functools import partial

# Import the performance profiler
from ..utils.performance_profiler import profile_execution, time_block, profiler

from .strategy_manager import Strategy, BaseStrategy, SignalDirection
from .indicator_engine import IndicatorEngine
from .multi_timeframe import MultiTimeframeStrategy, TimeframeManager
from ..ml.signal_validator import MLSignalValidator
from ..ml.adaptive_parameters import AdaptiveParameterManager, MarketRegimeClassifier
from ..data.data_source_factory import get_data_source_factory
from ..config.data_source_config import get_data_source_config
from ..common.utils import get_logger


class AdvancedTechnicalAnalysisAgent:
    """
    Enhanced Technical Analysis Agent with advanced features.
    
    This class integrates multi-timeframe analysis, machine learning signal validation,
    and adaptive parameter tuning to produce higher-quality trading signals.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the advanced technical analysis agent.
        
        Args:
            config: Configuration dictionary with parameters
                - strategies: List of strategy configurations
                - indicators: List of indicators to calculate
                - timeframes: List of timeframes to analyze
                - ml_validator: ML signal validator configuration
                - adaptive_parameters: Adaptive parameter configuration
                - profiling: Performance profiling configuration
                - data_source: Configuration for data source (mock vs real)
        """
        self.logger = get_logger("AdvancedTechnicalAnalysisAgent")
        self.config = config or {}
        
        # Initialize performance profiling
        profiling_config = self.config.get("profiling", {})
        self.enable_profiling = profiling_config.get("enabled", True)
        profiler.enabled = self.enable_profiling
        
        if self.enable_profiling:
            self.logger.info("Performance profiling enabled")
        
        # Initialize data source factory integration
        self.data_source_factory = get_data_source_factory()
        self.data_source_config = get_data_source_config()
        
        # Register as a listener for data source configuration changes
        self.data_source_config.register_listener(self._handle_data_source_change)
        
        # Extract configuration
        strategy_configs = self.config.get("strategies", [])
        indicator_list = self.config.get("indicators", [])
        self.timeframes = self.config.get("timeframes", ["1h", "4h", "1d"])
        
        # Initialize components
        indicator_config = self.config.get("indicator_config", {})
        self.indicator_engine = IndicatorEngine(indicator_config)
        
        # Initialize strategies
        self.strategies = self._init_strategies(strategy_configs)
        
        # Initialize data source
        data_source_config = self.config.get("data_source", {})
        if data_source_config:
            # If configuration is provided, update the global config
            self.data_source_config.update_config(data_source_config)
        
        # Initialize multi-timeframe manager
        self.timeframe_manager = TimeframeManager(self.timeframes)
        
        # Initialize ML signal validator
        ml_config = self.config.get("ml_validator", {})
        self.signal_validator = MLSignalValidator(ml_config)
        
        # Initialize adaptive parameter manager
        param_config = self.config.get("adaptive_parameters", {})
        self.parameter_manager = AdaptiveParameterManager(param_config)
        
        # Initialize market regime classifier
        self.regime_classifier = MarketRegimeClassifier(
            self.config.get("regime_classifier", {})
        )
        
        # Parallel processing configuration
        parallel_config = self.config.get("parallel_processing", {})
        self.enable_parallel = parallel_config.get("enabled", True)
        self.max_workers = parallel_config.get("max_workers", min(cpu_count(), 4))  # Default to 4 or CPU count
        self.min_timeframes_for_parallel = parallel_config.get("min_timeframes", 3)  # Minimum timeframes to use parallel
        self.min_symbols_for_parallel = parallel_config.get("min_symbols", 3)  # Minimum symbols to use parallel
        
        if self.enable_parallel:
            self.logger.info(f"Parallel processing enabled with {self.max_workers} workers")
        else:
            self.logger.info("Parallel processing disabled")
        
        # Set initial indicators list
        self.indicator_list = indicator_list
        
        # Extract indicator configurations
        self.indicator_configs = {}
        if "indicator_config" in self.config:
            self.indicator_configs = self.config["indicator_config"]
        
        # Metrics tracking
        self.metrics = {
            "signals_generated": 0,
            "signals_validated": 0,
            "signals_rejected": 0,
            "avg_signal_confidence": 0.0,
            "regime_changes": 0,
            "parameter_adaptations": 0,
            "timeframes_analyzed": len(self.timeframes),
            "last_execution_time_ms": 0
        }
        
        self.current_regime = "unknown"
        
        self.logger.info(
            f"Initialized Advanced Technical Analysis Agent with "
            f"{len(self.strategies)} strategies and {len(self.timeframes)} timeframes"
        )
    
    @profile_execution("AdvancedTechnicalAnalysisAgent")
    def _process_symbol_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                              indicator_list: List[Any]):
        """
        Process indicators for a specific symbol and timeframe combination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe identifier (e.g., "1h", "4h", "1d")
            data: Market data DataFrame
            indicator_list: List of indicators to calculate
            
        Returns:
            Tuple of (symbol, timeframe, indicator_results)
        """
        try:
            # Calculate all indicators
            with time_block(f"calculate_indicators_{symbol}_{timeframe}", "IndicatorEngine"):
                indicators = self.indicator_engine.calculate_all_indicators(data, indicator_list)
            return symbol, timeframe, indicators
        except Exception as e:
            self.logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
            return symbol, timeframe, {}
                
    @profile_execution("AdvancedTechnicalAnalysisAgent")
    def _process_symbol(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame],
                      indicator_list: List[Any], use_parallel: bool = False):
        """
        Process all timeframes for a specific symbol.
        
        Args:
            symbol: Trading symbol
            timeframe_data: Dictionary mapping timeframes to market data DataFrames
            indicator_list: List of indicators to calculate
            use_parallel: Whether to use parallel processing for timeframes
            
        Returns:
            Tuple of (symbol, timeframe_indicators)
        """
        # Store results for each timeframe
        timeframe_indicators = {}
        
        # Check if we should use parallel processing for timeframes
        if use_parallel and len(timeframe_data) >= self.min_timeframes_for_parallel:
            # Process timeframes in parallel
            tasks = []
            for tf, data in timeframe_data.items():
                tasks.append((symbol, tf, data, indicator_list))
                
            # Execute in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(lambda args: self._process_symbol_timeframe(*args), tasks))
                
            # Collect results
            for sym, tf, indicators in results:
                timeframe_indicators[tf] = indicators
        else:
            # Process timeframes sequentially
            for tf, data in timeframe_data.items():
                _, _, indicators = self._process_symbol_timeframe(symbol, tf, data, indicator_list)
                timeframe_indicators[tf] = indicators
                
        return symbol, timeframe_indicators
        
    def _init_strategies(self, strategy_configs: List[Dict[str, Any]]) -> Dict[str, Strategy]:
        """Initialize strategies from configuration."""
        strategies = {}
        
        # Import strategy classes
        from .strategy_manager import (
            MovingAverageCrossStrategy,
            RSIOverboughtOversoldStrategy
        )
        from .multi_timeframe import MultiTimeframeStrategy
        
        # Map strategy names to classes
        strategy_classes = {
            "MA_Cross": MovingAverageCrossStrategy,
            "RSI_OB_OS": RSIOverboughtOversoldStrategy,
            "Multi_TF": MultiTimeframeStrategy
        }
        
        # Initialize each strategy
        for config in strategy_configs:
            strategy_type = config.get("type")
            
            if strategy_type not in strategy_classes:
                self.logger.warning(f"Unknown strategy type: {strategy_type}")
                continue
                
            # Create the strategy instance
            strategy = strategy_classes[strategy_type](config)
            strategies[strategy.name] = strategy
            
        return strategies
    
    @profile_execution("AdvancedTechnicalAnalysisAgent")
    def _handle_data_source_change(self, updated_config: Dict[str, Any]) -> None:
        """
        Handle data source configuration changes.
        
        Args:
            updated_config: The updated configuration
        """
        self.logger.info(f"Data source changed to {'mock' if updated_config.get('use_mock_data', True) else 'real'}")
        
        # Could trigger a cache clear or other actions as needed
        # For now, just log the change
    
    def get_data_source_type(self) -> str:
        """
        Get the current data source type.
        
        Returns:
            'mock' or 'real' depending on current configuration
        """
        return "mock" if self.data_source_config.use_mock_data else "real"
    
    def toggle_data_source(self) -> str:
        """
        Toggle between mock and real data sources.
        
        Returns:
            String indicating the new data source type ('mock' or 'real')
        """
        new_state = self.data_source_factory.toggle_data_source()
        return "mock" if new_state else "real"
    
    def analyze(
        self, 
        market_data: Dict[str, Dict[str, pd.DataFrame]], 
        symbols: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze market data and generate validated trading signals.
        
        Args:
            market_data: Dictionary mapping symbols to timeframe-specific market data DataFrames
                Format: {symbol: {timeframe: DataFrame}}
            symbols: List of symbols to analyze, or None for all
            
        Returns:
            List of validated trading signal dictionaries
        """
        # Start timer for analysis
        start_time = time.time()
        
        # Log the data source being used
        data_source_type = self.get_data_source_type()
        self.logger.info(f"Using {data_source_type} data for analysis")
        
        self.logger.info(f"Starting analysis for {len(symbols or market_data.keys())} symbols")
        
        # If profiling enabled, start memory tracking
        if self.enable_profiling:
            profiler.start_memory_tracking()
            profiler.take_memory_snapshot("analyze_start")
        
        if symbols is None:
            symbols = list(market_data.keys())
            
        # Skip processing if no data
        if not market_data or not symbols:
            self.logger.warning("No market data or symbols provided")
            return []
            
        signals = []
        
        try:
            # Step 1: Organize data for multi-timeframe analysis (if needed)
            for symbol in list(symbols):  # Use list to avoid modifying during iteration
                if symbol not in market_data:
                    symbols.remove(symbol)
                    continue
                    
                # Convert single DataFrame to timeframe dict if needed
                if isinstance(market_data[symbol], pd.DataFrame):
                    market_data[symbol] = {self.timeframes[0]: market_data[symbol]}
            
            # Step 2: Calculate technical indicators for all timeframes using parallel processing
            self.logger.info(f"Calculating indicators for {len(symbols)} symbols across {len(self.timeframes)} timeframes")
            all_indicators = {}
            
            # Take memory snapshot before indicator calculation
            if self.enable_profiling:
                profiler.take_memory_snapshot("before_indicators")
            
            # Determine if we should use parallel processing for symbols
            use_parallel_symbols = self.enable_parallel and len(symbols) >= self.min_symbols_for_parallel
            
            if use_parallel_symbols:
                # Process all symbols in parallel
                tasks = []
                for symbol in symbols:
                    if symbol in market_data:
                        # Decide if we'll use parallel for timeframes too (nested parallelism)
                        use_parallel_timeframes = (self.enable_parallel and 
                                                len(market_data[symbol]) >= self.min_timeframes_for_parallel and
                                                len(symbols) < self.min_symbols_for_parallel)  # Don't use nested if too many symbols
                        
                        tasks.append((symbol, market_data[symbol], self.indicator_list, use_parallel_timeframes))
                        
                # Execute symbol processing in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Process each symbol
                    results = list(executor.map(lambda args: self._process_symbol(*args), tasks))
                    
                    # Collect results
                    for sym, tf_indicators in results:
                        all_indicators[sym] = tf_indicators
            else:
                # Process symbols sequentially
                for symbol in symbols:
                    if symbol in market_data:
                        # Still use parallel for timeframes if possible
                        use_parallel_timeframes = (self.enable_parallel and 
                                                len(market_data[symbol]) >= self.min_timeframes_for_parallel)
                        
                        sym, tf_indicators = self._process_symbol(
                            symbol, market_data[symbol], self.indicator_list, use_parallel_timeframes
                        )
                        all_indicators[sym] = tf_indicators
            
            indicator_time = time.time() - start_time
            self.logger.info(f"Indicator calculation completed in {indicator_time:.2f}s")
            
            # Take memory snapshot after indicator calculation
            if self.enable_profiling:
                profiler.take_memory_snapshot("after_indicators")
            
            # Step 3: Identify market regime for each symbol
            regimes = {}
            for symbol in symbols:
                if symbol not in market_data or symbol not in all_indicators:
                    continue
                    
                # Use primary timeframe for regime detection
                primary_tf = self.timeframes[0]
                if primary_tf in market_data[symbol] and primary_tf in all_indicators[symbol]:
                    regime = self.regime_classifier.classify_regime(
                        market_data[symbol][primary_tf], 
                        all_indicators[symbol][primary_tf]
                    )
                    regimes[symbol] = regime
                    
                    # Track metrics
                    if regime["regime"] != self.current_regime:
                        self.metrics["regime_changes"] += 1
                        self.current_regime = regime["regime"]
            
            # Step 4: Adjust strategy parameters based on market regime
            parameters = {}
            for symbol in symbols:
                if symbol not in regimes:
                    continue
                    
                # Adjust parameters for each strategy based on regime
                params = self.parameter_manager.get_optimal_parameters(
                    regimes[symbol], 
                    list(self.strategies.keys())
                )
                parameters[symbol] = params
                
                # Track metrics
                if params.get("adjusted", False):
                    self.metrics["parameter_adaptations"] += 1
            
            # Step 5: Generate signals using strategies
            raw_signals = []
            
            # Take memory snapshot before signal generation
            if self.enable_profiling:
                profiler.take_memory_snapshot("before_signal_generation")
            for strategy_name, strategy in self.strategies.items():
                # Apply multi-timeframe analysis if strategy supports it
                if isinstance(strategy, MultiTimeframeStrategy):
                    for symbol in symbols:
                        if symbol not in market_data or symbol not in all_indicators:
                            continue
                            
                        # Apply optimized parameters if available
                        if symbol in parameters and strategy_name in parameters[symbol]:
                            strategy.update_parameters(parameters[symbol][strategy_name])
                            
                        # Generate signals using multi-timeframe data
                        symbol_signals = strategy.generate_signals(
                            market_data[symbol],
                            all_indicators[symbol],
                            [symbol]
                        )
                        raw_signals.extend(symbol_signals)
                else:
                    # Use primary timeframe for regular strategies
                    primary_data = {}
                    primary_indicators = {}
                    
                    for symbol in symbols:
                        if symbol not in market_data or symbol not in all_indicators:
                            continue
                            
                        primary_tf = self.timeframes[0]
                        if primary_tf in market_data[symbol] and primary_tf in all_indicators[symbol]:
                            primary_data[symbol] = market_data[symbol][primary_tf]
                            primary_indicators[symbol] = all_indicators[symbol][primary_tf]
                            
                            # Apply optimized parameters if available
                            if symbol in parameters and strategy_name in parameters[symbol]:
                                strategy.update_parameters(parameters[symbol][strategy_name])
                    
                    # Generate signals for all symbols
                    if primary_data:
                        strategy_signals = strategy.generate_signals(
                            primary_data,
                            primary_indicators,
                            list(primary_data.keys())
                        )
                        raw_signals.extend(strategy_signals)
            
            # Track signal generation metrics
            self.metrics["signals_generated"] += len(raw_signals)
            
            # Step 6: Validate signals using ML validator
            validated_signals = []
            for signal in raw_signals:
                # Extract features for validation
                symbol = signal["symbol"]
                if symbol not in market_data or symbol not in all_indicators:
                    continue
                    
                # Use primary timeframe for feature extraction
                primary_tf = self.timeframes[0]
                if primary_tf not in market_data[symbol] or primary_tf not in all_indicators[symbol]:
                    continue
                    
                # Extract features for validation
                features = self.signal_validator.extract_features(
                    market_data[symbol][primary_tf],
                    all_indicators[symbol][primary_tf],
                    signal
                )
                
                # Validate signal
                validation_result = self.signal_validator.validate_signal(features, signal)
                
                # Track validation metrics
                if validation_result["status"] == "validated":
                    self.metrics["signals_validated"] += 1
                    
                    # Add validation result to signal
                    validated_signal = signal.copy()
                    validated_signal["validation"] = validation_result
                    validated_signal["confidence"] = validation_result["confidence"]
                    
                    # Add market regime information
                    if symbol in regimes:
                        validated_signal["market_regime"] = regimes[symbol]["regime"]
                        
                    # Format signal for output
                    formatted_signal = {
                        "type": "technical_signal",
                        "payload": validated_signal
                    }
                    validated_signals.append(formatted_signal)
                else:
                    self.metrics["signals_rejected"] += 1
            
            # Update signals list with validated signals
            signals = validated_signals
                    
            # Update average confidence metric
            if signals:
                confidences = [s["payload"].get("confidence", 0) for s in signals if "confidence" in s["payload"]]
                if confidences:
                    self.metrics["avg_signal_confidence"] = sum(confidences) / len(confidences)
                
            # Track overall metrics
            self.metrics["signals_generated"] += len(validated_signals)
            self.metrics["signals_validated"] += len([s for s in validated_signals if s.get("is_valid", False)])
            self.metrics["signals_rejected"] += len([s for s in validated_signals if not s.get("is_valid", False)])
            
            # Track data source in metrics
            self.metrics["data_source"] = self.get_data_source_type()
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self.metrics["last_execution_time_ms"] = total_time * 1000  # Convert to ms
            self.logger.info(f"Analysis completed in {total_time:.2f}s with {len(signals)} validated signals")
            
            # Take final memory snapshot and generate profiling report
            if self.enable_profiling:
                profiler.take_memory_snapshot("analyze_end")
                
                # Generate report
                profiling_report = profiler.generate_report()
                
                # Log bottlenecks
                bottlenecks = profiler.identify_bottlenecks()
                if bottlenecks:
                    self.logger.warning("Performance bottlenecks detected:")
                    for b in bottlenecks:
                        self.logger.warning(f"  {b['function']}: {b['avg_time_ms']:.2f}ms avg, {b['call_count']} calls")
                        
                # Log memory usage
                memory_growth = profiling_report.get("memory_growth", {})
                if memory_growth:
                    self.logger.info(f"Memory growth: {memory_growth.get('growth_mb', 0):.2f}MB ({memory_growth.get('growth_percent', 0):.2f}%)")
                    
                # Stop memory tracking
                profiler.stop_memory_tracking()
                
        except Exception as e:
            self.logger.error(f"Error in advanced technical analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return signals
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the advanced technical analysis agent."""
        metrics = self.metrics.copy()
        
        # Add profiling metrics if enabled
        if self.enable_profiling:
            metrics["profiling"] = {
                "bottlenecks": profiler.identify_bottlenecks(top_n=3),
                "memory_usage": profiler.get_memory_usage(),
            }
        
        # Add strategy metrics
        strategy_metrics = {}
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'metrics'):
                strategy_metrics[name] = strategy.metrics
        
        metrics["strategies"] = strategy_metrics
        metrics["current_regime"] = self.current_regime
        
        # Add signal validator metrics
        validator_metrics = self.signal_validator.get_metrics()
        metrics["signal_validator"] = validator_metrics
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "signals_generated": 0,
            "signals_validated": 0,
            "signals_rejected": 0,
            "avg_signal_confidence": 0.0,
            "regime_changes": 0,
            "parameter_adaptations": 0,
            "timeframes_analyzed": len(self.timeframes),
            "last_execution_time_ms": 0,
            "data_source": self.get_data_source_type()
        }
        
        # Reset profiling data if enabled
        if self.enable_profiling:
            profiler.reset()
        
        # Reset strategy metrics
        for strategy in self.strategies.values():
            if hasattr(strategy, 'reset_metrics'):
                strategy.reset_metrics()
    
    def save_state(self, directory: str) -> bool:
        """
        Save the current state of the agent to files.
        
        Args:
            directory: Directory to save files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save signal validator model
            model_path = os.path.join(directory, "signal_validator_model.pkl")
            self.signal_validator.save_model(model_path)
            
            # Save parameter manager performance history
            history_path = os.path.join(directory, "parameter_performance_history.json")
            self.parameter_manager.save_performance_history(history_path)
            
            # Save metrics
            metrics_path = os.path.join(directory, "agent_metrics.json")
            with open(metrics_path, 'w') as f:
                # Convert metrics to serializable format
                serializable_metrics = {}
                for k, v in self.get_metrics().items():
                    if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                        serializable_metrics[k] = v
                    elif isinstance(v, np.number):
                        serializable_metrics[k] = float(v)
                    else:
                        serializable_metrics[k] = str(v)
                
                json.dump(serializable_metrics, f, indent=2)
            
            self.logger.info(f"Saved agent state to {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent state: {str(e)}")
            return False
    
    def load_state(self, directory: str) -> bool:
        """
        Load the agent state from files.
        
        Args:
            directory: Directory with saved state files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                self.logger.warning(f"State directory not found: {directory}")
                return False
            
            # Load signal validator model
            model_path = os.path.join(directory, "signal_validator_model.pkl")
            if os.path.exists(model_path):
                self.signal_validator._load_model(model_path)
            
            # Load parameter manager performance history
            history_path = os.path.join(directory, "parameter_performance_history.json")
            if os.path.exists(history_path):
                self.parameter_manager.load_performance_history(history_path)
            
            self.logger.info(f"Loaded agent state from {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading agent state: {str(e)}")
            return False
