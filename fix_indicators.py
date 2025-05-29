"""
Script to fix the calculate_all_indicators method in indicator_engine.py
to ensure proper indentation and correct handling of LAG_FEATURES_RS_ indicators.
"""

import os
import re

def fix_calculate_all_indicators():
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the calculate_all_indicators method
    pattern = r'def calculate_all_indicators.*?:(.*?)def'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        print("Method not found")
        return
    
    # Create a fixed version of the method
    fixed_method = """    def calculate_all_indicators(self, market_data: Dict[str, pd.DataFrame], symbols: List[str]) -> Dict[str, Dict]:
        \"\"\"
        Calculate all enabled indicators for the provided market data.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            symbols: List of symbols to calculate indicators for
            
        Returns:
            Dictionary of indicator values organized by symbol and indicator type
        \"\"\"
        self.logger.debug(f"IndicatorEngine.calculate_all_indicators: Logger ID is {id(self.logger)}")
        overall_start_time = datetime.now()
        all_results: Dict[str, Dict[str, Any]] = {symbol: {} for symbol in symbols} # Initialize results for all symbols
        self.metrics["calculations_performed"] = 0 # Reset for this call
        self.metrics["calculation_errors"] = 0 # Reset for this call
        self.metrics["indicator_errors"] = 0 # Reset for this call
        self.metrics["indicators_calculated"] = 0 # Reset for this call
        self.metrics["indicator_calculation_time_ms"] = {} # Reset for this call

        if not market_data:
            self.logger.warning("Market data is empty. Cannot calculate indicators.")
            return all_results

        self.logger.debug(f"Starting calculation for symbols: {symbols}")
        
        for symbol in symbols:
            self.logger.debug(f"Processing symbol: {symbol}")
            try:
                if symbol not in market_data:
                    self.logger.warning(f"Symbol {symbol} not found in market data")
                    all_results[symbol] = {"error": f"Symbol {symbol} not found in market data"}
                    continue
                    
                df = market_data[symbol]
                if df.empty:
                    self.logger.warning(f"Market data for symbol {symbol} is empty, skipping.")
                    all_results[symbol] = {"error": "Market data is empty"}
                    continue
                
                # Initialize collections for this symbol
                symbol_indicators = {}
                calculated_indicator_names_for_symbol = []
                
                self.logger.debug(f"Processing symbol: {symbol} with data shape {df.shape}")
                
                # Process all registered indicators
                for indicator_name, settings in self.indicators.items():
                    self.logger.debug(f"[{symbol}] Processing indicator: {indicator_name}")
                    
                    # Check if indicator is enabled
                    defaulted_enabled_check = settings.get("enabled", False)
                    if not defaulted_enabled_check:
                        self.logger.debug(f"[{symbol}] Skipping disabled indicator: {indicator_name}")
                        continue
                    
                    # Get calculator function
                    calculator = settings.get("calculator")
                    if not calculator:
                        self.logger.warning(f"No calculator for indicator {indicator_name}")
                        continue
                    
                    # Proceed with calculation
                    try:
                        indicator_start_time = datetime.now()
                        
                        # Handle different indicator types
                        if indicator_name in ["sma", "ema", "rsi", "atr"]:
                            # Handle period-based indicators
                            periods = settings.get("periods", [])
                            indicator_results = {}
                            
                            for period in periods:
                                cache_key = f"{symbol}_{indicator_name}_{period}"
                                value_from_cache = self.cache_manager.get(cache_key)
                                
                                if value_from_cache is not None:
                                    indicator_results[str(period)] = value_from_cache
                                    self.metrics["cache_hits"] += 1
                                else:
                                    self.metrics["cache_misses"] += 1
                                    df_copy = df.copy()
                                    df_hash = hash(tuple(df_copy['close'].iloc[-10:].values))
                                    calc_value = calculator(df_copy, period=period)
                                    
                                    if calc_value is not None and not (isinstance(calc_value, pd.Series) and calc_value.empty):
                                        indicator_results[str(period)] = calc_value
                                        self.cache_manager.set(cache_key, calc_value, df_hash=df_hash)
                            
                            if indicator_results:
                                symbol_indicators[indicator_name] = indicator_results
                                calculated_indicator_names_for_symbol.append(indicator_name)
                        
                        elif indicator_name == "bollinger_bands":
                            # Handle Bollinger Bands
                            periods = settings.get("periods", [])
                            deviations = settings.get("deviations", 2)
                            bb_results_for_period = {}
                            
                            for period in periods:
                                cache_key = f"{symbol}_{indicator_name}_{period}_{deviations}"
                                value_from_cache = self.cache_manager.get(cache_key)
                                
                                if value_from_cache is not None:
                                    bb_results_for_period[str(period)] = value_from_cache
                                    self.metrics["cache_hits"] += 1
                                else:
                                    self.metrics["cache_misses"] += 1
                                    df_copy = df.copy()
                                    df_hash = hash(tuple(df_copy['close'].iloc[-10:].values))
                                    calc_value = calculator(df_copy, period, deviations)
                                    
                                    if isinstance(calc_value, dict) and all(isinstance(s, pd.Series) and not s.empty for s in calc_value.values()):
                                        bb_results_for_period[str(period)] = calc_value
                                        self.cache_manager.set(cache_key, calc_value, df_hash=df_hash)
                            
                            if bb_results_for_period:
                                symbol_indicators[indicator_name] = bb_results_for_period
                                calculated_indicator_names_for_symbol.append(indicator_name)
                        
                        elif indicator_name == "macd":
                            # Handle MACD
                            fast = settings.get("fast_period", 12)
                            slow = settings.get("slow_period", 26)
                            signal = settings.get("signal_period", 9)
                            cache_key = f"{symbol}_{indicator_name}_{fast}_{slow}_{signal}"
                            value_from_cache = self.cache_manager.get(cache_key)
                            
                            if value_from_cache is not None:
                                symbol_indicators[indicator_name] = value_from_cache
                                calculated_indicator_names_for_symbol.append(indicator_name)
                                self.metrics["cache_hits"] += 1
                            else:
                                self.metrics["cache_misses"] += 1
                                df_copy = df.copy()
                                df_hash = hash(tuple(df_copy['close'].iloc[-10:].values))
                                calc_value = calculator(df_copy, fast, slow, signal)
                                
                                if calc_value and isinstance(calc_value, dict) and all(isinstance(v, pd.Series) and not v.empty for v in calc_value.values()):
                                    symbol_indicators[indicator_name] = calc_value
                                    self.cache_manager.set(cache_key, calc_value, df_hash=df_hash)
                                    calculated_indicator_names_for_symbol.append(indicator_name)
                        
                        elif indicator_name == "lag_features":
                            # Handle regular lag features
                            lags = settings.get("lags", [])
                            if not lags:
                                continue
                                
                            lags_tuple_hash = hash(tuple(sorted(lags)))
                            cache_key = f"{symbol}_{indicator_name}_lags_{lags_tuple_hash}"
                            value_from_cache = self.cache_manager.get(cache_key)
                            
                            if value_from_cache is not None:
                                symbol_indicators[indicator_name] = value_from_cache
                                calculated_indicator_names_for_symbol.append(indicator_name)
                                self.metrics["cache_hits"] += 1
                            else:
                                self.metrics["cache_misses"] += 1
                                source_column = settings.get("source_column", "close")
                                df_copy = df.copy()
                                df_hash = hash(tuple(df_copy[source_column].iloc[-10:].values))
                                calc_value = calculator(df_copy, lags, source_column)
                                
                                if calc_value and isinstance(calc_value, dict) and all(isinstance(s, pd.Series) for s in calc_value.values()):
                                    symbol_indicators[indicator_name] = calc_value
                                    self.cache_manager.set(cache_key, calc_value, df_hash=df_hash)
                                    calculated_indicator_names_for_symbol.append(indicator_name)
                        
                        elif indicator_name.startswith("LAG_FEATURES_RS_"):
                            # Special handling for LAG_FEATURES_RS_ indicators
                            self.logger.debug(f"Processing LAG_FEATURES_RS_ indicator: {indicator_name}")
                            source_column = settings.get("source_column", "close")
                            lags_to_calculate = settings.get("lags_to_calculate", [])
                            
                            if not lags_to_calculate:
                                self.logger.warning(f"No lags provided for {indicator_name}")
                                continue
                            
                            # Calculate lag features
                            df_copy = df.copy()
                            lag_features_dict = self._calculate_lag_features(
                                df=df_copy, 
                                lags=lags_to_calculate,
                                source_column=source_column
                            )
                            
                            if lag_features_dict:
                                # Convert to DataFrame
                                lag_features_df = pd.DataFrame(lag_features_dict)
                                self.logger.debug(f"Created lag features DataFrame with shape {lag_features_df.shape}")
                                
                                # Store in results
                                symbol_indicators[indicator_name] = lag_features_df
                                calculated_indicator_names_for_symbol.append(indicator_name)
                                self.logger.debug(f"Added {indicator_name} to symbol_indicators")
                        
                        # Track calculation time
                        indicator_duration_ms = (datetime.now() - indicator_start_time).total_seconds() * 1000
                        self.metrics["indicator_calculation_time_ms"][indicator_name] = (
                            self.metrics["indicator_calculation_time_ms"].get(indicator_name, 0) + 
                            indicator_duration_ms
                        )
                        self.metrics["indicators_calculated"] = self.metrics.get("indicators_calculated", 0) + 1
                    
                    except Exception as e_inner:
                        self.logger.error(f"[{symbol}-{indicator_name}] Error: {e_inner}", exc_info=True)
                        self.metrics["indicator_errors"] = self.metrics.get("indicator_errors", 0) + 1
                
                # Store results for this symbol
                self.logger.debug(f"Processed indicators for {symbol}: {calculated_indicator_names_for_symbol}")
                if symbol_indicators:
                    all_results[symbol] = symbol_indicators.copy()  # Use copy to avoid reference issues
                    self.logger.info(f"Successfully calculated indicators for {symbol}: {list(symbol_indicators.keys())}")
                else:
                    all_results[symbol] = {}
                    self.logger.warning(f"No indicators calculated for {symbol}")
                
                self.last_calculation_time[symbol] = datetime.now().timestamp()
                
            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
                all_results[symbol] = {"error": f"Error processing symbol: {str(e)}"}
        
        # Calculation complete
        overall_duration_ms = (datetime.now() - overall_start_time).total_seconds() * 1000
        avg_time = overall_duration_ms / max(len(symbols), 1)
        self.logger.info(f"Calculated indicators for {len(symbols)} symbols in {overall_duration_ms:.2f}ms (avg: {avg_time:.2f}ms)")
        
        return all_results"""
    
    # Replace the old method with the new one
    pattern = r'(def calculate_all_indicators.*?:.*?)def'
    replacement = fixed_method + "\n\n    def"
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully updated {file_path} with fixed calculate_all_indicators method")

if __name__ == "__main__":
    fix_calculate_all_indicators()
