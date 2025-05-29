"""
This script directly tests the LAG_FEATURES_RS_ functionality with a simplified setup.
"""
import pandas as pd
import sys
import os
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the indicator engine directly
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

def main():
    print("\n==== Starting Direct LAG_FEATURES_RS_ Debug ====")
    
    # Create simplified test data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': range(10, 20),
    }, index=dates)
    
    # Specify the lags we want to test
    lags = [1, 2, 3]
    
    # Create a minimal config for LAG_FEATURES_RS_
    config = {
        "features": {
            "LAG_FEATURES_RS_debug": {
                "enabled": True,
                "lags_to_calculate": lags,
                "source_column": "close"
            }
        }
    }
    
    # Create the engine with our custom config
    print("\nCreating IndicatorEngine...")
    engine = IndicatorEngine(config)
    
    # Print registered indicators
    print(f"\nRegistered indicators: {list(engine.indicators.keys())}")
    
    # Check the LAG_FEATURES_RS_ indicator configuration
    if "LAG_FEATURES_RS_debug" in engine.indicators:
        print("\nLAG_FEATURES_RS_debug indicator configuration:")
        lag_rs_config = engine.indicators["LAG_FEATURES_RS_debug"]
        for key, value in lag_rs_config.items():
            if key != 'calculator':  # Skip calculator function
                print(f"  {key}: {value}")
    
    # Modify the calculate_all_indicators method to provide more detailed logging
    original_calculate = engine.calculate_all_indicators
    
    def debug_calculate(market_data, symbols):
        print("\n==== Starting calculate_all_indicators with Debug ====")
        print(f"Processing symbols: {symbols}")
        
        # Initialize the results dictionary directly
        all_results = {symbol: {} for symbol in symbols}
        
        for symbol in symbols:
            print(f"\nProcessing symbol: {symbol}")
            
            # Get the DataFrame for this symbol
            df = market_data[symbol]
            print(f"DataFrame shape: {df.shape}")
            
            # Track which indicators we process
            processed_indicators = []
            
            # Create a dictionary to store indicator results for this symbol
            symbol_indicators = {}
            
            # Process each registered indicator
            print("\nProcessing indicators:")
            for indicator_name, settings in engine.indicators.items():
                print(f"  Indicator: {indicator_name}")
                
                # Check if this indicator is enabled
                enabled = settings.get("enabled", False)
                print(f"  Enabled: {enabled}")
                
                if not enabled:
                    print(f"  Skipping disabled indicator: {indicator_name}")
                    continue
                
                # Get the calculator function for this indicator
                calculator = settings.get("calculator")
                if not calculator:
                    print(f"  No calculator for indicator: {indicator_name}")
                    continue
                
                print(f"  Using calculator: {calculator.__name__ if hasattr(calculator, '__name__') else type(calculator)}")
                
                # For LAG_FEATURES_RS_ indicators, we need special handling
                if indicator_name.startswith("LAG_FEATURES_RS_"):
                    print(f"\n  Special processing for LAG_FEATURES_RS_ indicator: {indicator_name}")
                    
                    # Get the parameters for this indicator
                    source_column = settings.get("source_column", "close")
                    lags_to_calculate = settings.get("lags_to_calculate", settings.get("lags", []))
                    
                    print(f"  Source column: {source_column}")
                    print(f"  Lags to calculate: {lags_to_calculate}")
                    
                    try:
                        # Calculate the lag features
                        print("  Calling _calculate_lag_features directly...")
                        lag_features_dict = engine._calculate_lag_features(
                            df=df.copy(),
                            lags=lags_to_calculate,
                            source_column=source_column
                        )
                        
                        # Check the result
                        print(f"  Result type: {type(lag_features_dict)}")
                        if isinstance(lag_features_dict, dict):
                            print(f"  Result keys: {list(lag_features_dict.keys())}")
                            
                            # Convert to DataFrame as in the original code
                            lag_features_df = pd.DataFrame(lag_features_dict)
                            print(f"  DataFrame shape: {lag_features_df.shape}")
                            print(f"  DataFrame columns: {lag_features_df.columns.tolist()}")
                            
                            # Store in symbol_indicators
                            symbol_indicators[indicator_name] = lag_features_df
                            processed_indicators.append(indicator_name)
                            print(f"  Added {indicator_name} to symbol_indicators")
                        else:
                            print(f"  Result is not a dictionary: {lag_features_dict}")
                    except Exception as e:
                        print(f"  Error calculating lag features: {e}")
            
            # After processing all indicators, update all_results
            print(f"\nProcessed indicators: {processed_indicators}")
            print(f"Symbol indicators keys: {list(symbol_indicators.keys())}")
            
            # Make sure all_results is updated correctly
            all_results[symbol] = symbol_indicators.copy()
            print(f"Final all_results[{symbol}] keys: {list(all_results[symbol].keys())}")
        
        print("\n==== Final Results ====")
        for symbol, indicators in all_results.items():
            print(f"Symbol: {symbol}")
            print(f"  Indicators: {list(indicators.keys())}")
        
        return all_results
    
    # Replace the method with our debug version
    engine.calculate_all_indicators = debug_calculate
    
    # Now run the calculation on our test data
    market_data = {"TEST_SYMBOL": data}
    symbols = ["TEST_SYMBOL"]
    
    print("\n==== Running calculate_all_indicators ====")
    results = engine.calculate_all_indicators(market_data, symbols)
    
    # Verify the results
    print("\n==== Verifying Results ====")
    if "TEST_SYMBOL" in results:
        print("✓ TEST_SYMBOL found in results")
        
        if "LAG_FEATURES_RS_debug" in results["TEST_SYMBOL"]:
            print("✓ LAG_FEATURES_RS_debug found in results")
            
            # Check the result type and structure
            result = results["TEST_SYMBOL"]["LAG_FEATURES_RS_debug"]
            print(f"Result type: {type(result)}")
            
            if isinstance(result, pd.DataFrame):
                print(f"DataFrame shape: {result.shape}")
                print(f"DataFrame columns: {result.columns.tolist()}")
                print("First few rows:")
                print(result.head(3))
            else:
                print(f"Result is not a DataFrame: {result}")
        else:
            print("✗ LAG_FEATURES_RS_debug NOT found in results")
            print(f"Available keys: {list(results['TEST_SYMBOL'].keys())}")
    else:
        print("✗ TEST_SYMBOL NOT found in results")
    
    print("\n==== Debug Complete ====")

if __name__ == "__main__":
    main()
