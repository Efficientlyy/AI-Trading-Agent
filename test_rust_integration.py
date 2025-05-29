import ai_trading_agent_rs

def main():
    print("Testing Rust extensions directly...")

    # Sample data
    series_data = [float(i) for i in range(1, 51)]  # e.g., 1.0, 2.0, ..., 50.0
    close_prices = series_data # For RSI, BB

    print(f"\nSample Series Data (first 5): {series_data[:5]}")

    # 1. Test Bollinger Bands from advanced_features.rs
    try:
        print("\nTesting create_bollinger_bands_rs...")
        bb_windows = [10, 20]
        bb_std_dev = 2.0
        bollinger_bands = ai_trading_agent_rs.create_bollinger_bands_rs(
            series=close_prices, 
            windows=bb_windows, 
            num_std_dev=bb_std_dev
        )
        print(f"Bollinger Bands (type: {type(bollinger_bands)}):")
        if isinstance(bollinger_bands, list) and bollinger_bands:
            print(f"  Number of window results: {len(bollinger_bands)}")
            for i, band_set_list in enumerate(bollinger_bands): # band_set is a list of 3 lists
                print(f"  For window {bb_windows[i]}:")
                if isinstance(band_set_list, list) and len(band_set_list) == 3:
                    print(f"    Upper (first 5): {band_set_list[0][:5] if band_set_list[0] else []}")
                    print(f"    Middle (first 5): {band_set_list[1][:5] if band_set_list[1] else []}")
                    print(f"    Lower (first 5): {band_set_list[2][:5] if band_set_list[2] else []}")
                else:
                    print(f"    Unexpected band_set_list type or structure: {type(band_set_list)}, content: {str(band_set_list)[:100]}")
        else:
            print(f"  Result: {str(bollinger_bands)[:200]}")

    except Exception as e:
        print(f"Error testing Bollinger Bands: {e}")

    # 2. Test EMA from moving_averages.rs
    try:
        print("\nTesting create_ema_features_rs...")
        ema_windows = [5, 10]
        ema_features = ai_trading_agent_rs.create_ema_features_rs(
            series=close_prices, 
            spans=ema_windows # Corrected keyword: spans
        )
        print(f"EMA Features (type: {type(ema_features)}):")
        if isinstance(ema_features, list) and ema_features:
            print(f"  Number of EMA series: {len(ema_features)}")
            for i, ema_list in enumerate(ema_features):
                print(f"  EMA for window {ema_windows[i]} (first 5): {ema_list[:5]}")
        else:
             print(f"  Result: {str(ema_features)[:200]}")
    except Exception as e:
        print(f"Error testing EMA: {e}")

    # 3. Test Lag Features from features.rs
    try:
        print("\nTesting create_lag_features_rs...")
        lags = [1, 2, 3]
        lag_features = ai_trading_agent_rs.create_lag_features_rs(
            series=series_data, 
            lags=lags
        )
        print(f"Lag Features (type: {type(lag_features)}):")
        if isinstance(lag_features, list) and lag_features:
            print(f"  Number of lag series: {len(lag_features)}")
            for i, lag_list in enumerate(lag_features):
                print(f"  Lag {lags[i]} series (first 5): {lag_list[:5]}")
        else:
            print(f"  Result: {str(lag_features)[:200]}")
    except Exception as e:
        print(f"Error testing Lag Features: {e}")
        
    # 4. Test RSI from advanced_features.rs
    try:
        print("\nTesting create_rsi_features_rs...")
        rsi_windows = [14, 28]
        rsi_values = ai_trading_agent_rs.create_rsi_features_rs(
            series=close_prices,
            windows=rsi_windows
        )
        print(f"RSI Values (type: {type(rsi_values)}):")
        if isinstance(rsi_values, list) and rsi_values:
            print(f"  Number of RSI series: {len(rsi_values)}")
            for i, rsi_list in enumerate(rsi_values):
                print(f"  RSI for window {rsi_windows[i]} (first 5): {rsi_list[:5]}")
        else:
            print(f"  Result: {str(rsi_values)[:200]}")
    except Exception as e:
        print(f"Error testing RSI: {e}")

    # 5. Test Stochastic Oscillator from advanced_features.rs
    try:
        print("\nTesting create_stochastic_oscillator_rs...")
        # Sample data for stochastic oscillator
        # Ensure highs >= lows, highs >= closes, lows <= closes for realistic data
        high_prices = [float(i + 2) for i in range(10, 60)] # e.g., 12.0, 13.0, ..., 61.0
        low_prices = [float(i) for i in range(10, 60)]    # e.g., 10.0, 11.0, ..., 59.0
        close_prices_stoch = [float(i + 1) for i in range(10, 60)] # e.g., 11.0, 12.0, ..., 60.0
        k_period = 14
        d_period = 3

        print(f"  Sample Highs (first 5): {high_prices[:5]}")
        print(f"  Sample Lows (first 5): {low_prices[:5]}")
        print(f"  Sample Closes (first 5): {close_prices_stoch[:5]}")

        stochastic_oscillator_result = ai_trading_agent_rs.create_stochastic_oscillator_rs(
            highs=high_prices,
            lows=low_prices,
            closes=close_prices_stoch,
            k_period=k_period,
            d_period=d_period
        )
        print(f"Stochastic Oscillator Result (type: {type(stochastic_oscillator_result)}):")
        if isinstance(stochastic_oscillator_result, dict):
            print(f"  Keys: {list(stochastic_oscillator_result.keys())}")
            percent_k = stochastic_oscillator_result.get("percent_k")
            percent_d = stochastic_oscillator_result.get("percent_d")
            
            print(f"  %K (type: {type(percent_k)}, length: {len(percent_k) if percent_k is not None else 'N/A'}):")
            if percent_k:
                print(f"    First 5 %K values: {percent_k[:5]}")
            
            print(f"  %D (type: {type(percent_d)}, length: {len(percent_d) if percent_d is not None else 'N/A'}):")
            if percent_d:
                print(f"    First 5 %D values: {percent_d[:5]}")
        else:
            print(f"  Result: {str(stochastic_oscillator_result)[:200]}")

    except Exception as e:
        print(f"Error testing Stochastic Oscillator: {e}")

if __name__ == "__main__":
    main()
