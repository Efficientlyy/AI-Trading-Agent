use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use serde::{Serialize, Deserialize};
use serde_json;

// Simple data structures for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub final_capital: f64,
    pub metrics: PerformanceMetrics,
}

// Helper function to safely convert C string to Rust string
unsafe fn c_str_to_string(c_str: *const c_char) -> Result<String, String> {
    if c_str.is_null() {
        return Err("Null pointer passed as C string".to_string());
    }
    
    CStr::from_ptr(c_str)
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("Invalid UTF-8 in C string: {}", e))
}

// Helper function to safely convert Rust string to C string
fn string_to_c_str(s: String) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

// Simple backtesting function that calculates returns based on a moving average crossover strategy
fn run_simple_backtest(data: Vec<OHLCVBar>, config: BacktestConfig) -> BacktestResult {
    let mut capital = config.initial_capital;
    let mut position = 0.0;
    let mut returns = Vec::new();
    let mut equity_curve = Vec::new();
    
    // Simple moving average parameters
    let fast_period = 10;
    let slow_period = 30;
    
    // Calculate moving averages
    for i in slow_period..data.len() {
        let fast_ma = data[i-fast_period..i].iter().map(|bar| bar.close).sum::<f64>() / fast_period as f64;
        let slow_ma = data[i-slow_period..i].iter().map(|bar| bar.close).sum::<f64>() / slow_period as f64;
        
        // Trading logic: Buy when fast MA crosses above slow MA, sell when it crosses below
        if fast_ma > slow_ma && position == 0.0 {
            // Buy signal
            let price = data[i].close * (1.0 + config.slippage);
            let shares = (capital * 0.95) / price; // Use 95% of capital
            let cost = shares * price * (1.0 + config.commission_rate);
            
            if cost <= capital {
                position = shares;
                capital -= cost;
            }
        } else if fast_ma < slow_ma && position > 0.0 {
            // Sell signal
            let price = data[i].close * (1.0 - config.slippage);
            let proceeds = position * price * (1.0 - config.commission_rate);
            
            capital += proceeds;
            position = 0.0;
        }
        
        // Calculate daily return
        let portfolio_value = capital + (position * data[i].close);
        if i > slow_period {
            let prev_value = equity_curve.last().unwrap_or(&config.initial_capital);
            let daily_return = (portfolio_value - prev_value) / prev_value;
            returns.push(daily_return);
        }
        
        equity_curve.push(portfolio_value);
    }
    
    // Calculate performance metrics
    let final_capital = capital + (position * data.last().unwrap_or(&OHLCVBar {
        timestamp: 0,
        open: 0.0,
        high: 0.0,
        low: 0.0,
        close: 0.0,
        volume: 0.0
    }).close);
    
    let total_return = (final_capital - config.initial_capital) / config.initial_capital;
    
    // Calculate Sharpe ratio (simplified)
    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_dev = (returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();
    let sharpe_ratio = if std_dev > 0.0 { avg_return / std_dev * (252.0_f64.sqrt()) } else { 0.0 };
    
    // Calculate max drawdown
    let mut max_drawdown = 0.0;
    let mut peak = config.initial_capital;
    
    for &value in &equity_curve {
        if value > peak {
            peak = value;
        } else {
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
    }
    
    BacktestResult {
        final_capital,
        metrics: PerformanceMetrics {
            total_return,
            sharpe_ratio,
            max_drawdown,
        }
    }
}

// C-compatible function for running a backtest
#[no_mangle]
pub extern "C" fn run_backtest_c(
    data_json: *const c_char,
    config_json: *const c_char,
) -> *mut c_char {
    // Convert C strings to Rust strings
    let data_str = unsafe {
        match c_str_to_string(data_json) {
            Ok(s) => s,
            Err(e) => return string_to_c_str(format!("{{\"error\": \"{}\"}}", e)),
        }
    };
    
    let config_str = unsafe {
        match c_str_to_string(config_json) {
            Ok(s) => s,
            Err(e) => return string_to_c_str(format!("{{\"error\": \"{}\"}}", e)),
        }
    };
    
    // Parse JSON strings
    let data: Result<Vec<OHLCVBar>, _> = serde_json::from_str(&data_str);
    let data = match data {
        Ok(d) => d,
        Err(e) => return string_to_c_str(format!("{{\"error\": \"Failed to parse data JSON: {}\"}}", e)),
    };
    
    let config: Result<BacktestConfig, _> = serde_json::from_str(&config_str);
    let config = match config {
        Ok(c) => c,
        Err(e) => return string_to_c_str(format!("{{\"error\": \"Failed to parse config JSON: {}\"}}", e)),
    };
    
    // Run the backtest
    let result = run_simple_backtest(data, config);
    
    // Convert result to JSON string
    let result_json = match serde_json::to_string(&result) {
        Ok(json) => json,
        Err(e) => format!("{{\"error\": \"Failed to serialize result to JSON: {}\"}}", e),
    };
    
    // Convert to C string and return
    string_to_c_str(result_json)
}

// Free a C string allocated by Rust
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = CString::from_raw(ptr);
        }
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
