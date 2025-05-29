// C-compatible API for the backtesting functionality
// This approach avoids using PyO3 and its Python detection mechanism

use std::ffi::{CStr, CString};
use std::os::raw::{c_char};
use std::collections::HashMap;
use serde_json;

use crate::backtesting::{Order, OHLCVBar, BacktestConfig, run_backtest};

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

/// C-compatible function for running a backtest
#[no_mangle]
pub extern "C" fn run_backtest_c(
    data_json: *const c_char,
    orders_json: *const c_char,
    config_json: *const c_char,
) -> *mut c_char {
    // Convert C strings to Rust strings
    let data_str = unsafe {
        match c_str_to_string(data_json) {
            Ok(s) => s,
            Err(e) => return string_to_c_str(format!("{{\"error\": \"{}\"}}", e)),
        }
    };
    
    let orders_str = unsafe {
        match c_str_to_string(orders_json) {
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
    let data: Result<HashMap<String, Vec<OHLCVBar>>, _> = serde_json::from_str(&data_str);
    let data = match data {
        Ok(d) => d,
        Err(e) => return string_to_c_str(format!("{{\"error\": \"Failed to parse data JSON: {}\"}}", e)),
    };
    
    let orders: Result<Vec<Order>, _> = serde_json::from_str(&orders_str);
    let orders = match orders {
        Ok(o) => o,
        Err(e) => return string_to_c_str(format!("{{\"error\": \"Failed to parse orders JSON: {}\"}}", e)),
    };
    
    let config: Result<BacktestConfig, _> = serde_json::from_str(&config_str);
    let config = match config {
        Ok(c) => c,
        Err(e) => return string_to_c_str(format!("{{\"error\": \"Failed to parse config JSON: {}\"}}", e)),
    };
    
    // Run the backtest
    let result = run_backtest(data, orders, config);
    
    let result_json = serde_json::to_string(&result)
        .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize result to JSON: {}\"}}", e));
    
    // Convert to JSON string and return as C string
    string_to_c_str(result_json)
}

#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = CString::from_raw(ptr);
        }
    }
}
