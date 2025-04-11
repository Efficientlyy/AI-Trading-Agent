// Simple Rust extension with C-compatible API
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

// C-compatible function to add two numbers
#[no_mangle]
pub extern "C" fn add_numbers(a: i64, b: i64) -> i64 {
    a + b
}

// C-compatible function to multiply two numbers
#[no_mangle]
pub extern "C" fn multiply_numbers(a: f64, b: f64) -> f64 {
    a * b
}

// C-compatible function to run a backtest
#[no_mangle]
pub extern "C" fn run_backtest_c(data_json: *const c_char, config_json: *const c_char) -> *mut c_char {
    // Convert C strings to Rust strings
    let data_str = unsafe {
        if data_json.is_null() {
            return ptr::null_mut();
        }
        match CStr::from_ptr(data_json).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };
    
    let config_str = unsafe {
        if config_json.is_null() {
            return ptr::null_mut();
        }
        match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };
    
    // For now, just return a simple result
    // In a real implementation, we would parse the JSON, run the backtest, and return the result
    let result = format!("{{\"final_capital\": 11500.0, \"metrics\": {{\"total_return\": 0.15, \"sharpe_ratio\": 1.2, \"max_drawdown\": 0.05}}}}");
    
    // Convert Rust string to C string
    match CString::new(result) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// Function to free memory allocated for C strings
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}
