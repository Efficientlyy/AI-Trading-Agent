use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ta::indicators::SimpleMovingAverage;
use ta::Next;

// Import feature engineering modules
mod features;
pub use features::*;

// Import lag features module
mod lag_features;

// C-compatible function for calculating SMA
#[no_mangle]
pub extern "C" fn calculate_sma_c(data_ptr: *const f64, data_len: usize, period: usize, result_ptr: *mut f64) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let result = unsafe { std::slice::from_raw_parts_mut(result_ptr, data_len) };
    
    // Create SMA indicator
    let sma_indicator_result = SimpleMovingAverage::new(period);
    if sma_indicator_result.is_err() {
        return -2;
    }
    
    let mut sma_indicator = sma_indicator_result.unwrap();

    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }

    // Calculate SMA for each data point
    for (i, &val) in data.iter().enumerate() {
        let sma_val = sma_indicator.next(val);
        
        // Only store values after we have enough data points
        if i >= period - 1 {
            result[i] = sma_val;
        }
    }

    0 // Success
}

// Free a C string allocated by Rust
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut std::os::raw::c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

/// Python module for rust extensions
#[pymodule]
fn rust_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register feature engineering functions
    features::register(_py, m)?;
    
    Ok()
}
