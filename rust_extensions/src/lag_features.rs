use rayon::prelude::*;
use std::slice;

/// Create lag features from a time series.
/// 
/// This function takes a time series and creates lag features for each lag period.
/// 
/// # Arguments
/// 
/// * `data_ptr` - Pointer to the input time series data
/// * `data_len` - Length of the input time series
/// * `lags_ptr` - Pointer to the lag periods
/// * `lags_len` - Number of lag periods
/// * `result_ptr` - Pointer to the output array (must be pre-allocated with size data_len * lags_len)
/// 
/// # Returns
/// 
/// * `0` on success
/// * `-1` if any pointer is null
/// * `-2` if any lag period is invalid (zero or negative)
/// 
/// # Safety
/// 
/// This function is unsafe because it dereferences raw pointers.
#[no_mangle]
pub extern "C" fn create_lag_features_c(
    data_ptr: *const f64,
    data_len: usize,
    lags_ptr: *const i32,
    lags_len: usize,
    result_ptr: *mut f64
) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || lags_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let lags = unsafe { slice::from_raw_parts(lags_ptr, lags_len) };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, data_len * lags_len) };
    
    // Check for invalid lag periods
    for &lag in lags {
        if lag <= 0 {
            return -2;
        }
    }
    
    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }
    
    // Calculate lag features in parallel for each lag period
    lags.par_iter().enumerate().for_each(|(i, &lag)| {
        let lag_usize = lag as usize;
        
        // For each lag, calculate the lag feature for each time point
        for j in lag_usize..data_len {
            // Calculate the index in the flattened result array
            let result_idx = j * lags_len + i;
            result[result_idx] = data[j - lag_usize];
        }
    });
    
    0 // Success
}

/// Create difference features from a time series.
/// 
/// This function takes a time series and creates difference features for each period.
/// 
/// # Arguments
/// 
/// * `data_ptr` - Pointer to the input time series data
/// * `data_len` - Length of the input time series
/// * `periods_ptr` - Pointer to the periods
/// * `periods_len` - Number of periods
/// * `result_ptr` - Pointer to the output array (must be pre-allocated with size data_len * periods_len)
/// 
/// # Returns
/// 
/// * `0` on success
/// * `-1` if any pointer is null
/// * `-2` if any period is invalid (zero or negative)
/// 
/// # Safety
/// 
/// This function is unsafe because it dereferences raw pointers.
#[no_mangle]
pub extern "C" fn create_diff_features_c(
    data_ptr: *const f64,
    data_len: usize,
    periods_ptr: *const i32,
    periods_len: usize,
    result_ptr: *mut f64
) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || periods_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let periods = unsafe { slice::from_raw_parts(periods_ptr, periods_len) };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, data_len * periods_len) };
    
    // Check for invalid periods
    for &period in periods {
        if period <= 0 {
            return -2;
        }
    }
    
    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }
    
    // Calculate difference features in parallel for each period
    periods.par_iter().enumerate().for_each(|(i, &period)| {
        let period_usize = period as usize;
        
        // For each period, calculate the difference feature for each time point
        for j in period_usize..data_len {
            // Calculate the index in the flattened result array
            let result_idx = j * periods_len + i;
            result[result_idx] = data[j] - data[j - period_usize];
        }
    });
    
    0 // Success
}

/// Create percentage change features from a time series.
/// 
/// This function takes a time series and creates percentage change features for each period.
/// 
/// # Arguments
/// 
/// * `data_ptr` - Pointer to the input time series data
/// * `data_len` - Length of the input time series
/// * `periods_ptr` - Pointer to the periods
/// * `periods_len` - Number of periods
/// * `result_ptr` - Pointer to the output array (must be pre-allocated with size data_len * periods_len)
/// 
/// # Returns
/// 
/// * `0` on success
/// * `-1` if any pointer is null
/// * `-2` if any period is invalid (zero or negative)
/// 
/// # Safety
/// 
/// This function is unsafe because it dereferences raw pointers.
#[no_mangle]
pub extern "C" fn create_pct_change_features_c(
    data_ptr: *const f64,
    data_len: usize,
    periods_ptr: *const i32,
    periods_len: usize,
    result_ptr: *mut f64
) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || periods_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let periods = unsafe { slice::from_raw_parts(periods_ptr, periods_len) };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, data_len * periods_len) };
    
    // Check for invalid periods
    for &period in periods {
        if period <= 0 {
            return -2;
        }
    }
    
    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }
    
    // Calculate percentage change features in parallel for each period
    periods.par_iter().enumerate().for_each(|(i, &period)| {
        let period_usize = period as usize;
        
        // For each period, calculate the percentage change feature for each time point
        for j in period_usize..data_len {
            // Calculate the index in the flattened result array
            let result_idx = j * periods_len + i;
            
            // Calculate percentage change
            let previous_value = data[j - period_usize];
            if previous_value != 0.0 {
                result[result_idx] = (data[j] - previous_value) / previous_value;
            }
            // If previous value is zero, result remains NaN
        }
    });
    
    0 // Success
}

/// Create rolling window features from a time series.
/// 
/// This function takes a time series and creates rolling window features for each window size.
/// 
/// # Arguments
/// 
/// * `data_ptr` - Pointer to the input time series data
/// * `data_len` - Length of the input time series
/// * `windows_ptr` - Pointer to the window sizes
/// * `windows_len` - Number of window sizes
/// * `feature_type` - Type of feature to calculate (0: mean, 1: std, 2: min, 3: max, 4: sum)
/// * `result_ptr` - Pointer to the output array (must be pre-allocated with size data_len * windows_len)
/// 
/// # Returns
/// 
/// * `0` on success
/// * `-1` if any pointer is null
/// * `-2` if any window size is invalid (zero or negative)
/// * `-3` if feature_type is invalid
/// 
/// # Safety
/// 
/// This function is unsafe because it dereferences raw pointers.
#[no_mangle]
pub extern "C" fn create_rolling_window_features_c(
    data_ptr: *const f64,
    data_len: usize,
    windows_ptr: *const i32,
    windows_len: usize,
    feature_type: i32,
    result_ptr: *mut f64
) -> i32 {
    // Safety check for null pointers
    if data_ptr.is_null() || windows_ptr.is_null() || result_ptr.is_null() {
        return -1;
    }
    
    // Convert raw pointers to Rust slices (unsafe)
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let windows = unsafe { slice::from_raw_parts(windows_ptr, windows_len) };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, data_len * windows_len) };
    
    // Check for invalid window sizes
    for &window in windows {
        if window <= 0 {
            return -2;
        }
    }
    
    // Check for invalid feature type
    if feature_type < 0 || feature_type > 4 {
        return -3;
    }
    
    // Initialize result array with NaN
    for val in result.iter_mut() {
        *val = f64::NAN;
    }
    
    // Calculate rolling window features in parallel for each window size
    windows.par_iter().enumerate().for_each(|(i, &window)| {
        let window_usize = window as usize;
        
        // For each window size, calculate the rolling window feature for each time point
        for j in window_usize - 1..data_len {
            // Calculate the index in the flattened result array
            let result_idx = j * windows_len + i;
            
            // Get the window data
            let window_data = &data[j - (window_usize - 1)..=j];
            
            // Calculate the feature based on the feature type
            match feature_type {
                0 => { // Mean
                    let sum: f64 = window_data.iter().sum();
                    result[result_idx] = sum / window_usize as f64;
                },
                1 => { // Standard deviation
                    let mean: f64 = window_data.iter().sum::<f64>() / window_usize as f64;
                    let variance: f64 = window_data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / window_usize as f64;
                    result[result_idx] = variance.sqrt();
                },
                2 => { // Min
                    result[result_idx] = window_data.iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                },
                3 => { // Max
                    result[result_idx] = window_data.iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                },
                4 => { // Sum
                    result[result_idx] = window_data.iter().sum();
                },
                _ => unreachable!(), // We already checked for invalid feature types
            }
        }
    });
    
    0 // Success
}
