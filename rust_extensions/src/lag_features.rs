// use rayon::prelude::*;
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
    // Collect (index, value) pairs first
    let indexed_values: Vec<(usize, f64)> = lags
        .iter()
        .enumerate()
        .flat_map(|(i, &lag)| {
            let lag_usize = lag as usize;
            let inner_collected: Vec<(usize, f64)> = (lag_usize..data_len).map(move |j| {
                let result_idx = j * lags_len + i;
                (result_idx, data[j - lag_usize])
            }).collect(); // Collect into a Vec
            inner_collected
        })
        .collect();

    // Fill the result slice (sequentially) from the collected indexed values
    for (idx, val) in indexed_values {
        if idx < result.len() { // Basic bounds check
            result[idx] = val;
        }
    }
    
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
    // Collect (index, value) pairs first
    let indexed_values: Vec<(usize, f64)> = periods
        .iter()
        .enumerate()
        .flat_map(|(i, &period)| {
            let period_usize = period as usize;
            let inner_collected: Vec<(usize, f64)> = (period_usize..data_len).map(move |j| {
                let result_idx = j * periods_len + i;
                (result_idx, data[j] - data[j - period_usize])
            }).collect(); // Collect into a Vec
            inner_collected
        })
        .collect();

    // Fill the result slice (sequentially) from the collected indexed values
    for (idx, val) in indexed_values {
        if idx < result.len() { // Basic bounds check
            result[idx] = val;
        }
    }
    
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
    // Collect (index, value) pairs first
    let indexed_values: Vec<(usize, f64)> = periods
        .iter()
        .enumerate()
        .flat_map(|(i, &period)| {
            let period_usize = period as usize;
            let inner_collected: Vec<(usize, f64)> = (period_usize..data_len).map(move |j| {
                let result_idx = j * periods_len + i;
                let previous_value = data[j - period_usize];
                let current_value = data[j];
                let value = if previous_value != 0.0 {
                    (current_value - previous_value) / previous_value
                } else {
                    f64::NAN
                };
                (result_idx, value)
            }).collect(); // Collect into a Vec
            inner_collected
        })
        .collect();

    // Fill the result slice (sequentially) from the collected indexed values
    for (idx, val) in indexed_values {
        if idx < result.len() { // Basic bounds check
            result[idx] = val;
        }
    }
    
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
    
    // Calculate (result_idx, value) pairs in parallel
    let indexed_values: Vec<(usize, f64)> = windows // windows is &[i32]
        .iter() // Produces &i32
        .enumerate() // Produces (usize, &i32), where usize is index `i`
        .flat_map(|(i, &window_val)| { // i is window_idx, window_val is the size from windows slice
            let window_usize = window_val as usize;
            // This inner part is sequential for each window configuration
            let inner_collected: Vec<(usize, f64)> = (window_usize - 1..data_len).map(move |j| { // j is data_idx (time point)
                let result_idx = j * windows_len + i; // Calculate flat index for result array
                let current_window_data_slice = &data[j - (window_usize - 1)..=j];
                
                let value = match feature_type {
                    0 => { // Mean
                        current_window_data_slice.iter().sum::<f64>() / window_usize as f64
                    }
                    1 => { // Standard deviation
                        let mean = current_window_data_slice.iter().sum::<f64>() / window_usize as f64;
                        let variance = current_window_data_slice.iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>() / window_usize as f64;
                        variance.sqrt()
                    }
                    2 => { // Min
                        current_window_data_slice.iter().fold(f64::INFINITY, |acc, &val| acc.min(val))
                    }
                    3 => { // Max
                        current_window_data_slice.iter().fold(f64::NEG_INFINITY, |acc, &val| acc.max(val))
                    }
                    4 => { // Sum
                        current_window_data_slice.iter().sum()
                    }
                    _ => unreachable!(), // Already checked for invalid feature types
                };
                (result_idx, value)
            }).collect(); // Collect into a Vec
            inner_collected
        })
        .collect();

    // Fill the result slice (sequentially) from the collected indexed values
    for (idx, val) in indexed_values {
        if idx < result.len() { // Basic bounds check
            result[idx] = val;
        }
    }
    
    0 // Success
}
