/*
 * Python bindings for the technical indicators
 */

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyDict};
use rust_decimal::Decimal;

use crate::technical::{
    SMA, EMA, WMA, MAType, MovingAverage, MACrossover, CrossoverSignal,
    calculate_sma, calculate_ema
};

/// Initialize the technical module
pub fn init_module(m: &PyModule) -> PyResult<()> {
    // Register functions
    m.add_function(wrap_pyfunction!(calc_sma, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ema, m)?)?;
    m.add_function(wrap_pyfunction!(detect_crossover, m)?)?;
    
    // Register indicator classes
    m.add_class::<PySMA>()?;
    m.add_class::<PyEMA>()?;
    m.add_class::<PyWMA>()?;
    m.add_class::<PyMACrossover>()?;
    
    // Register enums
    m.add("SMA_TYPE", "simple")?;
    m.add("EMA_TYPE", "exponential")?;
    m.add("WMA_TYPE", "weighted")?;
    
    m.add("BULLISH_CROSSOVER", "bullish_crossover")?;
    m.add("BEARISH_CROSSOVER", "bearish_crossover")?;
    m.add("NO_SIGNAL", "no_signal")?;
    
    Ok(())
}

/// Python wrapper for SMA
#[pyclass]
struct PySMA {
    inner: SMA,
}

#[pymethods]
impl PySMA {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: SMA::new(period),
        }
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        let decimal = Decimal::try_from(value).ok()?;
        self.inner.update(decimal).map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn current(&self) -> Option<f64> {
        self.inner.current().map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn reset(&mut self) {
        self.inner.reset();
    }
    
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }
}

/// Python wrapper for EMA
#[pyclass]
struct PyEMA {
    inner: EMA,
}

#[pymethods]
impl PyEMA {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: EMA::new(period),
        }
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        let decimal = Decimal::try_from(value).ok()?;
        self.inner.update(decimal).map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn current(&self) -> Option<f64> {
        self.inner.current().map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn reset(&mut self) {
        self.inner.reset();
    }
    
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }
}

/// Python wrapper for WMA
#[pyclass]
struct PyWMA {
    inner: WMA,
}

#[pymethods]
impl PyWMA {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: WMA::new(period),
        }
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        let decimal = Decimal::try_from(value).ok()?;
        self.inner.update(decimal).map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn current(&self) -> Option<f64> {
        self.inner.current().map(|d| d.to_f64().unwrap_or(0.0))
    }
    
    fn reset(&mut self) {
        self.inner.reset();
    }
    
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }
}

/// Python wrapper for MACrossover
#[pyclass]
struct PyMACrossover {
    inner: MACrossover,
}

#[pymethods]
impl PyMACrossover {
    #[new]
    fn new(fast_period: usize, slow_period: usize, fast_type: &str, slow_type: &str) -> PyResult<Self> {
        // Convert string types to MAType
        let fast_ma_type = match fast_type {
            "simple" => MAType::Simple,
            "exponential" => MAType::Exponential,
            "weighted" => MAType::Weighted,
            _ => return Err(PyValueError::new_err(format!("Invalid MA type: {}", fast_type)))
        };
        
        let slow_ma_type = match slow_type {
            "simple" => MAType::Simple,
            "exponential" => MAType::Exponential,
            "weighted" => MAType::Weighted,
            _ => return Err(PyValueError::new_err(format!("Invalid MA type: {}", slow_type)))
        };
        
        Ok(Self {
            inner: MACrossover::new(fast_period, slow_period, fast_ma_type, slow_ma_type),
        })
    }
    
    fn update(&mut self, price: f64) -> PyResult<String> {
        let decimal = Decimal::try_from(price)
            .map_err(|e| PyValueError::new_err(format!("Invalid price: {}", e)))?;
        
        let signal = self.inner.update(decimal);
        
        let result = match signal {
            CrossoverSignal::BullishCrossover => "bullish_crossover",
            CrossoverSignal::BearishCrossover => "bearish_crossover",
            CrossoverSignal::NoSignal => "no_signal",
        };
        
        Ok(result.to_string())
    }
    
    fn last_signal(&self) -> Option<String> {
        self.inner.last_signal().map(|signal| {
            match signal {
                CrossoverSignal::BullishCrossover => "bullish_crossover",
                CrossoverSignal::BearishCrossover => "bearish_crossover",
                CrossoverSignal::NoSignal => "no_signal",
            }.to_string()
        })
    }
    
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Calculate SMA for a list of values
#[pyfunction]
fn calc_sma(values: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    
    // Convert values to Decimal
    let decimal_values: Vec<Decimal> = values
        .iter()
        .filter_map(|&v| Decimal::try_from(v).ok())
        .collect();
    
    // Calculate SMA
    let sma_values = calculate_sma(&decimal_values, period);
    
    // Convert back to f64
    let result: Vec<f64> = sma_values
        .iter()
        .map(|d| d.to_f64().unwrap_or(0.0))
        .collect();
    
    Ok(result)
}

/// Calculate EMA for a list of values
#[pyfunction]
fn calc_ema(values: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    if period == 0 {
        return Err(PyValueError::new_err("Period must be greater than 0"));
    }
    
    // Convert values to Decimal
    let decimal_values: Vec<Decimal> = values
        .iter()
        .filter_map(|&v| Decimal::try_from(v).ok())
        .collect();
    
    // Calculate EMA
    let ema_values = calculate_ema(&decimal_values, period);
    
    // Convert back to f64
    let result: Vec<f64> = ema_values
        .iter()
        .map(|d| d.to_f64().unwrap_or(0.0))
        .collect();
    
    Ok(result)
}

/// Detect crossover between two moving averages
#[pyfunction]
fn detect_crossover(
    fast_ma_values: Vec<f64>,
    slow_ma_values: Vec<f64>
) -> PyResult<Vec<String>> {
    if fast_ma_values.len() != slow_ma_values.len() {
        return Err(PyValueError::new_err(
            "Fast and slow MA arrays must have the same length"
        ));
    }
    
    let mut result = Vec::with_capacity(fast_ma_values.len());
    
    // Need at least 2 values to detect a crossover
    if fast_ma_values.len() < 2 {
        return Ok(vec!["no_signal".to_string(); fast_ma_values.len()]);
    }
    
    // First value has no previous to compare
    result.push("no_signal".to_string());
    
    // Check for crossovers in the rest of the values
    for i in 1..fast_ma_values.len() {
        let prev_fast = fast_ma_values[i - 1];
        let prev_slow = slow_ma_values[i - 1];
        let curr_fast = fast_ma_values[i];
        let curr_slow = slow_ma_values[i];
        
        if curr_fast > curr_slow && prev_fast <= prev_slow {
            result.push("bullish_crossover".to_string());
        } else if curr_fast < curr_slow && prev_fast >= prev_slow {
            result.push("bearish_crossover".to_string());
        } else {
            result.push("no_signal".to_string());
        }
    }
    
    Ok(result)
} 