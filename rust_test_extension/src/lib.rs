use pyo3::prelude::*;

/// A simple function that adds two numbers
#[pyfunction]
fn add_numbers(a: i64, b: i64) -> PyResult<i64> {
    Ok(a + b)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_test_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    Ok(())
}
