use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod gouda {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: i32, b: i32) -> PyResult<i32> {
        Ok(a + b)
    }
}
