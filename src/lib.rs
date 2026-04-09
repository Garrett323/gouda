use pyo3::prelude::*;
mod knn;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
mod gouda {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<knn::KnnImputer>()?;
        Ok(())
    }
}
