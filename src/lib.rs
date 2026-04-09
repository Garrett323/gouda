use pyo3::prelude::*;
mod knn;
mod mice;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
mod gouda {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<knn::KnnImputer>()?;
        module.add_class::<mice::Mice>()?;
        Ok(())
    }
}
