use pyo3::prelude::*;
mod constant;
mod knn;
mod mice;
mod simple;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
mod gouda {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<knn::KnnImputer>()?;
        module.add_class::<mice::Mice>()?;
        module.add_class::<simple::SimpleImputer>()?;
        module.add_class::<constant::ConstantImputer>()?;
        Ok(())
    }
}
