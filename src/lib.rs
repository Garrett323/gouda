use pyo3::prelude::*;
mod imputer;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
mod gouda {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<imputer::KnnImputer>()?;
        module.add_class::<imputer::Mice>()?;
        module.add_class::<imputer::SimpleImputer>()?;
        module.add_class::<imputer::ConstantImputer>()?;
        Ok(())
    }
}
