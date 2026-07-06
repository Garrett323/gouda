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
        module.add_class::<imputer::SVMImputer>()?;
        module.add_class::<imputer::Mice>()?;
        module.add_class::<imputer::SimpleImputer>()?;
        module.add_class::<imputer::ConstantImputer>()?;
        module.add_class::<imputer::MissForest>()?;
        Ok(())
    }

    #[pyfunction]
    fn raise_if_nan_col(data: &Bound<'_, PyAny>) -> PyResult<()> {
        let (arr, _out, _enc) =
            utils::pyany_to_vec(data, &Some(utils::StringEncoding::LabelEncoding))?;
        utils::raise_if_nan_col(arr.view())?;
        Ok(())
    }
}
