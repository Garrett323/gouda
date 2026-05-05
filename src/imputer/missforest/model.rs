use crate::utils::{constants::NOT_FITTED_ERR, pyany_to_vec};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct MissForest {
    is_fitted: bool,
}

#[pymethods]
impl MissForest {
    #[new]
    pub fn new() -> MissForest {
        MissForest { is_fitted: false }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        {
            let mut inner = slf.borrow_mut(py);
            let data: Array2<f64> = Array2::from_shape_vec((nrows, ncols), vec).unwrap();
            inner.fit_impl(&data);
            inner.is_fitted = true;
        } // dropping inner here (releasing the mutex)
        Ok(slf)
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // check if fitted
        if !self.is_fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "{}",
                NOT_FITTED_ERR
            )));
        }
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        let imputed = self.impute(
            &Array2::from_shape_vec((nrows, ncols), vec)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        );
        // return python object
        Ok(imputed.into_pyarray(py))
    }
}

impl MissForest {
    fn fit_impl(&mut self, data: &Array2<f64>) -> &Self {
        self
    }
    fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        data.clone()
    }
}
