use crate::utils::{Data, pyany_to_vec};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct Mice {
    n_iterations: usize,
    is_fitted: bool,
}

#[pymethods]
impl Mice {
    #[new]
    #[pyo3(signature = (n_iterations=15))]
    pub fn new(n_iterations: usize) -> Mice {
        Mice {
            n_iterations,
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        {
            let mut inner = slf.borrow_mut(py);
            inner.solve(&Data::new(nrows, ncols, &vec));
            inner.is_fitted = true;
        } // dropping inner here (releasing the mutex)
        Ok(slf)
    }

    // TODO:
    // Does currently nothing
    // implement stuff
    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // check if fitted
        if !self.is_fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Imputer is not fitted",
            )));
        }
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        let imputed = vec;
        // return python object
        let array = ndarray::Array2::from_shape_vec((nrows, ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl Mice {
    fn solve(&self, data: &Data) {
        data.get_col(0);
    }
}
