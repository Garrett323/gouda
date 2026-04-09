use crate::utils::{Data, pyany_to_vec};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct Simple {
    sample_means: Option<Vec<f64>>,
    sample_mode: Option<Vec<f64>>,
    is_fitted: bool,
}

#[pymethods]
impl Simple {
    #[new]
    pub fn new() -> Simple {
        Simple {
            sample_means: None,
            sample_mode: None,
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        {
            let mut inner = slf.borrow_mut(py);
            let data = Data::new_colmayor(nrows, ncols, &vec);
            let means = inner.get_means(&data);
            inner.sample_means = Some(means);
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
                "Imputer is not fitted",
            )));
        }
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        let imputed = self.impute(Data::new_rowmayor(nrows, ncols, &vec));
        // return python object
        let array = ndarray::Array2::from_shape_vec((nrows, ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl Simple {
    fn get_means(&self, data: &Data) -> Vec<f64> {
        let mut means = vec![0.0; data.ncols];
        means
    }

    fn impute(&self, data: Data) -> Vec<f64> {
        data[0].to_vec()
    }
}
