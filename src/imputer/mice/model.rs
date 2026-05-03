use super::linear_regression::{Solver, SolverType};
use crate::imputer::SimpleImputer;
use crate::utils::pyany_to_vec;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct Mice {
    n_iterations: usize,
    backend: SolverType,
    is_fitted: bool,
}

#[pymethods]
impl Mice {
    #[new]
    #[pyo3(signature = (n_iterations=15, backend="Linear"))]
    pub fn new(n_iterations: usize, backend: &str) -> Mice {
        let backend = match backend.to_lowercase().as_str() {
            "linear" => SolverType::Linear,
            "ridge" => SolverType::Ridge,
            "bayesian" => SolverType::Bayesian,
            _ => panic!("Solver {backend} not supported!"),
        };
        Mice {
            n_iterations,
            backend: backend,
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        {
            let mut inner = slf.borrow_mut(py);
            // inner.solve(&Array2::from_shape_vec((nrows, ncols), vec).unwrap());
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
    fn _impute(&self, data: &Array2<f64>) {
        // initial mean imputation
        let _imputed = SimpleImputer::new().fit_impl(&data).impute(&data);
        for _i in 0..self.n_iterations {}
    }

    // fn solve(&self, data: &Array2<f64>) {
    //     data.column(0);
    // }
}
