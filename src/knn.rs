use crate::utils::pyany_to_vec;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ops::Index;

#[pyclass]
pub struct KnnImputer {
    #[pyo3(get, set)]
    k: usize,
    nrows: usize,
    ncols: usize,
    data: Option<Vec<f64>>,
    is_fitted: bool,
}

#[pymethods]
impl KnnImputer {
    #[new]
    #[pyo3(signature = (k=5))]
    pub fn new(k: usize) -> KnnImputer {
        KnnImputer {
            k,
            nrows: 0,
            ncols: 0,
            data: None,
            is_fitted: false,
        }
    }

    pub fn test(&self) -> i32 {
        let data = self.data.as_ref().unwrap();
        println!("{}, {}", data[0], data[1]);
        println!("shape: {}, {}", self.nrows, self.ncols);
        90
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        {
            let mut inner = slf.borrow_mut(py);
            inner.data = Some(vec);
            inner.nrows = nrows;
            inner.ncols = ncols;
            inner.is_fitted = true;
        } // dropping inner here (releasing the mutex)
        Ok(slf)
    }

    pub fn transform<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // check if fitted
        if !self.is_fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Imputer is not fitted",
            )));
        }
        // actual method
        let imputed = self.brute_force();
        // return python object
        let array = ndarray::Array2::from_shape_vec((self.nrows, self.ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl KnnImputer {
    fn brute_force(&self) -> Vec<f64> {
        let data = self.data.as_ref().unwrap();
        let mut imputed = self.data.as_ref().unwrap().clone();
        for (i, n) in data.iter().enumerate() {
            if n.is_nan() {
                // figure out point and impute
                let row = i / self.ncols;
                let col = i % self.ncols;
                let mut distances = Vec::with_capacity(self.nrows);
                for r in 0..self.nrows {
                    distances.push(self.nan_euclid(&self[row], &self[r]));
                }
                let mut indices: Vec<usize> = (0..self.nrows).collect();
                indices.sort_by(|&a, &b| distances[a].total_cmp(&distances[b]));
                indices.truncate(self.k);
                imputed[i] = self.average(&indices, col);
            }
        }
        imputed
    }

    // TODO:
    // Make it work with passing NaNs
    // write tests
    fn nan_euclid(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut nnans = 0;
        for (x, y) in a.iter().zip(b) {
            if x.is_nan() || y.is_nan() {
                nnans += 1;
            } else {
                total += (x - y).powi(2);
            }
        }
        total * ((self.ncols - nnans) as f64 / self.ncols as f64)
    }

    // TODO:
    // Implement weighting schemes
    fn average(&self, indices: &[usize], col: usize) -> f64 {
        0.0
    }
}

impl Index<usize> for KnnImputer {
    type Output = [f64];

    fn index(&self, row: usize) -> &[f64] {
        let offset = row * self.ncols;
        &self.data.as_ref().unwrap()[offset..offset + self.ncols]
    }
}
