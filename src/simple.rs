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
                "Imputer is not fitted",
            )));
        }
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
        let imputed = self.impute(&Data::new_rowmayor(nrows, ncols, &vec));
        // return python object
        let array = ndarray::Array2::from_shape_vec((nrows, ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl Simple {
    fn fit_impl(&mut self, data: &Data) -> &Self {
        let means = self.get_means(&data);
        self.sample_means = Some(means);
        return self;
    }
    fn get_means(&self, data: &Data) -> Vec<f64> {
        let mut means = vec![0.0; data.ncols];
        for i in 0..data.ncols {
            for entry in data.get_col(i) {
                if entry.is_nan() {
                    continue;
                }
                means[i] += entry;
            }
            means[i] /= data.nrows as f64;
        }
        means
    }

    fn impute(&self, data: &Data) -> Vec<f64> {
        // TODO:
        // check if rowmayor data is there and create if needed
        // return imputed vec
        data[0].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*; // has access to everything, including private

    #[test]
    fn test_means() {
        let mut simple = Simple::new();
        let data = Data::new_colmayor(5, 5, DATA);
        let imputed = simple.fit_impl(&data).impute(&data);
        for (exp, imp) in EXPECTED.iter().zip(imputed) {
            let diff = exp - imp;
            assert!(diff.abs() < 1e-10, "Expected: {}; Actual {}\n", exp, imp);
        }
    }

    const DATA: &[f64] = &[
        0.76052103,
        f64::NAN,
        0.4094729,
        0.9573324,
        f64::NAN,
        0.27839605,
        0.7338148,
        0.98359227,
        0.98189233,
        0.45384631,
        f64::NAN,
        0.22129885,
        0.8863533,
        0.50595314,
        0.5011135,
        f64::NAN,
        0.32309935,
        0.64573872,
        f64::NAN,
        f64::NAN,
        0.9317995,
        0.51597243,
        0.38054457,
        0.62366235,
        0.12229672,
        0.90547984,
        f64::NAN,
        0.68424979,
        0.55400964,
        0.55284803,
        0.68846839,
        0.53889275,
        0.44453843,
        0.43416536,
        0.18575075,
        0.13333331,
        0.8772666,
        0.64398646,
        f64::NAN,
        0.90529859,
        0.69819416,
        0.65251852,
        0.39663618,
        0.65702538,
        f64::NAN,
    ];
    const EXPECTED: &[f64] = &[
        1.0382174099275148,
        0.7912650658744038,
        0.0,
        0.41309417813332494,
        0.7905951937189456,
        0.2763805321428371,
        0.7077017509263522,
        1.042753531574897,
        0.8646734303095986,
    ];
}
