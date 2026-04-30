use crate::utils::{Matrix, constants::NOT_FITTED_ERR, pyany_to_vec};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct ConstantImputer {
    value: f64,
    is_fitted: bool,
}

#[pymethods]
impl ConstantImputer {
    #[new]
    pub fn new(value: f64) -> ConstantImputer {
        ConstantImputer {
            value,
            is_fitted: false,
        }
    }

    #[staticmethod]
    pub fn zero() -> ConstantImputer {
        ConstantImputer {
            value: 0.0,
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, _data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        {
            let mut inner = slf.borrow_mut(py);
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
        let imputed = self.impute(&Matrix::new(vec, nrows, ncols));
        // return python object
        let array = ndarray::Array2::from_shape_vec((nrows, ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl ConstantImputer {
    fn impute(&self, data: &Matrix) -> Vec<f64> {
        let data: &[f64] = data;
        let mut imputed = vec![self.value; data.len()];
        for (i, e) in data.iter().enumerate() {
            if !e.is_nan() {
                imputed[i] = *e;
            }
        }
        imputed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impute() {
        let imputer = ConstantImputer::new(7.0);
        let data = Matrix::new(DATA.to_vec(), 5, 5);
        let imputed = imputer.impute(&data);
        for i in 0..DATA.len() {
            if DATA[i].is_nan() {
                assert!((imputed[i] - 7.0).abs() < 1e-10);
            }
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
    ];
}
