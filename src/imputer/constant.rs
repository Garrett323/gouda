use crate::utils::{
    self, StringEncoding,
    constants::{ENCODING_WARN, NOT_FITTED_ERR},
    pyany_to_vec,
};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[pyclass]
pub struct ConstantImputer {
    value: f64,
    is_fitted: bool,
    string_encoding: Option<StringEncoding>,
}

#[pymethods]
impl ConstantImputer {
    #[new]
    #[pyo3(signature = (value=0.0, encoding=None))]
    pub fn new(value: f64, encoding: Option<&str>) -> ConstantImputer {
        ConstantImputer {
            value,
            string_encoding: match encoding {
                Some(_) => Some(StringEncoding::LabelEncoding),
                None => None,
            },
            is_fitted: false,
        }
    }

    #[staticmethod]
    pub fn zero() -> ConstantImputer {
        ConstantImputer {
            value: 0.0,
            string_encoding: None,
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, _data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        {
            let mut inner = slf.borrow_mut(py);
            if let Some(_) = inner.string_encoding {
                pyo3::PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    ENCODING_WARN,
                    1,
                )?;
            };
            inner.is_fitted = true;
        } // dropping inner here (releasing the mutex)
        Ok(slf)
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // check if fitted
        if !self.is_fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "{}",
                NOT_FITTED_ERR
            )));
        }
        let (arr, out, enc) = pyany_to_vec(data, &self.string_encoding)?;
        let imputed = self.impute(&arr);
        // return python object
        utils::arr_to_out(py, &imputed, out, enc)
    }
}

impl ConstantImputer {
    fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut imputed = data.clone(); //vec![self.value; data.len()];
        imputed.par_iter_mut().for_each(|e| {
            if e.is_nan() {
                *e = self.value;
            }
        });
        imputed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impute() {
        let imputer = ConstantImputer::new(7.0, None);
        let data = Array2::from_shape_vec((5, 5), DATA.to_owned()).unwrap();
        let imputed = imputer.impute(&data);
        for (i, &e) in imputed.iter().enumerate() {
            if DATA[i].is_nan() {
                assert!((e - 7.0).abs() < 1e-10);
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
