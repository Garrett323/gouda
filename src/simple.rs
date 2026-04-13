use crate::utils::{Data, NOT_FITTED_ERR, pyany_to_vec};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct SimpleImputer {
    sample_means: Option<Vec<f64>>,
    // sample_mode: Option<Vec<f64>>,// needed when implementing categoricals
    is_fitted: bool,
}

#[pymethods]
impl SimpleImputer {
    #[new]
    pub fn new() -> SimpleImputer {
        SimpleImputer {
            sample_means: None,
            // sample_mode: None,
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
                "{}",
                NOT_FITTED_ERR
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

impl SimpleImputer {
    pub fn fit_impl(&mut self, data: &Data) -> &Self {
        let means = self.get_means(&data);
        self.sample_means = Some(means);
        return self;
    }
    fn get_means(&self, data: &Data) -> Vec<f64> {
        let mut means = vec![0.0; data.ncols];
        for i in 0..data.ncols {
            let mut nnans = 0.0;
            for entry in data.get_col(i) {
                if entry.is_nan() {
                    nnans += 1.0;
                    continue;
                }
                means[i] += entry;
            }
            means[i] /= data.nrows as f64 - nnans;
        }
        means
    }

    pub fn impute(&self, data: &Data) -> Vec<f64> {
        let mut imputed = vec![0.0; data.nrows * data.ncols];
        let collapsed: &[f64] = data;
        for j in 0..data.nrows {
            for i in 0..data.ncols {
                let index = j * data.ncols + i;
                if collapsed[index].is_nan() {
                    imputed[index] = self.sample_means.as_ref().expect(NOT_FITTED_ERR)[i];
                } else {
                    imputed[index] = collapsed[index];
                }
            }
        }
        imputed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data() {
        assert_eq!(
            5 * 5,
            DATA.len(),
            "Expected: {} Actual {}",
            DATA.len(),
            5 * 5
        );
        assert_eq!(
            EXPECTED.len(),
            DATA.len(),
            "Expected: {} Actual {}",
            DATA.len(),
            EXPECTED.len(),
        );
    }

    #[test]
    fn test_impute() {
        let mut simple = SimpleImputer::new();
        let data = Data::new(5, 5, DATA);
        let imputed = simple.fit_impl(&data).impute(&data);
        println!("Means: {:?}", simple.sample_means.as_ref().unwrap());
        println!("Data: {:?}", DATA);
        println!("Expected: {:?}", EXPECTED);
        println!("Imputed: {:?}", imputed);
        for (id, (exp, imp)) in EXPECTED.iter().zip(imputed).enumerate() {
            let diff = exp - imp;
            assert!(
                diff.abs() < 1e-10,
                "ID: {} Expected: {}; Actual {}\n",
                id,
                exp,
                imp
            );
        }
    }

    #[test]
    fn test_means() {
        const MEANS: &[f64] = &[
            (0.76052103 + 0.27839605 + 0.9317995) / 3.0,
            (0.7338148 + 0.22129885 + 0.32309935 + 0.51597243) / 4.0,
            (0.4094729 + 0.98359227 + 0.8863533 + 0.64573872 + 0.38054457) / 5.0,
            (0.9573324 + 0.98189233 + 0.50595314 + 0.62366235) / 4.0,
            (0.45384631 + 0.5011135 + 0.12229672) / 3.0,
        ];
        let mut simple = SimpleImputer::new();
        let data = Data::new(5, 5, DATA);
        simple.fit_impl(&data);
        for (gt, estimate) in MEANS.iter().zip(simple.sample_means.as_ref().unwrap()) {
            let diff = gt - estimate;
            assert!(
                diff.abs() < 1e-10,
                "Expected: {}; Actual {}\n",
                gt,
                estimate
            );
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
    const EXPECTED: &[f64] = &[
        0.76052103,
        (0.7338148 + 0.22129885 + 0.32309935 + 0.51597243) / 4.0,
        0.4094729,
        0.9573324,
        (0.45384631 + 0.5011135 + 0.12229672) / 3.0,
        0.27839605,
        0.7338148,
        0.98359227,
        0.98189233,
        0.45384631,
        (0.76052103 + 0.27839605 + 0.9317995) / 3.0,
        0.22129885,
        0.8863533,
        0.50595314,
        0.5011135,
        (0.76052103 + 0.27839605 + 0.9317995) / 3.0,
        0.32309935,
        0.64573872,
        (0.9573324 + 0.98189233 + 0.50595314 + 0.62366235) / 4.0,
        (0.45384631 + 0.5011135 + 0.12229672) / 3.0,
        0.9317995,
        0.51597243,
        0.38054457,
        0.62366235,
        0.12229672,
    ];
}
