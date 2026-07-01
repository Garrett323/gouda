use crate::utils::{StringEncoding, arr_to_out, constants::NOT_FITTED_ERR, pyany_to_vec};
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;

#[pyclass]
pub struct SimpleImputer {
    sample_means: Option<Vec<f64>>,
    // sample_mode: Option<Vec<f64>>,// needed when implementing categoricals
    string_encoding: Option<StringEncoding>,
    is_fitted: bool,
}

#[pymethods]
impl SimpleImputer {
    #[new]
    #[pyo3(signature = (encoding=None))]
    pub fn new(encoding: Option<&str>) -> SimpleImputer {
        SimpleImputer {
            sample_means: None,
            // sample_mode: None,
            string_encoding: match encoding {
                Some(_) => Some(StringEncoding::LabelEncoding),
                None => None,
            },
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        {
            let mut inner = slf.borrow_mut(py);
            let (arr, _out, _enc) = pyany_to_vec(data, &inner.string_encoding)?;
            let ids = if let Some(enc) = _enc {
                Some(enc.string_column_indices)
            } else {
                None
            };
            inner.fit_impl(&arr, ids);
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
        arr_to_out(py, &imputed, out, enc)
    }

    pub fn fit_transform<'py>(
        slf: Py<Self>,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slf = Self::fit(slf, py, data)?;
        {
            let inner = slf.borrow_mut(py);
            inner.transform(py, data)
        }
    }
}

impl SimpleImputer {
    pub fn fit_impl(&mut self, data: &Array2<f64>, categories: Option<Vec<usize>>) -> &Self {
        let mut means = self.get_means(&data);
        if let Some(v) = categories {
            let modes = self.get_modes(data.view(), &v);
            for (&categorical, mode) in v.iter().zip(modes) {
                means[categorical] = mode;
            }
        }
        self.sample_means = Some(means);
        return self;
    }
    fn get_means(&self, data: &Array2<f64>) -> Vec<f64> {
        (0..data.ncols())
            .into_par_iter()
            .map(|i| {
                let mut nnans = 0.0;
                let mut mean = 0.0;
                for entry in data.column(i).iter() {
                    if entry.is_nan() {
                        nnans += 1.0;
                        continue;
                    }
                    mean += entry;
                }
                mean /= data.nrows() as f64 - nnans;
                mean
            })
            .collect()
    }

    fn get_modes(&self, data: ArrayView2<f64>, categories: &Vec<usize>) -> Vec<f64> {
        categories
            .par_iter()
            .map(|&idx| {
                let mut counts: HashMap<usize, usize> = HashMap::new();
                let col = data.column(idx);
                for &x in col {
                    *counts.entry(x as usize).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(val, _)| val)
                    .expect("Tried to find mode on empty column!") as f64
            })
            .collect()
    }

    pub fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut imputed = vec![0.0; data.shape()[0] * data.shape()[1]];
        for j in 0..data.shape()[0] {
            for i in 0..data.shape()[1] {
                let index = j * data.shape()[1] + i;
                if data[(j, i)].is_nan() {
                    imputed[index] = self.sample_means.as_ref().expect(NOT_FITTED_ERR)[i];
                } else {
                    imputed[index] = data[(j, i)];
                }
            }
        }
        Array2::from_shape_vec([data.nrows(), data.ncols()], imputed).unwrap()
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
        let mut simple = SimpleImputer::new(None);
        let data = Array2::from_shape_vec((5, 5), DATA.to_owned()).unwrap();
        let imputed = simple.fit_impl(&data, None).impute(&data);
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
        let mut simple = SimpleImputer::new(None);
        let data = Array2::from_shape_vec((5, 5), DATA.to_owned()).unwrap();
        simple.fit_impl(&data, None);
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
