use crate::utils::pyany_to_vec;
use core::f64;
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
    metric: String,
}

#[pymethods]
impl KnnImputer {
    #[new]
    #[pyo3(signature = (k=5, metric="nan_euclid"))]
    pub fn new(k: usize, metric: &str) -> KnnImputer {
        KnnImputer {
            k,
            nrows: 0,
            ncols: 0,
            data: None,
            is_fitted: false,
            metric: metric.to_owned(),
        }
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
        let (data, nrows, _) = pyany_to_vec(py, data)?;
        // actual method
        let dist = match self.metric.as_str() {
            "nan_euclid" => Self::nan_euclid,
            "expected_distance" => Self::expected_distance,
            m => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "{} is a unknown metric",
                    m
                )));
            }
        };
        let imputed = self.brute_force(&data, nrows, dist);
        // return python object
        let array = ndarray::Array2::from_shape_vec((self.nrows, self.ncols), imputed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}

impl KnnImputer {
    fn brute_force(
        &self,
        data: &[f64],
        nrows: usize,
        dist: fn(&KnnImputer, &[f64], &[f64]) -> f64,
    ) -> Vec<f64> {
        let mut imputed = data.to_vec();
        let mut i = 0;
        while i < data.len() {
            if data[i].is_nan() {
                // figure out point and impute
                let row = i / self.ncols;
                let col = i % self.ncols;
                let mut cols = Vec::with_capacity(20);
                for j in col..self.ncols {
                    let l = i + j - col;
                    if l >= data.len() {
                        break;
                    }
                    if data[l].is_nan() {
                        cols.push(j);
                    }
                }
                let mut distances = Vec::with_capacity(nrows);
                let p = &self[row];
                for r in 0..self.nrows {
                    if r == row {
                        distances.push(f64::MAX); // dont consider distance to self
                    } else {
                        distances.push(dist(&self, p, &self[r]));
                    }
                }
                let mut indices: Vec<usize> = (0..nrows).collect();
                // indices.sort_by(|&a, &b| distances[a].total_cmp(&distances[b]));
                indices.select_nth_unstable_by(self.k * 3 - 1, |&a, &b| {
                    distances[a].total_cmp(&distances[b])
                });
                let avgs = self.average(&indices, &cols);
                for (avg, c) in avgs.into_iter().zip(&cols) {
                    imputed[i + c - col] = avg;
                }
                i += self.ncols - col;
            } else {
                i += 1;
            }
        }
        imputed
    }

    fn average(&self, indices: &[usize], cols: &[usize]) -> Vec<f64> {
        let mut avg: Vec<f64> = vec![0.0; cols.len()];
        for (j, c) in cols.iter().enumerate() {
            let mut count = 0;
            for i in indices {
                let val = self[*i][*c];
                if val.is_nan() {
                    continue;
                }
                avg[j] += val;
                count += 1;
                if count >= self.k {
                    break;
                }
            }
            avg[j] = avg[j] * (1.0 / count as f64)
        }
        avg
    }
}

// Distance Functions
impl KnnImputer {
    // TODO:
    // write tests
    fn nan_euclid(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut nnans = 0;
        let ncols = a.len();
        for (x, y) in a.iter().zip(b) {
            if x.is_nan() || y.is_nan() {
                nnans += 1;
            } else {
                total += (x - y).powi(2);
            }
        }
        if nnans == ncols {
            return f64::INFINITY;
        }
        total * (ncols as f64 / (ncols - nnans) as f64)
    }

    fn expected_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut total_obs = 0.0;
        for (x, y) in a.iter().zip(b) {
            match (x.is_nan(), y.is_nan()) {
                (true, true) => total += 0.333,
                (true, false) => total += y.max(1.0 - y),
                (false, true) => total += x.max(1.0 - x),
                (false, false) => total_obs += (x - y).powi(2),
            }
        }
        total + total_obs.sqrt()
    }
}

impl Index<usize> for KnnImputer {
    type Output = [f64];

    fn index(&self, row: usize) -> &[f64] {
        let offset = row * self.ncols;
        &self.data.as_ref().unwrap()[offset..offset + self.ncols]
    }
}

#[cfg(test)]
mod tests {
    use super::*; // has access to everything, including private

    #[test]
    fn test_nan_euclid() {
        let knn = KnnImputer::new(5, "nan_euclid");
        let result = knn.nan_euclid(&[1.0, 2.0], &[3.0, 4.0]);
        assert!(
            (result - 8.0).abs() < 1e-10,
            "Expected: 8.0; Actual {}",
            result
        );
    }
    #[test]
    fn test_expected_distance() {
        let p = &[f64::NAN, 0.555556, f64::NAN, 0.555556];
        let points = &[
            [0.0, 0.777778, 0.0, 0.777778],
            [f64::NAN, 0.333333, 0.666667, 0.333333],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.88889, 0.0, 0.88889],
            [0.0, 0.44444, 0.0, 0.44444],
            [0.666667, f64::NAN, 0.666667, f64::NAN],
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            [f64::NAN, 0.555556, f64::NAN, 0.555556],
        ];
        let expected = &[
            2.314269366257674,
            1.3139377804712364,
            2.6285387325153478,
            2.4714054636000733,
            2.157141754196649,
            2.444446,
            1.777112,
            0.666,
        ];
        let knn = KnnImputer::new(5, "expected_distance");
        for (e, q) in expected.iter().zip(points) {
            let result = knn.expected_distance(p, q);
            assert!(
                (result - e).abs() < 1e-9,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
    }

    #[test]
    fn test_compare() {
        let knn = KnnImputer::new(5, "nan_euclid");
        let (a, b) = (&[1.0, 2.0], &[3.0, 4.0]);
        let diff = knn.nan_euclid(a, b).sqrt() - knn.expected_distance(a, b);
        assert!((diff).abs() < 1e-10, "Expected: 0.0; Actual {}", diff);

        let (a, b) = (&[1.0, f64::NAN], &[3.0, f64::NAN]);
        // 2.8284271247461903
        let euclid = knn.nan_euclid(a, b).sqrt();
        // 2 + 1/3
        let ed = knn.expected_distance(a, b);
        let diff = euclid - ed;
        assert!(
            (diff - 0.4954271247461901).abs() < 1e-10,
            "Expected: 0.4954271247461901 ; Actual {}\nEuclid: {}; ED: {}",
            diff,
            euclid,
            ed
        );
    }
}
