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
                indices.sort_unstable_by(|&a, &b| distances[a].total_cmp(&distances[b]));
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
        let mut valid = 0;
        let ncols = a.len();
        for i in 0..ncols {
            let (x, y) = (a[i], b[i]);
            if !(x.is_nan() || y.is_nan()) {
                let d = x - y;
                total += d * d;
                valid += 1;
            }
        }
        if valid == 0 {
            return f64::INFINITY;
        }
        total * (ncols as f64 / valid as f64)
    }

    fn expected_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut total_obs = 0.0;
        let ncols = a.len();
        for i in 0..ncols {
            let (x, y) = (a[i], b[i]);
            match (x.is_nan(), y.is_nan()) {
                (true, true) => total += 0.333,
                (true, false) => total += y.max(1.0 - y),
                (false, true) => total += x.max(1.0 - x),
                (false, false) => {
                    let d = x - y;
                    total_obs += d * d
                }
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
        let p = &[f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135];
        let points = &[
            [0.8184313, 0.21255369, 0.31404025, 0.94566382, 0.40544536],
            [0.76052103, f64::NAN, 0.4094729, 0.9573324, f64::NAN],
            [0.27839605, 0.7338148, 0.98359227, 0.98189233, 0.45384631],
            [f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135],
            [f64::NAN, 0.32309935, 0.64573872, f64::NAN, f64::NAN],
            [f64::NAN, 0.6991794, 0.9638419, 0.37826007, 0.73604386],
            [0.28932857, 0.45275616, 0.43649617, 0.7509847, 0.22084236],
            [0.9317995, 0.51597243, 0.38054457, 0.62366235, 0.12229672],
            [0.90547984, f64::NAN, 0.68424979, 0.55400964, 0.55284803],
            [0.68846839, 0.53889275, 0.44453843, 0.43416536, 0.18575075],
            [0.42542576, 0.80024827, 0.5310509, 0.15335542, 0.95273867],
            [0.57887984, 0.80655372, 0.84469441, 0.98941933, 0.11981212],
            [0.55863601, 0.7776858, 0.15676798, 0.10385082, 0.53160645],
            [0.67177456, 0.26571081, 0.37363354, 0.44251768, 0.36369568],
            [0.13333331, 0.8772666, 0.64398646, f64::NAN, 0.90529859],
            [0.69819416, 0.65251852, 0.39663618, 0.65702538, f64::NAN],
        ];
        let expected = &[
            0.8140305481190571,
            1.0382174099275148,
            0.7912650658744038,
            0.0,
            0.41309417813332494,
            0.6183365744290024,
            0.7022609100036123,
            0.7905951937189456,
            0.2763805321428371,
            0.7077017509263522,
            0.993549619679686,
            0.9509133938696136,
            1.1205340718678007,
            0.5997517090513087,
            1.042753531574897,
            0.8646734303095986,
        ];

        for (e, q) in expected.iter().zip(points) {
            let result = knn.nan_euclid(p, q).sqrt();
            assert!(
                (result - e).abs() < 1e-7,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
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
