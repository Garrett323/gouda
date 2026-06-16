use crate::utils::{self, SendPtr, StringEncoding};
use ndarray::{Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;

enum Weights {
    Uniform,
    Distance,
}

enum Metrics {
    NanEuclid,
    ExpectedDistance,
    Gower(Option<Vec<f64>>),
}

#[pyclass]
pub struct KnnImputer {
    #[pyo3(get, set)]
    k: usize,
    data: Option<Array2<f64>>,
    string_encoding: Option<StringEncoding>,
    cat_cols: Option<Vec<usize>>,
    num_cols: Option<Vec<usize>>,
    is_fitted: bool,
    metric: Metrics,
    weights: Weights,
}

const ALLOWED_WEIGHTS: [&str; 2] = ["uniform", "distance"];
const ALLOWED_METRICS: &[&str] = &["nan_euclid", "expected_distance", "gower"];

#[pymethods]
impl KnnImputer {
    #[new]
    #[pyo3(signature = (k=5, metric="nan_euclid", weights="uniform", encoding=None))]
    pub fn new(k: usize, metric: &str, weights: &str, encoding: Option<&str>) -> KnnImputer {
        assert!(ALLOWED_WEIGHTS.contains(&weights));
        KnnImputer::sanity_check(&metric, &weights);
        KnnImputer {
            k,
            data: None,
            is_fitted: false,
            metric: match metric {
                "nan_euclid" => Metrics::NanEuclid,
                "expected_distance" => Metrics::ExpectedDistance,
                "gower" => Metrics::Gower(None),
                _ => panic!("metric parameter not supported, {:?}", ALLOWED_METRICS),
            },
            weights: match weights {
                "uniform" => Weights::Uniform,
                "distance" => Weights::Distance,
                _ => panic!("weight parameter not supported, {:?}", ALLOWED_WEIGHTS),
            },
            cat_cols: None,
            num_cols: None,
            string_encoding: match encoding {
                None => None,
                Some(_) => Some(StringEncoding::LabelEncoding),
            },
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        {
            let mut inner = slf.borrow_mut(py);
            if let (Some(_), Metrics::ExpectedDistance | Metrics::NanEuclid) =
                (&inner.string_encoding, &inner.metric)
            {
                pyo3::PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    c"Passed Label encoding but didn't select a metric that supports categoricals! Please pass one of the metrics that support categoricals [gower]",
                    1,
                )?;
            };
            let (arr, _out, _enc) = utils::pyany_to_vec(py, data, &inner.string_encoding)?;
            if let Metrics::Gower(_) = inner.metric {
                inner.metric = Metrics::Gower(Some(inner.span(arr.view())));
                let indices = _enc
                    .expect("Passed gower but didnt pass encoding strategy")
                    .string_column_indices;
                inner.num_cols = Some(
                    (0..arr.ncols())
                        .filter(|idx| !indices.contains(idx))
                        .collect(),
                );
                inner.cat_cols = Some(indices);
            }
            inner.data = Some(arr);
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
                "Imputer is not fitted",
            )));
        }
        let (arr, out, enc) = utils::pyany_to_vec(py, data, &self.string_encoding)?;
        // actual method
        let dist = match self.metric {
            Metrics::NanEuclid => Self::nan_euclid,
            Metrics::ExpectedDistance => Self::expected_distance,
            Metrics::Gower(_) => Self::gower,
        };
        let imputed = self.brute_force(arr.view(), dist);
        // return python object
        utils::arr_to_out(py, &imputed, out, enc)
    }
}

impl KnnImputer {
    fn brute_force(
        &self,
        data: ArrayView2<f64>,
        dist: fn(&KnnImputer, ArrayView1<f64>, ArrayView1<f64>) -> f64,
    ) -> Array2<f64> {
        let mut imputed = data.to_owned();
        let imp_ptr = std::sync::Arc::new(SendPtr(imputed.as_mut_ptr()));
        let base = self.data.as_ref().unwrap();
        (0..data.nrows()).into_par_iter().for_each(|row| {
            let cols: Vec<usize> = (0..base.ncols())
                .filter(|&j| data[(row, j)].is_nan())
                .collect();
            let p = data.row(row);
            let distances: Vec<f64> = (0..data.nrows())
                .into_par_iter()
                .map(|r| {
                    if r == row {
                        f64::MAX
                    } else {
                        dist(&self, p, base.row(r))
                    }
                })
                .collect();
            let mut indices: Vec<_> = (0..data.nrows()).collect();
            indices.par_sort_unstable_by(|&a, &b| distances[a].total_cmp(&distances[b]));
            let avgs = self.average(&indices, &cols, &self.get_weights(&distances));
            let ptr = std::sync::Arc::clone(&imp_ptr);
            for (avg, c) in avgs.into_iter().zip(&cols) {
                // imputed[row * base.ncols() + c] = avg;
                unsafe {
                    *ptr.0.add(row * base.ncols() + c) = avg;
                }
            }
        });
        imputed
    }

    fn average(&self, indices: &[usize], cols: &[usize], weights: &[f64]) -> Vec<f64> {
        let base = self.data.as_ref().unwrap();
        let avg = |c: &usize| {
            let mut count = 0;
            let mut avg = 0.0;
            for i in indices {
                let val = base.row(*i)[*c];
                if val.is_nan() {
                    continue;
                }
                avg += val * weights[*i];
                count += 1;
                if count >= self.k {
                    break;
                }
            }
            avg
        };
        if cols.len() > 100 {
            cols.par_iter().map(avg).collect()
        } else {
            cols.iter().map(avg).collect()
        }
    }

    fn get_weights(&self, distances: &[f64]) -> Vec<f64> {
        match self.weights {
            Weights::Uniform => distances.iter().map(|_| 1.0 / self.k as f64).collect(),
            Weights::Distance => distances.iter().map(|d| 1.0 / d).collect(),
        }
    }

    fn span(&self, arr: ArrayView2<f64>) -> Vec<f64> {
        (0..arr.ncols())
            .into_par_iter()
            .map(|i| {
                let mut max = f64::NEG_INFINITY;
                let mut min = f64::INFINITY;
                arr.column(i).for_each(|&v| {
                    if v > max {
                        max = v
                    }
                    if v < min {
                        min = v
                    }
                });
                max - min
            })
            .collect()
    }

    fn sanity_check(metric: &str, weights: &str) {
        if !ALLOWED_WEIGHTS.contains(&weights) {
            panic!(
                "Please select a valid metric: [{:?}]\n{} is not supported",
                ALLOWED_WEIGHTS, weights
            );
        }
        if !ALLOWED_METRICS.contains(&metric) {
            panic!(
                "Please select a valid metric: [{:?}]\n{} is not supported",
                ALLOWED_METRICS, metric
            );
        }
    }
}

// Distance Functions
impl KnnImputer {
    fn nan_euclid(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
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

    fn expected_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
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

    fn gower(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let ranges = if let Metrics::Gower(v) = &self.metric {
            v.as_ref().unwrap().as_slice()
        } else {
            panic!("Set distance to gower when calling gower!");
        };
        let mut total = 0.0;
        let mut valid = 0;
        for &i in self.cat_cols.as_ref().unwrap() {
            if !(a[i].is_nan() || b[i].is_nan()) {
                total += (a[i] - b[i]).min(1.0);
                valid += 1;
            }
        }
        for &i in self.num_cols.as_ref().unwrap() {
            let (x, y) = (a[i] / ranges[i], b[i] / ranges[i]);
            if !(x.is_nan() || y.is_nan()) {
                let d = x - y;
                total += d * d;
                valid += 1;
            }
        }
        if valid == 0 { f64::INFINITY } else { total }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*; // has access to everything, including private

    #[test]
    fn gower() {
        // gower is same as nan_euclid for numeric only
        let knn = KnnImputer::new(5, "gower", "uniform", Some("label"));
        let p = &[f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135];

        for (e, q) in EXPECTED_EUCLID.iter().zip(POINTS_EUCLID) {
            let result = knn.nan_euclid(p.into(), q.into()).sqrt();
            assert!(
                (result - e).abs() < 1e-7,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
    }

    #[test]
    fn nan_euclid() {
        let knn = KnnImputer::new(5, "nan_euclid", "uniform", None);
        let p = &[f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135];

        for (e, q) in EXPECTED_EUCLID.iter().zip(POINTS_EUCLID) {
            let result = knn.nan_euclid(p.into(), q.into()).sqrt();
            assert!(
                (result - e).abs() < 1e-7,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
    }
    #[test]
    fn expected_distance() {
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
        let knn = KnnImputer::new(5, "expected_distance", "uniform", None);
        for (e, q) in expected.iter().zip(points) {
            let result = knn.expected_distance(
                Array1::from_vec(p.to_vec()).view(),
                Array1::from_vec(q.to_vec()).view(),
            );
            assert!(
                (result - e).abs() < 1e-9,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
    }

    #[test]
    fn compare() {
        let knn = KnnImputer::new(5, "nan_euclid", "uniform", None);
        let (a, b) = (&[1.0, 2.0], &[3.0, 4.0]);
        let diff = knn
            .nan_euclid(
                Array1::from_vec(a.to_vec()).view(),
                Array1::from_vec(b.to_vec()).view(),
            )
            .sqrt()
            - knn.expected_distance(
                Array1::from_vec(a.to_vec()).view(),
                Array1::from_vec(b.to_vec()).view(),
            );
        assert!((diff).abs() < 1e-10, "Expected: 0.0; Actual {}", diff);

        let (a, b) = (&[1.0, f64::NAN], &[3.0, f64::NAN]);
        // 2.8284271247461903
        let euclid = knn
            .nan_euclid(
                Array1::from_vec(a.to_vec()).view(),
                Array1::from_vec(b.to_vec()).view(),
            )
            .sqrt();
        // 2 + 1/3
        let ed = knn.expected_distance(
            Array1::from_vec(a.to_vec()).view(),
            Array1::from_vec(b.to_vec()).view(),
        );
        let diff = euclid - ed;
        assert!(
            (diff - 0.4954271247461901).abs() < 1e-10,
            "Expected: 0.4954271247461901 ; Actual {}\nEuclid: {}; ED: {}",
            diff,
            euclid,
            ed
        );
    }

    const POINTS_EUCLID: &[[f64; 5]] = &[
        [0.76052103, f64::NAN, 0.4094729, 0.9573324, f64::NAN],
        [0.27839605, 0.7338148, 0.98359227, 0.98189233, 0.45384631],
        [f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135],
        [f64::NAN, 0.32309935, 0.64573872, f64::NAN, f64::NAN],
        [0.9317995, 0.51597243, 0.38054457, 0.62366235, 0.12229672],
        [0.90547984, f64::NAN, 0.68424979, 0.55400964, 0.55284803],
        [0.68846839, 0.53889275, 0.44453843, 0.43416536, 0.18575075],
        [0.13333331, 0.8772666, 0.64398646, f64::NAN, 0.90529859],
        [0.69819416, 0.65251852, 0.39663618, 0.65702538, f64::NAN],
    ];
    const EXPECTED_EUCLID: &[f64; 9] = &[
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
