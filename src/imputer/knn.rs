use crate::utils::Errors::NotFitted;
use crate::utils::{self, Errors, StringEncoding};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
enum Weights {
    Uniform,
    Distance,
}

#[derive(Clone, Serialize, Deserialize)]
enum Metrics {
    NanEuclid,
    ExpectedDistance,
    Gower(Option<Vec<f64>>),
}

#[pyclass(name = "KnnImputerRS", module = "gouda.gouda")]
#[derive(Serialize, Deserialize)]
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

const ALLOWED_WEIGHTS: &[&str] = &["uniform", "distance"];
const ALLOWED_METRICS: &[&str] = &["nan_euclid", "expected_distance", "gower"];

#[pymethods]
impl KnnImputer {
    #[new]
    #[pyo3(signature = (k=5, metric="nan_euclid", weights="uniform", encoding=None))]
    pub fn new(
        k: usize,
        metric: &str,
        weights: &str,
        encoding: Option<&str>,
    ) -> PyResult<KnnImputer> {
        Ok(KnnImputer {
            k,
            data: None,
            is_fitted: false,
            metric: match metric.to_lowercase().as_str() {
                "nan_euclid" => Metrics::NanEuclid,
                "expected_distance" => Metrics::ExpectedDistance,
                "gower" => Metrics::Gower(None),
                s => {
                    return Err(Errors::UnsupportedValue {
                        parameter: "Knn.Metric",
                        value: s.to_owned(),
                        supported: Some(ALLOWED_METRICS),
                    }
                    .into());
                }
            },
            weights: match weights.to_lowercase().as_str() {
                "uniform" => Weights::Uniform,
                "distance" => Weights::Distance,
                s => {
                    return Err(Errors::UnsupportedValue {
                        parameter: "Knn.Weights",
                        value: s.to_owned(),
                        supported: Some(ALLOWED_WEIGHTS),
                    }
                    .into());
                }
            },
            cat_cols: None,
            num_cols: None,
            string_encoding: utils::process_labelencoding(encoding)?,
        })
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
            let (arr, _out, _enc) = utils::pyany_to_vec(data, &inner.string_encoding)?;
            utils::raise_if_nan_col(arr.view())?;
            if let Metrics::Gower(_) = inner.metric {
                inner.metric = Metrics::Gower(Some(inner.span(arr.view())));
                let indices = _enc.map_or(Vec::new(), |enc| enc.string_column_indices);
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
            return Err(Errors::NotFitted.into());
            // return Err(PyTypeError::new_err(format!("Imputer is not fitted",)));
        }
        let (arr, out, enc) = utils::pyany_to_vec(data, &self.string_encoding)?;
        // actual method
        utils::check_feature_mismatch(self.data.as_ref().ok_or(NotFitted)?.ncols(), arr.ncols())?;
        let imputed: Result<Array2<f64>, utils::Errors> = match &self.metric {
            Metrics::NanEuclid => self.brute_force(arr.view(), nan_euclid),
            Metrics::ExpectedDistance => self.brute_force(arr.view(), expected_distance),
            Metrics::Gower(None) => {
                return Err(Errors::NotFitted.into());
            }
            Metrics::Gower(Some(ranges)) => {
                let cat_cols = self.cat_cols.as_ref().ok_or(Errors::NotFitted)?;
                let num_cols = self.num_cols.as_ref().ok_or(Errors::NotFitted)?;
                self.brute_force(arr.view(), |a, b| gower(a, b, ranges, cat_cols, num_cols))
            }
        };
        // return python object
        utils::arr_to_out(py, &imputed?, out, enc.as_ref())
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

    #[getter]
    fn encoding(&self) -> Option<&str> {
        match self.string_encoding {
            None => None,
            Some(_) => Some("label"),
        }
    }

    #[getter]
    fn weights(&self) -> &str {
        match self.weights {
            Weights::Uniform => "uniform",
            Weights::Distance => "distance",
        }
    }

    #[getter]
    fn metric(&self) -> &str {
        match self.metric {
            Metrics::NanEuclid => "nan_euclid",
            Metrics::ExpectedDistance => "expected_distance",
            Metrics::Gower(_) => "gower",
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to pickle KnnImputerRS: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let decoded: KnnImputer = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to unpickle KnnImputerRS: {e}"))
        })?;
        self.k = decoded.k;
        self.data = decoded.data;
        self.string_encoding = decoded.string_encoding;
        self.cat_cols = decoded.cat_cols;
        self.num_cols = decoded.num_cols;
        self.is_fitted = decoded.is_fitted;
        self.metric = decoded.metric;
        self.weights = decoded.weights;
        Ok(())
    }
}

impl KnnImputer {
    fn brute_force<D>(&self, data: ArrayView2<f64>, dist: D) -> Result<Array2<f64>, utils::Errors>
    where
        D: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64 + Sync,
    {
        let mut imputed = data.to_owned();
        let base = self.data.as_ref().ok_or(utils::Errors::NotFitted)?;
        let res: Result<(), Errors> = imputed
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(nrow, mut row)| {
                let cols: Vec<usize> = (0..base.ncols())
                    .filter(|&j| data[(nrow, j)].is_nan())
                    .collect();
                if cols.is_empty() {
                    return Ok(());
                }

                let p = data.row(nrow);
                let mut neighbors: Vec<(usize, f64)> = (0..base.nrows())
                    .into_par_iter()
                    .filter(|&r| cols.iter().any(|&c| !base[(r, c)].is_nan()))
                    .map(|r| (r, dist(p, base.row(r))))
                    .collect();
                neighbors.par_sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let avgs = self.average(base.view(), &neighbors, &cols)?;
                for (avg, c) in avgs.into_iter().zip(&cols) {
                    row[*c] = avg;
                }
                Ok(())
            })
            .collect();
        res?;
        Ok(imputed)
    }

    fn average(
        &self,
        base: ArrayView2<f64>,
        neighbors: &[(usize, f64)],
        cols: &[usize],
    ) -> Result<Vec<f64>, Errors> {
        let avg = |&c: &usize| {
            let mut count = 0;
            let mut avg = 0.0;
            let mut weight_sum = 0.0;
            for &(i, distance) in neighbors {
                let val = unsafe { *base.row(i).uget(c) };
                if val.is_nan() {
                    continue;
                }
                let weight = self.weight(distance);
                avg += val * weight;
                weight_sum += weight;
                count += 1;
                if count >= self.k {
                    break;
                }
            }
            if count == 0 {
                Ok(f64::NAN)
            } else {
                Ok(avg / weight_sum)
            }
        };
        if cols.len() > 100 {
            cols.par_iter().map(avg).collect()
        } else {
            cols.iter().map(avg).collect()
        }
    }

    fn weight(&self, distance: f64) -> f64 {
        match self.weights {
            Weights::Uniform => 1.0,
            Weights::Distance => 1.0 / distance.max(f64::EPSILON),
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
}

// Distance Functions
fn nan_euclid(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let mut total = 0.0;
    let mut valid = 0;
    let ncols = a.len();
    for i in 0..ncols {
        let (x, y) = unsafe { (a.uget(i), b.uget(i)) };
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

fn expected_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let mut total = 0.0;
    let mut total_obs = 0.0;
    let ncols = a.len();
    for i in 0..ncols {
        let (x, y) = unsafe { (a.uget(i), b.uget(i)) };
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

fn gower(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    ranges: &[f64],
    cat_cols: &[usize],
    num_cols: &[usize],
) -> f64 {
    // These panics are intentional; it should not be possible to trigger this from the api
    let mut total = 0.0;
    let mut valid = 0;
    for &i in cat_cols {
        let (x, y) = unsafe { (a.uget(i), b.uget(i)) };
        if !(x.is_nan() || y.is_nan()) {
            // total += (x - y).abs().min(1.0);
            total += if x == y { 0.0 } else { 1.0 };
            valid += 1;
        }
    }
    for &i in num_cols {
        let (x, y) = unsafe {
            (
                a.uget(i) / ranges.get_unchecked(i),
                b.uget(i) / ranges.get_unchecked(i),
            )
        };
        if !(x.is_nan() || y.is_nan()) {
            total += (x - y).abs();
            valid += 1;
        }
    }
    if valid == 0 {
        f64::INFINITY
    } else {
        total / valid as f64
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*; // has access to everything, including private

    #[test]
    fn distance_weights_are_normalized() {
        let mut knn = KnnImputer::new(2, "nan_euclid", "distance", None).unwrap();
        knn.data = Some(array![[10.0], [20.0]]);

        // With distances 1 and 3, the weighted mean is:
        // (10 / 1 + 20 / 3) / (1 / 1 + 1 / 3) = 12.5.
        let neighbors = [(0, 1.0), (1, 3.0)];
        let actual = knn
            .average(knn.data.as_ref().unwrap().view(), &neighbors, &[0])
            .unwrap()[0];

        assert!((actual - 12.5).abs() < 1e-12, "actual: {actual}");
    }

    #[test]
    fn test_gower() {
        let mut knn = KnnImputer::new(5, "gower", "uniform", Some("label")).unwrap();
        // gower is same as nan_euclid for numeric only
        let train = Array2::from_shape_vec(
            [POINTS_EUCLID.len(), 5],
            POINTS_EUCLID.iter().flatten().copied().collect(),
        )
        .unwrap();

        let ranges = knn.span(train.view());
        knn.cat_cols = Some(vec![]);
        knn.num_cols = Some((0..5).collect());
        let p = &[f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135];

        for (e, q) in EXPECTED_GOWER.iter().zip(POINTS_EUCLID) {
            let result = gower(
                p.into(),
                q.into(),
                &ranges,
                &knn.cat_cols.as_ref().unwrap(),
                &knn.num_cols.as_ref().unwrap(),
            );
            assert!(
                (result - e).abs() < 1e-7,
                "Expected: {}; Actual: {}",
                e,
                result
            );
        }
    }

    #[test]
    fn test_nan_euclid() {
        let p = &[f64::NAN, 0.22129885, 0.8863533, 0.50595314, 0.5011135];

        for (e, q) in EXPECTED_EUCLID.iter().zip(POINTS_EUCLID) {
            let result = nan_euclid(p.into(), q.into()).sqrt();
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
        // let knn = KnnImputer::new(5, "expected_distance", "uniform", None).unwrap();
        for (e, q) in expected.iter().zip(points) {
            let result = expected_distance(
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
        let (a, b) = (&[1.0, 2.0], &[3.0, 4.0]);
        let diff = nan_euclid(
            Array1::from_vec(a.to_vec()).view(),
            Array1::from_vec(b.to_vec()).view(),
        )
        .sqrt()
            - expected_distance(
                Array1::from_vec(a.to_vec()).view(),
                Array1::from_vec(b.to_vec()).view(),
            );
        assert!((diff).abs() < 1e-10, "Expected: 0.0; Actual {}", diff);

        let (a, b) = (&[1.0, f64::NAN], &[3.0, f64::NAN]);
        // 2.8284271247461903
        let euclid = nan_euclid(
            Array1::from_vec(a.to_vec()).view(),
            Array1::from_vec(b.to_vec()).view(),
        )
        .sqrt();
        // 2 + 1/3
        let ed = expected_distance(
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

    const EXPECTED_GOWER: &[f64] = &[
        0.8074396249732037,
        0.46796509112089085,
        0.0,
        0.2770944551042174,
        0.49666980523195914,
        0.16298231711637567,
        0.43765590083192635,
        0.6393675528933603,
        0.581755451636246,
    ];
}
