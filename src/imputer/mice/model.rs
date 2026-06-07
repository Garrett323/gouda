use super::backend::{LinearRegression, PMM, Ridge, Solver};
use crate::imputer::SimpleImputer;
use crate::utils;
use ndarray::{Array1, Array2, Axis};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct Mice {
    max_iter: usize,
    backend: Box<dyn Solver>,
    models: Vec<Box<dyn Solver>>,
    is_fitted: bool,
}

#[pymethods]
impl Mice {
    #[new]
    #[pyo3(signature = (max_iter=10, backend="linear", alpha=1.0, pmm_backend="linear"))]
    pub fn new(max_iter: usize, backend: &str, alpha: f64, pmm_backend: &str) -> Mice {
        let backend = match backend.to_lowercase().as_str() {
            "linear" => Box::new(LinearRegression::new()) as Box<dyn Solver>,
            "ridge" => Box::new(Ridge::new(alpha)) as Box<dyn Solver>,
            "pmm" => Box::new(PMM::new(5, pmm_backend, Some(alpha))) as Box<dyn Solver>,
            _ => panic!("Solver {backend} not supported!"),
        };
        Mice {
            max_iter,
            backend: backend,
            models: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let ((vec, nrows, ncols), _out, _enc) = utils::pyany_to_vec(py, data, None)?;
        {
            let mut inner = slf.borrow_mut(py);
            inner.fit_impl(
                &Array2::from_shape_vec((nrows, ncols), vec)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            );
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
        let ((vec, nrows, ncols), out, enc) = utils::pyany_to_vec(py, data, None)?;
        let imputed = self.impute(
            &Array2::from_shape_vec((nrows, ncols), vec)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        );
        // return python object
        utils::arr_to_out(py, &imputed, out, enc)
    }
}

impl Mice {
    #[allow(unused)]
    fn linear(max_iter: usize) -> Self {
        Self::new(max_iter, "linear", 0.0, "linear")
    }
    #[allow(unused)]
    fn ridge(max_iter: usize, alpha: f64) -> Self {
        Self::new(max_iter, "ridge", alpha, "linear")
    }
    #[allow(unused)]
    fn pmm(max_iter: usize) -> Self {
        Self::new(max_iter, "pmm", 1.0, "linear")
    }

    fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        // initial mean imputation
        let mut imputed = SimpleImputer::new().fit_impl(&data).impute(&data);
        for _ in 0..self.max_iter {
            let imp_ptr = std::sync::Arc::new(SendPtr(imputed.as_mut_ptr()));
            let splits: Vec<_> = (0..data.ncols())
                .into_par_iter()
                .map(|j| split(&imputed, data, j))
                .collect();
            splits
                .into_par_iter()
                .enumerate()
                .for_each(|(j, (_, x_test, _, missing_indices))| {
                    let ptr = std::sync::Arc::clone(&imp_ptr);
                    let predictions = self.models[j].predict(&x_test);
                    for (k, v) in predictions.iter().enumerate() {
                        unsafe {
                            *ptr.0.add(missing_indices[k] * data.ncols() + j) = *v;
                        }
                    }
                });
        }
        imputed
    }

    fn fit_impl(&mut self, data: &Array2<f64>) -> &Self {
        let mut imputed = SimpleImputer::new().fit_impl(&data).impute(&data);
        let mut models: Vec<_> = (0..data.ncols())
            .into_iter()
            .map(|_| self.backend.clone())
            .collect();
        let imp_ptr = std::sync::Arc::new(SendPtr(imputed.as_mut_ptr()));
        for i in 0..self.max_iter {
            models.par_iter_mut().enumerate().for_each(|(j, m)| {
                let (x_train, x_test, y_train, missing_indices) = split(&imputed, data, j);
                m.fit(&x_train, &y_train);
                let ptr = std::sync::Arc::clone(&imp_ptr);
                for (k, v) in m.predict(&x_test).iter().enumerate() {
                    // let old = imputed[[missing_indices[k], j]];
                    // imputed[[missing_indices[k], j]] = *v;
                    unsafe {
                        *ptr.0.add(missing_indices[k] * data.ncols() + j) = *v;
                    }
                }
            });
        }
        self.models = models;
        self
    }
}

fn split(
    imputed: &Array2<f64>,
    missing: &Array2<f64>,
    col: usize,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Vec<usize>) {
    let x = ndarray::concatenate(
        Axis(1),
        &[
            imputed.slice(ndarray::s![.., ..col]),
            imputed.slice(ndarray::s![.., col + 1..]),
        ],
    )
    .expect("Failed concatenate views!");
    let y = missing.slice(ndarray::s![.., col]);
    // Build index lists
    let (missing_idx, observed_idx): (Vec<_>, Vec<_>) =
        y.iter().enumerate().partition(|(_, x)| x.is_nan());

    let missing_idx: Vec<usize> = missing_idx.into_iter().map(|(i, _)| i).collect();
    let observed_idx: Vec<usize> = observed_idx.into_iter().map(|(i, _)| i).collect();

    let x_train = x.select(Axis(0), &observed_idx);
    let x_test = x.select(Axis(0), &missing_idx);
    let y_train = y.select(Axis(0), &observed_idx);
    (x_train, x_test, y_train, missing_idx)
}

struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[cfg(test)]
mod test {
    use super::*; // has access to everything, including private

    #[test]
    fn move_away_from_initial() {
        let data = Array2::from_shape_vec((8, 4), POINTS.to_vec()).unwrap();
        let simple = SimpleImputer::new().fit_impl(&data).impute(&data);
        let mice = Mice::linear(2).fit_impl(&data).impute(&data);
        let diff = simple.iter().zip(&mice).any(|(p, q)| (p - q).abs() > 1e-6);
        println!("Simple: \n{}\n", &simple);
        println!("Mice: \n{}\n", &mice);
        assert!(diff, "Still at initial values :(");
    }

    #[test]
    fn ridge_neq_lin() {
        let data = Array2::from_shape_vec((8, 4), POINTS.to_vec()).unwrap();
        let ridge = Mice::ridge(2, 1.0).fit_impl(&data).impute(&data);
        let lin = Mice::linear(2).fit_impl(&data).impute(&data);
        let diff = ridge.iter().zip(&lin).any(|(p, q)| (p - q).abs() > 1e-6);
        println!("ridge: \n{}\n", &ridge);
        println!("linear: \n{}\n", &lin);
        assert!(diff, "Linear == Ridge values :(");
    }

    #[test]
    fn pmm() {
        let data = Array2::from_shape_vec((8, 4), POINTS.to_vec()).unwrap();
        let pmm = Mice::pmm(2).fit_impl(&data).impute(&data);
        println!("ridge: \n{}\n", &pmm);
        for e in &pmm {
            assert!(!e.is_nan());
        }
        let lin = Mice::linear(2).fit_impl(&data).impute(&data);
        let diff = pmm.iter().zip(&lin).any(|(p, q)| (p - q).abs() > 1e-6);
        assert!(diff, "Linear == PMM values :(");
    }

    const POINTS: &[f64] = &[
        0.0,
        0.777778,
        0.0,
        0.777778,
        f64::NAN,
        0.333333,
        0.666667,
        0.333333,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.88889,
        0.0,
        0.88889,
        0.0,
        0.44444,
        0.0,
        0.44444,
        0.666667,
        f64::NAN,
        0.666667,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0.555556,
        f64::NAN,
        0.555556,
    ];
}
