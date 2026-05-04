use super::linear_regression::{LinearRegression, Ridge, Solver};
use crate::imputer::SimpleImputer;
use crate::utils::pyany_to_vec;
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct Mice {
    n_iterations: usize,
    backend: Box<dyn Solver>,
    fitted: Vec<Box<dyn Solver>>,
    is_fitted: bool,
}

#[pymethods]
impl Mice {
    #[new]
    #[pyo3(signature = (n_iterations=15, backend="linear", alpha=1.0))]
    pub fn new(n_iterations: usize, backend: &str, alpha: f64) -> Mice {
        let backend = match backend.to_lowercase().as_str() {
            "linear" => Box::new(LinearRegression::new()) as Box<dyn Solver>,
            "ridge" => Box::new(Ridge::new(alpha)) as Box<dyn Solver>,
            _ => panic!("Solver {backend} not supported!"),
        };
        Mice {
            n_iterations,
            backend: backend,
            fitted: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let (vec, nrows, ncols) = pyany_to_vec(py, data)?;
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

    // TODO:
    // Does currently nothing
    // implement stuff
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
        let imputed = self.impute(
            &Array2::from_shape_vec((nrows, ncols), vec)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        );
        // return python object
        Ok(imputed.into_pyarray(py))
    }
}

impl Mice {
    fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        // initial mean imputation
        let mut imputed = SimpleImputer::new().fit_impl(&data).impute(&data);
        for _ in 0..self.n_iterations {
            for j in 0..data.shape()[1] {
                let (_, x_test, _, missing_indices) = split(&imputed, data, j);

                let predictions = self.fitted[j].predict(&x_test);
                for (k, v) in predictions.iter().enumerate() {
                    imputed[[missing_indices[k], j]] = *v;
                }
            }
        }
        imputed
    }

    fn fit_impl(&mut self, data: &Array2<f64>) -> &Self {
        self.fitted = Vec::with_capacity(data.shape()[1]);
        let mut imputed = SimpleImputer::new().fit_impl(&data).impute(&data);
        for i in 0..self.n_iterations {
            for j in 0..data.shape()[1] {
                let (x_train, x_test, y_train, missing_indices) = split(&imputed, data, j);
                self.backend.fit(&x_train, &y_train);
                for (k, v) in self.backend.predict(&x_test).iter().enumerate() {
                    // let old = imputed[[missing_indices[k], j]];
                    imputed[[missing_indices[k], j]] = *v;
                }
                if i == self.n_iterations - 1 {
                    self.fitted.push(self.backend.clone());
                }
            }
        }
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
    .unwrap();
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

#[cfg(test)]
mod test {
    use super::*; // has access to everything, including private

    #[test]
    fn move_away_from_initial() {
        let data = Array2::from_shape_vec((8, 4), POINTS.to_vec()).unwrap();
        let simple = SimpleImputer::new().fit_impl(&data).impute(&data);
        let mice = Mice::new(2, "linear", 1.0).fit_impl(&data).impute(&data);
        let diff = simple.iter().zip(&mice).any(|(p, q)| (p - q).abs() > 1e-6);
        println!("Simple: \n{}\n", &simple);
        println!("Mice: \n{}\n", &mice);
        assert!(diff, "Still at initial values :(");
    }

    #[test]
    fn ridge_neq_lin() {
        let data = Array2::from_shape_vec((8, 4), POINTS.to_vec()).unwrap();
        let ridge = Mice::new(2, "ridge", 1.0).fit_impl(&data).impute(&data);
        let lin = Mice::new(2, "linear", 1.0).fit_impl(&data).impute(&data);
        let diff = ridge.iter().zip(&lin).any(|(p, q)| (p - q).abs() > 1e-6);
        println!("ridge: \n{}\n", &ridge);
        println!("linear: \n{}\n", &lin);
        assert!(diff, "Still at initial values :(");
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
