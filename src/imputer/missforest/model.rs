use super::super::SimpleImputer;
use crate::{
    imputer::missforest::backend::{self, RandomForest},
    utils::{
        StringEncoding, arr_to_out,
        constants::{ENCODING_WARN, NOT_FITTED_ERR},
        pyany_to_vec,
    },
};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass]
pub struct MissForest {
    is_fitted: bool,
    init: SimpleImputer,
    forrests: Vec<RandomForest>,
    n_trees: usize,
    rng: StdRng,
    max_depth: usize,
    min_samples_leaf: usize,
    string_encoding: Option<StringEncoding>,
}

#[pymethods]
impl MissForest {
    #[new]
    #[pyo3(signature = (n_trees=15, max_depth=15, min_samples_leaf=5, seed=None, encoding=None))]
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_leaf: usize,
        seed: Option<u64>,
        encoding: Option<&str>,
    ) -> MissForest {
        let rng = if let Some(x) = seed {
            StdRng::seed_from_u64(x)
        } else {
            let mut rng = rand::rng();
            StdRng::seed_from_u64(rng.random())
        };
        MissForest {
            n_trees,
            is_fitted: false,
            init: SimpleImputer::new(encoding),
            forrests: Vec::new(),
            rng,
            max_depth,
            min_samples_leaf,
            string_encoding: match encoding {
                None => None,
                Some(_) => Some(StringEncoding::LabelEncoding),
            },
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        pyo3::PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            c"MissForest is currently experimental and may produce unexpected results.",
            1,
        )?;
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
            let (arr, _out, _enc) = pyany_to_vec(py, data, &inner.string_encoding)?;
            inner.fit_impl(&arr);
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
        let (arr, out, _enc) = pyany_to_vec(py, data, &self.string_encoding)?;
        let imputed = self.impute(&arr);
        // return python object
        arr_to_out(py, &imputed, out, _enc)
    }
}

impl MissForest {
    fn fit_impl(&mut self, data: &Array2<f64>) -> &Self {
        let ncols = data.ncols();
        self.forrests = (0..ncols)
            .map(|_| {
                backend::RandomForest::new(
                    self.n_trees,
                    self.rng.random(),
                    self.max_depth,
                    self.min_samples_leaf,
                )
            })
            .collect();

        self.init.fit_impl(data, None);
        let mut cur = self.init.impute(data);
        let mut nxt = cur.clone();

        let mut gamma_old = f64::MAX;
        let mut gamma_new = f64::MAX - f64::EPSILON;
        // let mut best_forests = self.forrests.clone();

        while gamma_old > gamma_new {
            gamma_old = gamma_new;
            // best_forests = self.forrests.clone();

            nxt.axis_iter_mut(Axis(1))
                .into_par_iter()
                .zip(self.forrests.par_iter_mut())
                .enumerate()
                .for_each(|(i, (mut col, forest))| {
                    let left = cur.slice(ndarray::s![.., ..i]);
                    let right = cur.slice(ndarray::s![.., i + 1..]);
                    let sliced = ndarray::concatenate(Axis(1), &[left, right]).unwrap();
                    forest.fit(&sliced, cur.column(i));
                    col.assign(&forest.transform(&sliced));
                });
            // for i in 0..ncols {
            //     let left  = cur.slice(ndarray::s![.., ..i]);
            //     let right = cur.slice(ndarray::s![.., i+1..]);
            //     let predictors = ndarray::concatenate(Axis(1), &[left, right]).unwrap();
            //
            //     self.forrests[i].fit(&predictors, cur.column(i));
            //     nxt.column_mut(i).assign(&self.forrests[i].transform(&predictors));
            // }
            //
            gamma_new = self.calc_gamma(cur.view(), nxt.view());
            std::mem::swap(&mut cur, &mut nxt); // free — just swaps pointers
        }
        // self.forrests = best_forests.clone();
        self
    }
    fn impute(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut imputed = self.init.impute(data);
        let static_imputed = imputed.clone();
        imputed
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(self.forrests.par_iter())
            .enumerate()
            .for_each(|(i, (mut col, forest))| {
                let left = static_imputed.slice(ndarray::s![.., ..i]);
                let right = static_imputed.slice(ndarray::s![.., i + 1..]);
                let sliced = ndarray::concatenate(Axis(1), &[left, right]).unwrap();
                col.assign(&forest.transform(&sliced));
            });
        imputed
    }

    fn calc_gamma(&self, old: ArrayView2<f64>, new: ArrayView2<f64>) -> f64 {
        let mut gamma = 0.0;
        assert!(old.ncols() == new.ncols());
        for index in 0..new.ncols() {
            gamma += criterion_num(old.column(index), new.column(index));
        }
        gamma
    }
}

fn criterion_num(old: ArrayView1<f64>, new: ArrayView1<f64>) -> f64 {
    let mut sum_of_squares = 0.0;
    let mut total = f64::EPSILON;
    for (n, o) in new.iter().zip(old) {
        sum_of_squares += (n - o) * (n - o);
        total += n.powi(2);
    }
    sum_of_squares / total
}

fn criterion_cat(n_miss: usize, old: &[u64], new: &[u64]) -> f64 {
    let equal: u64 = new.iter().zip(old).map(|(n, o)| (n == o) as u64).sum();
    equal as f64 / n_miss as f64
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn num_perfect_imputation() {
        // When new == old the numerator is 0 → γ = 0.
        let vals = Array1::from_vec([1.0, 2.0, 3.0].to_vec());
        assert_eq!(criterion_num(vals.view(), vals.view()), 0.0);
    }

    #[test]
    fn num_criterion() {
        // new = [2, 4], old = [1, 3]
        // numerator = (2-1)² + (4-3)² = 1 + 1 = 2
        // denominator = 2² + 4² = 4 + 16 = 20
        // γ = 2/20 = 0.1
        let new = Array1::from_vec([2.0_f64, 4.0].to_vec());
        let old = Array1::from_vec([1.0_f64, 3.0].to_vec());
        let g = criterion_num(old.view(), new.view());
        assert!((g - 0.1).abs() < 1e-12, "expected 0.1, got {g}");
    }
}
