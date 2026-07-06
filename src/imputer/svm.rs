use crate::imputer::SimpleImputer;
use crate::utils::{self, StringEncoding};
use libsvm_rs::{
    train, KernelType, SvmModel, SvmNode, SvmParameter, SvmParameterBuilder, SvmProblem, SvmType, set_quiet
};
use ndarray::{Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(name = "SVMImputerRS", module = "gouda.gouda")]
#[derive(Serialize, Deserialize)]
pub struct SVMImputer {
    models: Vec<SvmModel>,
    string_encoding: Option<StringEncoding>,
    _cat_cols: Vec<usize>,
    num_cols: Vec<usize>,
    init: SimpleImputer,
    #[pyo3(get)]
    is_fitted: bool,
    kernel: KernelType,
}

const ALLOWED_KERNELS: &[&str] = &["linear", "rbf", "sigmoid", "polynomial"];

#[pymethods]
impl SVMImputer {
    #[new]
    #[pyo3(signature = (kernel="linear", encoding=None))]
    pub fn new(kernel: &str, encoding: Option<&str>) -> SVMImputer {
        set_quiet(true);
        // assert!(ALLOWED_WEIGHTS.contains(&weights));
        // SVMImputer::sanity_check(&metric, &weights);
        SVMImputer {
            models: Vec::new(),
            is_fitted: false,
            kernel: match kernel.to_lowercase().as_str() {
                "linear" => KernelType::Linear,
                "rbf" => KernelType::Rbf,
                "sigmoid" => KernelType::Sigmoid,
                "polynomial" => KernelType::Polynomial,
                // "precomputed" => KernelType::Precomputed,
                _ => panic!("kernel parameter not supported, {:?}", ALLOWED_KERNELS),
            },
            _cat_cols: Vec::new(),
            num_cols: Vec::new(),
            string_encoding: match encoding {
                None => None,
                Some(_) => Some(StringEncoding::LabelEncoding),
            },
            init: SimpleImputer::new(encoding),
        }
    }

    pub fn fit(slf: Py<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        {
            let mut inner = slf.borrow_mut(py);
            let (arr, _out, _enc) = utils::pyany_to_vec(data, &inner.string_encoding)?;
            utils::raise_if_nan_col(arr.view())?;

            if let Some(enc) = _enc {
                let indices = enc.string_column_indices;
                inner.num_cols = (0..arr.ncols())
                    .into_iter()
                    .filter(|idx| !indices.contains(idx))
                    .collect();
                inner._cat_cols = indices;
            }
            inner.fit_model(arr.view());
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
            return Err(utils::raise_not_fitted(py));
        }
        let (arr, out, enc) = utils::pyany_to_vec(data, &self.string_encoding)?;
        utils::check_feature_mismatch(self.models.len(), arr.ncols())?;
        // actual method
        let imputed = self.impute(arr.view());
        // return python object
        utils::arr_to_out(py, &imputed, out, enc)
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
    fn kernel(&self) -> &str {
        match self.kernel {
            KernelType::Linear => "linear",
            KernelType::Rbf => "rbf",
            KernelType::Polynomial => "polynomial",
            KernelType::Sigmoid => "sigmoid",
            KernelType::Precomputed => "precomputed",
        }
    }

    #[getter]
    fn encoding(&self) -> Option<&str> {
        match self.string_encoding {
            None => None,
            Some(_) => Some("label"),
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to pickle SVMImputer: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let decoded: SVMImputer = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to unpickle SVMImputer: {e}"))
        })?;
        self.models = decoded.models;
        self.string_encoding = decoded.string_encoding;
        self.is_fitted = decoded.is_fitted;
        self._cat_cols = decoded._cat_cols;
        self.num_cols = decoded.num_cols;
        self.kernel = decoded.kernel;
        self.init = decoded.init;
        Ok(())
    }
}

impl SVMImputer {
    fn fit_model(&mut self, arr: ArrayView2<f64>) -> &SVMImputer {
        let imputed = self
            .init
            .fit_impl(
                arr,
                if self._cat_cols.len() == 0 {
                    None
                } else {
                    Some(&self._cat_cols)
                },
            )
            .impute(arr);
        self.models = (0..arr.ncols())
            .into_par_iter()
            .map(|i| {
                let training_params = self.get_model_params(i);
                let problem = create_problem(imputed.view(), (i, arr.column(i)), false);
                train::svm_train(&problem, &training_params)
            })
            .collect();
        self
    }
    fn impute(&self, arr: ArrayView2<f64>) -> Array2<f64> {
        let imputed = self.init.impute(arr);
        let imputed: Vec<f64> = (0..arr.ncols())
            .into_par_iter()
            .flat_map(|i| {
                let problem = create_problem(imputed.view(), (i, arr.column(i)), true);
                (0..problem.instances.len())
                    .into_par_iter()
                    .map(move |row| {
                        libsvm_rs::predict::predict(&self.models[i], &problem.instances[row])
                    })
            })
            .collect();
        let mut arr = arr.to_owned();
        let mut counter = 0;
        for v in &mut arr {
            if !v.is_nan() {
                continue;
            }
            *v = imputed[counter];
            counter += 1;
            if counter > imputed.len() {
                break;
            }
        }
        arr
    }

    fn get_model_params(&self, target_column: usize) -> SvmParameter {
        let svm_type = if self._cat_cols.len() > 0 && !self.num_cols.contains(&target_column) {
            SvmType::CSvc
        } else {
            SvmType::EpsilonSvr
        };
        SvmParameterBuilder::new()
            .svm_type(svm_type)
            .kernel_type(self.kernel)
            .build()
            .unwrap()
    }
}
fn create_problem(
    data: ArrayView2<f64>,
    target_column: (usize, ArrayView1<f64>),
    only_nans: bool,
) -> SvmProblem {
    let (left, right) = data.view().split_at(ndarray::Axis(1), target_column.0);
    let left_rows = left.rows();
    let right_rows = right.rows();

    let instances: Vec<Vec<SvmNode>> = left_rows
        .into_iter()
        .zip(right_rows)
        .enumerate()
        .filter(|(idx, _)| {
            if only_nans {
                target_column.1[*idx].is_nan()
            } else {
                !target_column.1[*idx].is_nan()
            }
        })
        .map(|(_, (lrow, rrow))| {
            let mut nodes: Vec<SvmNode> = Vec::new();
            // left side features: 1..i
            nodes.extend(
                lrow.iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan() && v.abs() > 0.0)
                    .map(|(j, &v)| SvmNode {
                        index: (j + 1) as i32,
                        value: v,
                    }),
            );

            // right side features: (i+1..)
            nodes.extend(
                rrow.iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan() && v.abs() > 0.0)
                    .map(|(j, &v)| SvmNode {
                        index: (target_column.0 + j + 1) as i32,
                        value: v,
                    }),
            );

            nodes
        })
        .collect();
    SvmProblem {
        labels: target_column.1.to_vec(),
        instances,
    }
}
// Distance Functions
impl SVMImputer {}

#[cfg(test)]
mod tests {
    use super::*; // has access to everything, including private
}
