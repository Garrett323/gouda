use crate::imputer::SimpleImputer;
use crate::utils::{self, Errors, StringEncoding};
use libsvm_rs::{
    KernelType, SvmError, SvmModel, SvmNode, SvmParameter, SvmParameterBuilder, SvmProblem,
    SvmType, set_quiet, train,
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
    cat_cols: Vec<usize>,
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
    pub fn new(kernel: &str, encoding: Option<&str>) -> PyResult<SVMImputer> {
        set_quiet(true);
        Ok(SVMImputer {
            models: Vec::new(),
            is_fitted: false,
            kernel: match kernel.to_lowercase().as_str() {
                "linear" => KernelType::Linear,
                "rbf" => KernelType::Rbf,
                "sigmoid" => KernelType::Sigmoid,
                "polynomial" => KernelType::Polynomial,
                // "precomputed" => KernelType::Precomputed,
                value => {
                    return Err(Errors::UnsupportedValue {
                        parameter: "SVM.Kernel",
                        value: value.to_owned(),
                        supported: Some(ALLOWED_KERNELS),
                    }
                    .into());
                }
            },
            cat_cols: Vec::new(),
            num_cols: Vec::new(),
            string_encoding: utils::process_labelencoding(encoding)?,
            init: SimpleImputer::new(encoding)?,
        })
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
                inner.cat_cols = indices;
            }
            inner.fit_model(arr.view())?;
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
        let imputed = self.impute(arr.view())?;
        // return python object
        utils::arr_to_out(py, &imputed, out, enc.as_ref())
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
        self.cat_cols = decoded.cat_cols;
        self.num_cols = decoded.num_cols;
        self.kernel = decoded.kernel;
        self.init = decoded.init;
        Ok(())
    }
}

impl SVMImputer {
    fn fit_model(&mut self, arr: ArrayView2<f64>) -> Result<&SVMImputer, Errors> {
        let imputed = self
            .init
            .fit_impl(
                arr,
                if self.cat_cols.len() == 0 {
                    None
                } else {
                    Some(&self.cat_cols)
                },
            )?
            .impute(arr)?;
        self.models = Vec::with_capacity(arr.ncols());
        let models: Vec<Result<SvmModel, SvmError>> = (0..arr.ncols())
            .into_par_iter()
            .map(|i| {
                let training_params = self.get_model_params(i)?;
                let problem = create_problem(imputed.view(), (i, arr.column(i)), false);
                Ok(train::svm_train(&problem, &training_params))
            })
            .collect();
        for (col, m) in models.into_iter().enumerate() {
            match m {
                Ok(model) => self.models.push(model),
                Err(err) => {
                    return Err(Errors::SvmTraining {
                        column: col,
                        message: err.to_string(),
                    });
                }
            }
        }
        Ok(self)
    }
    fn impute(&self, arr: ArrayView2<f64>) -> Result<Array2<f64>, Errors> {
        let initialized = self.init.impute(arr)?;
        let mut output = arr.to_owned();
        for column in 0..arr.ncols() {
            let target = arr.column(column);

            // Remember which rows are missing in this column.
            let missing_rows: Vec<usize> = target
                .iter()
                .enumerate()
                .filter_map(|(row, value)| if value.is_nan() { Some(row) } else { None })
                .collect();
            if missing_rows.is_empty() {
                continue;
            }
            // Build one prediction instance for each missing row.
            let problem = create_problem(initialized.view(), (column, target), true);
            debug_assert_eq!(missing_rows.len(), problem.instances.len(),);
            for (row, instance) in missing_rows.into_iter().zip(problem.instances.iter()) {
                let prediction = libsvm_rs::predict::predict(&self.models[column], instance);
                output[(row, column)] = prediction;
            }
        }
        Ok(output)
    }

    fn get_model_params(&self, target_column: usize) -> Result<SvmParameter, SvmError> {
        let svm_type = if self.cat_cols.len() > 0 && !self.num_cols.contains(&target_column) {
            SvmType::CSvc
        } else {
            SvmType::EpsilonSvr
        };
        SvmParameterBuilder::new()
            .svm_type(svm_type)
            .kernel_type(self.kernel)
            .build()
    }
}
fn create_problem(
    data: ArrayView2<f64>,
    target_column: (usize, ArrayView1<f64>),
    only_nans: bool,
) -> SvmProblem {
    let (left, right) = data.view().split_at(ndarray::Axis(1), target_column.0);
    let (_, right) = right.view().split_at(ndarray::Axis(1), 1);
    let right_rows = right.rows();
    let left_rows = left.rows();

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
        labels: target_column
            .1
            .iter()
            .filter(|&v| !v.is_nan())
            .copied()
            .collect(),
        instances,
    }
}
// Distance Functions
impl SVMImputer {}

#[cfg(test)]
mod tests {
    use super::*; // has access to everything, including private
    #[test]
    fn create_problem_does_not_leak_target_column() {
        let data = ndarray::array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0],];

        // Column 1 is the target: [10, 20, 30]
        let target_column = data.column(1);

        let problem = create_problem(data.view(), (1, target_column), false);

        // We should have one training instance per non-NaN target.
        assert_eq!(problem.instances.len(), 3);
        assert_eq!(problem.labels, vec![10.0, 20.0, 30.0]);

        for instance in &problem.instances {
            let values: Vec<_> = instance.iter().map(|node| node.value).collect();
            println!("{:?}", values);
            for v in values {
                for t in target_column {
                    assert!((v - t).abs() > 1e-8, "target feature leaked");
                }
            }
        }
    }
}
