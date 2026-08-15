pub mod constants;
use ndarray_linalg::lobpcg::LobpcgResult::Err;
use pyo3::prelude::*;
use std::fmt;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use pyglue::{StringEncoding, arr_to_out, pyany_to_vec};
pub use pyglue::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};

#[derive(Debug)]
pub enum Errors {
    NotFitted,
    NoValidOp {
        operation: String,
    },
    UnsupportedValue {
        parameter: &'static str,
        value: String,
        supported: Option<&'static [&'static str]>,
    },
    Shape(ndarray::ShapeError),
    LinearAlgebra(ndarray_linalg::error::LinalgError),
    SvmTraining {
        column: usize,
        message: String,
    },
}

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Errors::LinearAlgebra(err) => {
                write!(f, "Unable to compute least squares solution! {}", err)
            }
            Errors::NotFitted => {
                write!(f, "This imputer is not fitted. Call fit before transform.")
            }
            Errors::Shape(err) => write!(f, "Shape Mismatch: {}", err),
            Errors::UnsupportedValue {
                parameter,
                value,
                supported,
            } => {
                write!(
                    f,
                    "{}: [{}] is not supported {:?}",
                    parameter, value, supported
                )
            }
            Errors::NoValidOp { operation } => write!(f, "Not a valid operation: {}", operation),
            Errors::SvmTraining { column, message } => write!(
                f,
                "Failed to train SVM at column: {} because {}",
                column, message
            ),
        }
    }
}

impl std::error::Error for Errors {}

use pyo3::exceptions::{PyRuntimeError, PyValueError};

impl From<Errors> for PyErr {
    fn from(err: Errors) -> PyErr {
        match err {
            Errors::UnsupportedValue { .. } | Errors::Shape(_) => {
                PyValueError::new_err(err.to_string())
            }
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
