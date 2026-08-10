use ndarray::ArrayView2;
use rayon::prelude::*;
pub mod constants;
use std::collections::HashMap;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use pyglue::{StringEncoding, arr_to_out, pyany_to_vec};
pub use pyglue::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};
