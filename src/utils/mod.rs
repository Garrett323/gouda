use ndarray::ArrayView2;
use rayon::prelude::*;
pub mod constants;
mod python;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use python::{StringEncoding, arr_to_out, pyany_to_vec};
pub use python::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};

pub fn all_empty_column(arr: ArrayView2<f64>) -> Result<(), Vec<usize>> {
    let all_nan_cols: Vec<usize> = (0..arr.ncols())
        .into_par_iter()
        .filter_map(|i| {
            for v in arr.column(i) {
                if !v.is_nan() {
                    return None;
                }
            }
            Some(i)
        })
        .collect();
    if all_nan_cols.len() == 0 {
        Ok(())
    } else {
        Err(all_nan_cols)
    }
}
