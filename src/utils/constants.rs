use std::ffi::{CStr, CString};

pub const NOT_FITTED_ERR: &str = "Imputer not fitted, please call fit first";
pub const ENCODING_WARN: &CStr =
    c"Encoding Parameter is passed, but categorical handling is incomplete";

pub fn encoding_warn(model_name: &str) -> CString {
    CString::new(format!("{:?} [{}]", ENCODING_WARN.to_owned(), model_name))
        .unwrap_or(ENCODING_WARN.to_owned())
}
