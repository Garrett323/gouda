use std::ffi::CStr;

pub const NOT_FITTED_ERR: &str = "Imputer not fitted, please call fit first";
pub const ENCODING_WARN: &CStr =
    c"Encoding Parameter is passed, but categorical handling is incomplete";
