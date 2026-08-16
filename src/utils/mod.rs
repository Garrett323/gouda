pub mod constants;
mod errors;
pub use errors::Errors;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use pyglue::{StringEncoding, arr_to_out, pyany_to_vec};
pub use pyglue::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};

pub fn process_labelencoding(encoding: Option<&str>) -> Result<Option<StringEncoding>, Errors> {
    match encoding {
        None => Ok(None),
        Some("label") => Ok(Some(StringEncoding::LabelEncoding)),
        Some(value) => {
            return Err(Errors::UnsupportedValue {
                parameter: "LabelEncoding",
                value: value.to_owned(),
                supported: Some(&["label", "None"]),
            }
            .into());
        }
    }
}
