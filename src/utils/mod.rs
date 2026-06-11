pub mod constants;
mod python;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use python::{StringEncoding, arr_to_out, pyany_to_vec};
