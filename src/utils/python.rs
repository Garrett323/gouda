use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

const SUPPORTED_TYPES: &str = "numpy.ndarray, pandas.DataFrame";

pub fn pyany_to_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    // 1. numpy
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        // as_slice() gives the flat buffer directly if C-contiguous
        let data = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(), // non-contiguous fallback
        };
        return Ok((data, nrows, ncols));
    }

    // 2. pandas
    let pandas = py.import("pandas")?;
    if obj.is_instance(&pandas.getattr("DataFrame")?)? {
        let np_module = py.import("numpy")?;
        let np = np_module.call_method1("ascontiguousarray", (obj.call_method0("to_numpy")?,))?;
        let arr = np.extract::<PyReadonlyArray2<f64>>()?;
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let data = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok((data, nrows, ncols));
    }

    let type_name = obj
        .get_type()
        .qualname()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported type: '{}'. Supported types are: {}",
        type_name, SUPPORTED_TYPES
    )))
}
