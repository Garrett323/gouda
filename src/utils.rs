use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::ops::Index;

pub const NOT_FITTED_ERR: &str = "Imputer not fitted, please call fit first";
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

pub struct Data {
    pub nrows: usize,
    pub ncols: usize,
    data: Option<Vec<f64>>,
    data_cols: Option<Vec<f64>>,
}

impl Data {
    pub fn new(nrows: usize, ncols: usize, data: &[f64]) -> Data {
        let data_cols = Data::transpose(nrows, ncols, &data);
        Data {
            nrows,
            ncols,
            data: Some(data.to_vec()),
            data_cols: Some(data_cols),
        }
    }

    pub fn new_rowmayor(nrows: usize, ncols: usize, data: &[f64]) -> Data {
        Data {
            nrows,
            ncols,
            data: Some(data.to_vec()),
            data_cols: None,
        }
    }

    pub fn new_colmayor(nrows: usize, ncols: usize, data: &[f64]) -> Data {
        let data_cols = Data::transpose(nrows, ncols, &data);
        Data {
            nrows,
            ncols,
            data: None,
            data_cols: Some(data_cols),
        }
    }

    pub fn len(&self) -> usize {
        self.ncols * self.nrows
    }

    // fn iter(&self) -> std::slice::Iter<'_, f64> {
    //     match (self.data.as_ref(), self.data_cols.as_ref()) {
    //         (Some(v), None) => v.iter(),
    //         (None, Some(v)) => v.iter(),
    //         (Some(v), Some(_)) => v.iter(),
    //         (None, None) => panic!("No data available"),
    //     }
    // }

    fn transpose(nrows: usize, ncols: usize, data: &[f64]) -> Vec<f64> {
        let mut columns = vec![0.0; data.len()];
        for c in 0..ncols {
            for r in 0..nrows {
                columns[c + r * ncols] = data[c * nrows + r];
            }
        }
        columns
    }

    pub fn get_col(&self, col: usize) -> &[f64] {
        let offset = col * self.nrows;
        &self.data_cols.as_ref().unwrap()[offset..offset + self.nrows]
    }
}

impl Index<usize> for Data {
    type Output = [f64];

    fn index(&self, row: usize) -> &[f64] {
        let offset = row * self.ncols;
        &self.data.as_ref().unwrap()[offset..offset + self.ncols]
    }
}

impl std::ops::Deref for Data {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        match (&self.data, &self.data_cols) {
            (Some(v), _) => v,
            (None, Some(v)) => {
                println!("EEEE");
                v
            }
            (None, None) => panic!("No data to return"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DATA: &[f64] = &[
        0.27839605, 0.7338148, 0.98359227, 0.98189233, 0.45384631, 0.9317995, 0.51597243,
        0.38054457, 0.62366235, 0.12229672, 0.68846839, 0.53889275, 0.44453843, 0.43416536,
        0.18575075, 0.13333331, 0.8772666, 0.64398646, 0.1214414, 0.90529859, 0.69819416,
        0.65251852, 0.39663618, 0.65702538, 0.92424,
    ];

    #[test]
    fn test_transpose() {
        let data = Data::new(5, 5, TEST_DATA);
        let colmayor = &[
            0.27839605, 0.9317995, 0.68846839, 0.13333331, 0.69819416, 0.7338148, 0.51597243,
            0.53889275, 0.8772666, 0.65251852, 0.98359227, 0.38054457, 0.44453843, 0.64398646,
            0.39663618, 0.98189233, 0.62366235, 0.43416536, 0.1214414, 0.65702538, 0.45384631,
            0.12229672, 0.18575075, 0.90529859, 0.92424,
        ];
        let transpose = Data::transpose(data.nrows, data.ncols, colmayor);

        for i in 0..colmayor.len() {
            assert!((data.data_cols.as_ref().unwrap()[i] - colmayor[i]).abs() < 1e-10);
            assert!((data.data.as_ref().unwrap()[i] - transpose[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_getcol() {
        assert!(true);
    }
}
