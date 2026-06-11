use ndarray::Array2;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyString};
use std::collections::HashMap;

const SUPPORTED_TYPES: &str = "numpy.ndarray, pandas.DataFrame";
pub enum OUT {
    Numpy,
    DataFrame(Vec<String>),
}

#[derive(Debug)]
pub struct EncodingInfo {
    pub string_column_indices: Vec<usize>,
    pub _label_maps: HashMap<usize, HashMap<String, u64>>,
    pub reverse_maps: HashMap<usize, HashMap<u64, String>>,
}

pub enum StringEncoding {
    LabelEncoding,
}

fn label_encode(values: &[String]) -> (Vec<f64>, HashMap<String, u64>) {
    // Collect unique labels in sorted order.
    let mut unique: Vec<&str> = values.iter().map(String::as_str).collect();
    unique.sort_unstable();
    unique.dedup();

    let map: HashMap<String, u64> = unique
        .iter()
        .enumerate()
        .map(|(i, &s)| (s.to_owned(), i as u64))
        .collect();

    let encoded: Vec<f64> = values.iter().map(|s| map[s] as f64).collect();
    (encoded, map)
}
pub fn pyany_to_vec(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    string_encoding: &Option<StringEncoding>,
) -> PyResult<((Vec<f64>, usize, usize), OUT, Option<EncodingInfo>)> {
    // 1. numpy — try fast path first, fall back to buffer protocol
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let data = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok(((data, nrows, ncols), OUT::Numpy, None));
    }
    // buffer-protocol fallback for numpy (handles stale type cache)
    let type_name = obj
        .get_type()
        .qualname()
        .map(|s| s.to_string())
        .unwrap_or_default();
    if type_name == "ndarray" {
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        if shape.len() == 2 {
            let (nrows, ncols) = (shape[0], shape[1]);
            let dtype_name: String = obj.getattr("dtype")?.getattr("name")?.extract()?;
            return if dtype_name == "object" {
                match string_encoding {
                    None => panic!("Provide a way to encode String!"),
                    Some(enc) => {
                        let (data, enc_info) = encode_object_array(obj, nrows, ncols, enc);
                        Ok(((data, nrows, ncols), OUT::Numpy, Some(enc_info)))
                    }
                }
            } else {
                let data = read_via_buffer(py, obj, nrows, ncols)?;
                Ok(((data, nrows, ncols), OUT::Numpy, None))
            };
        }
    }

    // 2. pandas — never use isinstance on the derived numpy array
    let pandas = py.import("pandas")?;
    if obj.is_instance(&pandas.getattr("DataFrame")?)? {
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        let (nrows, ncols) = (shape[0], shape[1]);
        let columns: Vec<String> = obj
            .getattr("columns")?
            .try_iter()?
            .map(|item| {
                item.expect("no items found!")
                    .str()
                    .expect("Failed str conversion")
                    .to_string()
            })
            .collect();

        let has_strings = check_for_strs(&obj);
        return if has_strings {
            match string_encoding {
                Some(enc) => {
                    let (data, enc_info) = encode_dataframe(py, obj, nrows, ncols, &columns, enc)?;
                    Ok((
                        (data, nrows, ncols),
                        OUT::DataFrame(columns),
                        Some(enc_info),
                    ))
                }
                None => panic!("No String Encoding passed"),
            }
        } else {
            // All-numeric fast path.
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "float64")?;
            kwargs.set_item("copy", false)?;
            let np_any = obj.call_method("to_numpy", (), Some(&kwargs))?;
            let data = read_via_buffer(py, &np_any, nrows, ncols)?;
            Ok(((data, nrows, ncols), OUT::DataFrame(columns), None))
        };
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported type: '{}'. Supported types are: {}",
        type_name, SUPPORTED_TYPES
    )))
}

fn read_via_buffer(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Vec<f64>> {
    let buf = PyBuffer::<f64>::get(arr)?;
    if buf.dimensions() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Expected a 2-dimensional array",
        ));
    }
    let mut data = vec![0f64; nrows * ncols];
    buf.copy_to_slice(py, &mut data)?;
    Ok(data)
}

fn encode_object_array(
    arr: &Bound<'_, PyAny>,
    nrows: usize,
    ncols: usize,
    enc: &StringEncoding,
) -> (Vec<f64>, EncodingInfo) {
    let mut data = vec![0f64; nrows * ncols];
    let mut string_column_indices: Vec<usize> = Vec::new();
    let mut label_maps: HashMap<usize, HashMap<String, u64>> = HashMap::new();
    let mut reverse_maps: HashMap<usize, HashMap<u64, String>> = HashMap::new();

    for col_idx in 0..ncols {
        let mut numeric: Vec<f64> = Vec::with_capacity(nrows);
        // `strings` is only populated once a non-numeric element is found.
        let mut strings: Vec<String> = Vec::new();
        let mut is_string_col = false;

        for row_idx in 0..nrows {
            // numpy supports `arr[(row, col)]` integer tuple indexing.
            let elem = arr.get_item((row_idx, col_idx)).unwrap();

            if is_string_col {
                strings.push(elem.str().expect("cant convert to str").to_string());
            } else if let Ok(v) = elem.extract::<f64>() {
                numeric.push(v);
            } else {
                // First non-numeric element found — retroactively convert all
                // previously-seen numeric values to strings so the column is
                // encoded uniformly.
                is_string_col = true;
                strings = numeric.drain(..).map(|v| v.to_string()).collect();
                strings.push(elem.str().expect("cant convert to str").to_string());
            }
        }

        if is_string_col {
            let (encoded, map) = match enc {
                StringEncoding::LabelEncoding => label_encode(&strings),
            };
            for (row_idx, val) in encoded.into_iter().enumerate() {
                data[row_idx * ncols + col_idx] = val;
            }
            string_column_indices.push(col_idx);
            let reverse: HashMap<u64, String> = map.iter().map(|(k, &v)| (v, k.clone())).collect();
            label_maps.insert(col_idx, map);
            reverse_maps.insert(col_idx, reverse);
        } else {
            // Purely numeric column — write collected values directly.
            for (row_idx, val) in numeric.into_iter().enumerate() {
                data[row_idx * ncols + col_idx] = val;
            }
        }
    }

    (
        data,
        EncodingInfo {
            string_column_indices,
            _label_maps: label_maps,
            reverse_maps,
        },
    )
}

fn encode_dataframe(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    nrows: usize,
    ncols: usize,
    columns: &[String],
    enc: &StringEncoding, // only LabelEncoding exists today; kept for future variants
) -> PyResult<(Vec<f64>, EncodingInfo)> {
    let mut data = vec![0f64; nrows * ncols];
    let mut string_column_indices: Vec<usize> = Vec::new();
    let mut label_maps: HashMap<usize, HashMap<String, u64>> = HashMap::new();
    let mut reverse_maps: HashMap<usize, HashMap<u64, String>> = HashMap::new();

    let mut col_buffer = vec![0f64; nrows];
    for (col_idx, col_name) in columns.iter().enumerate() {
        let col = df.get_item(col_name)?; // pandas Series
        let dtype_name: String = col.getattr("dtype")?.getattr("name")?.extract()?;

        if ["object", "str"].contains(&dtype_name.as_str()) {
            // ── String column: iterate, encode ───────────────────────────────
            let raw: Vec<String> = col
                .try_iter()?
                .map(|item| item.and_then(|v| Ok(v.str()?.to_string())))
                .collect::<PyResult<_>>()?;

            let (encoded, map) = match enc {
                StringEncoding::LabelEncoding => label_encode(&raw),
            };
            for (row_idx, val) in encoded.into_iter().enumerate() {
                data[row_idx * ncols + col_idx] = val;
            }
            string_column_indices.push(col_idx);
            let reverse: HashMap<u64, String> = map.iter().map(|(k, &v)| (v, k.clone())).collect();
            reverse_maps.insert(col_idx, reverse);
            label_maps.insert(col_idx, map);
        } else {
            // ── Numeric column: buffer protocol ──────────────────────────────
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "float64")?;
            let col_np = col.call_method("to_numpy", (), Some(&kwargs))?;
            let buf = PyBuffer::<f64>::get(&col_np)?;
            buf.copy_to_slice(py, &mut col_buffer)?;
            for (row_idx, &val) in col_buffer.iter().enumerate() {
                data[row_idx * ncols + col_idx] = val;
            }
        }
    }

    Ok((
        data,
        EncodingInfo {
            string_column_indices,
            _label_maps: label_maps,
            reverse_maps,
        },
    ))
}
fn check_for_strs(obj: &Bound<'_, PyAny>) -> bool {
    let dtypes = obj.getattr("dtypes").expect("dtypes attr not found"); // pandas Series of dtype objects
    let mut found = false;
    for dtype_res in dtypes.try_iter().unwrap() {
        let dtype = dtype_res.expect("No dtypes found.");

        let kind: String = dtype
            .getattr("kind")
            .expect("kind attr not found")
            .extract()
            .unwrap();
        let name: String = dtype
            .getattr("name")
            .expect("name attr not found")
            .extract()
            .unwrap();

        if kind == "O" || name == "string" {
            found = true;
            break;
        }
    }
    found
}

pub fn arr_to_out<'py>(
    py: Python<'py>,
    arr: &Array2<f64>,
    out: OUT,
    enc_info: Option<EncodingInfo>,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        OUT::Numpy => {
            let mut out = arr.view().to_pyarray(py).into_any();
            if let Some(enc) = enc_info {
                // cast to object array then overwrite string columns
                let np = py.import("numpy")?;
                let obj_arr = np
                    .call_method1("array", (&out,))?
                    .call_method1("astype", ("object",))?;
                for &col_idx in &enc.string_column_indices {
                    let rev = &enc.reverse_maps[&col_idx];
                    for row_idx in 0..arr.nrows() {
                        let encoded = arr[(row_idx, col_idx)] as u64;
                        let label = rev.get(&encoded).map(String::as_str).unwrap_or("NaN");
                        obj_arr.call_method1("__setitem__", ((row_idx, col_idx), label))?;
                    }
                }
                out = obj_arr.into_any();
            }
            Ok(out)
        }
        OUT::DataFrame(columns) => {
            let pd = PyModule::import(py, "pandas")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("columns", &columns)?;
            let df = pd
                .getattr("DataFrame")?
                .call((arr.view().to_pyarray(py),), Some(&kwargs))?;

            if let Some(enc) = enc_info {
                for &col_idx in &enc.string_column_indices {
                    let rev = &enc.reverse_maps[&col_idx];
                    let decoded: Vec<Py<PyAny>> = (0..arr.nrows())
                        .map(|r| {
                            let v = arr[(r, col_idx)];
                            if v.is_nan() {
                                py.None()
                            } else {
                                rev.get(&(v as u64))
                                    .map(|s| PyString::new(py, s).into_any().unbind())
                                    .unwrap_or_else(|| py.None())
                            }
                        })
                        .collect();
                    // cast column to object dtype first
                    df.call_method1(
                        "__setitem__",
                        (
                            &columns[col_idx],
                            df.get_item(&columns[col_idx])?
                                .call_method1("astype", ("object",))?,
                        ),
                    )?;

                    // now assign strings
                    let loc = df.getattr("loc")?;
                    loc.set_item((pyo3::types::PySlice::full(py), &columns[col_idx]), decoded)?;
                }
            }
            Ok(df.into_any())
        }
    }
}
