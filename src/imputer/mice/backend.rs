use crate::utils::Errors::{LinearAlgebra, NotFitted};
use crate::utils::{Errors, SendPtr};
use ndarray::{Array1, Array2, ArrayView1, Axis, ShapeError};
use ndarray_linalg::{LeastSquaresSvd, SVD};
use serde::{Deserialize, Serialize};

use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
pub enum Solver {
    Linear(LinearRegression),
    Logistic(LogisticRegression),
    PMM(PMM),
    Ridge(Ridge),
}

fn default_rng() -> Mutex<SmallRng> {
    Mutex::new(SmallRng::seed_from_u64(42))
}

impl Solver {
    pub fn predict(&self, arr: &Array2<f64>) -> Result<Array1<f64>, Errors> {
        match self {
            Solver::Linear(model) => model.predict(arr),
            Solver::Logistic(model) => model.predict(arr),
            Solver::PMM(model) => model.predict(arr),
            Solver::Ridge(model) => model.predict(arr),
        }
    }

    pub fn fit(&mut self, arr: &Array2<f64>, target: &Array1<f64>) -> Result<(), Errors> {
        match self {
            Solver::Linear(model) => model.fit(arr, target),
            Solver::Logistic(model) => model.fit(arr, target),
            Solver::PMM(model) => model.fit(arr, target),
            Solver::Ridge(model) => model.fit(arr, target),
        }
    }

    pub fn coefficients(&self) -> Option<ArrayView1<'_, f64>> {
        match self {
            Solver::Linear(model) => model.coefficients(),
            Solver::Logistic(model) => model.coefficients(),
            Solver::PMM(model) => model.coefficients(),
            Solver::Ridge(model) => model.coefficients(),
        }
    }

    pub fn bias(&self) -> bool {
        match self {
            Solver::Linear(model) => model.bias(),
            Solver::Logistic(model) => model.bias(),
            Solver::PMM(model) => model.bias(),
            Solver::Ridge(model) => model.bias(),
        }
    }
}

pub trait Regressor: Send + Sync {
    fn bias(&self) -> bool;
    fn coefficients(&self) -> Option<ArrayView1<'_, f64>>;
    fn predict(&self, points: &Array2<f64>) -> Result<Array1<f64>, Errors> {
        let points = if self.bias() {
            &add_bias_column(points)
        } else {
            points
        };
        let weights = self.coefficients().ok_or(NotFitted)?;
        Ok(points.dot(&weights))
    }
}

#[derive(Serialize, Deserialize)]
pub struct PMM {
    n_neighbors: usize,
    pool: Option<Array1<f64>>,
    model: Box<Solver>,
    #[serde(skip, default = "default_rng")]
    rng: Mutex<rand::rngs::SmallRng>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    bias: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LogisticRegression {
    coefficients: Option<Array2<f64>>,
    n_classes: usize,
    bias: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Ridge {
    alpha: f64,
    coefficients: Option<Array1<f64>>,
    bias: bool,
}
impl Ridge {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            coefficients: None,
            bias: true,
        }
    }

    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> Result<(), Errors> {
        let x_mean = data.mean_axis(Axis(0)).ok_or(Errors::NoValidOp {
            operation: "compute mean on empty array".to_string(),
        })?; // original mean, keep this
        let data = data - &x_mean;

        let (u, e, v) = data
            .svd(true, true)
            .map_err(|err| Errors::LinearAlgebra(err))?;
        let u = u.ok_or(Errors::NoValidOp {
            operation: "Failed SVD decomposition!".to_string(),
        })?;
        let u = u.slice(ndarray::s![.., ..e.len()]);
        let v = v.ok_or(Errors::NoValidOp {
            operation: "Failed SVD decomposition!".to_string(),
        })?;
        let v = v.slice(ndarray::s![..e.len(), ..]);

        let d = &e / (&e.mapv(|x| x * x) + self.alpha);
        let uty = u.t().dot(target);
        let mut beta = v.t().dot(&(d * uty));

        if self.bias {
            let y_mean = target.mean().ok_or(Errors::NoValidOp {
                operation: "Computing mean on empty list!".to_string(),
            })?; // move up here too
            let intercept = Array1::from_elem(1, y_mean - x_mean.dot(&beta));
            beta = ndarray::concatenate(Axis(0), &[intercept.view(), beta.view()])
                .map_err(|err| Errors::Shape(err))?;
        }

        self.coefficients = Some(beta);
        Ok(())
    }
}
impl Regressor for Ridge {
    fn bias(&self) -> bool {
        self.bias
    }
    fn coefficients(&self) -> Option<ArrayView1<'_, f64>> {
        Some(self.coefficients.as_ref()?.view())
    }
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficients: None,
            bias: true,
        }
    }

    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> Result<(), Errors> {
        let data = if self.bias {
            &add_bias_column(data)
        } else {
            data
        };
        self.coefficients = Some(
            data.least_squares(&target)
                .map_err(|err| LinearAlgebra(err))?
                .solution,
        );
        Ok(())
    }
}
impl Regressor for LinearRegression {
    fn bias(&self) -> bool {
        self.bias
    }
    fn coefficients(&self) -> Option<ArrayView1<'_, f64>> {
        Some(self.coefficients.as_ref()?.view())
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        LogisticRegression {
            coefficients: None,
            n_classes: 0,
            bias: true,
        }
    }

    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> Result<(), Errors> {
        let target: Vec<u64> = target.par_iter().map(|&x| x as u64).collect();
        self.n_classes = target
            .iter()
            .map(|&x| x as u64)
            .collect::<HashSet<_>>()
            .len();
        let data = if self.bias {
            &add_bias_column(data)
        } else {
            data
        };
        let mut weights = Array2::<f64>::zeros((self.n_classes, data.ncols()));
        let weight_ptr = std::sync::Arc::new(SendPtr(weights.as_mut_ptr()));
        let res: Result<(), Errors> = (0..self.n_classes)
            .into_par_iter()
            .map(|i| {
                let ptr = std::sync::Arc::clone(&weight_ptr);
                let target: Array1<f64> = target
                    .iter()
                    .map(|&v| if v == i as u64 { 1.0 } else { 0.0 })
                    .collect();
                let coef = data
                    .least_squares(&target)
                    .map_err(|err| LinearAlgebra(err))?
                    .solution;
                for (j, &c) in coef.iter().enumerate() {
                    unsafe {
                        *ptr.0.add(i * weights.ncols() + j) = c;
                    }
                }
                Ok(())
            })
            .collect();
        self.coefficients = Some(weights);
        res
    }
    fn bias(&self) -> bool {
        self.bias
    }
    fn coefficients(&self) -> Option<ArrayView1<'_, f64>> {
        self.coefficients.as_ref().map(|c| c.row(0))
        // Some(self.coefficients.as_ref().unwrap().row(0))
    }

    fn predict(&self, points: &Array2<f64>) -> Result<Array1<f64>, Errors> {
        let points = if self.bias() {
            &add_bias_column(points)
        } else {
            points
        };
        let probabilites = points.dot(&self.coefficients.as_ref().ok_or(NotFitted)?.t());
        let predictions = (0..probabilites.nrows())
            .into_par_iter()
            .map(|i| -> Result<f64, Errors> {
                let row = probabilites.row(i);
                let slice = row.as_slice().ok_or(NotFitted)?;
                let (idx, _) = argmax(slice)?;
                Ok(idx as f64)
            })
            .collect::<Result<Vec<f64>, Errors>>()?;
        Ok(Array1::from(predictions))
    }
}

fn add_bias_column(x: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = x.dim();
    let mut out = Array2::ones((nrows, ncols + 1));
    out.slice_mut(ndarray::s![.., 1..]).assign(x);
    out
}

const PMM_BACKEND: &[&str] = &["linear", "ridge"];
impl PMM {
    pub fn new(n_neighbors: usize, backend: &str, alpha: Option<f64>) -> Result<PMM, Errors> {
        let model = match backend.to_lowercase().as_str() {
            "linear" => Box::new(Solver::Linear(LinearRegression::new())),
            "ridge" => Box::new(Solver::Ridge(Ridge::new(alpha.ok_or(
                Errors::UnsupportedValue {
                    parameter: "Ridge.alpha",
                    value: "None".to_string(),
                    supported: Some(&["0.0", "1.0"]),
                },
            )?))),
            val => {
                return Err(Errors::UnsupportedValue {
                    parameter: "PMM.Backend",
                    value: val.to_string(),
                    supported: Some(PMM_BACKEND),
                });
            }
        };
        Ok(PMM {
            n_neighbors,
            pool: None,
            model: model,
            rng: Mutex::new(rand::rngs::SmallRng::seed_from_u64(42)),
        })
    }

    fn sample(&self, arr: &[f64]) -> f64 {
        if arr.is_empty() {
            return f64::NAN;
        }
        let mut rng = self
            .rng
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        arr.choose(&mut rng).unwrap().clone()
    }
    #[allow(dead_code)]
    fn bias(&self) -> bool {
        self.model.bias()
    }
    #[allow(dead_code)]
    fn coefficients(&self) -> Option<ArrayView1<'_, f64>> {
        self.model.coefficients()
    }
    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> Result<(), Errors> {
        self.pool = Some(target.iter().filter(|e| !e.is_nan()).copied().collect());
        self.model.fit(data, target)?;
        Ok(())
    }

    fn predict(&self, points: &Array2<f64>) -> Result<Array1<f64>, Errors> {
        let predictions = self.model.predict(points)?.to_vec();
        let mut samples = Array1::zeros(points.nrows());
        let pool = self.pool.as_ref().ok_or(NotFitted)?;

        let res: Result<(), Errors> = predictions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if self.n_neighbors >= pool.len() {
                    samples[i] = self.sample(&pool.to_vec());
                    Ok(())
                } else {
                    let mut top_k: Vec<f64> = vec![f64::MAX; self.n_neighbors]; //Array1::ones(self.n_neighbors) * f64::MAX;
                    let distances: Vec<f64> =
                        pool.par_iter().map(|x| (x - p).powi(2).abs()).collect();
                    let (mut max_idx, mut max_val) = argmax(&top_k)?;
                    for (j, d) in distances.iter().enumerate() {
                        if d < &(max_val - p).abs() {
                            top_k[max_idx] = pool[j];
                            (max_idx, max_val) = argmax(&top_k)?;
                        }
                    }
                    top_k.retain(|&e| e < f64::MAX);
                    samples[i] = self.sample(&top_k);
                    Ok(())
                }
            })
            .collect();
        res?;
        Ok(samples)
    }
}

impl Clone for PMM {
    fn clone(&self) -> Self {
        let rng = self
            .rng
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self {
            n_neighbors: self.n_neighbors,
            pool: self.pool.clone(),
            model: self.model.clone(),
            rng: Mutex::new(rng.clone()),
        }
    }
}

fn argmax(arr: &[f64]) -> Result<(usize, f64), Errors> {
    let (id, v) = arr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(id, v)| (id, *v))
        .ok_or(Errors::NoValidOp {
            operation: "computing argmax on empty array!".to_string(),
        })?;
    Ok((id, v.clone()))
}

#[cfg(test)]
mod test {
    use super::*; // has access to everything, including private

    #[test]
    fn log_reg_estimate() {
        const DIM: usize = 5;
        const N_POINTS: usize = 4;
        const POINTS: &[f64] = &[1.0; DIM * N_POINTS];
        const N_CLASSES: usize = 3;

        let mut model = LogisticRegression::new();
        let mut rng = rand::rng();
        model.coefficients = Some(
            Array2::from_shape_vec(
                [N_CLASSES, DIM + 1],
                (0..N_CLASSES * (DIM + 1)).map(|_| rng.random()).collect(),
            )
            .expect("Failed to create coefficients"),
        );

        let estimates = model
            .predict(&Array2::from_shape_vec((4, 5), POINTS.to_vec()).expect("Predict failed"))
            .unwrap();
        assert!(N_POINTS == estimates.len());
        for &p in &estimates {
            assert!(
                &[0, 1, 2].contains(&(p as i32)),
                "expected: [0,1,2] actual: {:?}",
                &estimates
            );
        }
    }

    #[test]
    fn estimate() {
        const DIM: usize = 5;
        const N_POINTS: usize = 4;
        const POINTS: &[f64] = &[1.0; DIM * N_POINTS];
        const TOTAL: f64 = ((DIM - 1).pow(2) + (DIM - 1)) as f64 / 2.0;
        const EXPECTED: &[f64] = &[TOTAL; N_POINTS];

        let mut model = LinearRegression::new();
        model.coefficients = Some(
            Array1::from_shape_vec(
                DIM + 1,
                (0..=DIM)
                    .map(|x| if x < DIM { x as f64 } else { 0.0 })
                    .collect(),
            )
            .unwrap(),
        );

        let estimates = model
            .predict(&Array2::from_shape_vec((4, 5), POINTS.to_vec()).unwrap())
            .unwrap();
        assert!(EXPECTED.len() == estimates.len());
        for (p, q) in EXPECTED.iter().zip(&estimates) {
            assert!(
                (p - q).abs() < 1e-12,
                "expected: {:?} actual: {:?}",
                EXPECTED,
                &estimates
            );
        }
    }

    #[test]
    fn log_reg() {
        let x = Array2::from_shape_vec((20, 5), DATA.to_owned()).unwrap();
        let y: Array1<f64> = (0..20)
            .into_iter()
            .map(|i| if i < 10 { 1.0 } else { 0.0 })
            .collect();

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();
        // let estimate = model.predict(&x);
        assert!(model.coefficients.as_ref().map_or(false, |_| true));
        let coef = model.coefficients.as_ref().unwrap();
        println!("{:?}", coef.shape());
        assert!(coef.shape() == [2, 6]);
    }

    #[test]
    fn base() {
        let x = Array2::from_shape_vec((20, 5), DATA.to_owned()).unwrap();
        let y = Array1::from_shape_vec(20, TARGET.to_owned()).unwrap();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        let estimate = model.predict(&x).unwrap();
        println!("EXPECTED:{:?}\nActual{:?}", ESTIMATES, estimate);

        let error = ESTIMATES.iter().zip(estimate).map(|(p, q)| (p - q).abs());
        for e in error {
            assert!(e < 1e-6);
        }
    }

    #[test]
    fn ridge() {
        let x = Array2::from_shape_vec((20, 5), DATA.to_owned()).unwrap();
        let y = Array1::from_shape_vec(20, TARGET.to_owned()).unwrap();

        let mut model = Ridge::new(1.0);
        model.fit(&x, &y).unwrap();
        let estimate = model.predict(&x).unwrap();
        println!("EXPECTED:{:?}\nActual{:?}", RIDGE_ESTIMATE, estimate);

        let error = RIDGE_ESTIMATE
            .iter()
            .zip(estimate)
            .map(|(p, q)| (p - q).abs());
        for e in error {
            assert!(e < 1e-6);
        }
    }
    // shape (20, 5)
    const DATA: &[f64] = &[
        0.1503786, 0.91486817, 0.41649195, 0.7272033, 0.60286392, 0.12014579, 0.29518488,
        0.63478448, 0.46252395, 0.04233139, 0.65703806, 0.89071615, 0.49315059, 0.07182519,
        0.53814632, 0.53678706, 0.22718179, 0.52712324, 0.97307241, 0.61094238, 0.84030169,
        0.88514941, 0.3169348, 0.73315702, 0.58027396, 0.74525453, 0.44986027, 0.0393748,
        0.79530909, 0.78081087, 0.04067423, 0.28115197, 0.18434019, 0.93862817, 0.13327936,
        0.26004549, 0.60633788, 0.13466531, 0.48031966, 0.80228352, 0.40351739, 0.45063986,
        0.33203147, 0.17792964, 0.10084752, 0.69627007, 0.0549766, 0.21736543, 0.36073052,
        0.10840619, 0.8208764, 0.7387408, 0.15913283, 0.72609655, 0.61938197, 0.96613088,
        0.31940145, 0.56719364, 0.8310304, 0.63521956, 0.69346598, 0.19195907, 0.41359075,
        0.23997107, 0.78377297, 0.8027091, 0.67336775, 0.98602129, 0.19434108, 0.83498248,
        0.29357372, 0.44849007, 0.96261133, 0.5073289, 0.06963507, 0.95421079, 0.2498412,
        0.70698228, 0.4911054, 0.19317823, 0.51033988, 0.95581108, 0.7480978, 0.41532522,
        0.88782299, 0.43579289, 0.42107511, 0.21779886, 0.97376939, 0.87835809, 0.1854952,
        0.55386345, 0.75460027, 0.77770873, 0.182196, 0.13468564, 0.90096006, 0.64937967,
        0.82626406, 0.17179001,
    ];
    const TARGET: &[f64] = &[
        87.0, 16.0, 47.0, 61.0, 28.0, 8.0, 7.0, 14.0, 28.0, 59.0, 72.0, 53.0, 19.0, 11.0, 69.0,
        21.0, 57.0, 13.0, 33.0, 89.0,
    ];
    const ESTIMATES: &[f64] = &[
        52.19404633,
        35.67399759,
        44.9243519,
        31.92695929,
        59.76689661,
        28.74459203,
        36.27739888,
        22.43229942,
        33.78329041,
        21.40618744,
        48.72535239,
        38.69311356,
        7.21816035,
        37.73900914,
        50.62444755,
        42.30758618,
        47.63529625,
        27.12733115,
        54.39921427,
        70.40046928,
    ];
    const RIDGE_ESTIMATE: &[f64] = &[
        48.14373033,
        39.55579066,
        42.19152601,
        35.76898899,
        47.06610105,
        31.88531762,
        38.47347289,
        32.64043659,
        36.27600141,
        27.68503359,
        41.30167422,
        36.76651186,
        23.56929593,
        40.19006067,
        47.02423903,
        38.19606527,
        46.00349242,
        33.76244831,
        48.86090718,
        56.63890597,
    ];
}
