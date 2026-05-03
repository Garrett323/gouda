use crate::imputer::Mice;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{LeastSquaresSvd, SVD};

pub enum SolverType {
    Linear,
    Ridge,
    Bayesian,
}

pub trait Solver {
    fn bias(&self) -> bool;
    fn coefficients(&self) -> &Option<Array1<f64>>;
    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> &Self;
    fn predict(&self, points: &Array2<f64>) -> Array1<f64> {
        let points = if self.bias() {
            &add_bias_column(points)
        } else {
            points
        };
        let weights = self.coefficients().as_ref().unwrap();
        points.dot(weights)
    }
}

struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    bias: bool,
}

struct Ridge {
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
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficients: None,
            bias: true,
            // dim: dim + 1,
        }
    }
}

impl Solver for LinearRegression {
    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> &Self {
        let data = if self.bias {
            &add_bias_column(data)
        } else {
            data
        };
        self.coefficients = Some(data.least_squares(&target).unwrap().solution);
        self
    }
    fn bias(&self) -> bool {
        self.bias
    }
    fn coefficients(&self) -> &Option<Array1<f64>> {
        &self.coefficients
    }
}

impl Solver for Ridge {
    fn fit(&mut self, data: &Array2<f64>, target: &Array1<f64>) -> &Self {
        let x_mean = data.mean_axis(Axis(0)).unwrap(); // original mean, keep this
        let data = data - &x_mean;

        let (u, e, v) = data.svd(true, true).unwrap();
        let u = u.unwrap();
        let u = u.slice(ndarray::s![.., ..e.len()]);
        let v = v.unwrap();
        let v = v.slice(ndarray::s![..e.len(), ..]);

        let d = &e / (&e.mapv(|x| x * x) + self.alpha);
        let uty = u.t().dot(target);
        let mut beta = v.t().dot(&(d * uty));

        if self.bias {
            let y_mean = target.mean().unwrap(); // move up here too
            let intercept = Array1::from_elem(1, y_mean - x_mean.dot(&beta));
            beta = ndarray::concatenate(Axis(0), &[intercept.view(), beta.view()]).unwrap();
        }

        self.coefficients = Some(beta);
        self
    }
    fn bias(&self) -> bool {
        self.bias
    }
    fn coefficients(&self) -> &Option<Array1<f64>> {
        &self.coefficients
    }
}

fn add_bias_column(x: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = x.dim();
    let mut out = Array2::ones((nrows, ncols + 1));
    out.slice_mut(ndarray::s![.., 1..]).assign(x);
    out
}

#[cfg(test)]
mod test {
    use super::*; // has access to everything, including private

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

        let estimates = model.predict(&Array2::from_shape_vec((4, 5), POINTS.to_vec()).unwrap());
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
    fn base() {
        let x = Array2::from_shape_vec((20, 5), DATA.to_owned()).unwrap();
        let y = Array1::from_shape_vec(20, TARGET.to_owned()).unwrap();

        let estimate = LinearRegression::new().fit(&x, &y).predict(&x);
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

        let estimate = Ridge::new(1.0).fit(&x, &y).predict(&x);
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
